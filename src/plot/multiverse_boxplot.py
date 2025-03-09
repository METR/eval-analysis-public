import argparse
import itertools
import logging
import pathlib
from typing import Any, TypedDict

from collections import defaultdict
import dvc.api
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.dates import date2num, num2date
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import yaml

import src.utils.plots
from src.plot.logistic import fit_trendline
from src.wrangle.bootstrap import bootstrap_sample
from src.wrangle.logistic import run_logistic_regressions


class MultiverseParams(TypedDict):
    weightings: list[str]
    regularizations: list[float]
    include_agents: list[str]


class MultiverseRecord(TypedDict):
    coef: float
    intercept: float


def reg_to_dict(reg: LinearRegression) -> MultiverseRecord:
    assert len(reg.coef_) == 1
    assert isinstance(reg.intercept_, float)
    return MultiverseRecord(
        coef=reg.coef_[0],
        intercept=reg.intercept_,
    )


def process_agent_summaries(
    agent_summaries: pd.DataFrame, fig_params: MultiverseParams
) -> pd.DataFrame:
    agent_summaries = agent_summaries[agent_summaries["agent"] != "human"]
    agent_summaries = agent_summaries[
        agent_summaries["agent"].isin(fig_params["include_agents"])
    ]
    return agent_summaries


def leave_one_out_random(
    agent_summaries: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Drops a random row from the dataframe, for combined analysis"""
    idx = rng.choice(agent_summaries.index)
    return agent_summaries.drop(idx)


def fit_trendline_to_runs(
    df_runs: pd.DataFrame,
    release_dates: pathlib.Path,
    fig_params: MultiverseParams,
    wrangle_params: dict[str, Any],
    leave_one_out: bool = False,
    rng: np.random.Generator | None = None,
) -> MultiverseRecord:
    agent_summaries = run_logistic_regressions(
        df_runs,
        release_dates,
        wrangle_params,  # type: ignore
    )
    agent_summaries = process_agent_summaries(agent_summaries, fig_params)
    if leave_one_out:
        assert rng is not None
        agent_summaries = leave_one_out_random(agent_summaries, rng)
    reg, _ = fit_trendline(
        agent_summaries["p50"],
        pd.to_datetime(agent_summaries["release_date"]),
        log_scale=True,
    )
    return reg_to_dict(reg)


def get_leave_one_out_records(agent_summaries: pd.DataFrame) -> list[MultiverseRecord]:
    records = []
    for row in agent_summaries.index:
        df = agent_summaries.drop(row)
        reg, score = fit_trendline(
            df["p50"],
            pd.to_datetime(df["release_date"]),
            log_scale=True,
        )
        records.append(reg_to_dict(reg))
    return records


def get_weighting_regularization_records(
    df_runs: pd.DataFrame,
    release_dates: pathlib.Path,
    fig_params: MultiverseParams,
) -> list[MultiverseRecord]:
    records = []
    for weighting, regularization in tqdm(
        itertools.product(fig_params["weightings"], fig_params["regularizations"])
    ):
        wrangle_params = {
            "weighting": weighting,
            "regularization": regularization,
            "exclude": None,
            "success_percents": [50],
        }
        record = fit_trendline_to_runs(
            df_runs, release_dates, fig_params, wrangle_params
        )
        records.append(record)
    return records


def find_baseline_time_se(df_runs: pd.DataFrame) -> dict[str, float]:  # type: ignore
    human_runs = df_runs[df_runs["agent"] == "human"].copy()
    human_runs = human_runs[human_runs["score_binarized"] == 1]
    human_runs = human_runs[human_runs["completed_at"] > 0]

    human_runs["log_run_minutes"] = np.log(human_runs["completed_at"] / (60 * 1000))
    # Get mean log minutes for each task
    task_means = human_runs.groupby("task_id")["log_run_minutes"].mean()
    logging.info(f"task means: {task_means}")
    # Calculate deviations from task means
    human_runs["log_deviation"] = human_runs.apply(
        lambda row: row["log_run_minutes"] - task_means[row["task_id"]], axis=1
    )
    assert not any(human_runs["log_deviation"].isna())
    logging.info(f"log deviation: {human_runs['log_deviation'].describe()}")

    # Calculate pooled standard deviation with sample size correction, ignoring tasks with only 1 baseline
    task_counts = human_runs.groupby("task_id").size()
    multi_baseline_tasks = task_counts[task_counts > 1].index
    multi_baseline_runs = human_runs[human_runs["task_id"].isin(multi_baseline_tasks)]
    n = len(multi_baseline_runs)
    pooled_std = np.sqrt(np.sum(multi_baseline_runs["log_deviation"] ** 2) / (n - 1))

    logging.info(f"Pooled standard deviation: factor of {np.exp(pooled_std)}x")

    # Create task stats with standard error
    task_stats = human_runs.groupby("task_id").size().reset_index()
    task_stats.columns = ["task_id", "run_count"]
    missing_task_ids = set(df_runs["task_id"].unique()) - set(task_stats["task_id"])
    # Missing tasks estimated to have 1.44x the sd of a baseline
    missing_tasks = pd.DataFrame({"task_id": list(missing_task_ids), "run_count": 0.5})
    # Concatenate with existing task_stats
    task_stats = pd.concat([task_stats, missing_tasks], ignore_index=True)
    task_stats["log_se"] = pooled_std / np.sqrt(task_stats["run_count"])
    median_se = task_stats["log_se"].median()
    task_stats["log_se"] = task_stats["log_se"].fillna(median_se)

    assert not any(task_stats["log_se"].isna())
    return dict(zip(task_stats["task_id"], task_stats["log_se"]))


def add_noise_to_runs(
    df_runs: pd.DataFrame, task_se_dict: dict[str, float], gen: np.random.Generator
) -> pd.DataFrame:
    noisy_runs = df_runs.copy()
    # Vectorized noise application
    task_ses = noisy_runs["task_id"].map(task_se_dict)
    assert not any(task_ses.isna())
    noise = gen.normal(0, task_ses, size=len(noisy_runs))

    log_minutes = np.log(noisy_runs["human_minutes"])
    noisy_runs["human_minutes"] = np.exp(log_minutes + noise)
    assert not any(noisy_runs["human_minutes"].isna())
    return noisy_runs


def get_baseline_noise_records(
    df_runs: pd.DataFrame,
    release_dates: pathlib.Path,
    fig_params: MultiverseParams,
    gen: np.random.Generator,
) -> list[MultiverseRecord]:
    """Calculate noise-based uncertainty in human performance times and generate records."""
    records = []
    n_samples = 100

    task_se_dict = find_baseline_time_se(df_runs)

    for _ in range(n_samples):
        noisy_runs = add_noise_to_runs(df_runs, task_se_dict, gen)

        wrangle_params = {
            "weighting": "invsqrt_task_weight",
            "regularization": 0.1,
            "exclude": None,
            "success_percents": [50],
        }
        record = fit_trendline_to_runs(
            noisy_runs, release_dates, fig_params, wrangle_params, leave_one_out=False
        )
        records.append(record)

    return records


def get_bootstrap_records(
    bootstrap_file: pathlib.Path,
    fig_params: MultiverseParams,
    s_release_dates: pd.Series,  # type: ignore
) -> list[MultiverseRecord]:
    bootstrap_results = pd.read_csv(bootstrap_file)
    agent_cols = [col for col in bootstrap_results.columns if col.endswith("_p50")]
    included_cols = [
        col
        for col in agent_cols
        if col.replace("_p50", "") in fig_params["include_agents"]
    ]
    assert len(included_cols) == len(s_release_dates)

    bootstrap_results = bootstrap_results[included_cols]
    records = []
    for _, row in bootstrap_results.iterrows():
        horizons = pd.Series(row)
        if horizons.isna().any():
            continue
        reg, _ = fit_trendline(
            horizons,
            s_release_dates,
            log_scale=True,
        )
        records.append(reg_to_dict(reg))
    return records


def get_total_records(
    df_runs: pd.DataFrame,
    release_dates: pathlib.Path,
    fig_params: MultiverseParams,
    gen: np.random.Generator,
) -> list[MultiverseRecord]:
    records = []
    n_samples = 100

    category_dict = {
        "f": "task_family",
        "t": "task_id",
        "r": "run_id",
    }
    categories = list(category_dict.values())
    task_se_dict = find_baseline_time_se(df_runs)
    for _ in range(n_samples):
        noisy_runs = add_noise_to_runs(df_runs, task_se_dict, gen)
        sampled_runs = bootstrap_sample(noisy_runs, categories, gen)
        wrangle_params = {
            "weighting": gen.choice(fig_params["weightings"]),
            "regularization": gen.choice(fig_params["regularizations"]),
            "exclude": None,
            "success_percents": [50],
        }
        record = fit_trendline_to_runs(
            sampled_runs,
            release_dates,
            fig_params,
            wrangle_params,
            leave_one_out=True,
            rng=gen,
        )
        records.append(record)
    return records


def log_metrics(df_records: pd.DataFrame, output_metrics_file: pathlib.Path) -> None:
    metrics = defaultdict(dict)
    for key in df_records["record_type"].unique():
        df_filtered = df_records[df_records["record_type"] == key]
        ci_width = df_filtered["predicted_date"].quantile(0.9) - df_filtered[
            "predicted_date"
        ].quantile(0.1)
        ci_width_years = round(ci_width.days / 365.25, 3)
        metrics["predicted_date"][key.replace("\n", " ")] = {
            "ci_width_years": ci_width_years,
            "q10": df_filtered["predicted_date"].quantile(0.1),
            "q50": df_filtered["predicted_date"].quantile(0.5),
            "q90": df_filtered["predicted_date"].quantile(0.9),
        }
    with open(output_metrics_file, "w") as f:
        yaml.dump(dict(metrics), f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-file", type=pathlib.Path, required=True)
    parser.add_argument("--logistic-fits-file", type=pathlib.Path, required=True)
    parser.add_argument("--bootstrap-file", type=pathlib.Path, required=True)
    parser.add_argument("--release-dates-file", type=pathlib.Path, required=True)
    parser.add_argument("--output-file", type=pathlib.Path, required=True)
    parser.add_argument("--output-metrics-file", type=pathlib.Path, required=True)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for noise generation"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    params = dvc.api.params_show(stages="plot_multiverse_boxplot")
    fig_params = params["figs"]["plot_multiverse_boxplot"]
    rng = np.random.default_rng(args.seed)

    runs = pd.read_json(args.runs_file, lines=True)
    agent_summaries = pd.read_csv(args.logistic_fits_file)
    agent_summaries = process_agent_summaries(agent_summaries, fig_params)

    leave_one_out_records = get_leave_one_out_records(agent_summaries)
    bootstrap_records = get_bootstrap_records(
        bootstrap_file=args.bootstrap_file,
        fig_params=fig_params,
        s_release_dates=pd.to_datetime(agent_summaries["release_date"]),
    )
    weightings_regularizations_records = get_weighting_regularization_records(
        df_runs=runs,
        release_dates=args.release_dates_file,
        fig_params=fig_params,
    )
    baseline_noise_records = get_baseline_noise_records(
        df_runs=runs,
        release_dates=args.release_dates_file,
        fig_params=fig_params,
        gen=rng,
    )
    total_records = get_total_records(
        df_runs=runs,
        release_dates=args.release_dates_file,
        fig_params=fig_params,
        gen=rng,
    )

    record_dict = {
        "Total": total_records,
        "IID Baseline\nNoise": baseline_noise_records,
        "Weighting/\nRegularization": weightings_regularizations_records,
        "Leave-one-out": leave_one_out_records,
        "Bootstrap": bootstrap_records,
    }

    records_df = pd.DataFrame(
        [
            {**record, "record_type": record_type}
            for record_type, records in record_dict.items()
            for record in records
        ]
    )

    # Create figure and axis
    plt.figure(figsize=(10, 6))

    # Convert coefficients to predicted dates for 1 month (167 hours)
    target_minutes = 167 * 60
    records_df["predicted_date"] = records_df.apply(
        lambda row: num2date((np.log(target_minutes) - row["intercept"]) / row["coef"]),
        axis=1,
    ).dt.date
    records_df["date_num"] = date2num(records_df["predicted_date"])

    colors = ["#3498db", "#2ecc71", "#9b59b6", "#e74c3c"]
    boxplot = plt.boxplot(
        [
            records_df[records_df["record_type"] == rt]["date_num"]
            for rt in record_dict.keys()
        ],
        vert=False,
        whis=(10, 90),
        patch_artist=True,
        meanline=False,
        showmeans=False,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 2},
        boxprops={"alpha": 0.8},
        whiskerprops={"linewidth": 1.5, "linestyle": "-"},
        capprops={"linewidth": 1.5},
    )

    for box, color in zip(boxplot["boxes"], colors):
        box.set(facecolor=color)

    # plt.figure(plt.gcf().number).patch.set_alpha(0.0)
    # plt.gca().patch.set_alpha(0.9)

    plt.yticks(
        range(1, len(record_dict.keys()) + 1),
        labels=list(record_dict.keys()),
        fontsize=12,
        fontweight="bold",
    )

    plt.gca().xaxis_date()
    plt.gcf().autofmt_xdate()

    plt.xlabel("Projected date of 1-month AI")
    plt.title("Uncertainty in projected date of 1-month AI")

    # Add grid for easier visualization
    plt.grid(True, axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()

    src.utils.plots.save_or_open_plot(args.output_file, params["plot_format"])

    log_metrics(records_df, args.output_metrics_file)
    logging.info(f"Logged metrics to {args.output_metrics_file}")


if __name__ == "__main__":
    main()
