import argparse
import itertools
import json
import logging
import pathlib
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression

from horizon.plot.logistic import fit_trendline
from horizon.wrangle.bootstrap import bootstrap_sample
from horizon.wrangle.logistic import run_logistic_regressions


def reg_to_dict(reg: LinearRegression) -> Dict[str, float]:
    assert len(reg.coef_) == 1
    assert isinstance(reg.intercept_, float)
    return {
        "coef": float(reg.coef_[0]),
        "intercept": float(reg.intercept_),
    }


def process_agent_summaries(
    agent_summaries: pd.DataFrame, fig_params: Dict[str, Any]
) -> pd.DataFrame:
    agent_summaries = agent_summaries[agent_summaries["agent"] != "human"]
    agent_summaries = agent_summaries[
        agent_summaries["agent"].isin(fig_params["include_agents"])
    ]
    return agent_summaries


def bootstrap_models(
    agent_summaries: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Resamples all the models with replacement"""
    return agent_summaries.sample(
        n=len(agent_summaries), replace=True, random_state=rng
    )


def fit_trendline_to_runs(
    df_runs: pd.DataFrame,
    release_dates: pathlib.Path,
    fig_params: Dict[str, Any],
    wrangle_params: dict[str, Any],
    bootstrap_over_models: bool = False,
    rng: np.random.Generator | None = None,
    only_2024_agents: bool = False,
) -> Dict[str, float]:
    agent_summaries = run_logistic_regressions(
        df_runs,
        release_dates,
        wrangle_params,  # type: ignore
        include_empirical_rates=False,
        # We dont' ensure that weights sum to 1.0 when we bootstrap over models, as we don't renormalize weights
        # here, nor expect them to sum to 1.0
        ensure_weights_sum_to_1=False,
    )
    agent_summaries = process_agent_summaries(agent_summaries, fig_params)
    if only_2024_agents:
        agent_summaries = agent_summaries[
            agent_summaries["agent"].isin(fig_params["agents_2024"])
        ]
    if bootstrap_over_models:
        assert rng is not None
        agent_summaries = bootstrap_models(agent_summaries, rng)
    reg, _ = fit_trendline(
        agent_summaries["p50"],
        pd.to_datetime(agent_summaries["release_date"]),
        log_scale=True,
    )
    return reg_to_dict(reg)


def process_in_parallel(
    func: Callable[..., Dict[str, float]],
    n_samples: int,
    gen: np.random.Generator,
    n_jobs: int = -1,
    batch_size: int = 10,
    **kwargs: Any,
) -> List[Dict[str, float]]:
    """Process a function in parallel with batching.

    Args:
        func: Function to process in parallel
        n_samples: Number of samples to process
        gen: Random number generator to derive seeds from
        n_jobs: Number of jobs to run in parallel
        batch_size: Number of samples to process per batch
        **kwargs: Additional arguments to pass to func

    Returns:
        List of results from func
    """
    n_batches = (n_samples + batch_size - 1) // batch_size  # Round up division
    seeds = gen.integers(0, 2**31, size=n_samples)

    def process_batch(batch_idx: int) -> List[Dict[str, float]]:
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        batch_results = []
        for i in range(start_idx, end_idx):
            rng = np.random.default_rng(seeds[i])
            result = func(i, rng, **kwargs)
            batch_results.append(result)
        return batch_results

    # Run parallel computations in batches
    batched_results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(process_batch)(i) for i in range(n_batches)
    )
    assert all(isinstance(batch, list) for batch in batched_results)
    # Flatten results
    return [result for batch in batched_results for result in batch]  # type: ignore


def get_bootstrap_models_records(
    agent_summaries: pd.DataFrame, rng: np.random.Generator, n_samples: int = 100
) -> List[Dict[str, float]]:
    def process_sample(
        idx: int, rng: np.random.Generator, agent_summaries: pd.DataFrame
    ) -> Dict[str, float]:
        df = bootstrap_models(agent_summaries, rng)
        reg, score = fit_trendline(
            df["p50"],
            pd.to_datetime(df["release_date"]),
            log_scale=True,
        )
        return reg_to_dict(reg)

    return process_in_parallel(
        process_sample, n_samples, rng, agent_summaries=agent_summaries
    )


def get_weighting_regularization_records(
    df_runs: pd.DataFrame,
    release_dates: pathlib.Path,
    fig_params: Dict[str, Any],
    gen: np.random.Generator,
) -> List[Dict[str, float]]:
    combinations = list(
        itertools.product(fig_params["weightings"], fig_params["regularizations"])
    )
    n_samples = len(combinations)

    def process_sample(
        idx: int,
        rng: np.random.Generator,
        df_runs: pd.DataFrame,
        release_dates: pathlib.Path,
        fig_params: Dict[str, Any],
        combinations: list[tuple[str, float]],
    ) -> Dict[str, float]:
        weighting, regularization = combinations[idx]
        wrangle_params = {
            "weighting": weighting,
            "regularization": regularization,
            "exclude": None,
            "success_percents": [50],
            "confidence_level": 0.95,
        }
        return fit_trendline_to_runs(df_runs, release_dates, fig_params, wrangle_params)

    return process_in_parallel(
        process_sample,
        n_samples,
        gen,
        df_runs=df_runs,
        release_dates=release_dates,
        fig_params=fig_params,
        combinations=combinations,
    )


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
    # Generate one noise value per unique task ID
    unique_task_ids = noisy_runs["task_id"].unique()
    task_noise = {
        task_id: gen.normal(0, task_se_dict[task_id]) for task_id in unique_task_ids
    }

    # Apply noise by mapping task IDs to their noise values
    noise = noisy_runs["task_id"].map(task_noise)
    assert not any(noise.isna())

    log_minutes = np.log(noisy_runs["human_minutes"])
    noisy_runs["human_minutes"] = np.exp(log_minutes + noise)
    assert not any(noisy_runs["human_minutes"].isna())
    return noisy_runs


def get_baseline_noise_records(
    df_runs: pd.DataFrame,
    release_dates: pathlib.Path,
    fig_params: Dict[str, Any],
    gen: np.random.Generator,
) -> List[Dict[str, float]]:
    """Calculate noise-based uncertainty in human performance times and generate records."""
    task_se_dict = find_baseline_time_se(df_runs)

    def process_sample(
        idx: int,
        rng: np.random.Generator,
        df_runs: pd.DataFrame,
        release_dates: pathlib.Path,
        fig_params: Dict[str, Any],
        task_se_dict: dict[str, float],
    ) -> Dict[str, float]:
        noisy_runs = add_noise_to_runs(df_runs, task_se_dict, rng)
        wrangle_params = {
            "weighting": "invsqrt_task_weight",
            "regularization": 0.1,
            "exclude": None,
            "success_percents": [50],
            "confidence_level": 0.95,
        }
        return fit_trendline_to_runs(
            noisy_runs,
            release_dates,
            fig_params,
            wrangle_params,
            bootstrap_over_models=False,
        )

    return process_in_parallel(
        process_sample,
        fig_params["n_bootstrap"],
        gen,
        df_runs=df_runs,
        release_dates=release_dates,
        fig_params=fig_params,
        task_se_dict=task_se_dict,
    )


def get_bootstrap_records(
    df_runs: pd.DataFrame,
    gen: np.random.Generator,
    fig_params: Dict[str, Any],
    release_dates: pathlib.Path,
    categories: list[str],
) -> List[Dict[str, float]]:
    def process_sample(
        idx: int,
        rng: np.random.Generator,
        df_runs: pd.DataFrame,
        fig_params: Dict[str, Any],
        release_dates: pathlib.Path,
        categories: list[str],
    ) -> Dict[str, float]:
        sampled_runs = bootstrap_sample(df_runs, categories, rng)
        wrangle_params = {
            "weighting": rng.choice(fig_params["weightings"]),
            "regularization": rng.choice(fig_params["regularizations"]),
            "exclude": None,
            "success_percents": [50],
            "confidence_level": 0.95,
        }
        return fit_trendline_to_runs(
            sampled_runs,
            release_dates,
            fig_params,
            wrangle_params,
            bootstrap_over_models=True,
            rng=rng,
        )

    return process_in_parallel(
        process_sample,
        fig_params["n_bootstrap"],
        gen,
        df_runs=df_runs,
        fig_params=fig_params,
        release_dates=release_dates,
        categories=categories,
    )


def get_total_records(
    df_runs: pd.DataFrame,
    release_dates: pathlib.Path,
    fig_params: Dict[str, Any],
    gen: np.random.Generator,
    only_2024_agents: bool = False,
) -> List[Dict[str, float]]:
    categories = ["task_family", "task_id", "run_id"]
    task_se_dict = find_baseline_time_se(df_runs)

    def process_sample(
        idx: int,
        rng: np.random.Generator,
        df_runs: pd.DataFrame,
        release_dates: pathlib.Path,
        fig_params: Dict[str, Any],
        categories: list[str],
        task_se_dict: dict[str, float],
    ) -> Dict[str, float]:
        noisy_runs = add_noise_to_runs(df_runs, task_se_dict, rng)
        sampled_runs = bootstrap_sample(noisy_runs, categories, rng)
        wrangle_params = {
            "weighting": rng.choice(fig_params["weightings"]),
            "regularization": rng.choice(fig_params["regularizations"]),
            "exclude": None,
            "success_percents": [50],
            "confidence_level": 0.95,
        }
        return fit_trendline_to_runs(
            sampled_runs,
            release_dates,
            fig_params,
            wrangle_params,
            bootstrap_over_models=True,
            rng=rng,
            only_2024_agents=only_2024_agents,
        )

    return process_in_parallel(
        process_sample,
        fig_params["n_bootstrap"],
        gen,
        df_runs=df_runs,
        release_dates=release_dates,
        fig_params=fig_params,
        categories=categories,
        task_se_dict=task_se_dict,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-file", type=pathlib.Path, required=True)
    parser.add_argument("--logistic-fits-file", type=pathlib.Path, required=True)
    parser.add_argument("--release-dates-file", type=pathlib.Path, required=True)
    parser.add_argument("--output-records-file", type=pathlib.Path, required=True)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for noise generation"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Load parameters from DVC
    import dvc.api

    params = dvc.api.params_show(stages="wrangle_multiverse_boxplot")
    fig_params = params["figs"]["plot_multiverse_boxplot"]
    rng = np.random.default_rng(args.seed)

    # Load data
    runs = pd.read_json(args.runs_file, lines=True)
    runs.rename(columns={"alias": "agent"}, inplace=True)
    agent_summaries = pd.read_csv(args.logistic_fits_file)
    agent_summaries = process_agent_summaries(agent_summaries, fig_params)

    # Generate all the different types of records
    bootstrap_models_records = get_bootstrap_models_records(
        agent_summaries, rng, n_samples=fig_params["n_bootstrap"]
    )
    bootstrap_task_records = get_bootstrap_records(
        df_runs=runs,
        gen=rng,
        fig_params=fig_params,
        release_dates=args.release_dates_file,
        categories=["task_family", "task_id"],
    )
    bootstrap_run_records = get_bootstrap_records(
        df_runs=runs,
        gen=rng,
        fig_params=fig_params,
        release_dates=args.release_dates_file,
        categories=["run_id"],
    )
    weightings_regularizations_records = get_weighting_regularization_records(
        df_runs=runs,
        release_dates=args.release_dates_file,
        fig_params=fig_params,
        gen=rng,
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
        only_2024_agents=False,
    )
    total_2024_records = get_total_records(
        df_runs=runs,
        release_dates=args.release_dates_file,
        fig_params=fig_params,
        gen=rng,
        only_2024_agents=True,
    )

    # Organize records by type
    record_dict = {
        "Overall (2024-2025 trend)": total_2024_records,
        "Overall (2019-2025 trend)": total_records,
        "IID Baseline Noise": baseline_noise_records,
        "Weighting/Regularization": weightings_regularizations_records,
        "Bootstrap (models)": bootstrap_models_records,
        "Bootstrap (runs)": bootstrap_run_records,
        "Bootstrap (tasks)": bootstrap_task_records,
    }

    # Save records to JSON file
    args.output_records_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_records_file, "w") as f:
        json.dump(record_dict, f)

    logging.info(f"Saved records to {args.output_records_file}")


if __name__ == "__main__":
    main()
