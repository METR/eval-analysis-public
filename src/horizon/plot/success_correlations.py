import argparse
import datetime
import pathlib
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy.special import expit

BAD_AGENTS = ["gpt2", "davinci-002", "gpt-3.5-turbo-instruct"]


def _load_release_dates(release_dates_path: pathlib.Path) -> dict[str, str]:
    with open(release_dates_path, "r") as f:
        return yaml.safe_load(f)["date"]


def _filter_out_agents(runs_df: pd.DataFrame, agents: list[str]) -> pd.DataFrame:
    return runs_df[~runs_df["alias"].isin(agents)]


def _predict_success_rate(
    coefficient: float, intercept: float, human_minutes: int
) -> float:
    y = expit(coefficient * np.log2(human_minutes) + intercept)
    return y


def _get_predicted_success_rates(
    all_runs: pd.DataFrame, logistic_fits: pd.DataFrame
) -> pd.DataFrame:
    task_ids = all_runs["task_id"].unique()
    aliases = all_runs["alias"].unique()

    predicted_success_rates = pd.DataFrame(index=task_ids, columns=aliases, dtype=float)

    for alias in aliases:
        agent_info = logistic_fits[logistic_fits["agent"] == alias]
        for task_id in task_ids:
            human_minutes = all_runs[all_runs["task_id"] == task_id][
                "human_minutes"
            ].values[0]
            predicted_success_rates.loc[task_id, alias] = _predict_success_rate(
                agent_info.iloc[0]["coefficient"],
                agent_info.iloc[0]["intercept"],
                human_minutes,
            )
    return predicted_success_rates


def _get_observed_success_rates(all_runs: pd.DataFrame) -> pd.DataFrame:
    return all_runs.groupby(["task_id", "alias"])["score_binarized"].mean().unstack()


def _make_average_and_sort(
    df: pd.DataFrame, release_dates: dict[str, str]
) -> pd.DataFrame:
    # Sort all columns except 'average' by release date
    cols = [col for col in df.columns if col != "average"]
    sorted_cols = sorted(cols, key=lambda x: release_dates[x])

    # Calculate and add average column
    df["average"] = df.mean(axis=1)

    # Reorder columns with average first
    df = df[["average"] + sorted_cols]

    # Sort rows by average
    df = df.sort_values(by="average", ascending=False)

    return df


def _make_excess_success_rates(
    observed_success_rates: pd.DataFrame,
    predicted_success_rates: pd.DataFrame,
    release_dates: dict[str, str],
    output_data_dir: pathlib.Path,
) -> pd.DataFrame:
    excess_success_rates = observed_success_rates - predicted_success_rates
    excess_success_rates = _make_average_and_sort(excess_success_rates, release_dates)
    excess_success_rates.to_csv(output_data_dir / "excess_success_rates.csv")
    return excess_success_rates


def _make_fractional_excess_success_rates(
    observed_success_rates: pd.DataFrame,
    predicted_success_rates: pd.DataFrame,
    release_dates: dict[str, str],
    output_data_dir: pathlib.Path,
) -> pd.DataFrame:
    fractional_excess_success_rates = (
        observed_success_rates - predicted_success_rates
    ) / (predicted_success_rates)
    # Replace -inf with nan
    fractional_excess_success_rates = fractional_excess_success_rates.replace(
        [np.inf, -np.inf], np.nan
    )
    fractional_excess_success_rates = _make_average_and_sort(
        fractional_excess_success_rates, release_dates
    )
    fractional_excess_success_rates.to_csv(
        output_data_dir / "fractional_excess_success_rates.csv"
    )
    return fractional_excess_success_rates


def _sort_cols_by_release_dates(
    df: pd.DataFrame, release_dates: dict[str, str]
) -> pd.DataFrame:
    DEFAULT_DATE = datetime.date(2100, 1, 1)
    return df[
        sorted(
            df.columns,
            key=lambda x: release_dates[x] if x != "average" else DEFAULT_DATE,
        )
    ]


def _corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    # Calculate correlation matrix between agents (excluding 'average' column)
    agent_columns = [col for col in df.columns if col != "average"]
    correlation_matrix = df[agent_columns].corr()
    return correlation_matrix


def _plot_corr_matrix(
    corr_matrix: pd.DataFrame, name: str, output_plots_dir: pathlib.Path
) -> None:
    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, ax=ax)
    nice_name = name.replace("_", " ").title()
    average_corr = _get_average_correlation(corr_matrix)
    ax.set_title(
        f"Correlation of {nice_name} Between Agents\nAverage Correlation: {average_corr:.2f}"
    )
    fig.tight_layout()
    print(
        f"Saving {name} correlations to {output_plots_dir / f'{name}_correlations.png'}"
    )
    fig.savefig(output_plots_dir / f"{name}_correlations.png", dpi=300)


def _get_average_absolute_correlation(corr_matrix: pd.DataFrame) -> float:
    return corr_matrix.abs().mean().mean()


def _get_average_correlation(corr_matrix: pd.DataFrame) -> float:
    return corr_matrix.mean().mean()


def _print_correlation_stats(
    corr_matrix: pd.DataFrame, name: str, without_bad_agents_str: bool = False
) -> None:
    if without_bad_agents_str:
        print(f"Without {BAD_AGENTS}")
    average_abs_corr = _get_average_absolute_correlation(corr_matrix)
    print(f"Average absolute correlation for {name}: {average_abs_corr}")
    average_corr = _get_average_correlation(corr_matrix)
    print(f"Average correlation for {name}: {average_corr}")


def main(
    runs_file: pathlib.Path,
    release_dates_file: pathlib.Path,
    logistic_file: pathlib.Path,
    output_plots_dir: pathlib.Path,
    output_data_dir: pathlib.Path,
    exclude_agents: List[str],
) -> None:
    all_runs = pd.read_json(runs_file, lines=True)
    # all_runs = _fill_missing_gpt2_runs(all_runs[all_runs["alias"] != "human"])
    all_runs = _filter_out_agents(all_runs, ["human", "o3-mini"] + exclude_agents)

    release_dates = _load_release_dates(release_dates_file)
    logistic_fits = pd.read_csv(logistic_file)

    observed_success_rates = _get_observed_success_rates(all_runs=all_runs)
    predicted_success_rates = _get_predicted_success_rates(
        all_runs=all_runs,
        logistic_fits=logistic_fits,
    )

    excess_success_rates = _make_excess_success_rates(
        observed_success_rates=observed_success_rates,
        predicted_success_rates=predicted_success_rates,
        release_dates=release_dates,
        output_data_dir=output_data_dir,
    )
    excess_success_rates = _sort_cols_by_release_dates(
        df=excess_success_rates, release_dates=release_dates
    )
    name = "excess_success_rates"
    print(f"Saving {name} to {output_data_dir / f'{name}.csv'}")
    excess_success_rates.to_csv(output_data_dir / f"{name}.csv")
    _plot_corr_matrix(
        corr_matrix=_corr_matrix(excess_success_rates),
        name="excess_success_rates",
        output_plots_dir=output_plots_dir,
    )
    fractional_excess_success_rates = _make_fractional_excess_success_rates(
        observed_success_rates=observed_success_rates,
        predicted_success_rates=predicted_success_rates,
        release_dates=release_dates,
        output_data_dir=output_data_dir,
    )
    fractional_excess_success_rates = _sort_cols_by_release_dates(
        df=fractional_excess_success_rates, release_dates=release_dates
    )
    name = "fractional_excess_success_rates"
    print(f"Saving {name} to {output_data_dir / f'{name}.csv'}")
    fractional_excess_success_rates.to_csv(output_data_dir / f"{name}.csv")
    _plot_corr_matrix(
        corr_matrix=_corr_matrix(fractional_excess_success_rates),
        name="fractional_excess_success_rates",
        output_plots_dir=output_plots_dir,
    )
    # Sort columns by release date
    observed_success_rates = _sort_cols_by_release_dates(
        df=observed_success_rates, release_dates=release_dates
    )
    name = "observed_success_rates"
    print(f"Saving {name} to {output_data_dir / f'{name}.csv'}")
    observed_success_rates.to_csv(output_data_dir / f"{name}.csv")
    _plot_corr_matrix(
        corr_matrix=_corr_matrix(observed_success_rates),
        name="observed_success_rates",
        output_plots_dir=output_plots_dir,
    )


def _make_dirs_if_not_exists(
    output_plots_dir: pathlib.Path,
    output_data_dir: pathlib.Path,
) -> Tuple[pathlib.Path, pathlib.Path]:
    # make output dir if it doesn't exist
    output_plots_dir.mkdir(parents=True, exist_ok=True)

    output_data_dir.mkdir(parents=True, exist_ok=True)
    return output_plots_dir, output_data_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-file", type=str, required=True)
    parser.add_argument("--release-dates", type=str, required=True)
    parser.add_argument("--logistic-file", type=str, required=True)
    parser.add_argument("--output-plots-dir", type=str, required=True)
    parser.add_argument("--output-data-dir", type=str, required=True)
    parser.add_argument("--exclude-agent", action="append", default=None)
    args = parser.parse_args()
    output_plots_path, output_data_path = _make_dirs_if_not_exists(
        output_plots_dir=pathlib.Path(args.output_plots_dir),
        output_data_dir=pathlib.Path(args.output_data_dir),
    )
    main(
        runs_file=pathlib.Path(args.runs_file),
        release_dates_file=pathlib.Path(args.release_dates),
        logistic_file=pathlib.Path(args.logistic_file),
        output_plots_dir=output_plots_path,
        output_data_dir=output_data_path,
        exclude_agents=args.exclude_agent,
    )
