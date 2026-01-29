from __future__ import annotations

import argparse
import logging
import os
import pathlib
from typing import Any, Dict, List, Literal, Tuple, TypedDict

import dvc.api
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray

# Reuse styling constants from logistic.py
titlepad = 0
xlabelpad = 10
ax_label_fontsize = 14
ylabelpad = 10

TASKS_WITHOUT_HUMAN_TIMES = []

# Binary factors to sum (excluding implicit generalizability which needs special handling)
BINARY_FACTORS = [
    "real life source",
    # "# sourced from metr internal work",
    "resource limited",
    "not easily resettable",
    "irreversible mistake availability",
    "dynamic environment",
    "difficult counterfactuals",
    "not purely automatic scoring",
    "non explicit scoring description",
    "is suboptimal behavior exploited",
    "no provided verification mechanisms",
    "realtime coordination",
    "self modification required",
    "self improvement required",
    "information seeking required",
    "novel situation",
]

IMPLICIT_GENERALIZABILITY_FACTORS = [
    "implicit generalizability required",
    "implicit generalizability requirements are scored",
]


def create_factor_sum_column(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a factor sum column to the dataframe."""
    # First check the data types and values
    logging.debug("Factor values before conversion:")
    for factor in BINARY_FACTORS:
        unique_vals = df[factor].unique()
        logging.debug(
            f"{factor}: unique values = {unique_vals}, dtype = {df[factor].dtype}"
        )
    binary_factor_sum = df[BINARY_FACTORS].sum(axis=1)

    implicit_gen = df[IMPLICIT_GENERALIZABILITY_FACTORS].all(axis=1)
    logging.debug(
        f"Implicit generalizability: unique values = {implicit_gen.unique()}, dtype = {implicit_gen.dtype}"
    )

    factor_sum = binary_factor_sum + implicit_gen
    logging.debug(
        f"Factor sum: unique values = {factor_sum.unique()}, dtype = {factor_sum.dtype}"
    )
    return df.assign(factor_sum=factor_sum)


def exclude_ai_rd_tasks(df: pd.DataFrame) -> pd.DataFrame:
    # If it contains ai_rd_ in the task_id, exclude it
    logging.info(
        f"Number of ai_rd_ tasks in runs: {len(df[df['task_id'].str.contains('ai_rd_')]['task_id'].unique())}"
    )
    return df[~df["task_id"].str.contains("ai_rd_")]


def bootstrap_weighted_success(
    successes: NDArray[np.float64],
    weights: NDArray[np.float64],
    n_bootstrap: int = 1000,
    confidence: float = 0.9,
) -> Tuple[float, float]:
    """Calculate confidence intervals using bootstrap resampling."""
    n_samples = len(successes)
    bootstrap_estimates = []

    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.randint(0, n_samples, size=n_samples)
        resampled_successes = successes[indices]
        resampled_weights = weights[indices]

        # Normalize weights
        resampled_weights = resampled_weights / resampled_weights.sum()

        # Calculate weighted success
        weighted_success = (resampled_successes * resampled_weights).sum()
        bootstrap_estimates.append(float(weighted_success))  # Convert to float

    # Calculate confidence intervals
    lower = float(np.percentile(bootstrap_estimates, (1 - confidence) * 100 / 2))
    upper = float(np.percentile(bootstrap_estimates, (1 + confidence) * 100 / 2))

    return lower, upper


def calculate_weighted_success(
    agent_runs: pd.DataFrame,
    task_messiness: pd.DataFrame,
    weighting_scheme: Literal["combined", "task_only"],
) -> Tuple[float, float, float]:
    """Calculate weighted success using specified weighting scheme."""
    # Get the list of task IDs we want to include
    included_task_ids = set(task_messiness["task id"].unique())

    # Filter agent runs to only include those tasks
    agent_runs = agent_runs[agent_runs["task_id"].isin(included_task_ids)]

    # If no tasks remain after filtering, return zeros
    if len(agent_runs) == 0:
        return 0.0, 0.0, 0.0

    # Merge with messiness data
    agent_runs = agent_runs.merge(
        task_messiness[["task id"]],
        left_on="task_id",
        right_on="task id",
        how="inner",
    )

    # First, calculate mean success rate per task_id
    task_means = (
        agent_runs.groupby("task_id")
        .agg(
            {
                "score_binarized": "mean",
                "human_minutes": "mean",  # Take mean of human_minutes per task
                "invsqrt_task_weight": "first",  # Weight is same per task
            }
        )
        .reset_index()
    )

    if weighting_scheme == "combined":
        human_time_weights = task_means["human_minutes"].to_numpy()
        human_time_weights = human_time_weights / human_time_weights.sum()
        np.testing.assert_allclose(human_time_weights.sum(), 1.0, rtol=1e-10)

        task_weights = task_means["invsqrt_task_weight"].to_numpy()
        task_weights = task_weights / task_weights.sum()
        np.testing.assert_allclose(task_weights.sum(), 1.0, rtol=1e-10)

        combined_weights = human_time_weights * task_weights
        combined_weights = combined_weights / combined_weights.sum()
        np.testing.assert_allclose(combined_weights.sum(), 1.0, rtol=1e-10)
    else:  # task_only
        combined_weights = task_means["invsqrt_task_weight"].to_numpy()
        combined_weights = combined_weights / combined_weights.sum()
        np.testing.assert_allclose(combined_weights.sum(), 1.0, rtol=1e-10)

    # Calculate weighted success using task-level means
    weighted_success = (
        task_means["score_binarized"].to_numpy() * combined_weights
    ).sum()

    # Calculate confidence intervals using task-level data
    ci_lower, ci_upper = bootstrap_weighted_success(
        task_means["score_binarized"].to_numpy(),
        combined_weights,
        n_bootstrap=1000,
        confidence=0.9,
    )

    return weighted_success, ci_lower, ci_upper


def get_tasks(
    df: pd.DataFrame,
    messiness_tasks_path: pathlib.Path,
) -> List[str]:
    print(df.head())
    assert "task_id" in df.columns
    tasks_in_df = set(df["task_id"].unique())
    messiness_df = pd.read_csv(messiness_tasks_path)
    print(messiness_df.head())
    assert "task_id" in messiness_df.columns
    tasks_in_messiness = set(messiness_df["task_id"].unique())
    tasks_in_both = tasks_in_df & tasks_in_messiness
    logging.info(f"Number of tasks in both df and messiness: {len(tasks_in_both)}")
    return list(tasks_in_both)


def exclude_tasks_not_in_column(
    df: pd.DataFrame,
    column: str,
    messiness_tasks_path: pathlib.Path,
) -> pd.DataFrame:
    # EXPANDED_TEST_SET_FOR_BLOG_POST = list(
    #     set(EXPANDED_TEST_SET_FOR_BLOG_POST_FULL) - set(TASKS_WITHOUT_HUMAN_TIMES)
    # )
    # Task ids are all the unique task ids in all_runs.jsonl and messiness.csv
    tasks_in_df = set(df["task_id"].unique())
    tasks_in_messiness = set(pd.read_csv(messiness_tasks_path)["task_id"].unique())
    tasks_in_both = tasks_in_df & tasks_in_messiness
    logging.info(f"Number of tasks in both df and messiness: {len(tasks_in_both)}")

    df_messiness_tasks = pd.read_csv(messiness_tasks_path)
    if column == "in_expanded_test_set":
        in_test_set = df_messiness_tasks[
            df_messiness_tasks["task_id"].isin(tasks_in_both)
        ]["task_id"].unique()
    elif column == "in_original_test_set":
        # Find all the tasks where the column is true
        in_test_set = df_messiness_tasks[df_messiness_tasks[column]]["task_id"].unique()
    else:
        raise ValueError(f"Unknown column: {column}")
    logging.info(f"Number of tasks in {column}: {len(in_test_set)}")
    logging.info(
        f"Excluding {len(df_messiness_tasks['task_id'].unique()) - len(in_test_set)} tasks not in {column}"
    )
    return df[df["task id"].isin(in_test_set)]


def exclude_tasks_marked_for_exclusion_by_person(
    df: pd.DataFrame,
    person: str = "Megan Kinniment",
    column: str = "exclude task from analysis",
) -> pd.DataFrame:
    person_labels = df[df["labeller"] == person]
    tasks_marked_for_exclusion_by_person = person_labels[
        person_labels[column].fillna(False)
    ]["task id"].unique()
    return df[~df["task id"].isin(tasks_marked_for_exclusion_by_person)]


def plot_success_trend_by_messiness_and_length(
    runs_file: pathlib.Path,
    release_dates_file: pathlib.Path,
    messiness_file: pathlib.Path,
    quantile: float,
    messiness_tasks_file: pathlib.Path,
    output_file: pathlib.Path,
    plot_params: Dict[str, Any],
    agent_styling: Dict[str, AgentStyle],
    exclude_agents: List[str],
    weighting_scheme: Literal["combined", "task_only"],
    label_type: Literal["megan", "contractor", "combined"],
    test_set: Literal["original", "expanded"],
) -> None:
    logging.info("Reading input files...")
    # Read all runs
    df_runs = pd.read_json(runs_file, lines=True, orient="records")
    tasks_in_both = get_tasks(df_runs, messiness_tasks_file)
    print(f"Tasks in both: {len(tasks_in_both)}")
    for task in tasks_in_both:
        print(task)
    logging.info(f"Read {len(df_runs)} runs")
    logging.info(
        f"NUMBER OF EXPANDED TEST SET FOR BLOG POST TASKS THAT ALSO HAVE A RUN: {len(df_runs[df_runs['task_id'].isin(tasks_in_both)]['task_id'].unique())}"
    )
    logging.info(
        f"NUMBER OF EXPANDED TEST SET FOR BLOG POST TASKS THAT DO NOT HAVE A RUN:{len(set(tasks_in_both) - set(df_runs['task_id'].unique()))}"
    )
    logging.info(
        f"LIST OF EXPANDED TEST SET FOR BLOG POST TASKS THAT DO NOT HAVE A RUN:\n{set(tasks_in_both) - set(df_runs['task_id'].unique())}"
    )
    # Read messiness data
    df_messiness = pd.read_csv(messiness_file)
    tasks_in_messiness_and_blog_post = df_messiness[
        df_messiness["task id"].isin(tasks_in_both)
    ]["task id"].unique()
    df_runs = df_runs[~df_runs["alias"].isin(exclude_agents)]
    tasks_in_runs_and_blog_post = df_runs[df_runs["task_id"].isin(tasks_in_both)][
        "task_id"
    ].unique()
    tasks_in_both = set(tasks_in_messiness_and_blog_post) & set(
        tasks_in_runs_and_blog_post
    )

    list_tasks = df_messiness["task id"].unique()
    tasks_in_runs = df_runs["task_id"].unique()
    tasks_in_runs_in_test_set = [task for task in tasks_in_runs if task in list_tasks]
    logging.info(
        f"Number of task ids in run set that are in EXPANDED TEST SET FOR BLOG POST: {len(tasks_in_runs_in_test_set)} / {len(tasks_in_both)}"
    )
    logging.info(
        f"Tasks that are in EXPANDED TEST SET FOR BLOG POST but not in run set:\n{set(tasks_in_both) - set(tasks_in_runs_in_test_set)}"
    )
    # Filter and process based on label type
    if label_type == "megan":
        df_messiness = df_messiness[df_messiness["labeller"] == "Megan Kinniment"]
        title_source = "Megan's Labels"
    elif label_type == "contractor":
        # Filter to contractor labels
        df_messiness = df_messiness[df_messiness["labeller"] != "Megan Kinniment"]

        # Average the raw factors for each task
        df_messiness = (
            df_messiness.groupby("task id")
            .agg(
                {
                    **{factor: "mean" for factor in BINARY_FACTORS},
                    "implicit generalizability required": "mean",
                    "implicit generalizability requirements are scored": "mean",
                }
            )
            .reset_index()
        )

        title_source = "Contractor Labels (Averaged)"
    else:  # combined
        title_source = "Combined Labels (Averaged)"

    logging.info(f"Using {len(df_messiness)} records for {label_type} analysis")

    df_messiness = create_factor_sum_column(df_messiness)
    # Aggregate messiness factor sum as mean over labellers
    df_messiness = df_messiness.groupby("task id")["factor_sum"].mean().reset_index()

    # Create time buckets in df_runs
    df_runs["time_bucket"] = pd.cut(
        df_runs["human_minutes"],
        bins=[0, 60, float("inf")],
        labels=["< 1 hour", "1+ hours"],
    )

    # Get the time bucket for each task by taking the mean human_minutes per task
    task_time_buckets = (
        df_runs.groupby("task_id").agg({"human_minutes": "mean"}).reset_index()
    )
    task_time_buckets["time_bucket"] = pd.cut(
        task_time_buckets["human_minutes"],
        bins=[0, 60, float("inf")],
        labels=["< 1 hour", "1+ hours"],
    )

    # Merge time buckets into df_messiness
    df_messiness = df_messiness.merge(
        task_time_buckets[["task_id", "time_bucket"]],
        left_on="task id",
        right_on="task_id",
        how="left",
    )

    # Now we can calculate median factor sum by time bucket
    median_factor_sum_by_time_bucket = df_messiness.groupby("time_bucket")[
        "factor_sum"
    ].quantile(quantile)
    logging.info(
        f"Median factor sum by time bucket: {median_factor_sum_by_time_bucket}"
    )
    # Load release dates
    release_dates = yaml.safe_load(release_dates_file.read_text())["date"]
    logging.info(f"Loaded {len(release_dates)} release dates")

    # Create figure with 1x2 subplots (1 row for high/low factor sum)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 12),
        gridspec_kw={"hspace": 0.4, "wspace": 0.2},
    )

    total_tasks = 0
    list_present_tasks = []

    # Process each factor sum group
    for row_idx, (factor_filter, factor_label) in enumerate(
        [
            (lambda x: x < median_factor_sum_by_time_bucket[row_idx], "Low"),
            (lambda x: x >= median_factor_sum_by_time_bucket[row_idx], "High"),
        ]
    ):
        # Filter tasks based on factor sum
        plot_subset = df_messiness[factor_filter(df_messiness["factor_sum"])]

        total_tasks += len(plot_subset)
        list_present_tasks.extend(plot_subset["task id"].unique())
        ax = axes[row_idx]

        # Calculate weighted success for each agent
        results = []
        for alias in df_runs["alias"].unique():
            agent_runs = df_runs[df_runs["alias"] == alias]

            weighted_success, ci_lower, ci_upper = calculate_weighted_success(
                agent_runs, plot_subset, weighting_scheme
            )

            # Get release date
            release_date = pd.to_datetime(release_dates.get(alias))

            if release_date is not None:
                results.append(
                    {
                        "agent": alias,
                        "weighted_success": weighted_success,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "release_date": release_date,
                    }
                )

        df_results = pd.DataFrame(results)

        # Sort by legend order
        df_results = df_results.copy()
        df_results["legend_order"] = df_results["agent"].map(
            {agent: i for i, agent in enumerate(plot_params["legend_order"])}
        )
        df_results = df_results.sort_values("legend_order")

        # Plot scatter points with error bars
        legend_handles = []
        legend_labels = []

        for _, row in df_results.iterrows():
            agent = row["agent"]
            if agent in agent_styling:
                marker = agent_styling[agent]["marker"]
                color = agent_styling[agent]["lab_color"]

                # Plot error bars
                yerr_lower = max(0, float(row["weighted_success"] - row["ci_lower"]))
                yerr_upper = max(0, float(row["ci_upper"] - row["weighted_success"]))
                ax.errorbar(
                    row["release_date"],
                    row["weighted_success"],
                    yerr=[[yerr_lower], [yerr_upper]],
                    fmt="none",
                    color="grey",
                    capsize=2,
                    alpha=1,
                    zorder=9,
                    linewidth=1.5,
                    capthick=1.5,
                )
                ax.grid(True, which="major", linestyle="-", alpha=0.25, color="black")

                # Plot scatter point
                scatter = ax.scatter(
                    row["release_date"],
                    row["weighted_success"],
                    marker=marker,
                    color=color,
                    s=150,
                    zorder=10,
                    edgecolor="black",
                    linewidth=0.5,
                )

                # Add to legend only if model is in the data and not excluded
                if (
                    row_idx == 0 and agent not in exclude_agents
                ):  # Only add legend to first column
                    legend_handles.append(scatter)
                    legend_labels.append(agent)

        # Customize subplot
        if row_idx == 1:  # Only bottom row gets x-label
            ax.set_xlabel(
                "Model Release Date", fontsize=ax_label_fontsize, labelpad=xlabelpad
            )

        ax.set_ylabel(
            "Weighted Success Rate", fontsize=ax_label_fontsize, labelpad=ylabelpad
        )

        factor_boundary = f"Messiness score {'>=' if factor_label == 'High' else '<'} {median_factor_sum_by_time_bucket[row_idx]:.1f}"
        most_or_least = "most" if factor_label == "High" else "least"
        if most_or_least == "least":
            quantile_str = f"{quantile * 100:.0f}%"
        elif most_or_least == "most":
            quantile_str = f"{(100 - quantile * 100):.0f}%"
        else:
            raise ValueError(f"Invalid most_or_least: {most_or_least}")

        ax.set_title(
            f"{quantile_str} {most_or_least} Messy Tasks\n({factor_boundary}, N tasks = {len(plot_subset)})",
            fontsize=ax_label_fontsize,
            pad=titlepad,
        )

        # Format x-axis dates
        ax.grid(True, which="major", linestyle="-", alpha=0.1, color="gray")
        ax.grid(True, which="minor", linestyle=":", alpha=0.05, color="gray")
        ax.set_axisbelow(True)

        # Set date range
        ax.set_xlim(
            float(mdates.date2num(pd.Timestamp("2022-01-01"))),
            float(mdates.date2num(pd.Timestamp("2025-05-01"))),
        )

        # Format x-axis ticks
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        # Set y-axis range from 0 to 1
        ax.set_ylim(0, 1)

        # Add legend only to first column
        if row_idx == 0:
            ax.legend(
                legend_handles,
                legend_labels,
                loc="upper left",
                fontsize=10,
            )
    # print(f"Tasks which were dropped: {set(list_tasks) - set(list_present_tasks)}")
    # tasks_dropped = set(list_tasks) - set(list_present_tasks)
    # for task in tasks_dropped:
    #     assert (
    #         task not in df_runs["task_id"].unique()
    #     ), f"Dropped task {task} is in df_runs"

    # Add overall title
    test_set_string = "(Original Test Set)" if test_set == "original" else ""
    if title_source == "Combined Labels (Averaged)":
        title_source = ""
    title_str = "Model Success Rates over Time, split by Messiness"
    overall_title = f"{title_str} {title_source} {test_set_string}\n"
    if weighting_scheme == "combined":
        overall_title += "Weighted by normalized human time × task diversity"
    else:
        overall_title += "(Weighted by Task Diversity)"
    fig.suptitle(overall_title, fontsize=18, y=0.98)

    # Save plot
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_file, bbox_inches="tight", dpi=300)
    logging.info(f"Saved plot to {output_file}")


class AgentStyle(TypedDict):
    lab_color: str
    marker: str
    unique_color: str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-file", type=pathlib.Path, required=True)
    parser.add_argument("--release-dates", type=pathlib.Path, required=True)
    parser.add_argument("--messiness-file", type=pathlib.Path, required=True)
    parser.add_argument("--messiness-tasks-file", type=pathlib.Path, required=True)
    parser.add_argument("--messiness-quantile-boundary", type=float, required=True)
    parser.add_argument("--output-file", type=pathlib.Path, required=True)
    parser.add_argument("--params-file", type=pathlib.Path, required=True)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--exclude-agent", action="append", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    agent_styling: Dict[str, AgentStyle] = yaml.safe_load(args.params_file.read_text())[
        "plots"
    ]["agent_styling"]

    pathlib.Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    try:
        base_output = args.output_file.with_suffix("")

        # Generate plots for each test set
        test_sets: tuple[Literal["original", "expanded"], ...] = (
            # "original",
            "expanded",
        )
        for test_set in test_sets:
            # Generate plots for each label type
            label_types: tuple[Literal["megan", "contractor", "combined"], ...] = (
                #      "megan",
                #     "contractor",
                "combined",
            )
            for label_type in label_types:
                # Task weights only version
                logging.info(
                    f"Generating task-only weights version for {label_type} labels on {test_set} test set..."
                )
                plot_success_trend_by_messiness_and_length(
                    runs_file=args.runs_file,
                    release_dates_file=args.release_dates,
                    messiness_file=args.messiness_file,
                    messiness_tasks_file=args.messiness_tasks_file,
                    quantile=args.messiness_quantile_boundary,
                    output_file=base_output.with_name(f"{base_output.name}.png"),
                    plot_params=dvc.api.params_show(stages="plot_logistic_regression")[
                        "plots"
                    ],
                    agent_styling=agent_styling,
                    exclude_agents=args.exclude_agent,
                    weighting_scheme="task_only",
                    label_type=label_type,
                    test_set=test_set,
                )

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
