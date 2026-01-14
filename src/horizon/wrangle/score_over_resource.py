import argparse
import pathlib
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml

HUMAN_ACTIONS_PER_MINUTE = 1

SUCCESS_RATE_COLUMN = "success_rate_at_cost"
WEIGHTED_SUCCESS_RATE_COLUMN = "weighted_success_rate_at_cost"

WEIGHT_COL_NAMES = {
    "equal_task_weight": "Weighting Tasks Equally",
    "invsqrt_task_weight": "Weighted by Task Diversity",
}


def _get_weighted_success_rate_at_cost_slow(
    df: pd.DataFrame,
    agent: str,
    score_column: str,
    usage_column: str,
    weighting_column: str,
    success_rate_column: str,
    weighted_success_rate_column: str,
) -> pd.DataFrame:
    """Keeping this function here as a reference for what the faster version is doing.
    Have tested that results are the same on one agent."""

    df_agent = df[df["alias"] == agent]
    # Sort in ascending order of generation_cost
    df_agent = df_agent.sort_values(by=usage_column)
    df_final = df_agent.copy()

    total_weight = df_agent[weighting_column].sum()
    assert abs(total_weight - 1) < 1e-6

    for cost in sorted(df_agent[usage_column].unique().tolist()):
        df_cost = df_agent.copy()

        # Set all runs with generation_cost greater than cost to 0
        df_cost[score_column] = df_cost.apply(
            lambda row: row[score_column] if row[usage_column] <= cost else 0,
            axis=1,
        )

        # Calculate success rate per task
        success_rate_per_task = df_cost.groupby("task_id")[score_column].mean()

        df_final.loc[df_final[usage_column] == cost, success_rate_column] = (
            success_rate_per_task.mean()
        )
        if not weighting_column:
            # Each task has the same weight
            task_weight_per_task = 1 / len(success_rate_per_task)
        else:
            task_weight_per_task = df_cost.groupby("task_id")[weighting_column].sum()

        # Calculate weighted success rate
        weighted_success_rate = (success_rate_per_task * task_weight_per_task).sum()

        if weighting_column:
            df_final.loc[
                df_final[usage_column] == cost, weighted_success_rate_column
            ] = weighted_success_rate

    return df_final


def _process_action_count(df: pd.DataFrame) -> pd.DataFrame:
    """Adds 1 to all non NaN action counts, and sets all NaNs to 0.
    (because submit isn't counted as an action)"""
    df.loc[df["action_count"].notna(), "action_count"] += 1
    df.loc[df["action_count"].isna(), "action_count"] = 0
    df["action_count"] = df["action_count"].astype(int)
    return df


def _action_per_dollar_for_humans() -> float:
    """Get the hourly wage for humans from the params file."""
    with open("../params.yaml", "r") as f:
        params = yaml.safe_load(f)

    hourly_wage = params["stages"]["merge_data"]["hourly_wage"]
    actions_per_hour = HUMAN_ACTIONS_PER_MINUTE * 60
    return actions_per_hour / hourly_wage


def _add_best_human_agent(df: pd.DataFrame, score_column: str) -> pd.DataFrame:
    best_human_rows = []
    for task_id in df["task_id"].unique():
        # Find the best existing human row for this task
        human_rows = df[(df["task_id"] == task_id) & (df["alias"] == "human")]
        if human_rows.empty:
            continue
        best_human_row = (
            human_rows.sort_values(by=score_column, ascending=False).iloc[0].copy()
        )
        best_human_row["alias"] = "Best Human for Each Task"
        for i in range(len(human_rows)):
            best_human_rows.append(best_human_row)
    best_human_df = pd.DataFrame(best_human_rows)
    df = pd.concat([df, best_human_df])
    return df


def handle_human_agent(
    df_original: pd.DataFrame, plot_human: bool, score_column: str, x: str
) -> pd.DataFrame:
    df = df_original.copy()
    if not plot_human:
        df = df[~df["alias"].isin(["human"])]
    else:
        if x == "generation_cost":
            # Replace generation_cost with human_cost for human agent
            df.loc[df["alias"] == "human", "generation_cost"] = df.loc[
                df["alias"] == "human", "human_cost"
            ]
            # Also make best human alias, which takes the best human score for each task
            df = _add_best_human_agent(df, score_column)
        elif x == "action_count":
            # Convert human cost to estimated action count using hourly rate
            human_mask = df["alias"] == "human"
            df.loc[human_mask, "action_count"] = (
                (
                    df.loc[human_mask, "human_cost"].fillna(0)
                    * _action_per_dollar_for_humans()
                )
                .round(0)
                .astype(int)
            )
            # Create best human baseline using action counts
            df = _add_best_human_agent(df, score_column)
        elif x == "tokens_count":
            raise ValueError(
                "Human baseline is not supported when plotting by tokens_count"
            )
        else:
            raise ValueError(f"Usage column {x} is not supported for human plotting")
    return df


def _add_invsqrt_task_weight(original_df: pd.DataFrame) -> pd.DataFrame:
    """Weight the task families for an agent by the inverse square root of the number of tasks in each family."""
    df = original_df.copy()
    task_family_counts = defaultdict(int)
    for task_id in df["task_id"].unique():
        family = task_id.split("/")[0]
        task_family_counts[family] += 1

    df["invsqrt_task_weight"] = df["task_family"].apply(
        lambda x: 1 / np.sqrt(task_family_counts[x]) * 1 / len(df["task_id"].unique())
    )
    df["invsqrt_task_weight"] = (
        df["invsqrt_task_weight"] / df["invsqrt_task_weight"].sum()
    )
    return df


def _add_equal_task_weight(df: pd.DataFrame) -> pd.DataFrame:
    """Weight the tasks equally."""
    df["equal_task_weight"] = 1
    df["equal_task_weight"] = df["equal_task_weight"] / df["equal_task_weight"].sum()
    return df


def get_weighted_success_rate_at_cost(
    df: pd.DataFrame,
    agent: str,
    usage_column: str,
    score_column: str,
    weighting_column: str,
    success_rate_column: str,
    weighted_success_rate_column: str,
) -> pd.DataFrame:
    df_agent = df[df["alias"] == agent].sort_values(by=usage_column)
    assert not df_agent.empty, f"No data for agent '{agent}'"

    df_final = df_agent.copy()
    if weighting_column == "invsqrt_task_weight":
        df_agent = _add_invsqrt_task_weight(df_agent)
    elif weighting_column == "equal_task_weight":
        df_agent = _add_equal_task_weight(df_agent)
    else:
        raise ValueError(
            f"Weighting column {weighting_column} is not supported. Choose from equal_task_weight or invsqrt_task_weight"
        )
    total_weight = df_agent[weighting_column].sum()
    assert abs(total_weight - 1) < 1e-6, f"Total weight is {total_weight}"

    costs = df_agent[usage_column].to_numpy()
    scores = df_agent[score_column].to_numpy()
    tasks = df_agent["task_id"].to_numpy()
    weights = df_agent[weighting_column].to_numpy()

    unique_costs = np.sort(np.unique(costs))
    unique_tasks = np.unique(tasks)

    task_to_idx = {t: i for i, t in enumerate(unique_tasks)}
    task_indices = np.array([task_to_idx[t] for t in tasks])

    # Create a cost comparison matrix (runs × cost thresholds)
    cost_matrix = costs[:, np.newaxis] <= unique_costs

    # Create a task matrix (runs × tasks) where each cell is 1 if the run is for that task
    task_matrix = np.zeros((len(scores), len(unique_tasks)), dtype=bool)
    task_matrix[np.arange(len(task_indices)), task_indices] = True

    # Multiply scores by cost matrix to zero out scores above threshold
    weighted_scores = scores[:, np.newaxis] * cost_matrix

    # Sum scores per task per cost threshold
    task_scores = task_matrix.T @ weighted_scores

    task_counts = np.bincount(task_indices, minlength=len(unique_tasks))
    task_counts = task_counts[:, np.newaxis]

    task_success_rates = task_scores / task_counts

    task_weight_per_task = np.zeros((len(unique_tasks), 1))
    for task_idx, weight in zip(task_indices, weights):
        task_weight_per_task[task_idx] += weight

    mean_success_rates = np.nanmean(task_success_rates, axis=0)
    weighted_success_rates = np.sum(task_success_rates * task_weight_per_task, axis=0)

    # Create mappings for quick lookup
    cost_to_rate = {c: r for c, r in zip(unique_costs, mean_success_rates)}
    cost_to_weighted_rate = {c: r for c, r in zip(unique_costs, weighted_success_rates)}

    # Apply mappings vectorized
    df_final[success_rate_column] = df_final[usage_column].map(cost_to_rate)
    if weighting_column:
        df_final[weighted_success_rate_column] = df_final[usage_column].map(
            cost_to_weighted_rate
        )

    df_final = df_final[df_final[success_rate_column].notna()]
    if weighting_column:
        df_final = df_final[df_final[weighted_success_rate_column].notna()]

    return df_final


def process_runs_file(
    df: pd.DataFrame,
    plot_human: bool,
    score_column: str,
    weighting_column: str,
    x: str,
) -> pd.DataFrame:
    df = df[df["task_source"].isin(["HCAST", "RE-Bench"])]
    df = handle_human_agent(df, plot_human, score_column, x)
    if x == "action_count":
        df = _process_action_count(df)
    df.loc[df[score_column].isna(), score_column] = 0

    if x == "tokens_count":
        assert "tokens_count" in df.columns, "tokens_count column missing"
        df = df[df["tokens_count"].notna()]
        assert not df.empty, "No rows remain after filtering for tokens_count"
        assert (df["tokens_count"] >= 0).all(), "tokens_count must be non-negative"

    agent_dfs = []
    for agent in df["alias"].unique():
        agent_df = get_weighted_success_rate_at_cost(
            df=df,
            agent=agent,
            usage_column=x,
            score_column=score_column,
            weighting_column=weighting_column,
            success_rate_column=SUCCESS_RATE_COLUMN,
            weighted_success_rate_column=WEIGHTED_SUCCESS_RATE_COLUMN,
        )
        assert not agent_df.empty, f"No data for agent '{agent}' after processing"

        agent_dfs.append(agent_df)
    return pd.concat(agent_dfs)


def main(
    runs_file: pathlib.Path,
    output_file: pathlib.Path,
    include_human: bool,
    score_column: str,
    weighting_column: str,
    x: str,
) -> None:
    df = pd.read_json(runs_file, lines=True, orient="records")
    df = process_runs_file(
        df=df,
        plot_human=include_human,
        score_column=score_column,
        weighting_column=weighting_column,
        x=x,
    )
    pathlib.Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_file, orient="records", lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-file",
        type=pathlib.Path,
        required=True,
        help="Path to the runs.jsonl file",
    )
    parser.add_argument(
        "--output-file",
        type=pathlib.Path,
        help="Path to save the output figure",
        required=True,
    )
    parser.add_argument(
        "--include-human",
        action="store_true",
        help="Whether to include the human agent in the data",
    )
    parser.add_argument(
        "--score-column",
        type=str,
        required=True,
        help="Column to use for scoring. E.g. score_binarized",
    )
    parser.add_argument(
        "--weighting-column",
        type=str,
        required=True,
        help="Column to use for weighting (either equal_task_weight or invsqrt_task_weight)",
    )
    parser.add_argument(
        "--x",
        required=True,
        type=str,
        help="Column to use for x-axis. E.g. generation_cost or action_count",
    )
    args = parser.parse_args()
    main(**vars(args))
