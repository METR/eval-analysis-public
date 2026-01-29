import argparse
import logging
import pathlib

import pandas as pd


def wrangle_costs(
    df_agent_runs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_aggregated = df_agent_runs.groupby(["alias", "task_id"]).agg(
        {
            "human_cost": "first",
            "human_minutes": "first",
            "generation_cost": "mean",
            "score_binarized": "mean",
            "started_at": "first",
            "completed_at": "first",
        }
    )

    # Calculate successful run times separately
    successful_runs = df_agent_runs[df_agent_runs["score_binarized"] == 1]
    successful_times = successful_runs.groupby(["alias", "task_id"]).agg(
        time_successful_runs=(
            "completed_at",
            lambda x: (x - successful_runs.loc[x.index, "started_at"]).mean(),
        )
    )
    df_aggregated = df_aggregated.join(successful_times, how="left")

    # Calculate duration in minutes for each run
    df_aggregated["duration_minutes"] = (
        df_aggregated["completed_at"] - df_aggregated["started_at"]
    ) / (60 * 1000)

    time_buckets = [[0, 4], [4, 16], [16, 256], [256, 1024], [1024, 4096]]

    def categorize_minutes(minutes: int) -> str | None:
        # For each value in the series, find which bucket it belongs to
        for bucket in time_buckets:
            if minutes > bucket[0] and minutes <= bucket[1]:
                return f"{bucket[0]}-{bucket[1]}"
        return None

    df_aggregated["actual_cost"] = (
        df_aggregated["score_binarized"] * df_aggregated["generation_cost"]
    ) + ((1 - df_aggregated["score_binarized"]) * df_aggregated["human_cost"])
    df_aggregated["human_minutes_bucket"] = df_aggregated["human_minutes"].apply(
        categorize_minutes
    )

    # Calculate average durations
    duration_stats = (
        df_aggregated.groupby("alias")
        .agg(
            avg_duration_all=("duration_minutes", "mean"),
            avg_duration_successful=(
                "duration_minutes",
                lambda x: x[df_aggregated["score_binarized"] == 1].mean(),
            ),
        )
        .round(2)
    )

    savings = df_aggregated.groupby(["alias", "human_minutes_bucket"]).agg(
        {"actual_cost": "sum", "human_cost": "sum"}
    )
    savings["cost_ratio"] = savings["actual_cost"] / savings["human_cost"]

    savings_non_bucketed = df_aggregated.groupby(["alias"]).agg(
        {"actual_cost": "sum", "human_cost": "sum"}
    )
    savings_non_bucketed["cost_ratio"] = (
        savings_non_bucketed["actual_cost"] / savings_non_bucketed["human_cost"]
    )

    # Add duration stats to savings_non_bucketed
    savings_non_bucketed = pd.merge(
        savings_non_bucketed,
        duration_stats,
        left_index=True,
        right_index=True,
    )

    return df_aggregated, savings, savings_non_bucketed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-file", type=pathlib.Path, required=True
    )  # data/external/all_runs.jsonl
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    runs = pd.read_json(
        args.runs_file, lines=True, orient="records", convert_dates=False
    )
    df_agent_runs = runs[runs["alias"] != "human"]
    df_aggregated, savings, savings_non_bucketed = wrangle_costs(df_agent_runs)

    pathlib.Path("data/processed/wrangled/costs").mkdir(parents=True, exist_ok=True)
    df_aggregated.to_csv("data/processed/wrangled/costs/cost_info.csv")
    pathlib.Path("metrics/costs").mkdir(parents=True, exist_ok=True)
    savings.to_csv("metrics/costs/savings_info.csv")
    savings_non_bucketed.to_csv("metrics/costs/savings_non_bucketed_info.csv")


if __name__ == "__main__":
    main()
