import argparse
import logging

import numpy as np
import pandas as pd


def _calculate_percentiles_across_tasks(
    df: pd.DataFrame,
    score_column: str,
    task_column: str = "task_id",
    percentiles: list[float] = list(range(1, 101)),
) -> pd.DataFrame:
    """Returns per-task percentiles and their cross-task means."""
    percentile_names = [f"p{p}" for p in percentiles]

    task_percentiles = [
        {
            task_column: task,
            **dict(
                zip(
                    percentile_names,
                    np.nanpercentile(
                        df[df[task_column] == task][score_column], percentiles
                    ),
                )
            ),
        }
        for task in df[task_column].unique()
    ]

    task_percentiles_df = pd.DataFrame(task_percentiles)
    mean_percentiles = pd.DataFrame(
        [
            {
                task_column: "average",
                **{col: task_percentiles_df[col].mean() for col in percentile_names},
            }
        ]
    )

    return pd.concat([task_percentiles_df, mean_percentiles], ignore_index=True)


def _calculate_final_quantiles(human_scores: pd.DataFrame) -> pd.DataFrame:
    """Returns quantiles for final scores of each task."""

    return _calculate_percentiles_across_tasks(human_scores, score_column="score_cont")


def wrangle_quantiles(
    interpolated_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate average of per-task quantiles at each time point and final scores."""
    human_scores = interpolated_scores[interpolated_scores["alias"] == "human"]

    final_quantiles = _calculate_final_quantiles(human_scores)

    return final_quantiles


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wrangle quantiles from interpolated scores."
    )
    parser.add_argument(
        "--interpolated-scores",
        type=str,
        required=True,
        help="Path to the interpolated scores file (JSONL).",
    )
    parser.add_argument(
        "--output-percentiles",
        type=str,
        required=True,
        help="Path to save the final quantiles data as JSONL.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    interpolated_scores = pd.read_json(args.interpolated_scores, lines=True)
    logging.info(
        f"Loaded {len(interpolated_scores)} rows from {args.interpolated_scores}"
    )

    final_quantiles = wrangle_quantiles(interpolated_scores)

    final_quantiles.to_json(args.output_percentiles, orient="records", lines=True)
    logging.info(f"Final quantiles data saved to {args.output_percentiles}")


if __name__ == "__main__":
    main()
