#!/usr/bin/env python3
"""Compare task distributions across two datasets by human_minutes.

This script plots histograms of task counts by human_minutes (log-binned)
for two different datasets, allowing comparison of task difficulty distributions.
"""

import argparse
import logging
import pathlib

import dvc.api
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from horizon.utils.plots import (
    format_time_label,
    get_logarithmic_bins,
)

logger = logging.getLogger(__name__)


def load_unique_tasks(file_path: pathlib.Path) -> pd.DataFrame:
    """Load JSONL file and get unique task_id/human_minutes combinations."""
    df = pd.read_json(file_path, lines=True)

    # Get unique combinations of task_id and human_minutes
    unique_tasks = df[["task_id", "human_minutes"]].drop_duplicates()

    logger.info(f"Loaded {len(df)} runs from {file_path.name}")
    logger.info(f"Found {len(unique_tasks)} unique tasks")

    return unique_tasks


def plot_task_distributions(
    tasks1: pd.DataFrame,
    tasks2: pd.DataFrame,
    label1: str,
    label2: str,
    output_file: pathlib.Path,
    color1: str,
    color2: str,
) -> None:
    """Plot overlaid histograms of task counts by human_minutes for two datasets."""
    # Get combined min/max to ensure same bins for both
    all_times = pd.concat([tasks1["human_minutes"], tasks2["human_minutes"]])
    min_time = all_times.min()
    max_time = all_times.max()

    logger.info("Data range:")
    logger.info(f"  Min time: {format_time_label(min_time * 60)}")
    logger.info(f"  Max time: {format_time_label(max_time * 60)}")

    # Check for long tasks:
    for suite, times in [
        (label1, tasks1["human_minutes"]),
        (label2, tasks2["human_minutes"]),
    ]:
        tasks_beyond_8_hrs = times[times > 480]
        logger.info(f"{suite} has {len(tasks_beyond_8_hrs)} tasks longer than 8 hours")
        tasks_beyond_16_hrs = times[times > 960]
        logger.info(
            f"{suite} has {len(tasks_beyond_16_hrs)} tasks longer than 16 hours"
        )
        logger.info(f"    Longest task: {format_time_label(times.max() * 60)}")

    bins = get_logarithmic_bins(min_time, max_time)

    # Create figure with single axis
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Calculate histograms
    counts1, _ = np.histogram(tasks1["human_minutes"], bins=bins)
    counts2, _ = np.histogram(tasks2["human_minutes"], bins=bins)
    width = np.diff(bins)
    centers = bins[:-1]

    # Plot both datasets on same axis
    ax.bar(
        centers,
        counts1,
        width=width,
        alpha=0.6,
        color=color1,
        align="edge",
        edgecolor="black",
        linewidth=0.5,
        label=f"{label1} (n={len(tasks1)} tasks)",
    )
    ax.bar(
        centers,
        counts2,
        width=width,
        alpha=0.4,
        color=color2,
        align="edge",
        edgecolor="black",
        linewidth=0.5,
        label=f"{label2} (n={len(tasks2)} tasks)",
    )

    ax.set_ylabel("Number of Tasks", fontsize=14)
    ax.set_xlabel("Task Length (Human Time)", fontsize=14)
    ax.set_title("Task Distribution Comparison", fontsize=16, pad=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xscale("log")
    ax.legend(fontsize=12, framealpha=0.9)

    # Set x-axis ticks with formatted labels
    tick_positions = bins[::2]  # Show every other tick to avoid crowding
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([format_time_label(t * 60) for t in tick_positions], rotation=45)

    fig.tight_layout()

    # Save figure
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    logger.info(f"Saved plot to {output_file}")

    # Print summary statistics
    logger.info(f"{label1} statistics:")
    logger.info(f"  Min time: {format_time_label(tasks1['human_minutes'].min() * 60)}")
    logger.info(f"  Max time: {format_time_label(tasks1['human_minutes'].max() * 60)}")
    logger.info(
        f"  Median time: {format_time_label(tasks1['human_minutes'].median() * 60)}"
    )

    logger.info(f"{label2} statistics:")
    logger.info(f"  Min time: {format_time_label(tasks2['human_minutes'].min() * 60)}")
    logger.info(f"  Max time: {format_time_label(tasks2['human_minutes'].max() * 60)}")
    logger.info(
        f"  Median time: {format_time_label(tasks2['human_minutes'].median() * 60)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare task distributions by human_minutes across two datasets"
    )
    parser.add_argument(
        "file1",
        type=pathlib.Path,
        help="Path to first JSONL file (e.g., filtered_runs_without_weights.jsonl)",
    )
    parser.add_argument(
        "file2",
        type=pathlib.Path,
        help="Path to second JSONL file",
    )
    parser.add_argument(
        "--label1",
        type=str,
        default="Dataset 1",
        help="Label for first dataset in plot",
    )
    parser.add_argument(
        "--label2",
        type=str,
        default="Dataset 2",
        help="Label for second dataset in plot",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=pathlib.Path,
        default=pathlib.Path("task_distribution_comparison.png"),
        help="Output file path for plot",
    )
    parser.add_argument("--log-level", type=str, default="INFO")

    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Load colors from DVC params
    params = dvc.api.params_show()
    color1 = params.get("comparison_color_1", "blue")
    color2 = params.get("comparison_color_2", "orange")

    # Load unique tasks from both files
    tasks1 = load_unique_tasks(args.file1)
    tasks2 = load_unique_tasks(args.file2)

    # Plot distributions
    plot_task_distributions(
        tasks1,
        tasks2,
        args.label1,
        args.label2,
        args.output,
        color1,
        color2,
    )


if __name__ == "__main__":
    main()
