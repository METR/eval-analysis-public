#!/usr/bin/env python3
"""Compare doubling times between two model reports.

This script reads bootstrap results and agent summaries from two model reports,
computes doubling times for each, and creates overlapping histograms for comparison.

NOTE: if used to compare e.g. Time Horizon 1.0 and 1.1, this is a bit misleading: the
full bootstrap is trying to incorporate more variance than is actually fair, since
the two suites heavily overlap. This is included here mainly for exploration and
further work.
"""

import argparse
import logging
import pathlib

import dvc.api
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from horizon.compute_trendline_ci import get_sota_agents
from horizon.plot.bootstrap_ci import compute_bootstrap_confidence_region

logger = logging.getLogger(__name__)


def compute_doubling_times_for_report(
    bootstrap_file: pathlib.Path,
    agent_summaries_file: pathlib.Path,
    release_dates: dict[str, str],
    after_date: str,
    before_date: str | None,
) -> tuple[list[float], list[str], float]:
    """Compute doubling times for a single model report.

    Returns:
        Tuple of (all_doubling_times, sota_agents, point_estimate)
    """
    bootstrap_results = pd.read_csv(bootstrap_file)
    agent_summaries = pd.read_csv(agent_summaries_file)

    sota_agents = get_sota_agents(
        agent_summaries, release_dates, after_date, before_date
    )
    logger.info(f"SOTA agents for {bootstrap_file.parent.parent.name}: {sota_agents}")

    # Filter to agents with bootstrap results
    sota_agents_with_data = [
        agent for agent in sota_agents if f"{agent}_p50" in bootstrap_results.columns
    ]
    assert len(sota_agents_with_data) == len(sota_agents), (
        "Some SOTA agents are missing bootstrap data: "
        f"{set(sota_agents) - set(sota_agents_with_data)}"
    )

    agent_summaries_for_fitting = agent_summaries[
        agent_summaries["agent"].isin(sota_agents_with_data)
    ]
    bootstrap_results_for_fitting = bootstrap_results[
        [f"{agent}_p50" for agent in sota_agents_with_data]
    ]

    # Wrap release_dates in the expected format
    release_dates_wrapped = {
        "date": {k: pd.to_datetime(v).date() for k, v in release_dates.items()}
    }

    # Use before_date if provided, otherwise a far future date
    # Note: max_date only affects trendline visualization, not the doubling time calculation
    max_date = (
        pd.to_datetime(before_date) if before_date else pd.to_datetime("2030-01-01")
    )

    stats, _, _, _ = compute_bootstrap_confidence_region(
        agent_summaries=agent_summaries_for_fitting,
        bootstrap_results=bootstrap_results_for_fitting,
        release_dates=release_dates_wrapped,
        after_date=after_date,
        max_date=max_date,
        confidence_level=0.95,
    )

    return stats.all_doubling_times, sota_agents_with_data, stats.point_estimate


def plot_overlapping_histograms(
    doubling_times_1: list[float],
    doubling_times_2: list[float],
    label_1: str,
    label_2: str,
    point_estimate_1: float,
    point_estimate_2: float,
    output_file: pathlib.Path,
    plot_format: str,
    color_1: str,
    color_2: str,
) -> None:
    """Create overlapping histograms of doubling times from two reports."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Determine common bin range (log-spaced for log-normal distributions)
    all_times = doubling_times_1 + doubling_times_2
    min_time = min(all_times)
    max_time = max(all_times)
    bins = np.logspace(np.log10(min_time), np.log10(max_time), 50)
    ax.set_xscale("log")

    ax.hist(
        doubling_times_1,
        bins=bins,
        alpha=0.6,
        label=label_1,
        color=color_1,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.hist(
        doubling_times_2,
        bins=bins,
        alpha=0.6,
        label=label_2,
        color=color_2,
        edgecolor="white",
        linewidth=0.5,
    )

    # Add vertical lines for medians
    median_1 = np.median(doubling_times_1)
    median_2 = np.median(doubling_times_2)

    ax.axvline(
        float(median_1),
        color=color_1,
        linestyle="--",
        linewidth=2,
        label=f"{label_1} median: {median_1:.0f} days",
    )
    ax.axvline(
        float(median_2),
        color=color_2,
        linestyle="--",
        linewidth=2,
        label=f"{label_2} median: {median_2:.0f} days",
    )

    # Compute 95% CI bounds
    ci_1_lower = np.percentile(doubling_times_1, 2.5)
    ci_1_upper = np.percentile(doubling_times_1, 97.5)
    ci_2_lower = np.percentile(doubling_times_2, 2.5)
    ci_2_upper = np.percentile(doubling_times_2, 97.5)

    # Add tick marks on x-axis for 95% CI bounds and point estimates
    tick_y = ax.get_ylim()[0]
    ax.plot(
        [ci_1_lower, ci_1_upper],
        [tick_y, tick_y],
        marker="|",
        markersize=12,
        color=color_1,
        linestyle="none",
        clip_on=False,
        markeredgewidth=2,
    )
    ax.plot(
        [point_estimate_1],
        [tick_y],
        marker="^",
        markersize=8,
        color=color_1,
        linestyle="none",
        clip_on=False,
    )
    ax.plot(
        [ci_2_lower, ci_2_upper],
        [tick_y, tick_y],
        marker="|",
        markersize=12,
        color=color_2,
        linestyle="none",
        clip_on=False,
        markeredgewidth=2,
    )
    ax.plot(
        [point_estimate_2],
        [tick_y],
        marker="^",
        markersize=8,
        color=color_2,
        linestyle="none",
        clip_on=False,
    )

    # Add to legend
    ax.plot([], [], marker="|", color="gray", linestyle="none", label="95% CI bounds")
    ax.plot([], [], marker="^", color="gray", linestyle="none", label="Point estimate")

    ax.set_xlabel("Doubling Time (days)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(
        "Comparison of Bootstrapped Doubling Times\n(2023+ SOTA Models)", fontsize=14
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, format=plot_format, dpi=150, bbox_inches="tight")
    logger.info(f"Plot saved to {output_file}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare doubling times between two model reports."
    )
    parser.add_argument(
        "--bootstrap-file-1",
        type=pathlib.Path,
        required=True,
        help="Bootstrap results CSV for first model report",
    )
    parser.add_argument(
        "--agent-summaries-file-1",
        type=pathlib.Path,
        required=True,
        help="Agent summaries CSV for first model report",
    )
    parser.add_argument(
        "--label-1",
        type=str,
        required=True,
        help="Label for first model report",
    )
    parser.add_argument(
        "--bootstrap-file-2",
        type=pathlib.Path,
        required=True,
        help="Bootstrap results CSV for second model report",
    )
    parser.add_argument(
        "--agent-summaries-file-2",
        type=pathlib.Path,
        required=True,
        help="Agent summaries CSV for second model report",
    )
    parser.add_argument(
        "--label-2",
        type=str,
        required=True,
        help="Label for second model report",
    )
    parser.add_argument(
        "--release-dates",
        type=pathlib.Path,
        required=True,
        help="Release dates YAML file",
    )
    parser.add_argument(
        "--after-date",
        type=str,
        default="2023-01-01",
        help="Only include SOTA models released on or after this date",
    )
    parser.add_argument(
        "--before-date",
        type=str,
        default=None,
        help="Only include SOTA models released before this date",
    )
    parser.add_argument(
        "--output-plot-file",
        type=pathlib.Path,
        required=True,
        help="Output file for histogram plot",
    )
    parser.add_argument(
        "--plot-format",
        type=str,
        default="png",
        help="Plot format (png, pdf, svg)",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )

    # Load colors from DVC params
    params = dvc.api.params_show()
    color_1 = params.get("comparison_color_1", "blue")
    color_2 = params.get("comparison_color_2", "orange")

    release_dates_data = yaml.safe_load(args.release_dates.read_text())
    release_dates = release_dates_data["date"]

    logger.info(f"Computing doubling times for {args.label_1}...")
    doubling_times_1, _, point_estimate_1 = compute_doubling_times_for_report(
        bootstrap_file=args.bootstrap_file_1,
        agent_summaries_file=args.agent_summaries_file_1,
        release_dates=release_dates,
        after_date=args.after_date,
        before_date=args.before_date,
    )

    logger.info(f"Computing doubling times for {args.label_2}...")
    doubling_times_2, _, point_estimate_2 = compute_doubling_times_for_report(
        bootstrap_file=args.bootstrap_file_2,
        agent_summaries_file=args.agent_summaries_file_2,
        release_dates=release_dates,
        after_date=args.after_date,
        before_date=args.before_date,
    )

    # Create plot
    plot_overlapping_histograms(
        doubling_times_1=doubling_times_1,
        doubling_times_2=doubling_times_2,
        label_1=args.label_1,
        label_2=args.label_2,
        point_estimate_1=point_estimate_1,
        point_estimate_2=point_estimate_2,
        output_file=args.output_plot_file,
        plot_format=args.plot_format,
        color_1=color_1,
        color_2=color_2,
    )


if __name__ == "__main__":
    main()
