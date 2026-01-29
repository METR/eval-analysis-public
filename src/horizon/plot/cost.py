"""Plot cost ratios for different time buckets and overall costs."""

import argparse
import logging
import pathlib
import random

import dvc.api
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from horizon.plot.individual_histograms import ScriptParams
from horizon.utils import plots


def _plot_bucketed_cost_ratios(
    savings: pd.DataFrame,
    plot_params: plots.PlotParams,
    release_dates: dict[str, str],
    script_params: ScriptParams,
) -> None:
    savings = savings.copy()
    """Plot cost ratios for different time buckets."""
    # Store legend order
    legend_order = plot_params["legend_order"]

    # Plot the cost ratio for each time bucket with the datapoint for each model
    buckets = sorted(
        savings.index.get_level_values("human_minutes_bucket").unique(),
        key=lambda x: int(x.split("-")[0]),
    )
    n_rows = (len(buckets) + 1) // 2  # Calculate number of rows needed for 2 columns
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 4 * n_rows))
    axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing

    savings["release_date"] = [release_dates[m[0]] for m in savings.index]
    savings.sort_values(by="release_date", inplace=True)

    # Create subplots in a grid
    legend_handles = []
    legend_labels = []

    for i, bucket in enumerate(buckets):
        data = savings.xs(bucket, level="human_minutes_bucket")

        # Apply grid styling
        axes[i].grid(**plot_params["scatter_styling"]["grid"])

        # Plot points for each agent
        for agent in script_params["include_agents"]:
            agent_style = plot_params["agent_styling"][agent]
            scatter = axes[i].scatter(
                data.loc[agent, "release_date"],
                data.loc[agent, "cost_ratio"],
                color=agent_style["lab_color"],
                marker=agent_style["marker"],
                **plot_params["scatter_styling"]["scatter"],
            )

            # Only add to legend once
            if agent not in legend_labels:
                legend_handles.append(scatter)
                legend_labels.append(agent)

        # Style the axes
        axes[i].set_title(
            f"Time bucket: {bucket} minutes",
            fontsize=plot_params["title_fontsize"],
            pad=plot_params["xlabelpad"],
        )
        axes[i].set_xlabel(
            "Release Date",
            fontsize=plot_params["ax_label_fontsize"],
            labelpad=plot_params["xlabelpad"],
        )
        axes[i].set_ylabel(
            "Cost Ratio",
            fontsize=plot_params["ax_label_fontsize"],
            labelpad=plot_params["ylabelpad"],
        )
        axes[i].set_ylim(0, 1.5)
        axes[i].tick_params(axis="x", rotation=45)

    # Remove any empty subplots
    for i in range(len(buckets), len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle(
        "Cost Ratio by Time Bucket", fontsize=plot_params["title_fontsize"] + 2, y=1.02
    )

    # Sort legend elements according to the predefined order
    legend_elements = sorted(
        zip(legend_handles, legend_labels),
        key=lambda x: (
            legend_order.index(x[1]) if x[1] in legend_order else float("inf")
        ),
    )
    handles, labels = zip(*legend_elements)

    # Add the legend below the subplots
    fig.legend(
        handles=handles,
        labels=labels,
        loc="center right",
        bbox_to_anchor=(0.98, 0.1),
        fontsize=plot_params["ax_label_fontsize"],
    )

    plt.tight_layout()
    pathlib.Path("plots/cost").mkdir(parents=True, exist_ok=True)
    plt.savefig(
        "plots/cost/bucketed.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def _plot_overall_cost_ratio(
    savings_non_bucketed: pd.DataFrame,
    plot_params: plots.PlotParams,
    release_dates: dict[str, str],
    script_params: ScriptParams,
) -> None:
    """Plot the overall cost ratio for each model."""
    savings_non_bucketed = savings_non_bucketed.copy()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Add release dates to non-bucketed data
    savings_non_bucketed["release_date"] = [
        release_dates[m] for m in savings_non_bucketed.index
    ]
    savings_non_bucketed.sort_values(by="release_date", inplace=True)

    # Apply grid styling
    ax.grid(**plot_params["scatter_styling"]["grid"])

    # Plot points for each agent
    legend_handles = []
    legend_labels = []

    for agent in script_params["include_agents"]:
        agent_style = plot_params["agent_styling"].get(
            agent, plot_params["agent_styling"]["default"]
        )
        scatter = ax.scatter(
            np.array([savings_non_bucketed.loc[agent, "release_date"]]),
            np.array([savings_non_bucketed.loc[agent, "cost_ratio"]]),
            color=agent_style["lab_color"],
            marker=agent_style["marker"],
            **plot_params["scatter_styling"]["scatter"],
        )
        legend_handles.append(scatter)
        legend_labels.append(agent)

    # Style the axes
    ax.set_title(
        "Cost to complete all tasks once,\nusing models for tasks which they\nare capable of completing",
        fontsize=plot_params["title_fontsize"],
        pad=plot_params["xlabelpad"],
    )
    ax.set_xlabel(
        "Release Date",
        fontsize=plot_params["ax_label_fontsize"],
        labelpad=plot_params["xlabelpad"],
    )
    ax.set_ylabel(
        "Cost (1 = all tasks performed by a human)",
        fontsize=plot_params["ax_label_fontsize"],
        labelpad=plot_params["ylabelpad"],
    )
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(0, 1.2)

    # Sort legend elements according to the predefined order
    legend_elements = sorted(
        zip(legend_handles, legend_labels),
        key=lambda x: (
            plot_params["legend_order"].index(x[1])
            if x[1] in plot_params["legend_order"]
            else float("inf")
        ),
    )
    handles, labels = zip(*legend_elements)

    # Add the legend
    ax.legend(
        handles=handles,
        labels=labels,
        loc="center left",
        bbox_to_anchor=(1.15, 0.5),
        fontsize=plot_params["ax_label_fontsize"],
    )

    plt.tight_layout()
    plt.savefig(
        "plots/cost/non_bucketed.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def _plot_agent_cost_grid(
    savings: pd.DataFrame,
    plot_params: plots.PlotParams,
    script_params: ScriptParams,
) -> None:
    """Plot cost ratios in a grid with agents as rows and time buckets as columns."""
    savings = savings.copy()
    agents = script_params["include_agents"]
    buckets = sorted(
        savings.index.get_level_values("human_minutes_bucket").unique(),
        key=lambda x: int(x.split("-")[0]),
    )
    n_cols = 4
    n_rows = (len(agents) + n_cols - 1) // n_cols  # Ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten()

    for i, agent in enumerate(agents):
        agent_data = savings.xs(agent, level=0)
        agent_style = plot_params["agent_styling"][agent]

        # Ensure cost ratios are in the same order as buckets
        cost_ratios = [agent_data.loc[bucket, "cost_ratio"] for bucket in buckets]

        axes[i].grid(**plot_params["scatter_styling"]["grid"])
        axes[i].bar(
            range(len(buckets)),
            cost_ratios,
            color=agent_style["lab_color"],
            alpha=0.7,
        )

        axes[i].set_title(
            agent,
            fontsize=plot_params["ax_label_fontsize"],
            pad=plot_params["xlabelpad"],
        )
        axes[i].set_xticks(range(len(buckets)))
        axes[i].set_xticklabels(buckets, rotation=45, ha="right")
        axes[i].set_ylim(0, 1.5)
        axes[i].set_ylabel("Cost Ratio")

    # Remove empty subplots
    for i in range(len(agents), len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle(
        "Cost Ratio by Agent and Time Bucket",
        fontsize=plot_params["title_fontsize"] + 2,
        y=1.02,
    )

    plt.tight_layout()
    pathlib.Path("plots/cost").mkdir(parents=True, exist_ok=True)
    plt.savefig(
        "plots/cost/agent_grid.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def _plot_cost_ratio_histograms(
    costs: pd.DataFrame,
    plot_params: plots.PlotParams,
    script_params: ScriptParams,
) -> None:
    """Plot histograms of cost ratios for each time bucket."""
    costs = costs.copy()
    costs = costs[
        costs["score_binarized"] != 0
    ]  # just tasks where it succeeded at least once
    costs["cost_ratio"] = costs["generation_cost"] / costs["human_cost"]

    buckets = sorted(
        costs["human_minutes_bucket"].unique(),
        key=lambda x: int(x.split("-")[0]),
    )
    n_rows = (len(buckets) + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, bucket in enumerate(buckets):
        bucket_data = costs[costs["human_minutes_bucket"] == bucket]
        total_count = len(bucket_data)

        # Calculate percentage of values outside range
        below_min = (bucket_data["cost_ratio"] < 0).sum()
        above_max = (bucket_data["cost_ratio"] > 2).sum()
        outside_pct = ((below_min + above_max) / total_count) * 100

        # Calculate histogram values for data within range
        in_range_data = bucket_data[
            (bucket_data["cost_ratio"] >= 0) & (bucket_data["cost_ratio"] <= 2)
        ]["cost_ratio"]
        counts, bins = np.histogram(
            in_range_data, bins=np.logspace(np.log10(1e-7), np.log10(2), 30)
        )

        # Convert to percentages (relative to total count including out-of-range values)
        percentages = (counts / total_count) * 100

        axes[i].grid(**plot_params["scatter_styling"]["grid"])
        axes[i].bar(
            bins[:-1],  # Left edges of bins
            percentages,
            width=np.diff(bins),  # Width of each bar
            align="edge",
            color="steelblue",
            alpha=0.7,
        )

        axes[i].set_title(
            f"Time bucket: {bucket} minutes\n"
            f"n={total_count} ({outside_pct:.1f}% outside [0,2])",
            fontsize=plot_params["ax_label_fontsize"],
            pad=plot_params["xlabelpad"],
        )
        axes[i].set_xlabel(
            "Cost Ratio (Model Cost / Human Cost)",
            fontsize=plot_params["ax_label_fontsize"],
            labelpad=plot_params["xlabelpad"],
        )
        axes[i].set_ylabel(
            "Percentage of Tasks",
            fontsize=plot_params["ax_label_fontsize"],
            labelpad=plot_params["ylabelpad"],
        )
        axes[i].set_ylim(0, 100)  # Set a reasonable y-limit for percentages
        axes[i].set_xscale("log")

    # Remove any empty subplots
    for i in range(len(buckets), len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle(
        "Distribution of Cost Ratios by Time Bucket",
        fontsize=plot_params["title_fontsize"] + 2,
        y=1.02,
    )

    plt.tight_layout()
    pathlib.Path("plots/cost").mkdir(parents=True, exist_ok=True)
    plt.savefig(
        "plots/cost/ratio_histograms.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    # Plot overall histogram in a separate figure
    fig, ax = plt.subplots(figsize=(10, 6))
    total_count = len(costs)
    below_min = (costs["cost_ratio"] < 0).sum()
    above_max = (costs["cost_ratio"] > 2).sum()
    outside_pct = ((below_min + above_max) / total_count) * 100

    in_range_data = costs[(costs["cost_ratio"] >= 0) & (costs["cost_ratio"] <= 2)][
        "cost_ratio"
    ]
    counts, bins = np.histogram(in_range_data, bins=30, range=(0, 2))
    percentages = (counts / total_count) * 100

    ax.grid(**plot_params["scatter_styling"]["grid"])
    ax.bar(
        bins[:-1],
        percentages,
        width=np.diff(bins),
        align="edge",
        color="steelblue",
        alpha=0.7,
    )

    ax.set_title(
        f"Overall Distribution of Cost Ratios\n"
        f"n={total_count} ({outside_pct:.1f}% outside [0,2])",
        fontsize=plot_params["ax_label_fontsize"],
        pad=plot_params["xlabelpad"],
    )
    ax.set_xlabel(
        "Cost Ratio (Model Cost / Human Cost)",
        fontsize=plot_params["ax_label_fontsize"],
        labelpad=plot_params["xlabelpad"],
    )
    ax.set_ylabel(
        "Percentage of Tasks",
        fontsize=plot_params["ax_label_fontsize"],
        labelpad=plot_params["ylabelpad"],
    )
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(
        "plots/cost/ratio_histogram_overall.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def _plot_duration_stats(
    costs: pd.DataFrame,
    plot_params: plots.PlotParams,
    script_params: ScriptParams,
) -> None:
    """Plot histograms of cost ratios for each time bucket."""
    costs = costs.copy()
    costs = costs[costs["time_successful_runs"].notna()]
    costs["time_ratio"] = (costs["time_successful_runs"] / 1000 / 60) / costs[
        "human_minutes"
    ]
    buckets = sorted(
        costs["human_minutes_bucket"].unique(),
        key=lambda x: int(x.split("-")[0]),
    )
    n_rows = (len(buckets) + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, bucket in enumerate(buckets):
        bucket_data = costs[costs["human_minutes_bucket"] == bucket]
        total_count = len(bucket_data)

        # Calculate percentage of values outside range
        below_min = (bucket_data["time_ratio"] < 0).sum()
        above_max = (bucket_data["time_ratio"] > 2).sum()
        outside_pct = ((below_min + above_max) / total_count) * 100

        # Calculate histogram values for data within range
        in_range_data = bucket_data[
            (bucket_data["time_ratio"] >= 0) & (bucket_data["time_ratio"] <= 2)
        ]["time_ratio"]
        counts, bins = np.histogram(in_range_data, bins=30, range=(0, 2))

        # Convert to percentages (relative to total count including out-of-range values)
        percentages = (counts / total_count) * 100

        axes[i].grid(**plot_params["scatter_styling"]["grid"])
        axes[i].bar(
            bins[:-1],  # Left edges of bins
            percentages,
            width=np.diff(bins),  # Width of each bar
            align="edge",
            color="steelblue",
            alpha=0.7,
        )

        axes[i].set_title(
            f"Time bucket: {bucket} minutes\n"
            f"n={total_count} ({outside_pct:.1f}% outside [0,2])",
            fontsize=plot_params["ax_label_fontsize"],
            pad=plot_params["xlabelpad"],
        )
        axes[i].set_xlabel(
            "Time Ratio (Model Time / Human Time)",
            fontsize=plot_params["ax_label_fontsize"],
            labelpad=plot_params["xlabelpad"],
        )
        axes[i].set_ylabel(
            "Percentage of Tasks",
            fontsize=plot_params["ax_label_fontsize"],
            labelpad=plot_params["ylabelpad"],
        )
        axes[i].set_ylim(0, 100)  # Set a reasonable y-limit for percentages

    # Remove any empty subplots
    for i in range(len(buckets), len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle(
        "Distribution of Time Ratios by Time Bucket",
        fontsize=plot_params["title_fontsize"] + 2,
        y=1.02,
    )

    plt.tight_layout()
    pathlib.Path("plots/cost").mkdir(parents=True, exist_ok=True)
    plt.savefig(
        "plots/cost/time_histograms.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def _plot_cost_ratio_vs_length(
    costs: pd.DataFrame,
    plot_params: plots.PlotParams,
    script_params: ScriptParams,
) -> None:
    """Plot scatter of cost ratio vs task length with logarithmic axes."""
    costs = costs.copy()
    costs = costs[
        costs["score_binarized"] != 0
    ]  # just tasks where it succeeded at least once
    costs["cost_ratio"] = costs["generation_cost"] / costs["human_cost"]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Filter to reasonable ranges for visualization
    plot_data = costs[
        (costs["cost_ratio"] > 0)  # Exclude 0 for log scale
        & (costs["cost_ratio"] <= 2)
    ]

    # Calculate percentage of filtered points
    total_count = len(costs)
    filtered_count = len(plot_data)
    filtered_pct = ((total_count - filtered_count) / total_count) * 100

    ax.grid(**plot_params["scatter_styling"]["grid"])

    # Add jitter to both dimensions
    rng = np.random.default_rng(seed=42)
    jitter_amount = 0.03  # Adjust this value to control jitter magnitude
    x_jitter = rng.normal(0, jitter_amount, size=len(plot_data))
    y_jitter = rng.normal(0, jitter_amount, size=len(plot_data))

    ax.scatter(
        plot_data["human_minutes"] * (1 + x_jitter),
        plot_data["cost_ratio"] * (1 + y_jitter),
        alpha=0.5,
        color="steelblue",
        s=50,
    )

    # Set up logarithmic axes with proper formatting
    plots.log_x_axis(ax, unit="minutes")
    ax.set_yscale("log")

    ax.set_title(
        f"Cost Ratio vs Task Length\n"
        f"n={filtered_count} ({filtered_pct:.1f}% outside visible range)",
        fontsize=plot_params["ax_label_fontsize"],
        pad=plot_params["xlabelpad"],
    )
    ax.set_xlabel(
        "Task Length",
        fontsize=plot_params["ax_label_fontsize"],
        labelpad=plot_params["xlabelpad"],
    )
    ax.set_ylabel(
        "Cost Ratio (Model Cost / Human Cost)",
        fontsize=plot_params["ax_label_fontsize"],
        labelpad=plot_params["ylabelpad"],
    )

    # Set y-axis limits to show range from ~0.01 to 2
    ax.set_ylim(1e-6, 2)

    plt.tight_layout()
    pathlib.Path("plots/cost").mkdir(parents=True, exist_ok=True)
    plt.savefig(
        "plots/cost/ratio_vs_length.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def _plot_cost_ratio_vs_length_swarm(
    costs: pd.DataFrame,
    plot_params: plots.PlotParams,
    script_params: ScriptParams,
) -> None:
    """Plot swarm plot of cost ratio vs task length buckets."""
    costs = costs.copy()
    costs = costs[
        costs["score_binarized"] != 0
    ]  # just tasks where it succeeded at least once
    costs["cost_ratio"] = costs["generation_cost"] / costs["human_cost"]

    # Filter to reasonable ranges for visualization
    plot_data = costs[
        (costs["cost_ratio"] > 0)  # Exclude 0 for log scale
        & (costs["cost_ratio"] <= 2)
    ]

    # Calculate percentage of filtered points
    total_count = len(costs)
    filtered_count = len(plot_data)
    filtered_pct = ((total_count - filtered_count) / total_count) * 100

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.grid(**plot_params["scatter_styling"]["grid"])

    # Create swarm plot using seaborn
    import seaborn as sns

    sns.swarmplot(
        data=plot_data,
        x="human_minutes_bucket",
        y="cost_ratio",
        size=5,
        alpha=0.6,
        color="steelblue",
    )
    sns.stripplot(
        data=plot_data,
        x="human_minutes_bucket",
        y="cost_ratio",
        size=5,
        alpha=0.3,
        color="steelblue",
        jitter=0.2,
    )

    ax.set_yscale("log")
    ax.set_title(
        f"Cost Ratio vs Task Length (Swarm Plot)\n"
        f"n={filtered_count} ({filtered_pct:.1f}% outside visible range)",
        fontsize=plot_params["ax_label_fontsize"],
        pad=plot_params["xlabelpad"],
    )
    ax.set_xlabel(
        "Task Length Bucket",
        fontsize=plot_params["ax_label_fontsize"],
        labelpad=plot_params["xlabelpad"],
    )
    ax.set_ylabel(
        "Cost Ratio (Model Cost / Human Cost)",
        fontsize=plot_params["ax_label_fontsize"],
        labelpad=plot_params["ylabelpad"],
    )
    ax.tick_params(axis="x", rotation=45)

    # Set y-axis limits to show range from ~0.01 to 2
    ax.set_ylim(0.01, 2)

    plt.tight_layout()
    pathlib.Path("plots/cost").mkdir(parents=True, exist_ok=True)
    plt.savefig(
        "plots/cost/ratio_vs_length_swarm.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--savings-info-file", type=pathlib.Path, required=True
    )  # metrics/costs/savings_info.csv
    parser.add_argument(
        "--savings-non-bucketed-info-file", type=pathlib.Path, required=True
    )  # metrics/costs/savings_non_bucketed_info.csv
    parser.add_argument(
        "--release-dates-file", type=pathlib.Path, required=True
    )  # data/external/release_dates.yaml
    parser.add_argument(
        "--cost-info-file",
        type=pathlib.Path,
        default="data/processed/wrangled/costs/cost_info.csv",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    params = dvc.api.params_show(stages="plot_cost", deps=True)

    # Load savings data
    savings = pd.read_csv(
        args.savings_info_file, index_col=[0, 1]
    )  # metrics/costs/savings_info.csv
    savings_non_bucketed = pd.read_csv(args.savings_non_bucketed_info_file, index_col=0)

    # Load raw cost data
    costs = pd.read_csv(args.cost_info_file)

    # Load release dates
    with open(args.release_dates_file, "r") as f:
        release_dates = yaml.safe_load(f)["date"]

    _plot_bucketed_cost_ratios(
        savings, params["plots"], release_dates, params["plots"]["plot_cost"]
    )
    _plot_overall_cost_ratio(
        savings_non_bucketed,
        params["plots"],
        release_dates,
        params["plots"]["plot_cost"],
    )
    _plot_agent_cost_grid(
        savings,
        params["plots"],
        params["plots"]["plot_cost"],
    )
    _plot_cost_ratio_vs_length(
        costs,
        params["plots"],
        params["plots"]["plot_cost"],
    )
    _plot_cost_ratio_vs_length_swarm(
        costs,
        params["plots"],
        params["plots"]["plot_cost"],
    )
    _plot_cost_ratio_histograms(
        costs,
        params["plots"],
        params["plots"]["plot_cost"],
    )
    _plot_duration_stats(
        costs,
        params["plots"],
        params["plots"]["plot_cost"],
    )


if __name__ == "__main__":
    main()
