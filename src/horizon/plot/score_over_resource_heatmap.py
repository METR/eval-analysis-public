import argparse
import datetime
import pathlib
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib.figure import Figure

COST_BINS = [
    (0.00, 0.01),
    (0.01, 0.1),
    (0.1, 1),
    (1, 10),
    (10, 100),
    (100, 1000),
    (1000, 10000),
]

SUCCESS_RATE_COLUMN = "success_rate_at_cost"
WEIGHTED_SUCCESS_RATE_COLUMN = "weighted_success_rate_at_cost"
WEIGHT_COL_NAMES = {
    "equal_task_weight": "Weighting Tasks Equally",
    "invsqrt_task_weight": "Weighted by Task Diversity",
}
FIGSIZE = (10, 6)
TITLE = {
    "generation_cost": "Agent Performance on HCAST & RE-Bench by Cost",
    "action_count": "Agent Performance on HCAST & RE-Bench by Actions",
}
YLABEL = "Weighted Success Rate"
XLABEL = {
    "generation_cost": "Allowed Cost per Run ($)",
    "action_count": "Allowed Actions per Run",
}


def _get_order_from_release_dates(
    df: pd.DataFrame, release_dates: Dict[str, datetime.date]
) -> list[str]:
    release_dates_unix = {
        agent: datetime.datetime.combine(date, datetime.time()).timestamp()
        for agent, date in release_dates.items()
    }
    order = sorted(release_dates_unix.keys(), key=lambda x: release_dates_unix[x])
    release_date_ordered_agents = [
        agent for agent in order if agent in df["alias"].unique()
    ]
    for human in ["human", "best human for each task"]:
        if human in df["alias"].unique():
            release_date_ordered_agents.append(human)
    return release_date_ordered_agents


def _score_heatmap(
    df: pd.DataFrame,
    agents: list[str],
    release_dates: Dict[str, datetime.date],
    usage_column: str,
    weighting_column: str,
) -> Figure:
    # Make the figure taller
    fig, ax = plt.subplots(figsize=(10, 10))

    all_agent_data = []

    # Calculate weighted success rate at cost for each agent
    for agent in agents:
        df_agent = df[df["alias"] == agent]
        if df_agent.empty:
            continue

        agent_data = df_agent[
            ["alias", usage_column, WEIGHTED_SUCCESS_RATE_COLUMN]
        ].copy()
        all_agent_data.append(agent_data)

    if not all_agent_data:
        raise ValueError("No valid agent data found")

    # Combine all agent data
    heatmap_df = pd.concat(all_agent_data)

    # Handle duplicate combinations of alias and usage_column by taking the last value
    # (consistent with how the line chart handles duplicates)
    heatmap_df = heatmap_df.drop_duplicates(subset=["alias", usage_column], keep="last")

    # Use the predefined COST_BINS instead of dynamically creating bins
    if usage_column == "generation_cost":
        bins = [min_cost for min_cost, _ in COST_BINS] + [COST_BINS[-1][1]]
        bin_labels = [
            f"${min_cost:.2f}-{max_cost:.2f}" for min_cost, max_cost in COST_BINS
        ]
    else:
        raise ValueError(
            f"Usage column {usage_column} is not supported for heatmap yet"
        )

    # Apply binning
    heatmap_df["usage_bin"] = pd.cut(
        heatmap_df[usage_column],
        bins=bins,
        labels=bin_labels,
        include_lowest=True,
        ordered=True,  # Ensure bins are ordered
    )

    # For each agent and bin, keep the row with maximum weighted success rate
    grouped = (
        heatmap_df.groupby(["alias", "usage_bin"], observed=True)[
            WEIGHTED_SUCCESS_RATE_COLUMN
        ]
        .max()
        .reset_index()
    )

    # Create pivot table for heatmap
    pivot_df = grouped.pivot(
        index="alias", columns="usage_bin", values=WEIGHTED_SUCCESS_RATE_COLUMN
    )

    # Ensure columns are ordered by bin value, not alphabetically
    pivot_df = pivot_df.reindex(columns=bin_labels)

    # Fill NaNs with most recent usage value for that agent (forward fill)
    pivot_df = pivot_df.ffill(axis=1)

    # Reindex pivot table with custom order
    custom_order = _get_order_from_release_dates(df, release_dates)
    pivot_df = pivot_df.reindex(custom_order)

    # Format annotations to show percentages
    annot_matrix = pivot_df.apply(
        lambda col: col.apply(lambda x: f"{x:.0%}" if x > 0 else "0%")
    )

    # Plot heatmap with annotations
    _ = sns.heatmap(
        pivot_df,
        ax=ax,
        cmap="viridis",
        vmin=0,
        vmax=1,
        cbar_kws={"label": YLABEL},
        annot=annot_matrix,  # Add cell labels
        fmt="",  # Use formatted annot_matrix
        annot_kws={"size": 10},  # Adjust annotation text size
    )

    ax.set_title(f"{TITLE[usage_column]}\n({WEIGHT_COL_NAMES[weighting_column]})")
    ax.set_xlabel(XLABEL[usage_column])
    ax.set_ylabel("Agent")

    # Improve x-axis appearance by showing bin boundaries
    # Get positions for the middle of each bin cell
    num_bins = len(bin_labels)

    # Create labels showing the actual bin boundaries
    boundary_labels = [f"${cost:.2f}" for cost in bins]

    # Set custom tick positions and labels
    ax.set_xticks(np.arange(num_bins + 1))
    ax.set_xticklabels(boundary_labels, rotation=45, ha="right")

    # Add more space at the bottom for rotated x labels
    plt.tight_layout()
    return fig


def _load_release_dates(release_dates_file: pathlib.Path) -> Dict[str, datetime.date]:
    return yaml.safe_load(open(str(release_dates_file)))["date"]


def main(
    wrangled_resource_file: pathlib.Path,
    output_file: pathlib.Path,
    weighting_column: str,
    release_dates_file: pathlib.Path,
    x: str,
) -> None:
    release_dates = _load_release_dates(release_dates_file)
    df = pd.read_json(wrangled_resource_file, lines=True, orient="records")
    heatmap_fig = _score_heatmap(
        df,
        agents=list(df["alias"].unique()),
        usage_column=x,
        weighting_column=weighting_column,
        release_dates=release_dates,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving figure to {output_file}")
    heatmap_fig.savefig(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wrangled-resource-file",
        type=pathlib.Path,
        required=True,
        help="Path to the wrangled resource data file (jsonl format)",
    )
    parser.add_argument(
        "--release-dates-file",
        type=pathlib.Path,
        required=True,
        help="Path to the release dates file (yaml format)",
    )
    parser.add_argument(
        "--output-file",
        type=pathlib.Path,
        help="Path to save the output figure",
        required=True,
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
        help="Column to use for x-axis",
    )

    args = parser.parse_args()
    main(**vars(args))
