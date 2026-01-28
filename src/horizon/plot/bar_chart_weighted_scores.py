from __future__ import annotations

import argparse
import logging
import pathlib
from typing import Any

import dvc.api
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from horizon.utils import plots


def _get_title(script_params: plots.ScriptParams) -> str:
    # Get included task groups
    task_group_names = ["HCAST", "SWAA", "RE-Bench"]
    included_task_groups = []
    for name in task_group_names:
        if name not in script_params["exclude"]:
            included_task_groups.append(name)

    # Make title
    task_groups_string = "+ ".join(included_task_groups)
    title = f"Average Success Rate for {task_groups_string} Tasks"
    return title


def plot_weighted_scores(
    params: Any,
    stage_params: Any,
    df: pd.DataFrame,
    release_dates: dict[str, Any],
    focus_agents: list[str] | None = None,
    ylabel: str = "Average success rate",
) -> Figure:
    """Plot a bar chart showing weighted scores for different agents.

    Args:
        plot_params: Plot parameters including colors and legend order
        df: DataFrame containing agent scores and confidence intervals
        release_dates: Dictionary mapping agent names to release dates
        title: Title for the plot
        focus_agents: Optional list of agents to focus on
        ylabel: Label for y-axis

    Returns:
        matplotlib.figure.Figure: The generated plot
    """
    df = df.copy()

    # Add release dates to DataFrame
    df["release_date"] = pd.to_datetime(
        [release_dates["date"].get(agent, None) for agent in df["agent"]]
    )

    # Filter for focus agents if specified
    if focus_agents:
        df = df[df["agent"].isin(focus_agents)]
        for agent in focus_agents:
            assert agent in df["agent"].unique(), f"Agent {agent} not found in df"

    # Sort agents by release date
    df = df.sort_values("release_date")
    agent_ordering = df["agent"].tolist()
    logging.info(f"Plotting agents: {agent_ordering}")

    # Get colors for each agent
    agent_colors = [
        plots.get_agent_color(params["plots"], agent) for agent in agent_ordering
    ]

    # Create figure with appropriate width
    fig, ax = plt.subplots(
        figsize=(10, 5),
        tight_layout=True,
    )

    # Plot bars for each agent
    bar_offset = 0  # Offset for bar x positions
    for idx, agent in enumerate(agent_ordering):
        agent_data = df[df["agent"] == agent]
        score = agent_data["average"].item()
        color = agent_colors[idx]

        # Plot bar without error bars
        ax.bar(
            idx + bar_offset,
            score,
            color=color,
            edgecolor=color,
            lw=1.5,
            zorder=4,
        )

    # Configure axes
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=1, which="both", alpha=0.5)

    plt.xticks(
        [x + bar_offset for x in range(len(agent_ordering))],
        agent_ordering,
        rotation=45,
        ha="right",
    )

    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)  # Success rate goes from 0 to 1
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: "{:.0%}".format(y))
    )  # Format as percentage

    title = _get_title(stage_params)

    possible_weightings = params["weighting"]
    for possible_weighting in possible_weightings:
        if possible_weighting["weight_col"] == stage_params["weighting"]:
            weighting = possible_weighting
            break
    else:
        raise ValueError(f"Weighting {stage_params['weighting']} not found")
    subtitle = weighting["graph_snippet"]

    ax.set_title(f"{title}\n{subtitle}")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig-name", type=str, required=True)
    parser.add_argument("--metrics-file", type=pathlib.Path, required=True)
    parser.add_argument("--release-dates", type=pathlib.Path, required=True)
    parser.add_argument("--output-file", type=pathlib.Path, required=True)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Load metrics data from logistic regression results
    metrics_df = pd.read_csv(args.metrics_file)
    logging.info("Loaded metrics data")

    # Load release dates
    release_dates = yaml.safe_load(args.release_dates.read_text())
    logging.info("Loaded release dates")

    # Load plot parameters
    params = dvc.api.params_show(stages="plot_bar_chart_weighted_scores", deps=True)

    stage_params = params["figs"]["plot_bar_chart_weighted_scores"]
    # print(f"focusing on {stage_params.get('focus_agents')}")

    plot_weighted_scores(
        params,
        stage_params,
        metrics_df,
        release_dates,
        focus_agents=stage_params.get("focus_agents"),
    )

    plots.save_or_open_plot(args.output_file, params["plot_format"])


if __name__ == "__main__":
    main()
