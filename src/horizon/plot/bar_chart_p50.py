import argparse
import logging
import pathlib
import textwrap

import dvc.api
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def overall_bar_chart_weighted(
    df: pd.DataFrame,
    ordered_agents: list[str],
    highlight_agent: str | None = None,
) -> Figure:
    agent_data = {}

    for agent in ordered_agents:
        if agent not in df["agent"].values:
            raise ValueError(f"Agent {agent} not found in data")

        agent_row = df[df["agent"] == agent]

        agent_data[agent] = {
            "median": agent_row["p50"].iloc[0],
            "ci_low": agent_row["p50q0.025"].iloc[0],
            "ci_high": agent_row["p50q0.975"].iloc[0],
        }

    min_ci = min(agent_dict["ci_low"] for agent_dict in agent_data.values())
    max_ci = max(agent_dict["ci_high"] for agent_dict in agent_data.values())

    y_min = min_ci * 0.8
    y_max = max_ci * 1.2

    fig, ax = plt.subplots(
        figsize=(max(8, 1.5 * len(ordered_agents)), 6),
    )
    plt.subplots_adjust(bottom=0.25)

    for idx_agent, agent in enumerate(ordered_agents):
        data = agent_data[agent]
        median = data["median"]
        ci_low = data["ci_low"]
        ci_high = data["ci_high"]

        bar_color = "#cccccc" if agent != highlight_agent else "#4F6BFE"

        yerr_lower = median - ci_low
        yerr_upper = ci_high - median

        ax.bar(
            idx_agent + 0.66,
            median,
            color=bar_color,
            lw=1.5,
            yerr=[[yerr_lower], [yerr_upper]],
            error_kw=dict(capsize=7.5, lw=1.5, capthick=1.5),
            zorder=1,
            alpha=1.0,
        )

    ax.set_axisbelow(False)
    ax.spines["bottom"].set_zorder(3)

    wrapped_labels = [textwrap.fill(label, width=12) for label in ordered_agents]
    plt.xticks([x + 0.66 for x in range(0, len(ordered_agents))], wrapped_labels)

    ax.set_ylabel("Task duration\n(measured in hours of human baseliner time)")
    ax.set_ylim(y_min, y_max)
    ax.set_yscale("log")

    # Used to format time labels on y-axis
    def format_time_label(seconds: float) -> str:
        seconds = round(seconds)
        hours = seconds / 3600
        if hours >= 24:
            return f"{int(hours / 24)}d"
        elif hours >= 1:
            remainder = seconds % 3600
            minutes_str = (
                (", " + format_time_label(remainder)) if remainder > 60 else ""
            )
            return f"{int(hours)} hr" + ("s" if int(hours) > 1 else "") + minutes_str
        elif hours >= 1 / 60:
            return f"{int(hours * 60)} min"
        else:
            return f"{int(seconds)} sec"

    standard_ticks = np.array(
        [
            1 / 60,  # 1 second
            5 / 60,  # 5 seconds
            15 / 60,  # 15 seconds
            30 / 60,  # 30 seconds
            1,  # 1 minute
            2,  # 2 minutes
            4,  # 4 minutes
            8,  # 8 minutes
            15,  # 15 minutes
            30,  # 30 minutes
            60,  # 1 hour
            2 * 60,  # 2 hours
            4 * 60,  # 4 hours
            8 * 60,  # 8 hours
            12 * 60,  # 12 hours
            24 * 60,  # 1 day
            2 * 24 * 60,  # 2 days
            3 * 24 * 60,  # 3 days
            7 * 24 * 60,  # 1 week
        ]
    )

    # Filter to standard ticks within our range and use them if we have enough
    standard_in_range = standard_ticks[
        (standard_ticks >= y_min) & (standard_ticks <= y_max)
    ]
    if len(standard_in_range) >= 4:
        yticks = standard_in_range
    else:
        log_min, log_max = np.log10(y_min), np.log10(y_max)
        num_ticks = max(5, min(9, int(log_max - log_min) * 3))
        yticks = np.logspace(log_min, log_max, num_ticks)
        logger.info(
            f"Not enough standard ticks in range, using calculated ticks: {standard_in_range}"
        )

    multiplier = 60  # minutes to seconds
    ax.set_yticks(yticks)
    ytick_labels = [format_time_label(tick * multiplier) for tick in yticks]
    ax.set_yticklabels(ytick_labels)

    logger.info(f"Y-axis ticks: {yticks}")
    logger.info(f"Y-axis tick labels: {ytick_labels}")

    ax.set_title("Task duration where we predict 50% chance of AI success")
    fig.tight_layout()

    return fig


def main(
    input_file: pathlib.Path,
    output_file: pathlib.Path,
    log_level: str,
) -> None:
    logging.basicConfig(level=log_level.upper())

    params = dvc.api.params_show(stages="plot_bar_chart", deps=True)
    bar_chart_config = params["figs"]["plot_bar_chart"]

    ordered_agents = bar_chart_config["bar_chart_agents_ordered"]
    highlight_agent = bar_chart_config["bar_chart_highlight_agent"]

    df = pd.read_csv(input_file)

    fig = overall_bar_chart_weighted(
        df,
        ordered_agents=ordered_agents,
        highlight_agent=highlight_agent,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=pathlib.Path, required=True)
    parser.add_argument("--output-file", type=pathlib.Path, required=True)
    parser.add_argument("--log-level", type=str, required=True)
    args = parser.parse_args()

    main(**vars(args))
