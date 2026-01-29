import argparse
import logging
import pathlib
from typing import Sequence

import dvc.api
import matplotlib.axes
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.special import expit

from horizon.utils import plots


def plot_logistic_function(
    ax: matplotlib.axes.Axes, agent_info: pd.DataFrame, color: str
) -> None:
    x = np.logspace(0, np.log(32), 1000)
    y = expit(
        agent_info.iloc[0]["coefficient"] * np.log2(x) + agent_info.iloc[0]["intercept"]
    )
    ax.plot(x, y, label=agent_info.iloc[0]["agent"], color=color)


def plot_logistic_regression(
    plot_params: plots.PlotParams,
    agent_summaries: pd.DataFrame,
    focus_agents: Sequence[str],
    show_example_p50: bool = False,
    show_empirical_rates: bool = False,
) -> None:
    agent_summaries = agent_summaries[agent_summaries["agent"].isin(focus_agents)]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot logistic curves for each agent
    for i, (_, agent_info) in enumerate(agent_summaries.groupby("agent")):
        agent_color = plots.get_agent_color(plot_params, agent_info.iloc[0]["agent"])
        plot_logistic_function(ax, agent_info, agent_color)

    ax.hlines(
        0.5, ax.get_xlim()[0], ax.get_xlim()[1], color="gray", linestyle="--", alpha=0.5
    )  # type: ignore[reportArgumentType]
    # Add p50 reference lines if requested
    if show_example_p50:
        p50 = agent_summaries["p50"].iloc[0]
        ax.vlines(p50, 0, 0.5, color="gray", linestyle="--", alpha=0.5)  # type: ignore[reportArgumentType]
        ax.scatter(p50, 0.5, color="gray", marker="o", alpha=0.8)  # type: ignore[reportArgumentType]
        ax.text(
            p50,
            0.05,
            f"50% @\n{p50:.0f} minutes",
            color="black",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    ax.set_xscale("log")
    # Add empirical data points if requested
    if show_empirical_rates:
        time_buckets = [1, 4, 16, 64, 256, 960]

        for i, (_, agent_info) in enumerate(agent_summaries.groupby("agent")):
            agent = agent_info.iloc[0]["agent"]
            agent_color = plots.get_agent_color(plot_params, agent)
            for j in range(len(time_buckets) - 1):
                start = time_buckets[j]
                end = time_buckets[j + 1]
                rate = agent_info.iloc[0][f"{start}-{end} min"]

                # ax.hlines(rate, start, end, color=colors[i], linestyle="--")
                ax.bar(
                    (start + end) / 2,
                    rate,
                    width=end - start,
                    color=agent_color,
                    align="center",
                    alpha=0.3,
                )

    # Customize axis appearance
    ax.set_xticks([1, 4, 15, 60, 4 * 60, 16 * 60])
    ax.set_xticklabels(["1 min", "4 min", "15 min", "1 hr", "4 hrs", "16 hrs"])
    ax.set_xlabel("Human time-to-complete")
    ax.set_ylabel("Probability of success")
    ax.set_title("Agent success rate vs time horizon, with logistic fit")

    ax.set_ylim(0, 1)
    plots.create_sorted_legend(ax, plot_params["legend_order"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=pathlib.Path, required=True)
    parser.add_argument("--output-file", type=pathlib.Path, required=True)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--show-p50", type=str, default="false")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    agent_summaries = pd.read_csv(args.input_file)
    params = dvc.api.params_show(stages="plot_logistic_individual")
    focus_agents = [
        # "Claude 3.5 Sonnet (New)",
        # "Claude 3.5 Sonnet (Old)",
        # "GPT-4 0314",
        # "GPT-4 Turbo",
        # "GPT-4o",
        # "gpt-3.5-turbo-instruct",
        # "o1-preview",
        "o1",
    ]

    logging.info("Loaded input data")

    plot_logistic_regression(
        params["plots"],
        agent_summaries,
        focus_agents,
        show_empirical_rates=True,
        show_example_p50=args.show_p50 == "true",
    )

    plots.save_or_open_plot(args.output_file, params["plot_format"])


if __name__ == "__main__":
    main()
