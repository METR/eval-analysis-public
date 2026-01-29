import argparse
import pathlib

import dvc.api
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

from horizon.utils import plots


def plot_task_distribution(
    runs_df: pd.DataFrame,
    plot_params: plots.PlotParams,
) -> None:
    """Plot a vertical histogram of the human time estimates for each run."""

    fig, ax = plt.subplots()
    data = runs_df.groupby("task_id")[["human_minutes", "task_source"]].first()

    data["color"] = data["task_source"]
    # Make sure we use the same size bins regardless of the range we're plotting
    seconds_bins = [x / 60 for x in [1, 2, 4, 8, 15, 30]]
    bins = seconds_bins + [1, 2, 4, 8, 15, 30, 60, 120, 240, 480, 960]

    mplrc = yaml.safe_load(open("matplotlibrc"))
    sns.set_theme(rc=mplrc)

    sns.histplot(
        data,
        x="human_minutes",
        bins=bins,  # type: ignore
        hue="task_source",
        multiple="stack",
        legend=True,
        ax=ax,
    )  # type: ignore
    ax.set_xlabel(
        "Number of tasks",
        fontsize=plot_params["ax_label_fontsize"],
        labelpad=plot_params["xlabelpad"],
    )

    legend = ax.get_legend()
    assert legend is not None, "Expected legend to exist on axis"
    handles = legend.legend_handles
    labels = [t.get_text() for t in legend.get_texts()]
    ax.legend(
        title="Task Source",
        title_fontsize=14,
        handles=handles,
        labels=labels,
        loc="upper right",
        bbox_to_anchor=(1, 1),
    )

    ax.grid(**plot_params["task_distribution_styling"]["grid"])

    plots.log_x_axis(ax)
    xticks = ax.get_xticks()
    xticklabels = ax.get_xticklabels()

    # Keep only every other label by setting others to empty string
    ax.set_xticks(xticks[::2])
    ax.set_xticklabels(xticklabels[::2])

    ymin, ymax = ax.get_ylim()
    yticks = range(0, int(ymax) + 5, 5)
    ax.set_yticks(yticks)

    ax.set_xlabel("Human task time")
    ax.set_ylabel("Number of tasks (stacked)")

    ax.set_title(
        "Distribution of Task Difficulty",
        fontsize=plot_params["title_fontsize"],
        pad=plot_params["xlabelpad"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-file", type=str, required=True)
    parser.add_argument("--output-file", type=pathlib.Path, required=True)

    args = parser.parse_args()

    with open(args.runs_file, "r") as f:
        runs_df = pd.read_json(f, lines=True)

    plot_params = dvc.api.params_show(stages="plot_task_distribution")["plots"]

    plot_task_distribution(
        runs_df=runs_df,
        plot_params=plot_params,
    )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output_file)


if __name__ == "__main__":
    main()
