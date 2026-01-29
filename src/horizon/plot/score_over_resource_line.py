import argparse
import pathlib
from typing import Any, Literal, cast

import dvc.api
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from horizon.utils.plots import (
    PlotParams,
    get_agent_color,
    make_y_axis,
)

FIGSIZE = (9, 5)


def _get_plot_column(weighting_column: str) -> str:
    """Map weighting_column to the actual column to plot on y-axis."""
    if weighting_column in ("invsqrt_task_weight", "equal_task_weight"):
        return "weighted_success_rate_at_cost"
    else:
        return weighting_column


def _plot_agent_line(
    df_agent: pd.DataFrame,
    ax: Axes,
    usage_column: str,
    weighting_column: str,
    color: str = "black",
    label: str | None = None,
) -> None:
    assert not df_agent.empty
    graph_df = df_agent.copy()
    # If multiple with same usage, keep the last one
    graph_df = graph_df.drop_duplicates(subset=usage_column, keep="last")
    graph_df = graph_df.sort_values(by=usage_column)

    plot_column = _get_plot_column(weighting_column)

    if graph_df.empty:
        return

    ax.plot(
        graph_df[usage_column],
        graph_df[plot_column],
        linewidth=2.25,
        color=color,
        alpha=0.95,
        label=label,
        drawstyle="steps-post",
        dash_capstyle="round",
    )


def _score_line_chart(
    df: pd.DataFrame,
    plot_params: dict[str, Any],
    weighting_column: str,
    usage_column: str,
    score_column: str,
    y_scale: Literal["log", "linear"],
    y_lim_lower: float | None,
    y_lim_upper: float | None,
) -> Figure:
    line_params = plot_params["score_over_resource_line"]
    y_params = line_params["y_params"]
    x_params = line_params["x_params"]

    fig, ax = plt.subplots(figsize=(FIGSIZE[0], FIGSIZE[1]))

    last_entries = df.groupby("alias").last().reset_index()
    agent_order = last_entries.sort_values(by=score_column, ascending=False)["alias"]

    for agent in agent_order:
        df_agent = df[df["alias"] == agent]
        _plot_agent_line(
            df_agent=df_agent,
            ax=ax,
            usage_column=usage_column,
            weighting_column=weighting_column,
            label=str(agent),
            color=get_agent_color(cast(PlotParams, plot_params), agent),
        )

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.subplots_adjust(right=0.5, bottom=0.2)

    ax.set_ylabel(y_params[weighting_column]["y_label"], wrap=True)
    ax.set_title(
        f"Agent Performance on HCAST & RE-Bench {x_params[usage_column]['by_title']}\n({y_params[weighting_column]['subtitle']})"
    )

    default_ylims = y_params[weighting_column]["y_col_lims"]
    ax.set_ylim(
        default_ylims[0] if y_lim_lower is None else y_lim_lower,
        default_ylims[1] if y_lim_upper is None else y_lim_upper,
    )

    if weighting_column == "p50":
        ax.set_yscale(y_scale)
        make_y_axis(ax, unit="minutes", scale=y_scale)

    ax.set_xlabel(x_params[usage_column]["x_label"])
    ax.set_xscale("log")
    ax.set_xlim(x_params[usage_column]["x_col_lims"])

    fig.tight_layout()
    return fig


def main(
    wrangled_resource_file: pathlib.Path,
    output_file: pathlib.Path,
    params_file: pathlib.Path,
    weighting_column: str,
    score_column: str,
    x: str,
    y_scale: Literal["log", "linear"],
    y_lim_lower: float | None,
    y_lim_upper: float | None,
    fig_name: str | None = None,
) -> None:
    df = pd.read_json(wrangled_resource_file, lines=True, orient="records")
    params = yaml.safe_load(open(params_file))

    if fig_name is not None:
        dvc_params = dvc.api.params_show(
            stages="plot_score_over_resource_horizon", deps=True
        )
        fig_params = dvc_params["figs"]["plot_score_over_resource_horizon"][fig_name]
        exclude_agent = fig_params["exclude_agents"]
        df = df[~df["alias"].isin(exclude_agent)]

    fig = _score_line_chart(
        df,
        plot_params=params["plots"],
        usage_column=x,
        score_column=score_column,
        weighting_column=weighting_column,
        y_scale=y_scale,
        y_lim_lower=y_lim_lower,
        y_lim_upper=y_lim_upper,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving figure to {output_file}")
    fig.savefig(output_file)


def _float_or_str(x: float | str) -> float | None:
    if isinstance(x, str) and x.lower() == "none":
        return None
    return float(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wrangled-resource-file",
        type=pathlib.Path,
        required=True,
        help="Path to the wrangled resource data file (jsonl format)",
    )
    parser.add_argument(
        "--output-file",
        type=pathlib.Path,
        help="Path to save the output figure",
        required=True,
    )
    parser.add_argument(
        "--params-file",
        type=pathlib.Path,
        help="Path to the params file",
        required=True,
    )
    parser.add_argument(
        "--score-column",
        type=str,
        required=True,
        help="Column to use for scoring. E.g. score_binarized",
    )
    parser.add_argument(
        "--weighting-column",
        type=str,
        required=True,
        help="Column to use for weighting (either equal_task_weight or invsqrt_task_weight, or p50)",
    )
    parser.add_argument(
        "--y-scale",
        type=str,
        required=False,
        default="log",
        help="Scale to use for y-axis. E.g. log or linear",
    )
    parser.add_argument(
        "--y-lim-lower",
        type=_float_or_str,
        required=True,
        help="Lower limit for y-axis",
    )
    parser.add_argument(
        "--y-lim-upper",
        type=_float_or_str,
        required=True,
        help="Upper limit for y-axis",
    )
    parser.add_argument(
        "--fig-name",
        type=str,
        default=None,
        help="Fig params key for exclude_agents",
    )
    parser.add_argument(
        "--x",
        required=True,
        type=str,
        help="Column to use for x-axis. E.g. generation_cost or action_count",
    )
    args = parser.parse_args()
    main(**vars(args))
