import logging
import pathlib
from typing import TypedDict

import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.markers import MarkerStyle
from typing_extensions import Literal


class PlotColorsParams(TypedDict):
    default: str


class ErrorBarParams(TypedDict):
    color: str
    fmt: str
    capsize: int
    alpha: float
    zorder: int
    linewidth: float
    capthick: float


class GridParams(TypedDict):
    which: Literal["major", "minor", "both"]
    linestyle: str
    alpha: float
    color: str
    zorder: int


class ScatterParams(TypedDict):
    s: int
    edge_color: str
    linewidth: float
    zorder: int


class AgentStylingParams(TypedDict):
    lab_color: str
    marker: MarkerStyle
    unique_color: str


class ScatterStylingParams(TypedDict):
    error_bar: ErrorBarParams
    grid: GridParams
    scatter: ScatterParams


class HistParams(TypedDict):
    edgecolor: str
    color: str
    alpha: float
    linewidth: float


class TaskDistributionStylingParams(TypedDict):
    hist: HistParams
    grid: GridParams


class TrendlineAnnotationParams(TypedDict):
    color: str


class TrendlineLineParams(TypedDict):
    color: str
    alpha: float
    linewidth: float


class TrendlineStylingParams(TypedDict):
    annotation: TrendlineAnnotationParams
    line: TrendlineLineParams


class PerformanceOverTimeTrendlineStylingParams(TypedDict):
    linear: TrendlineStylingParams
    exponential: TrendlineStylingParams
    hyperbolic: TrendlineStylingParams
    default: TrendlineStylingParams


class PlotParams(TypedDict):
    agent_styling: AgentStylingParams
    scatter_styling: ScatterStylingParams
    performance_over_time_trendline_styling: PerformanceOverTimeTrendlineStylingParams
    ax_label_fontsize: int
    colors: PlotColorsParams
    legend_order: list[str]
    suptitle_fontsize: int
    task_distribution_styling: TaskDistributionStylingParams
    title_fontsize: int
    xlabelpad: int
    ylabelpad: int


def format_time_label(seconds: float) -> str:
    seconds = round(seconds)
    hours = seconds / 3600
    if hours >= 24:
        return f"{int(hours / 24)}d"
    if hours >= 1:
        return f"{int(hours)} hr" + ("s" if int(hours) > 1 else "")
    if hours >= 1 / 60:
        return f"{int(hours * 60)} min"
    return f"{int(seconds)} sec"


possible_ticks = np.array(
    [
        1 / 60,
        2 / 60,
        4 / 60,
        8 / 60,
        15 / 60,
        30 / 60,
        1,
        2,
        4,
        8,
        15,
        30,
        60,
        120,
        240,
        480,
        960,
        24 * 60,
        2 * 24 * 60,
        4 * 24 * 60,
        8 * 24 * 60,
    ]
)


def log_x_axis(
    ax: matplotlib.axes.Axes, low_limit: int | None = None, unit: str = "minutes"
) -> None:
    ax.set_xscale("log")
    x_min, x_max = ax.get_xlim()

    multiplier = 60 if unit == "minutes" else 3600
    if low_limit is not None:
        x_min = max(x_min, low_limit / multiplier)
        ax.set_xlim(left=x_min)

    xticks = possible_ticks[(possible_ticks >= x_min) & (possible_ticks <= x_max)]
    labels = [format_time_label(tick * multiplier) for tick in xticks]

    ax.set_xticks(xticks)
    ax.set_xticklabels(labels)
    ax.xaxis.set_major_locator(
        matplotlib.ticker.FixedLocator([float(x) for x in xticks])
    )
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def log_y_axis(ax: matplotlib.axes.Axes, unit: str = "minutes") -> None:
    ax.set_yscale("log")
    y_min, y_max = ax.get_ylim()
    multiplier = 60 if unit == "minutes" else 3600
    yticks = possible_ticks[(possible_ticks >= y_min) & (possible_ticks <= y_max)]
    labels = [format_time_label(tick * multiplier) for tick in yticks]

    ax.set_yticks(yticks)
    ax.set_yticklabels(labels)
    ax.yaxis.set_major_locator(
        matplotlib.ticker.FixedLocator([float(x) for x in yticks])
    )
    # ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def get_agent_color(
    plot_params: PlotParams,
    agent: str = "default",
    color_type: Literal["lab_color", "individual"] = "individual",
) -> str:
    """Get color for agent, falling back to default if not found."""
    if "human" in agent.lower():
        agent = "human"

    assert "default" in plot_params["agent_styling"]

    if color_type == "lab_color":
        return plot_params["agent_styling"].get(
            agent, plot_params["agent_styling"]["default"]
        )["color"]
    else:
        return plot_params["agent_styling"].get(
            agent, plot_params["agent_styling"]["default"]
        )["unique_color"]


def create_sorted_legend(ax: matplotlib.axes.Axes, legend_order: list[str]) -> None:
    handles, labels = ax.get_legend_handles_labels()
    legend_elements = sorted(
        zip(handles, labels),
        key=lambda x: (
            legend_order.index(x[1]) if x[1] in legend_order else float("inf")
        ),
    )
    handles, labels = zip(*legend_elements)

    ax.legend(handles=handles, labels=labels)


def save_or_open_plot(
    output_file: pathlib.Path | None = None, plot_format: str = "png"
) -> None:
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, format=plot_format, bbox_inches="tight")
        logging.info(f"Plot saved to {output_file}")
    else:
        plt.show()
    plt.close()
