import io
import logging
import pathlib
from typing import Any, TypedDict

import cairosvg
import matplotlib.axes
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.markers import MarkerStyle
from matplotlib.offsetbox import AnnotationBbox, HPacker, OffsetImage, TextArea
from PIL import Image
from typing_extensions import Literal, NotRequired


class TrendlineStyling(TypedDict):
    linewidth: float
    alpha: float
    linestyle: str


class TrendlineParams(TypedDict):
    fit_type: Literal["auto", "default", "exponential", "linear"]
    after_date: str
    color: str
    line_start_date: str | None
    line_end_date: str
    display_r_squared: bool
    data_file: str | None
    styling: TrendlineStyling | None
    caption: str | None
    skip_annotation: bool


class ScriptParams(TypedDict):
    parameter_group_name: str
    lower_y_lim: float
    upper_y_lim: float
    exclude: list[str]
    title: NotRequired[str]
    subtitle: str
    weighting: str
    include_task_distribution: str
    weight_key: str | None
    trendlines: list[TrendlineParams]
    exclude_agents: list[str]
    xlabel: NotRequired[str]
    ylabel: NotRequired[str]
    legend_fontsize: NotRequired[int]
    ax_label_fontsize: int
    title_fontsize: NotRequired[int]
    y_ticks_skip: NotRequired[int]
    hide_regression_info: NotRequired[bool]
    annotation_fontsize: NotRequired[int]
    legend_frameon: NotRequired[bool]
    xticks_skip: NotRequired[int]
    rename_legend_labels: NotRequired[dict[str, str]]
    show_y_label: bool


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
    agent_styling: dict[str, AgentStylingParams]
    scatter_styling: ScatterStylingParams
    performance_over_time_trendline_styling: PerformanceOverTimeTrendlineStylingParams
    ax_label_fontsize: int
    colors: PlotColorsParams
    legend_order: list[str]
    suptitle_fontsize: int
    suptitle_position: Literal["left", "center"]
    annotation_fontsize: int
    task_distribution_styling: TaskDistributionStylingParams
    title_fontsize: int
    xlabelpad: int
    ylabelpad: int


def format_time_label(seconds: float) -> str:
    seconds = round(seconds)
    hours = seconds / 3600
    if hours >= 1:
        # Always show as hours, even if > 24 hours
        # Remove decimal point if it's a whole number
        if hours == int(hours):
            return f"{int(hours)}h"
        else:
            return f"{hours:.1f}h"
    if hours >= 1 / 60:
        return f"{int(hours * 60)}m"
    return f"{int(seconds)}s"


linear_ticks = np.linspace(0, 120, 9)
logarithmic_ticks = np.array(
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
        2 * 60,
        4 * 60,
        8 * 60,
        16 * 60,
        32 * 60,
        64 * 60,
        128 * 60,
        256 * 60,
    ]
)


def get_logarithmic_bins(
    min_time: float, max_time: float
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Get logarithmic bins that cover the range [min_time, max_time].

    The right edge extends past max_time so all data points fall within a bin.
    """
    start_idx = np.searchsorted(logarithmic_ticks, min_time, side="right") - 1
    end_idx = np.searchsorted(logarithmic_ticks, max_time, side="right")

    assert (
        start_idx >= 0
    ), f"min_time {min_time} is less than smallest tick {logarithmic_ticks[0]}"
    assert max_time <= logarithmic_ticks[-1], (
        f"max_time {max_time} exceeds largest tick {logarithmic_ticks[-1]}; "
        "data at max_time would be excluded from histogram; "
        "add more values to logarithmic_ticks"
    )

    return np.array(logarithmic_ticks[start_idx : end_idx + 1])


def log_x_axis(
    ax: matplotlib.axes.Axes, low_limit: int | None = None, unit: str = "minutes"
) -> None:
    ax.set_xscale("log")
    x_min, x_max = ax.get_xlim()

    multiplier = 60 if unit == "minutes" else 3600
    if low_limit is not None:
        x_min = max(x_min, low_limit / multiplier)
        ax.set_xlim(left=x_min)

    xticks = logarithmic_ticks[
        (logarithmic_ticks >= x_min) & (logarithmic_ticks <= x_max)
    ]
    labels = [format_time_label(tick * multiplier) for tick in xticks]

    ax.set_xticks(xticks)
    ax.set_xticklabels(labels)
    ax.xaxis.set_major_locator(
        matplotlib.ticker.FixedLocator([float(x) for x in xticks])
    )
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def make_quarterly_xticks(
    ax: matplotlib.axes.Axes, start_year: int, end_year: int, skip: int = 1
) -> None:
    major_ticks = np.array(
        [pd.Timestamp(f"{y}-01-01") for y in range(start_year, end_year)]
    )
    minor_ticks = np.array(
        [
            pd.Timestamp(f"{y}-{m:02d}-01")
            for y in range(start_year, end_year)
            for m in [4, 7, 10]
        ]
    )
    minor_ticks = np.array(
        [
            t
            for t in minor_ticks
            if t >= pd.Timestamp(start_year) and t <= pd.Timestamp(end_year)
        ]
    )

    ax.set_xticks(major_ticks[::skip])
    ax.set_xticklabels([x.strftime("%Y") for x in major_ticks[::skip]])
    ax.set_xticks(minor_ticks, minor=True)


def add_monthly_minor_ticks(
    ax: matplotlib.axes.Axes,
    start_year: int,
    end_year: int,
    months_between: int,
    graph_start: pd.Timestamp,
    graph_end: pd.Timestamp,
) -> None:
    """Add evenly spaced month subticks (e.g. every 2 months)."""
    valid_months = {1, 2, 3, 4, 6}
    assert (
        months_between in valid_months
    ), f"months_between must be one of {sorted(valid_months)}. Got {months_between}"
    subticks: list[pd.Timestamp] = []
    for y in range(start_year, end_year + 1):
        for month in range(1 + months_between, 13, months_between):
            tick = pd.Timestamp(f"{y}-{month:02d}-01")
            if graph_start <= tick <= graph_end:
                subticks.append(tick)

    if not subticks:
        return

    subtick_nums = list(mdates.date2num(subticks))
    existing_minor = ax.get_xticks(minor=True)
    all_minor = list(existing_minor) + subtick_nums

    ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(all_minor))
    ax.tick_params(axis="x", which="minor", length=4, width=1, color="gray")
    ax.grid(
        True, which="minor", axis="x", linestyle="-", alpha=0.15, color="gray", zorder=3
    )
    ax.set_axisbelow(False)


def make_y_axis(
    ax: matplotlib.axes.Axes,
    unit: str = "minutes",
    scale: Literal["log", "linear"] = "log",
    script_params: ScriptParams | None = None,
) -> None:
    ticks_to_use = []
    if scale == "log":
        ticks_to_use = logarithmic_ticks
        ax.set_yscale("log")
    else:
        ticks_to_use = linear_ticks
        ax.set_yscale("linear")
    if script_params:
        ticks_to_use = ticks_to_use[:: script_params.get("y_ticks_skip", 1)]
    y_min, y_max = ax.get_ylim()
    multiplier = 60 if unit == "minutes" else 3600

    yticks = ticks_to_use[(ticks_to_use >= y_min) & (ticks_to_use <= y_max)]
    labels = [format_time_label(tick * multiplier) for tick in yticks]

    ax.set_yticks(yticks)
    ax.set_yticklabels(labels)
    ax.yaxis.set_major_locator(
        matplotlib.ticker.FixedLocator([float(x) for x in yticks])
    )
    if script_params and not script_params.get("show_minor_xticks", True):
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())


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
        )["lab_color"]
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


def add_watermark(
    fig: Figure,
    logo_path: pathlib.Path,
    legend_on_right: bool = False,
) -> None:
    """Add METR watermark to the figure.

    Args:
        fig: matplotlib figure
        logo_path: path to the logo file
        legend_on_right: if True, the legend occupies the right-hand gutter, so we
            shift the watermark further right to avoid overlap.
    """
    if logo_path.suffix == ".svg":
        png_data = cairosvg.svg2png(url=str(logo_path))
        logo = Image.open(io.BytesIO(png_data)).convert("RGBA")  # type: ignore
    else:
        logo = Image.open(logo_path).convert("RGBA")

    logo_array = np.array(logo)

    ax = fig.axes[0]

    imagebox = OffsetImage(logo_array, zoom=0.15, alpha=0.6)
    text = TextArea("METR", textprops=dict(color="black", alpha=0.6, fontsize=22))
    watermark = HPacker(children=[imagebox, text], align="center", pad=5, sep=10)

    x_pos = 1.25 if legend_on_right else 0.90

    ab = AnnotationBbox(
        watermark, (x_pos, 1.1), xycoords="axes fraction", frameon=False
    )
    ax.add_artist(ab)

    website_text = TextArea(
        "metr.org", textprops=dict(color="#2c7c58", alpha=0.6, fontsize=16)
    )
    website_box = AnnotationBbox(
        website_text, (x_pos, -0.1), xycoords="axes fraction", frameon=False
    )
    ax.add_artist(website_box)

    cc_text = TextArea("CC-BY", textprops=dict(color="#2c7c58", alpha=0.6, fontsize=16))
    cc_box = AnnotationBbox(
        cc_text, (-0.05, -0.1), xycoords="axes fraction", frameon=False
    )
    ax.add_artist(cc_box)
