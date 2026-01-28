import argparse
import json
import logging
import pathlib
from collections import defaultdict
from typing import TypedDict

import dvc.api
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import yaml
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.dates import date2num, num2date
from matplotlib.figure import Figure

from horizon.utils import plots


class MultiverseParams(TypedDict):
    weightings: list[str]
    regularizations: list[float]
    include_agents: list[str]
    n_bootstrap: int
    agents_2024: list[str]


class MultiverseRecord(TypedDict):
    coef: float
    intercept: float


def _darken_color(color: str, factor: float = 0.7) -> tuple[float, float, float]:
    """Darken a color by multiplying RGB values by a factor."""
    rgb = to_rgb(color)
    return tuple(x * factor for x in rgb)  # type: ignore


DARKEN_AMOUNT = 0.4
COLOR_OVERALL = "#1E76C1"
COLOR_2024 = "#C4A3A3"
DARKER_COLOR = _darken_color(COLOR_OVERALL, DARKEN_AMOUNT)
DARKER_COLOR_2024 = _darken_color(COLOR_2024, DARKEN_AMOUNT)


def log_metrics(df_records: pd.DataFrame, output_metrics_file: pathlib.Path) -> None:
    metrics = defaultdict(dict)
    for key in df_records["record_type"].unique():
        df_filtered = df_records[df_records["record_type"] == key]
        ci_width = df_filtered["predicted_date"].quantile(0.9) - df_filtered[
            "predicted_date"
        ].quantile(0.1)
        ci_width_years = round(ci_width.days / 365.25, 3)
        metrics["predicted_date"][key.replace("\n", " ")] = {
            "ci_width_years": float(ci_width_years),
            "q10": df_filtered["predicted_date"].quantile(0.1).strftime("%Y-%m-%d"),
            "q50": df_filtered["predicted_date"].quantile(0.5).strftime("%Y-%m-%d"),
            "q90": df_filtered["predicted_date"].quantile(0.9).strftime("%Y-%m-%d"),
        }
    with open(output_metrics_file, "w") as f:
        yaml.dump(dict(metrics), f)


def _setup_total_uncertainty_plot(ax: Axes) -> None:
    """Create and setup the basic figure and axes for total uncertainty plots."""

    # Setup date axis
    ax.xaxis_date()

    # Get updated limits
    xlims = ax.get_xlim()
    start_year = num2date(xlims[0]).year
    end_year = num2date(xlims[1]).year + 1
    plots.make_quarterly_xticks(ax, start_year, end_year)
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    # Add grid
    ax.grid(True, axis="x", linestyle="-", color="grey", alpha=0.7)
    ax.grid(True, axis="x", which="minor", linestyle="-", color="grey", alpha=0.2)

    ax.set_xlabel("Projected date (95% CI)")
    ax.set_title(
        "Projected creation date of AI which can complete one month (167 hour) tasks"
    )


def _save_and_close(fig: Figure, output_file: pathlib.Path, plot_format: str) -> None:
    """Save the plot and close the figure.

    Args:
        fig: Figure to save
        output_file: Path to save the plot to
        plot_format: Format for saving the plot
    """
    plots.save_or_open_plot(output_file, plot_format)
    plt.close(fig)


def create_total_uncertainty_violin_plot(
    records_df: pd.DataFrame, output_file: pathlib.Path, plot_format: str
) -> None:
    """Create a violin plot showing the total uncertainty distribution.

    Args:
        records_df: DataFrame containing all uncertainty records
        output_file: Path to save the output plot
        plot_format: Format for the plot output
    """

    record_types = [
        "Overall (2024-2025 trend)",
        "Overall (2019-2025 trend)",
    ]
    total_records = records_df[records_df["record_type"].isin(record_types)]
    fig, ax = plt.subplots(figsize=(10, 1.5))

    # Get the same percentiles used in box plot for consistency
    data = total_records["date_num"]
    q025, q975 = np.percentile(data, [2.5, 97.5])
    clipped_data = data[(data >= q025) & (data <= q975)]

    # Create violin plot with clipped data
    violin = ax.violinplot(
        [clipped_data[records_df["record_type"] == rt] for rt in record_types],
        vert=False,
        showmeans=False,
        showextrema=True,
        showmedians=True,
        positions=[1, 2],  # Explicitly set positions for y-axis labels
    )

    # Style the violin plot
    violin["bodies"][0].set_facecolor(COLOR_2024)  # type: ignore
    violin["bodies"][0].set_alpha(1.0)  # type: ignore
    violin["bodies"][0].set_edgecolor(DARKER_COLOR_2024)  # type: ignore
    violin["bodies"][1].set_facecolor(COLOR_OVERALL)  # type: ignore
    violin["bodies"][1].set_alpha(1.0)  # type: ignore
    violin["bodies"][1].set_edgecolor(DARKER_COLOR)  # type: ignore
    violin["cmins"].set_color(DARKER_COLOR)
    violin["cmaxes"].set_color(DARKER_COLOR)
    violin["cbars"].set_color(DARKER_COLOR)
    violin["cmedians"].set_color(DARKER_COLOR)
    violin["cmedians"].set_linewidth(2)

    # Add y-axis labels with colors matching the violins
    ax.set_yticks([1, 2])
    ax.set_yticklabels(
        ["2024-2025\ntrend", "2019-2025\ntrend"],
    )

    # Make the left spine visible for the y-axis labels
    ax.spines["left"].set_visible(True)

    _setup_total_uncertainty_plot(ax)

    violin_output = output_file.with_stem(output_file.stem + "_violin")
    _save_and_close(fig, violin_output, plot_format)


def create_total_uncertainty_box_plot(
    records_df: pd.DataFrame, output_file: pathlib.Path, plot_format: str
) -> None:
    """Create a box plot showing the total uncertainty distribution.

    Args:
        records_df: DataFrame containing all uncertainty records
        output_file: Path to save the output plot
        plot_format: Format for the plot output
    """
    record_types = [
        "Overall (2024-2025 trend)",
        "Overall (2019-2025 trend)",
    ]
    total_records = records_df[records_df["record_type"].isin(record_types)]
    fig, ax = plt.subplots(figsize=(10, 1.5))

    LINE_WIDTH = 2
    MEDIAN_LINE_WIDTH = 2
    boxplot = ax.boxplot(
        [
            total_records[total_records["record_type"] == rt]["date_num"]
            for rt in record_types
        ],
        vert=False,
        whis=(2.5, 97.5),
        patch_artist=True,
        meanline=False,
        showmeans=False,
        showfliers=False,
        medianprops={
            "color": DARKER_COLOR,
            "linewidth": MEDIAN_LINE_WIDTH,
        },
        boxprops={"alpha": 1},
        widths=0.6,
        whiskerprops={
            "linewidth": LINE_WIDTH,
            "linestyle": "-",
            "color": DARKER_COLOR,
        },
        capprops={
            "linewidth": LINE_WIDTH,
            "color": DARKER_COLOR,
        },
        positions=[1, 2],  # Explicitly set positions for y-axis labels
    )

    boxplot["boxes"][0].set(
        facecolor=COLOR_2024,
        edgecolor=DARKER_COLOR_2024,
        linewidth=LINE_WIDTH,
    )

    boxplot["boxes"][1].set(
        facecolor=COLOR_OVERALL,
        edgecolor=DARKER_COLOR,
        linewidth=LINE_WIDTH,
    )

    # Style the whiskers, caps, and medians for each box
    for i, color in enumerate([DARKER_COLOR_2024, DARKER_COLOR]):
        boxplot["whiskers"][i * 2].set(color=color, linewidth=LINE_WIDTH)
        boxplot["whiskers"][i * 2 + 1].set(color=color, linewidth=LINE_WIDTH)
        boxplot["caps"][i * 2].set(color=color, linewidth=LINE_WIDTH)
        boxplot["caps"][i * 2 + 1].set(color=color, linewidth=LINE_WIDTH)
        boxplot["medians"][i].set(color=color, linewidth=MEDIAN_LINE_WIDTH)

    # Add y-axis labels with colors matching the boxes
    ax.set_yticks([1, 2])
    ax.set_yticklabels(
        ["2024-2025\ntrend", "2019-2025\ntrend"],
    )

    # Make the left spine visible for the y-axis labels
    ax.spines["left"].set_visible(True)

    _setup_total_uncertainty_plot(ax)

    box_output = output_file.with_stem(output_file.stem + "_box")
    _save_and_close(fig, box_output, plot_format)


def create_total_uncertainty_plot(
    records_df: pd.DataFrame, output_file: pathlib.Path, plot_format: str
) -> None:
    """Create separate violin and box plots showing the total uncertainty distribution.

    Args:
        records_df: DataFrame containing all uncertainty records
        output_file: Path to save the output plots
        plot_format: Format for the plot output
    """
    create_total_uncertainty_violin_plot(records_df, output_file, plot_format)
    create_total_uncertainty_box_plot(records_df, output_file, plot_format)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-file", type=pathlib.Path, required=True)
    parser.add_argument("--output-file", type=pathlib.Path, required=True)
    parser.add_argument("--output-metrics-file", type=pathlib.Path, required=True)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    params = dvc.api.params_show(stages="plot_multiverse_boxplot")
    plot_format = params["plot_format"]

    # Load records from JSON file
    with open(args.records_file, "r") as f:
        record_dict = json.load(f)

    # Convert record_dict to DataFrame
    records_df = pd.DataFrame(
        [
            {**record, "record_type": record_type}
            for record_type, records in record_dict.items()
            for record in records
        ]
    )

    print(records_df.head())

    print(records_df["record_type"].unique())

    # Create figure and axis
    plt.figure()

    # Convert coefficients to predicted dates for 1 month (167 hours)
    target_minutes = 167 * 60

    # Clip coefficients to avoid negative slopes
    records_df["coef"] = records_df["coef"].clip(lower=1e-5)
    records_df["intercept"] = records_df["intercept"].clip(
        upper=np.log(target_minutes) - 1
    )
    assert (records_df["coef"] > 0).all()
    difference = np.log(target_minutes) - records_df["intercept"]
    predicted_num = difference / records_df["coef"]
    records_df["predicted_date"] = predicted_num.apply(num2date)
    records_df["date_num"] = date2num(records_df["predicted_date"])
    LINE_WIDTH = 1.75
    MEDIAN_LINE_WIDTH = 2
    colors = [
        COLOR_2024,
        COLOR_OVERALL,
        "#28B463",
        "#8E44AD",
        "#8B4513",
        "#F39C12",
        "#16A085",
    ]
    fig, ax = plt.subplots()
    boxplot = ax.boxplot(
        [
            records_df[records_df["record_type"] == rt]["date_num"]
            for rt in record_dict.keys()
        ],
        vert=False,
        whis=(10, 90),
        patch_artist=True,
        meanline=False,
        showmeans=False,
        showfliers=False,
        medianprops={"linewidth": MEDIAN_LINE_WIDTH},
        boxprops={"alpha": 1},
        whiskerprops={"linewidth": LINE_WIDTH, "linestyle": "-"},
        capprops={"linewidth": LINE_WIDTH},
    )

    for i, (box, color) in enumerate(zip(boxplot["boxes"], colors)):
        box.set(
            facecolor=color,
            edgecolor=_darken_color(color, DARKEN_AMOUNT),
            linewidth=LINE_WIDTH,
        )
        # Each box has 2 whiskers, 2 caps, and 1 median
        boxplot["whiskers"][i * 2].set(
            color=_darken_color(color, DARKEN_AMOUNT), linewidth=LINE_WIDTH
        )
        boxplot["whiskers"][i * 2 + 1].set(
            color=_darken_color(color, DARKEN_AMOUNT), linewidth=LINE_WIDTH
        )
        boxplot["caps"][i * 2].set(
            color=_darken_color(color, DARKEN_AMOUNT), linewidth=LINE_WIDTH
        )
        boxplot["caps"][i * 2 + 1].set(
            color=_darken_color(color, DARKEN_AMOUNT), linewidth=LINE_WIDTH
        )
        boxplot["medians"][i].set(color=_darken_color(color, DARKEN_AMOUNT))

    # horizontal dotted line separating the first two boxplots (2024-2025 trend and 2019-2025 trend)
    y_separator = 1.5
    xlims = ax.get_xlim()
    ax.axhline(
        y=y_separator,
        xmin=0,
        xmax=1,
        linestyle=":",
        color="black",
        linewidth=2.0,
        alpha=0.7,
        zorder=1,
    )

    ax.set_yticks(
        range(1, len(record_dict.keys()) + 1),
        labels=list(record_dict.keys()),
        fontsize=14,
    )

    ax.xaxis_date()
    xlims = plt.gca().get_xlim()
    start_year = num2date(xlims[0]).year
    end_year = num2date(xlims[1]).year + 1
    plots.make_quarterly_xticks(ax, start_year, end_year)
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    ax.set_xlabel("Extrapolated date of 1 month AI (167 working hours)")
    ax.set_title("Uncertainty in extrapolated date of 1 month AI (167 working hours)")

    # Add grid for easier visualization
    ax.grid(True, axis="x", linestyle="-", color="grey", alpha=0.7)  # Major grid lines
    ax.grid(
        True, axis="x", which="minor", linestyle="-", color="grey", alpha=0.2
    )  # Minor grid lines
    fig.tight_layout()

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    plots.save_or_open_plot(args.output_file, plot_format)

    output_total_file = args.output_file.with_stem(args.output_file.stem + "_total")
    # Create the total uncertainty plot
    create_total_uncertainty_plot(records_df, output_total_file, plot_format)

    log_metrics(records_df, args.output_metrics_file)
    logging.info(f"Logged metrics to {args.output_metrics_file}")


if __name__ == "__main__":
    main()
