import argparse
import logging
import pathlib
from datetime import date
from typing import Any, List

import dvc.api
import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.dates import date2num

import src.utils.plots
from src.plot.logistic import (
    _get_title,
    _process_agent_summaries,
    fit_trendline,
    plot_horizon_graph,
    plot_trendline,
)

logger = logging.getLogger(__name__)


def add_bootstrap_confidence_region(
    ax: Axes,
    plot_params: src.utils.plots.PlotParams,
    bootstrap_results: pd.DataFrame,
    release_dates: dict[str, dict[str, date]],
    after_date: str,
    script_params: dict[str, Any],
    max_date: pd.Timestamp,
    confidence_level: float,
    exclude_agents: list[str],
) -> tuple[List[float], Any]:
    """Add bootstrap confidence intervals and region to an existing plot.

    Args:
        ax: matplotlib axes
        bootstrap_results: DataFrame with columns for each agent containing p50s
        release_dates: Dictionary mapping agent names to release dates

    Returns:
        List of doubling times from the trendlines
    """
    dates = release_dates["date"]
    focus_agents = sorted(list(dates.keys()), key=lambda x: dates[x])
    focus_agents = [agent for agent in focus_agents if agent not in exclude_agents]
    doubling_times = []
    # Calculate and plot the confidence region
    n_bootstraps = len(bootstrap_results)

    # Create time points for prediction
    time_points = pd.date_range(
        start=pd.to_datetime(after_date),
        end=max_date,
        freq="D",
    )
    predictions = np.zeros((n_bootstraps, len(time_points)))
    # Calculate predictions for each bootstrap sample
    for sample_idx in range(n_bootstraps):
        # Collect valid p50 values and dates for this sample
        valid_p50s = []
        valid_dates = []

        for agent in focus_agents:
            if f"{agent}_p50" not in bootstrap_results.columns:
                continue

            p50 = pd.to_numeric(
                bootstrap_results[f"{agent}_p50"].iloc[sample_idx], errors="coerce"
            )

            if pd.isna(p50) or np.isinf(p50) or p50 < 1e-3:
                continue

            valid_p50s.append(p50)
            valid_dates.append(dates[agent])

        if len(valid_p50s) < 2:
            continue

        # Fit exponential trend
        try:
            reg, _ = fit_trendline(
                pd.Series(valid_p50s),
                pd.Series(pd.to_datetime(valid_dates)),
                log_scale=True,
            )
            time_x = date2num(time_points)
            predictions[sample_idx] = np.exp(reg.predict(time_x.reshape(-1, 1)))
            slope = reg.coef_[0]
            doubling_time = np.log(2) / slope
            if doubling_time > 0:  # Only include positive doubling times
                doubling_times.append(doubling_time)
        except Exception as e:
            print(e)
            continue

    # Calculate confidence bounds
    low_q = (1 - confidence_level) / 2
    high_q = 1 - low_q
    lower_bound = np.nanpercentile(predictions, low_q * 100, axis=0)
    upper_bound = np.nanpercentile(predictions, high_q * 100, axis=0)

    # Plot confidence region
    handle = ax.fill_between(
        time_points,
        lower_bound,
        upper_bound,
        color="gray",
        alpha=0.12,
        # label="80% CR for trendline",
    )

    return doubling_times, handle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig-name", type=str, required=True)
    parser.add_argument("--input-file", type=pathlib.Path, required=True)
    parser.add_argument("--agent-summaries-file", type=pathlib.Path, required=True)
    parser.add_argument("--release-dates", type=pathlib.Path, required=True)
    parser.add_argument("--output-file", type=pathlib.Path, required=True)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument(
        "--y-scale",
        choices=["log", "linear"],
        default="log",
        help="Scale type for y-axis",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    params = dvc.api.params_show(stages="plot_bootstrap_ci", deps=True)
    plot_params = params["plots"]
    script_params = params["figs"]["plot_logistic_regression"][args.fig_name]

    confidence_level = 0.95

    # Load data
    bootstrap_results = pd.read_csv(args.input_file)
    agent_summaries = pd.read_csv(args.agent_summaries_file)
    release_dates = yaml.safe_load(args.release_dates.read_text())
    agent_summaries = _process_agent_summaries(
        script_params["exclude_agents"], agent_summaries, release_dates
    )

    subtitle = script_params["subtitle"] or ""
    title = _get_title(script_params, script_params.get("success_percent", 50))

    # Create plot with two subplots
    if script_params.get("show_boxplot", False):
        fig, axs = plt.subplots(1, 2, width_ratios=[6, 1], figsize=(12, 6))
    else:
        fig, axs = plt.subplots(1, 1, figsize=(12, 6))
        axs = [axs]

    end_date = script_params["x_lim_end"]
    upper_y_lim = script_params["upper_y_lim"]
    trendline_end_date = script_params["x_lim_end"]
    if args.y_scale == "linear":
        end_date = agent_summaries["release_date"].max() + pd.Timedelta(days=60)
        upper_y_lim = agent_summaries["p50"].max() * 1.2
        trendline_end_date = agent_summaries["release_date"].max()
    # Plot the main horizon graph
    plot_horizon_graph(
        plot_params,
        agent_summaries,
        title=title,
        release_dates=release_dates,
        runs_df=pd.DataFrame(),  # Empty DataFrame since we don't need task distribution
        subtitle=subtitle,
        lower_y_lim=script_params["lower_y_lim"],
        upper_y_lim=upper_y_lim,
        x_lim_start=script_params["x_lim_start"],
        x_lim_end=end_date,
        include_task_distribution="none",
        weight_key=script_params["weighting"],
        trendlines=None,
        exclude_agents=script_params["exclude_agents"],
        fig=fig,
        success_percent=script_params.get("success_percent", 50),
        y_scale=args.y_scale,
        script_params=script_params,
    )

    # Add trendline from agent_summaries
    reg, score = fit_trendline(
        agent_summaries[f"p{script_params.get('success_percent', 50)}"],
        pd.to_datetime(agent_summaries["release_date"]),
        log_scale=True,
    )
    dashed_outside = (
        agent_summaries["release_date"].min(),
        agent_summaries["release_date"].max(),
    )
    plot_trendline(
        ax=axs[0],
        dashed_outside=dashed_outside,
        plot_params=plot_params,
        trendline_params={
            "after_date": script_params["trendlines"][0]["line_start_date"],
            "color": "blue",
            "line_start_date": None,
            "line_end_date": trendline_end_date,
            "display_r_squared": True,
            "data_file": None,
            "styling": None,
            "caption": None,
            "skip_annotation": False,
            "fit_type": "exponential",
        },
        reg=reg,
        score=score,
        log_scale=True,
        method="OLS",
    )

    reg_doubling_time = np.log(2) / reg.coef_[0]

    # Add bootstrap confidence region
    doubling_times, handle = add_bootstrap_confidence_region(
        ax=axs[0],
        plot_params=plot_params,
        bootstrap_results=bootstrap_results,
        release_dates=release_dates,
        after_date=script_params["trendlines"][0]["line_start_date"],
        script_params=script_params,
        max_date=trendline_end_date,
        confidence_level=confidence_level,
        exclude_agents=script_params["exclude_agents"],
    )

    # Add confidence region and existing scatter points to legend
    handles, labels = axs[0].get_legend_handles_labels()
    sorted_items = sorted(
        zip(handles, labels), key=lambda x: plot_params["legend_order"].index(x[1])
    )
    handles, labels = zip(*sorted_items)
    rename_map = script_params.get("rename_legend_labels", {})
    # Apply rename map to legend labels
    labels = [rename_map.get(label, label) for label in labels]
    axs[0].legend(
        handles,
        labels,
        loc="upper left",
        fontsize=script_params.get("legend_fontsize", 12),
        frameon=script_params.get("legend_frameon", True),
    )
    # Turn off grid
    axs[0].grid(script_params.get("show_grid", True))

    if args.y_scale == "log":
        low_q = (1 - confidence_level) / 2
        high_q = 1 - low_q
        qlow, qhigh = np.nanpercentile(doubling_times, [100 * low_q, 100 * high_q])
        confidence_str = f"{confidence_level * 100:.0f}%"
        if not script_params.get("hide_regression_info", False):
            extra_text = f"\n{confidence_str} CI: {qlow:.0f} to {qhigh:.0f} days\nR²: {score:.2f}"
        else:
            extra_text = ""
        axs[0].text(
            0.95,
            0.05,
            f"Doubling time: {(reg_doubling_time/30):.0f} months{extra_text}",
            transform=axs[0].transAxes,
            fontsize=script_params.get(
                "annotation_fontsize", plot_params["annotation_fontsize"]
            ),
            verticalalignment="bottom",
            horizontalalignment="right",
        )

    if script_params.get("show_boxplot", False):
        # Plot doubling times boxplot
        axs[1].boxplot([doubling_times], vert=True, showfliers=False, whis=(10, 90))
        axs[1].set_xticklabels(["Doubling times\n(days)"])
        axs[1].set_ylim(0, 365)

    # Save plot
    src.utils.plots.save_or_open_plot(args.output_file, params["plot_format"])


if __name__ == "__main__":
    main()
