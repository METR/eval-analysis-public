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

    doubling_times = []
    # Calculate and plot the confidence region
    n_bootstraps = len(bootstrap_results)

    max_date = script_params["x_lim_end"]
    # Create time points for prediction
    time_points = pd.date_range(
        start=pd.to_datetime(after_date),
        end=max_date,
        freq="D",
    )
    predictions = np.zeros((n_bootstraps, len(time_points)))

    # Calculate predictions for each bootstrap sample
    for sample_idx in range(n_bootstraps):
        # Create a DataFrame for this bootstrap sample
        sample_data = []
        for agent in focus_agents:
            if agent not in bootstrap_results.columns:
                continue
            p50 = pd.to_numeric(
                bootstrap_results[agent].iloc[sample_idx], errors="coerce"
            )
            if pd.isna(p50) or np.isinf(p50) or p50 < 1e-3:
                continue
            sample_data.append(
                {
                    "agent": agent,
                    "release_date": dates[agent],
                    "50%": p50,
                    "50_low": p50,
                    "50_high": p50,
                }
            )

        sample_df = pd.DataFrame(sample_data)
        if len(sample_df) < 2:
            continue

        # Fit exponential trend
        try:
            reg, _ = fit_trendline(sample_df, after_date, log_scale=True)
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
    lower_bound = np.nanpercentile(predictions, 10, axis=0)
    upper_bound = np.nanpercentile(predictions, 90, axis=0)

    # Plot confidence region
    handle = ax.fill_between(
        time_points,
        lower_bound,
        upper_bound,
        color="gray",
        alpha=0.2,
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
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    params = dvc.api.params_show("public/params.yaml", deps=True)
    plot_params = params["plots"]
    script_params = params["figs"]["plot_logistic_regression"]["headline"]

    # Load data
    bootstrap_results = pd.read_csv(args.input_file)
    agent_summaries = pd.read_csv(args.agent_summaries_file)
    release_dates = yaml.safe_load(args.release_dates.read_text())
    agent_summaries = _process_agent_summaries(
        script_params["exclude_agents"], agent_summaries, release_dates
    )

    possible_weightings = params["weighting"]
    for possible_weighting in possible_weightings:
        if possible_weighting["weight_col"] == script_params["weighting"]:
            weighting = possible_weighting
            break
    else:
        raise ValueError(f"Weighting {script_params['weighting']} not found")
    subtitle = weighting["graph_snippet"]
    title = _get_title(script_params)

    # Create plot with two subplots
    if script_params.get("show_boxplot", False):
        fig, axs = plt.subplots(1, 2, width_ratios=[6, 1], figsize=(12, 6))
    else:
        fig, axs = plt.subplots(1, 1, figsize=(12, 6))
        axs = [axs]

    # Plot the main horizon graph
    plot_horizon_graph(
        plot_params,
        agent_summaries,
        title=title,
        release_dates=release_dates,
        runs_df=pd.DataFrame(),  # Empty DataFrame since we don't need task distribution
        subtitle=subtitle,
        lower_y_lim=script_params["lower_y_lim"],
        upper_y_lim=script_params["upper_y_lim"],
        x_lim_start=script_params["x_lim_start"],
        x_lim_end=script_params["x_lim_end"],
        include_task_distribution="none",
        weight_key=weighting["weight_col"],
        trendlines=None,
        exclude_agents=script_params["exclude_agents"],
        fig=fig,
    )

    # Add trendline from agent_summaries
    reg, score = fit_trendline(
        agent_summaries=agent_summaries,
        after="2019-01-01",
        log_scale=True,
        method="OLS",
    )
    plot_trendline(
        ax=axs[0],
        agent_summaries=agent_summaries,
        plot_params=plot_params,
        after="2019-01-01",
        reg=reg,
        score=score,
        line_end_date=script_params["x_lim_end"],
        log_scale=True,
        annotate=False,
        fit_type="exponential",
        fit_color="black",
        method="OLS",
    )

    reg_doubling_time = np.log(2) / reg.coef_[0]

    # Add bootstrap confidence region
    doubling_times, handle = add_bootstrap_confidence_region(
        ax=axs[0],
        plot_params=plot_params,
        bootstrap_results=bootstrap_results,
        release_dates=release_dates,
        after_date="2019-01-01",
        script_params=script_params,
    )

    # Add confidence region and existing scatter points to legend
    handles, labels = axs[0].get_legend_handles_labels()
    sorted_items = sorted(
        zip(handles, labels), key=lambda x: plot_params["legend_order"].index(x[1])
    )
    handles, labels = zip(*sorted_items)
    axs[0].legend([handle, *handles], ["80% confidence region", *labels], loc="best")

    p10, p50, p90 = np.nanpercentile(doubling_times, [10, 50, 90])
    axs[0].text(
        0.95,
        0.05,
        f"Doubling time {reg_doubling_time:.0f} days\n80% CI: {p10:.0f} to {p90:.0f} days\nPoint CIs are 80%",
        transform=axs[0].transAxes,
        fontsize=plot_params["title_fontsize"],
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
