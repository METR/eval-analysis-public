from __future__ import annotations

import argparse
import logging
import pathlib
from datetime import date
from typing import Any

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import ruamel.yaml
import yaml
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.dates import date2num, num2date

from horizon.plot.logistic import (
    _process_agent_summaries,
    fit_trendline,
)
from horizon.utils import plots as utils_plots
from horizon.wrangle.out_of_sample_extrapolation import (
    get_frontier_agents,
)

logger = logging.getLogger(__name__)


def _add_baseline_confidence_region(
    ax: Axes,
    bootstrap_results: pd.DataFrame,
    release_dates: dict[str, dict[str, date]],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    confidence_level: float,
) -> None:
    """Shade CI for baseline trend by re-fitting OLS to each bootstrap sample."""
    dates = release_dates["date"]
    focus_agents = [col.removesuffix("_p50") for col in bootstrap_results.columns]
    agent_dates = pd.Series([dates[a] for a in focus_agents])

    n_bootstraps = len(bootstrap_results)
    time_points = pd.date_range(start=start_date, end=end_date, freq="D")
    time_x = date2num(time_points).reshape(-1, 1)
    predictions = np.zeros((n_bootstraps, len(time_points)))

    bootstrap_vals = bootstrap_results.values

    for sample_idx in range(n_bootstraps):
        vals = bootstrap_vals[sample_idx].astype(float)
        valid = ~(np.isnan(vals) | np.isinf(vals) | (vals < 1e-3))
        if valid.sum() < 2:
            continue

        reg, _ = fit_trendline(
            pd.Series(vals[valid]),
            pd.to_datetime(agent_dates[valid]),
            log_scale=True,
        )
        predictions[sample_idx] = np.exp(reg.predict(time_x))

    low_q = (1 - confidence_level) / 2
    ax.fill_between(
        time_points,
        np.nanpercentile(predictions, low_q * 100, axis=0),
        np.nanpercentile(predictions, (1 - low_q) * 100, axis=0),
        color="#d2dfd7",
        alpha=0.4,
        zorder=1,
    )


def _plot_baseline_points(
    ax: Axes,
    agent_summaries: pd.DataFrame,
    success_percent: int,
    confidence_level: float,
    lower_y_lim: float,
) -> None:
    """Plot observed model time horizons as scatter with bootstrap error bars."""
    low_q = (1 - confidence_level) / 2
    high_q = 1 - low_q

    y = agent_summaries[f"p{success_percent}"]
    y_clipped = y.clip(lower_y_lim, np.inf)
    y_low = agent_summaries.get(f"p{success_percent}q{low_q:.3f}", y)
    y_high = agent_summaries.get(f"p{success_percent}q{high_q:.3f}", y)

    yerr = np.array([y - y_low, y_high - y])
    yerr = np.clip(yerr, 0, np.inf)

    ax.errorbar(
        pd.to_datetime(agent_summaries["release_date"]),
        y_clipped,
        yerr=yerr,
        fmt="none",
        color="grey",
        capsize=2,
        alpha=0.8,
        zorder=9,
        linewidth=1.5,
    )

    ax.scatter(
        pd.to_datetime(agent_summaries["release_date"]),
        y_clipped,
        color="#2c7c58",
        marker="o",
        s=80,
        edgecolor="black",
        linewidth=0.5,
        zorder=10,
    )


def _plot_baseline_trendline(
    ax: Axes,
    agent_summaries: pd.DataFrame,
    success_percent: int,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> None:
    """Plot OLS trendline through observed time horizons."""
    reg, _ = fit_trendline(
        agent_summaries[f"p{success_percent}"],
        pd.to_datetime(agent_summaries["release_date"]),
        log_scale=True,
    )

    x_range = np.linspace(date2num(start_date), date2num(end_date), 120)
    y_pred = reg.predict(x_range.reshape(-1, 1))
    y_values = np.exp(y_pred)
    x_dates = np.array(num2date(x_range))

    ax.plot(
        x_dates,
        y_values,
        color="#2c7c58",
        linewidth=2,
        alpha=0.8,
        linestyle="-",
        label="External Deployment Trend",
    )


def _plot_extrapolation_bands(
    ax: Axes,
    samples_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    baseline_date: pd.Timestamp,
) -> None:
    """Plot extrapolation mean and CI bands using samples from wrangle stage.

    Unlike baseline, these samples already incorporate gap uncertainty.
    Bands trace percentile trajectories.
    """
    target_dates = pd.to_datetime(summary_df["target_date"]).sort_values()
    dates = [baseline_date] + list(target_dates)[1:]
    x_values = date2num(np.array([pd.Timestamp(d) for d in dates]))

    unique_samples = samples_df.groupby("sample_idx").first().reset_index()
    baseline_x = date2num(baseline_date)
    log_h_baseline = unique_samples["intercept"] + unique_samples["slope"] * (
        baseline_x + unique_samples["gap_days"]
    )
    unique_samples = unique_samples.iloc[log_h_baseline.argsort()]
    n = len(unique_samples)

    def _trajectory(q: float) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
        s = unique_samples.iloc[int(q * (n - 1))]
        return np.exp(s["intercept"] + s["slope"] * (x_values + s["gap_days"]))  # type: ignore[return-value]

    dates_arr = np.array(dates)
    ax.fill_between(
        dates_arr, _trajectory(0.025), _trajectory(0.975), color="#e63946", alpha=0.2
    )

    mean_gap_days = float(unique_samples["gap_days"].mean())
    mean_line = np.exp(
        unique_samples["intercept"].mean()
        + unique_samples["slope"].mean() * (x_values + mean_gap_days)
    )
    ax.plot(
        dates_arr,
        mean_line,
        color="#e63946",
        linewidth=2.5,
        linestyle="--",
        label="Extrapolation Geometric Mean",
        zorder=11,
    )


def _add_extrapolation_annotation(
    ax: Axes,
    samples_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    baseline_date: pd.Timestamp,
    x_lim_start: pd.Timestamp,
    target_months: int = 6,
) -> None:
    """Add horizontal line and text annotation for a specific extrapolation point."""
    target_dates = pd.to_datetime(summary_df["target_date"])
    target_date = baseline_date + pd.DateOffset(months=target_months)

    mean_gap_days = samples_df["gap_days"].mean()
    mean_val = np.exp(
        samples_df["intercept"].mean()
        + samples_df["slope"].mean() * (date2num(target_date) + mean_gap_days)
    )

    closest_idx = int((target_dates - target_date).abs().idxmin())  # type: ignore[arg-type]
    low = float(summary_df.loc[closest_idx, "q0.025"])  # type: ignore[arg-type]
    high = float(summary_df.loc[closest_idx, "q0.975"])  # type: ignore[arg-type]

    ax.hlines(
        mean_val,
        mdates.date2num(x_lim_start),
        mdates.date2num(target_date),
        colors="#e63946",
        linestyles="--",
        linewidth=1.5,
        alpha=0.7,
    )

    text = (
        f"Extrapolation to {target_date.strftime('%b %Y')}: {mean_val/60:.1f} hours "
        f"[{low/60:.1f} hours - {high/60:.1f} hours]"
    )
    annotation_x = float(mdates.date2num(x_lim_start + pd.Timedelta(days=30)))
    ax.annotate(
        text,
        xy=(annotation_x, float(mean_val)),
        xytext=(0, 4),
        textcoords="offset points",
        fontsize=12,
        color="#e63946",
        va="bottom",
        ha="left",
    )


def plot_time_horizon_with_extrapolation(
    ax: Axes,
    agent_summaries: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    samples_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    release_dates: dict[str, Any],
    fig_params: dict[str, Any],
) -> None:
    """Plot observed time horizon trend with future extrapolation."""
    success_percent = fig_params.get("success_percent", 50)
    confidence_level = fig_params.get("confidence_level", 0.95)
    x_lim_start = pd.Timestamp(fig_params["x_lim_start"])
    x_lim_end = pd.Timestamp(fig_params["x_lim_end"])
    lower_y_lim = fig_params["lower_y_lim"]
    upper_y_lim = fig_params["upper_y_lim"]

    baseline_date = pd.to_datetime(agent_summaries["release_date"]).max()

    _add_baseline_confidence_region(
        ax,
        bootstrap_df[[f"{a}_p50" for a in agent_summaries["agent"]]],
        release_dates,
        x_lim_start,
        baseline_date,
        confidence_level,
    )
    _plot_baseline_points(
        ax, agent_summaries, success_percent, confidence_level, lower_y_lim
    )
    _plot_baseline_trendline(
        ax, agent_summaries, success_percent, x_lim_start, baseline_date
    )

    _plot_extrapolation_bands(ax, samples_df, summary_df, baseline_date)
    _add_extrapolation_annotation(
        ax, samples_df, summary_df, baseline_date, x_lim_start
    )

    ax.set_yscale("log")
    ax.set_ylim(lower_y_lim, upper_y_lim)
    ax.set_xlim(float(mdates.date2num(x_lim_start)), float(mdates.date2num(x_lim_end)))
    utils_plots.make_y_axis(ax, scale="log")
    utils_plots.make_quarterly_xticks(ax, x_lim_start.year, x_lim_end.year + 1)

    ax.set_xlabel("Model release date", fontsize=14)
    ax.set_ylabel(
        f"Task time (for humans) that model completes\nwith {success_percent}% success rate",
        fontsize=14,
    )
    ax.set_title(
        fig_params.get("title", "Time Horizon with Extrapolation"), fontsize=16, pad=10
    )
    ax.grid(which="major", linestyle="-", alpha=0.2, color="grey")
    ax.legend(loc="lower right", fontsize=11)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logistic-fits-file", type=pathlib.Path, required=True)
    parser.add_argument("--bootstrap-file", type=pathlib.Path, required=True)
    parser.add_argument("--samples-file", type=pathlib.Path, required=True)
    parser.add_argument("--summary-file", type=pathlib.Path, required=True)
    parser.add_argument("--release-dates-file", type=pathlib.Path, required=True)
    parser.add_argument("--output-file", type=pathlib.Path, required=True)
    parser.add_argument("--fig-params-file", type=pathlib.Path, required=True)
    parser.add_argument("--script-parameter-group", type=str, required=True)
    parser.add_argument("--plot-format", type=str, default="png")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s - %(levelname)s - %(message)s"
    )

    yaml_loader = ruamel.yaml.YAML(typ="safe")
    with open(args.fig_params_file) as f:
        params = yaml_loader.load(f)
    fig_params = params["figs"]["plot_out_of_sample_extrapolation"][
        args.script_parameter_group
    ]

    release_dates = yaml.safe_load(args.release_dates_file.read_text())
    logistic_fits_df = pd.read_csv(args.logistic_fits_file)
    success_percent = fig_params.get("success_percent", 50)
    frontier_agents = get_frontier_agents(
        logistic_fits_df, release_dates, success_percent
    )
    frontier_fits_df = logistic_fits_df[logistic_fits_df["agent"].isin(frontier_agents)]
    agent_summaries = _process_agent_summaries(
        fig_params.get("exclude_agents", []),
        frontier_fits_df,
        release_dates,
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_time_horizon_with_extrapolation(
        ax,
        agent_summaries,
        pd.read_csv(args.bootstrap_file),
        pd.read_csv(args.samples_file),
        pd.read_csv(args.summary_file),
        release_dates,
        fig_params,
    )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    utils_plots.save_or_open_plot(args.output_file, args.plot_format)
    logger.info(f"Saved plot to {args.output_file}")


if __name__ == "__main__":
    main()
