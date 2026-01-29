import argparse
import logging
import pathlib
from dataclasses import dataclass
from datetime import date
from typing import Any, List, Literal

import dvc.api
import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.dates import date2num
from numpy.typing import NDArray

import horizon.utils.plots as src_utils_plots
from horizon.plot.logistic import (
    _get_title,
    _process_agent_summaries,
    fit_trendline,
    plot_horizon_graph,
    plot_trendline,
)
from horizon.utils.plots import (
    PlotParams,
    TrendlineParams,
    add_watermark,
)

logger = logging.getLogger(__name__)


def _load_overlay_results(
    overlay_path: pathlib.Path,
    success_percent: int = 50,
) -> pd.DataFrame:
    """Load overlay results from a benchmark_results.yaml file.

    Args:
        overlay_path: Path to benchmark_results.yaml file
        success_percent: Which p-value to use (50 or 80)

    Returns:
        DataFrame with columns: agent_key, release_date, p_value, ci_low, ci_high, has_inspect
    """
    with open(overlay_path) as f:
        data = yaml.safe_load(f)

    records = []
    for agent_key, agent_data in data["results"].items():
        if agent_key == "human":
            continue

        metrics = agent_data["metrics"]

        # Only include SOTA agents
        if not metrics.get("is_sota", False):
            continue

        p_key = f"p{success_percent}_horizon_length"
        if p_key not in metrics:
            logger.warning(f"Skipping {agent_key}: no {p_key} in metrics")
            continue

        p_data = metrics[p_key]
        release_date = pd.to_datetime(agent_data["release_date"])
        p_value = p_data["estimate"]
        ci_low = p_data.get("ci_low", p_value)
        ci_high = p_data.get("ci_high", p_value)

        has_inspect = "_inspect" in agent_key or "(Inspect)" in agent_key

        records.append(
            {
                "agent_key": agent_key,
                "release_date": release_date,
                "p_value": p_value,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "has_inspect": has_inspect,
            }
        )

    return pd.DataFrame(records)


def _add_overlay_results(
    ax: Axes,
    overlay_df: pd.DataFrame,
    overlay_color: str,
) -> None:
    """Add overlay points to the plot.

    Args:
        ax: matplotlib axes to plot on
        overlay_df: DataFrame from _load_overlay_results
        overlay_color: Color to use for overlay points
    """
    for _, row in overlay_df.iterrows():
        yerr_low = row["p_value"] - row["ci_low"]
        yerr_high = row["ci_high"] - row["p_value"]

        ax.errorbar(
            row["release_date"],
            row["p_value"],
            yerr=[[yerr_low], [yerr_high]],
            fmt="o",
            color=overlay_color,
            markersize=8,
            alpha=0.8,
            markeredgewidth=1,
            capsize=2,
            zorder=10,
        )


def _add_overlay_and_trendline(
    axs: list[Axes],
    script_params: dict[str, Any],
    agent_summaries_for_fitting: pd.DataFrame,
    trendline_end_date: pd.Timestamp | str,
    plot_params: PlotParams,
    base_color: str,
    overlay_color: str,
) -> None:
    """Add overlay results and fit a second trendline through overlay + common points.

    This function handles the logic for overlaying a second set of results on the plot,
    identifying which agents are common to both fits, re-plotting those points with
    split colors, and fitting a trendline through the combined data.

    Args:
        axs: List of matplotlib axes
        script_params: Script parameters from config
        agent_summaries_for_fitting: Agent summaries used for fitting
        trendline_end_date: End date for trendline
        plot_params: Plot parameters from config
        base_color: Color for base trendline
        overlay_color: Color for overlay trendline
    """
    overlay_results_path = script_params.get("overlay_results", None)
    if overlay_results_path is None:
        return

    overlay_path = pathlib.Path(overlay_results_path)
    overlay_df = _load_overlay_results(
        overlay_path,
        success_percent=script_params.get("success_percent", 50),
    )

    # Find earliest overlay date
    earliest_overlay_date = overlay_df["release_date"].min()

    # Get agents that are common to both fits (before the earliest overlay date)
    common_to_both_fits = agent_summaries_for_fitting[
        pd.to_datetime(agent_summaries_for_fitting["release_date"])
        < earliest_overlay_date
    ]
    logger.info(f"Earliest overlay date: {earliest_overlay_date}")
    logger.info(f"Agents common to both fits: {common_to_both_fits['agent'].tolist()}")

    # Re-plot those points with split colors (left half base, right half overlay)
    # to show they're used in both trendline fits
    success_col = f"p{script_params.get('success_percent', 50)}"
    for _, row in common_to_both_fits.iterrows():
        ax = axs[0]
        logger.info(
            f"Plotting split point for {row['agent']} at {row['release_date']}, p50={row[success_col]:.2f}"
        )
        # Plot left half in base color
        ax.plot(
            pd.to_datetime(row["release_date"]),
            row[success_col],
            marker="o",
            color=base_color,
            markersize=10,
            fillstyle="left",
            markeredgewidth=0,
            linestyle="None",
            zorder=15,
        )
        # Plot right half in overlay color
        ax.plot(
            pd.to_datetime(row["release_date"]),
            row[success_col],
            marker="o",
            color=overlay_color,
            markersize=10,
            fillstyle="right",
            markeredgewidth=0,
            linestyle="None",
            zorder=15,
        )

    # Add overlay points
    _add_overlay_results(axs[0], overlay_df, overlay_color)

    # Fit overlay trendline through overlay + common_to_both_fits points
    # Combine the data
    overlay_for_fit = overlay_df[["release_date", "p_value"]].rename(
        columns={"p_value": success_col}
    )
    common_for_fit = common_to_both_fits[["release_date", success_col]].copy()
    common_for_fit["release_date"] = pd.to_datetime(common_for_fit["release_date"])
    combined_for_fit = pd.concat([common_for_fit, overlay_for_fit], ignore_index=True)

    logger.info(
        f"Combined data for overlay trendline fit ({len(combined_for_fit)} points):"
    )
    for _, row in combined_for_fit.iterrows():
        logger.info(f"  {row['release_date']}: {row[success_col]:.2f}")

    if len(combined_for_fit) >= 2:
        overlay_reg, overlay_score = fit_trendline(
            combined_for_fit[success_col],
            pd.to_datetime(combined_for_fit["release_date"]),
            log_scale=True,
        )
        overlay_doubling_time = np.log(2) / overlay_reg.coef_[0]
        logger.info(
            f"Overlay trendline doubling time: {overlay_doubling_time:.1f} days, R²={overlay_score:.3f}"
        )

        # Plot overlay trendline
        # Convert dates to proper types
        min_date_raw = combined_for_fit["release_date"].min()
        max_date_raw = combined_for_fit["release_date"].max()
        assert not pd.isna(min_date_raw) and not pd.isna(max_date_raw)
        min_date = pd.Timestamp(min_date_raw)
        max_date = pd.Timestamp(max_date_raw)
        overlay_dashed_outside = (min_date, max_date)

        # Convert trendline_end_date to string if it's a Timestamp
        end_date_str = (
            trendline_end_date.strftime("%Y-%m-%d")
            if isinstance(trendline_end_date, pd.Timestamp)
            else trendline_end_date
        )

        trendline_params: TrendlineParams = {
            "after_date": script_params["trendlines"][0]["after_date"],
            "color": overlay_color,
            "line_start_date": min_date.strftime("%Y-%m-%d"),
            "line_end_date": end_date_str,
            "display_r_squared": True,
            "data_file": None,
            "styling": None,
            "caption": None,
            "skip_annotation": False,
            "fit_type": "exponential",
        }

        overlay_annotation = plot_trendline(
            ax=axs[0],
            dashed_outside=overlay_dashed_outside,
            plot_params=plot_params,
            trendline_params=trendline_params,
            reg=overlay_reg,
            score=overlay_score,
            log_scale=True,
        )

        # Add overlay annotation above the base one
        if (
            not script_params.get("hide_regression_info", False)
            and overlay_annotation is not None
        ):
            overlay_annotation["color"] = overlay_color
            axs[0].annotate(
                **overlay_annotation,
                xy=(0.98, 0.18),
                xycoords="axes fraction",
                ha="right",
                va="bottom",
            )


def _add_time_markers(ax: Axes) -> None:
    """Add reference time markers on the y-axis."""
    time_markers = {
        15 / 60: "Answer question",
        2: "Count words in passage",
        10: "Find fact on web",
        49: "Train classifier",
        4 * 60: "Train adversarially robust image model",
    }

    for seconds, label in time_markers.items():
        ax.axhline(
            y=seconds,
            color="#2c7c58",
            linestyle="-",
            alpha=0.4,
            zorder=1,
            xmin=0,
            xmax=0.01,
        )
        ax.text(
            0.02,
            seconds,
            label,
            transform=ax.get_yaxis_transform(),
            verticalalignment="center",
            horizontalalignment="left",
            fontsize=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.5),
        )


def _add_individual_labels(
    axs: list[Axes],
    agent_summaries: pd.DataFrame,
    script_params: dict[str, Any],
    scale: Literal["log", "linear"],
) -> None:
    # Label first and last points
    sorted_data = agent_summaries.sort_values("release_date")
    rename_map = script_params["rename_legend_labels"]

    def rename_with_default(x: str) -> str:
        return rename_map.get(x, x)

    sorted_data["agent"] = sorted_data["agent"].map(rename_with_default)
    first_point = sorted_data.iloc[0]
    last_point = sorted_data.iloc[-1]

    # TODO would be great to map from agent names to label position, and ideally make this configurable in the yaml
    second_to_last = sorted_data.iloc[-2]
    log_agents_to_label = [
        (first_point, (4, -6), None),
        (sorted_data.iloc[1], (8, 0), None),
        (sorted_data.iloc[2], (8, 0), None),
        (sorted_data.iloc[3], (-6, 0), None),  # gpt 4
        (sorted_data.iloc[5], (4, -2), None),  # gpt 4
        (sorted_data.iloc[9], (4, -10), None),
        (second_to_last, (8, -10), None),
        (last_point, (8, -10), None),
    ]

    linear_agents_to_label = [
        (point, label_pos, None)
        for point, label_pos in [
            (first_point, (0, 6)),
            (sorted_data.iloc[1], (0, 6)),
            (sorted_data.iloc[2], (0, 6)),
            (sorted_data.iloc[3], (0, 6)),  # gpt 4
            (sorted_data.iloc[4], (0, 6)),
            (sorted_data.iloc[5], (6, -2)),  # gpt 4o
            (sorted_data.iloc[6], (-12, 0.1)),
            (sorted_data.iloc[7], (-12, 0.1)),
            (sorted_data.iloc[8], (-12, 0.1)),
            (sorted_data.iloc[9], (-12, 0.1)),
            (second_to_last, (-12, 0.1)),
            (last_point, (-12, 0.1)),
        ]
    ]
    if scale == "linear":
        agents_to_label = linear_agents_to_label
    else:
        agents_to_label = log_agents_to_label

    for point, label_pos, horizontal_alignment in agents_to_label:
        axs[0].annotate(
            point["agent"],
            xy=(
                pd.to_datetime(point["release_date"]),
                point[f"p{script_params.get('success_percent', 50)}"],
            ),
            xytext=label_pos,
            textcoords="offset points",
            ha=horizontal_alignment or ("left" if label_pos[0] > 0 else "right"),
            va="bottom" if label_pos[1] > 0 else "top",
            fontsize=12,
            color="grey",
        )


@dataclass
class DoublingTimeStats:
    point_estimate: float
    median: float
    ci_lower: float
    ci_upper: float
    pct_above: float
    pct_below: float
    n_samples: int
    all_doubling_times: List[float]


def _compute_doubling_time_and_predictions_from_p50s(
    p50s: List[float],
    dates: List[pd.Timestamp],
    time_points: pd.DatetimeIndex,
) -> tuple[float, NDArray[Any]]:
    reg, _ = fit_trendline(
        pd.Series(p50s),
        pd.Series(pd.to_datetime(dates)),
        log_scale=True,
    )
    time_x = date2num(time_points)
    predictions = np.exp(reg.predict(time_x.reshape(-1, 1)))
    slope = reg.coef_[0]
    doubling_time = np.log(2) / slope
    return float(doubling_time), predictions


def compute_bootstrap_confidence_region(
    agent_summaries: pd.DataFrame,
    bootstrap_results: pd.DataFrame,
    release_dates: dict[str, dict[str, date]],
    after_date: str,
    max_date: pd.Timestamp,
    confidence_level: float,
) -> tuple[DoublingTimeStats, pd.DatetimeIndex, NDArray[Any], NDArray[Any]]:
    """Compute bootstrap confidence intervals from bootstrap samples.

    Args:
        bootstrap_results: DataFrame with columns for each agent containing p50s.
        release_dates: Dictionary mapping agent names to release dates
        after_date: Start date for trendline
        max_date: End date for trendline
        confidence_level: Confidence level for the interval (e.g. 0.95)

    Returns:
        Tuple of (doubling_time_stats, time_points, lower_bound, upper_bound)
    """
    dates = release_dates["date"]

    focus_agents = [
        colname.removesuffix("_p50") for colname in bootstrap_results.columns
    ]

    doubling_times = []
    n_bootstraps = len(bootstrap_results)

    time_points = pd.date_range(
        start=pd.to_datetime(after_date),
        end=max_date,
        freq="D",
    )
    predictions = np.zeros((n_bootstraps, len(time_points)))
    assert n_bootstraps > 0
    for sample_idx in range(n_bootstraps):
        valid_p50s = []
        valid_dates = []

        for agent in focus_agents:
            p50 = pd.to_numeric(
                bootstrap_results[f"{agent}_p50"].iloc[sample_idx], errors="coerce"
            )

            if pd.isna(p50) or np.isinf(p50) or p50 < 1e-3:
                continue

            valid_p50s.append(p50)
            valid_dates.append(dates[agent])

        if len(valid_p50s) < 2:
            continue

        doubling_time, predictions_for_sample = (
            _compute_doubling_time_and_predictions_from_p50s(
                valid_p50s, valid_dates, time_points
            )
        )
        if doubling_time > 0:
            doubling_times.append(doubling_time)
        predictions[sample_idx] = predictions_for_sample

    # Fit a single trendline to the agent summaries, to compute the point estimate for the trendline
    point_estimate_doubling_time, _ = _compute_doubling_time_and_predictions_from_p50s(
        agent_summaries["p50"].tolist(),
        agent_summaries["release_date"].tolist(),
        time_points,
    )

    low_q = (1 - confidence_level) / 2
    high_q = 1 - low_q
    lower_bound = np.nanpercentile(predictions, low_q * 100, axis=0)
    upper_bound = np.nanpercentile(predictions, high_q * 100, axis=0)

    median = float(np.median(doubling_times))
    ci_lower = float(np.percentile(doubling_times, low_q * 100))
    ci_upper = float(np.percentile(doubling_times, high_q * 100))

    stats = DoublingTimeStats(
        point_estimate=point_estimate_doubling_time,
        median=median,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        pct_above=(ci_upper - median) / median,
        pct_below=(median - ci_lower) / median,
        n_samples=len(doubling_times),
        all_doubling_times=doubling_times,
    )

    return stats, time_points, lower_bound, upper_bound


def add_bootstrap_confidence_region(
    ax: Axes,
    agent_summaries: pd.DataFrame,
    bootstrap_results: pd.DataFrame,
    release_dates: dict[str, dict[str, date]],
    after_date: str,
    max_date: pd.Timestamp,
    confidence_level: float,
    color: str = "#d2dfd7",
) -> DoublingTimeStats:
    """Add bootstrap confidence intervals and region to an existing plot.

    Args:
        ax: matplotlib axes
        bootstrap_results: DataFrame with columns for each agent containing p50s.
        release_dates: Dictionary mapping agent names to release dates
        after_date: Start date for trendline
        max_date: End date for trendline
        confidence_level: Confidence level for the interval (e.g. 0.95)

    Returns:
        DoublingTimeStats with median and confidence interval
    """
    stats, time_points, lower_bound, upper_bound = compute_bootstrap_confidence_region(
        agent_summaries,
        bootstrap_results,
        release_dates,
        after_date,
        max_date,
        confidence_level,
    )

    ax.fill_between(
        time_points,
        lower_bound,
        upper_bound,
        color=color,
        alpha=0.4,
    )

    return stats


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

    # Load comparison colors from params
    comparison_color_1 = params.get("comparison_color_1", "blue")
    comparison_color_2 = params.get("comparison_color_2", "orange")

    confidence_level = 0.95

    # Load data
    bootstrap_results = pd.read_csv(args.input_file)
    agent_summaries = pd.read_csv(args.agent_summaries_file)
    release_dates = yaml.safe_load(args.release_dates.read_text())

    after_date = script_params["trendlines"][0].get("after_date", None)
    before_date = script_params["trendlines"][0].get("before_date", None)

    # Filter agents for display: exclude if in exclude_agents
    exclude_agents_for_display = script_params["exclude_agents"]
    agent_summaries_for_display = _process_agent_summaries(
        exclude_agents_for_display, agent_summaries, release_dates
    )
    # Filter agents for fitting trendline: exclude if in exclude_agents OR in exclude_agents_from_all_fits
    exclude_agents_from_all_fits = script_params.get("exclude_agents_from_all_fits", [])
    exclude_agents_from_all_fits = (
        script_params["exclude_agents"] + exclude_agents_from_all_fits
    )
    agent_summaries_for_fitting = _process_agent_summaries(
        exclude_agents_from_all_fits,
        agent_summaries,
        release_dates,
        after_date,
        before_date,
    )

    subtitle = script_params["subtitle"] or ""
    title = _get_title(script_params, script_params.get("success_percent", 50))

    # Create plot with two subplots
    if script_params.get("show_boxplot", False):
        fig, axs = plt.subplots(1, 2, width_ratios=[6, 1], figsize=(12, 6))
    else:
        fig, axs = plt.subplots(1, 1, figsize=(12, 6))
        axs = [axs]

    if "linear_overrides" in script_params and args.y_scale == "linear":
        for key in script_params["linear_overrides"]:
            if isinstance(script_params["linear_overrides"][key], dict):
                if key not in script_params:
                    script_params[key] = {}
                script_params[key] = {
                    **script_params[key],
                    **script_params["linear_overrides"][key],
                }
            else:
                script_params[key] = script_params["linear_overrides"][key]
    if "plot_style_overrides" in script_params:
        for key in script_params["plot_style_overrides"]:
            plot_params[key] = plot_params[key] if key in plot_params else {}
            for subkey in script_params["plot_style_overrides"][key]:
                if subkey not in plot_params[key]:
                    plot_params[key][subkey] = {}
                plot_params[key][subkey] = {
                    **plot_params[key][subkey],
                    **script_params["plot_style_overrides"][key][subkey],
                }

    end_date = script_params["x_lim_end"]
    upper_y_lim = script_params["upper_y_lim"]
    trendline_end_date = script_params["x_lim_end"]
    if args.y_scale == "linear":
        max_release_date = agent_summaries_for_display["release_date"].dropna().max()
        end_date = max_release_date + pd.Timedelta(days=60)
        upper_y_lim = agent_summaries_for_display["p50"].max() * 1.2
        trendline_end_date = max_release_date

    # If there's an overlay, override agent colors to use comparison_color_1
    if script_params.get("overlay_results") is not None:
        for agent in plot_params["agent_styling"]:
            if agent != "default":
                plot_params["agent_styling"][agent]["lab_color"] = comparison_color_1

    plot_horizon_graph(
        plot_params,
        agent_summaries_for_display,
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
        marker_override="o",
    )

    if script_params.get("show_example_tasks", False):
        _add_time_markers(axs[0])

    if script_params.get("show_watermark", False):
        parent_dir = pathlib.Path(__file__).parent.parent.parent
        add_watermark(fig, parent_dir / pathlib.Path("metr-logo.svg"))

    if script_params["individual_labels"]:
        _add_individual_labels(
            axs, agent_summaries_for_display, script_params, args.y_scale
        )

    stats = None
    if not script_params.get("hide_trendline", False):
        logger.info(
            f"Agents to be used for trendline central estimate:  {agent_summaries_for_fitting.sort_values('release_date')['agent'].values.tolist()}"
        )
        reg, score = fit_trendline(
            agent_summaries_for_fitting[f"p{script_params.get('success_percent', 50)}"],
            pd.to_datetime(agent_summaries_for_fitting["release_date"]),
            log_scale=True,
        )
        dashed_outside = (
            agent_summaries_for_fitting["release_date"].min(),
            agent_summaries_for_fitting["release_date"].max(),
        )

        # Use comparison_color_1 if there's an overlay, otherwise use default color
        base_trendline_color = (
            comparison_color_1
            if script_params.get("overlay_results") is not None
            else "#2c7c58"
        )

        annotation = plot_trendline(
            ax=axs[0],
            dashed_outside=dashed_outside,
            plot_params=plot_params,
            trendline_params={
                "after_date": script_params["trendlines"][0]["after_date"],
                "color": base_trendline_color,
                "line_start_date": script_params["trendlines"][0]["line_start_date"],
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
        )
        if (
            not script_params.get("hide_regression_info", False)
            and annotation is not None
        ):
            # Position annotation at bottom right
            axs[0].annotate(
                **annotation,
                xy=(0.98, 0.02),
                xycoords="axes fraction",
                ha="right",
                va="bottom",
            )

        # Use comparison_color_1 if there's an overlay, otherwise use default color
        confidence_region_color = (
            comparison_color_1
            if script_params.get("overlay_results") is not None
            else "#d2dfd7"
        )

        stats = add_bootstrap_confidence_region(
            ax=axs[0],
            agent_summaries=agent_summaries_for_fitting,
            bootstrap_results=bootstrap_results[
                [f"{agent}_p50" for agent in agent_summaries_for_fitting["agent"]]
            ],
            release_dates=release_dates,
            after_date=script_params["trendlines"][0]["line_start_date"],
            max_date=trendline_end_date,
            confidence_level=confidence_level,
            color=confidence_region_color,
        )
        logger.info(
            f"95% CI for doubling times: [{stats.ci_lower:.0f}, {stats.ci_upper:.0f}] days "
            f"(+{stats.pct_above:.0%}/-{stats.pct_below:.0%})"
        )

    # Add overlay results and trendline if provided in script_params
    _add_overlay_and_trendline(
        axs=axs,
        script_params=script_params,
        agent_summaries_for_fitting=agent_summaries_for_fitting,
        trendline_end_date=trendline_end_date,
        plot_params=plot_params,
        base_color=comparison_color_1,
        overlay_color=comparison_color_2,
    )

    # Add confidence region and existing scatter points to legend
    handles, labels = axs[0].get_legend_handles_labels()
    sorted_items = sorted(
        zip(handles, labels), key=lambda x: plot_params["legend_order"].index(x[1])
    )
    handles, labels = zip(*sorted_items)
    rename_map = script_params["rename_legend_labels"]
    labels = [rename_map.get(label, label) for label in labels]

    if script_params["individual_labels"]:
        legend = axs[0].get_legend()
        if legend is not None:
            legend.remove()
    else:
        axs[0].legend(
            handles,
            labels,
            loc="upper left",
            fontsize=script_params["legend_fontsize"],
            frameon=script_params["legend_frameon"],
        )
    axs[0].grid(script_params.get("show_grid", True))
    axs[0].grid(which="minor", linestyle=":", alpha=0.6, color="#d2dfd7")

    if script_params.get("show_boxplot", False):
        assert stats is not None  # show_boxplot implies not hide_trendline
        axs[1].boxplot(
            [stats.all_doubling_times], vert=True, showfliers=False, whis=(10, 90)
        )
        axs[1].set_xticklabels(["Doubling times\n(days)"])
        axs[1].set_ylim(0, 365)

    src_utils_plots.save_or_open_plot(args.output_file, params["plot_format"])


if __name__ == "__main__":
    main()
