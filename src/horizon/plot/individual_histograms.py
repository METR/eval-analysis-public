import argparse
import logging
import pathlib
from typing import Any, Literal, Sequence, Union

import dvc.api
import matplotlib.axes
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, to_rgb
from matplotlib.ticker import FuncFormatter
from scipy.special import expit
from typing_extensions import TypedDict

from horizon.utils import plots
from horizon.utils.plots import get_logarithmic_bins


class horizontalLineStyling(TypedDict):
    color: str
    linestyle: str
    linewidth: float
    alpha: float


class HorizontalLine(TypedDict):
    p_success: float
    styling: horizontalLineStyling


class ScriptParams(TypedDict):
    parameter_group_name: str
    logistic_file: str
    weighting: str
    title: str
    regularization: float
    categories: str
    n_subplot_cols: int
    horizontal_lines: list[HorizontalLine]
    annotate_p50: bool
    exclude: list[Literal["General Autonomy", "SWAA", "RE-Bench"]]
    include_agents: list[str]
    type: list[Literal["stacked", "overlaid", "normal"]]
    width: float
    rename_labels: dict[str, str]


def _darken_color(
    color: Union[str, np.ndarray[Any, np.dtype[np.float64]]], factor: float = 0.7
) -> tuple[float, float, float]:
    """Darken a color, ensuring RGB values remain between 0 and 1."""
    # If color is already RGBA (e.g., from colormap), extract RGB
    if isinstance(color, (tuple, list)) and len(color) >= 3:
        r, g, b = color[:3]
    else:
        assert isinstance(color, str)
        r, g, b = to_rgb(color)

    # Convert to numpy array and apply adjustments
    rgb = np.array([r, g, b], dtype=np.float64)
    adjustments = np.array([0.0, -0.15, 0.1], dtype=np.float64)
    rgb_adjusted = rgb + adjustments

    # Clamp values to [0, 1] range using numpy
    rgb_clamped = np.clip(rgb_adjusted, 0.0, 1.0)

    return tuple(rgb_clamped)


def _get_title(script_params: ScriptParams) -> str:
    return script_params["title"]


def _remove_empty_subplots(
    agent_summaries: pd.DataFrame,
    axes: list[matplotlib.axes.Axes],
    focus_agents: list[str],
) -> None:
    num_subplots = len(agent_summaries)

    if num_subplots < len(axes):
        # Remove axes starting from the end
        for idx in range(len(axes) - 1, num_subplots - 1, -1):
            axes[idx].remove()


def _get_all_agents_min_max_time(
    all_runs: pd.DataFrame, focus_agents: Sequence[str]
) -> tuple[float, float]:
    all_agents_runs = all_runs[all_runs["alias"].isin(focus_agents)]
    all_agents_min_time = all_agents_runs["human_minutes"].min()
    all_agents_max_time = all_agents_runs["human_minutes"].max()
    return all_agents_min_time, all_agents_max_time


def _remove_excluded_task_groups(
    all_runs: pd.DataFrame, script_params: ScriptParams
) -> pd.DataFrame:
    # Exclude tasks from runs_df
    if "General Autonomy" in script_params["exclude"]:
        raise ValueError(
            "Exclusion of general autonomy has not been implemented in logistic.py, panic"
        )

    if "SWAA" in script_params["exclude"]:
        if "run_id" not in all_runs.columns:
            raise ValueError(
                "Trying to exclude SWAA, which needs run_id column, but runs_df does not have run_id column"
            )
        all_runs = all_runs[
            ~all_runs["run_id"].astype(str).str.contains("small_tasks_")
        ]

    if "RE-Bench" in script_params["exclude"]:
        all_runs = all_runs[~all_runs["task_id"].astype(str).str.contains("ai_rd_")]
    return all_runs


def plot_logistic_regression_on_histogram(
    plot_params: plots.PlotParams,
    agent_summaries: pd.DataFrame,
    all_runs: pd.DataFrame,
    focus_agents: Sequence[str],
    output_file: pathlib.Path,
    script_params: ScriptParams,
) -> None:
    """Create subplots showing logistic regression curves with empirical histograms for each agent.

    Args:
        plot_params: Plot parameters including colors
        agent_summaries: DataFrame with agent regression parameters
        all_runs: DataFrame with individual run results
        focus_agents: List of agents to plot
        output_file: Path to save the figure
        script_params: Script parameters
    """
    agent_summaries = agent_summaries[agent_summaries["agent"].isin(focus_agents)]
    n_agents = len(agent_summaries["agent"].unique())
    n_cols = script_params["n_subplot_cols"]
    n_rows = (n_agents + n_cols - 1) // n_cols  # Ceiling division

    alpha = 0.5
    plot_type = script_params.get("type", "normal")
    if plot_type == "overlaid":
        n_rows = 1
        n_cols = 1
        width = script_params.get("width", 10)
        alpha = 0.9
    else:
        width = script_params.get("width", 15)
    figsize = (width, 4 * n_rows)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        sharey=True,
        height_ratios=[0.8] * n_rows,
    )

    # Make some room above plot for title
    fig.subplots_adjust(top=0.7)
    fig.suptitle(
        script_params["title"],
        fontsize=plot_params["suptitle_fontsize"] + 4,
        y=1.0,
    )

    # Turn axes into a 1D array, regardless of its current shape
    if hasattr(axes, "flatten"):
        axes = axes.flatten()  # Flatten to make indexing easier
    elif not hasattr(axes, "__len__"):
        axes = [axes]

    all_agents_min_time, all_agents_max_time = _get_all_agents_min_max_time(
        all_runs, focus_agents
    )

    # Order agents by focus agents list
    grouped_agent_summaries = agent_summaries.groupby("agent")
    ordered_agent_summaries = [
        grouped_agent_summaries.get_group(agent) for agent in focus_agents
    ]

    # Get release dates and normalize them for color mapping
    release_dates = pd.to_datetime(agent_summaries["release_date"], errors="coerce")
    valid_dates = release_dates.dropna()

    min_date = valid_dates.min()
    max_date = valid_dates.max()

    if pd.isna(min_date) or pd.isna(max_date):
        logging.warning("Could not determine valid date range for color mapping.")
        norm = Normalize(vmin=0, vmax=1)
    else:
        norm = Normalize(vmin=min_date.timestamp(), vmax=max_date.timestamp())
    sm = plt.cm.ScalarMappable(
        cmap=plt.get_cmap("YlGn"),
        norm=norm,
    )

    for idx, agent_info in enumerate(ordered_agent_summaries):
        if plot_type == "overlaid":
            ax_idx = 0
        else:
            ax_idx = idx
        agent = agent_info.iloc[0]["agent"]

        if plot_type == "overlaid":
            agent_color = sm.to_rgba(
                pd.Timestamp(agent_info.iloc[0]["release_date"]).timestamp()  # type: ignore
            )
        else:
            agent_color = plots.get_agent_color(plot_params=plot_params, agent=agent)

        agent_runs = all_runs[all_runs["alias"] == agent]

        times = agent_runs["human_minutes"]
        successes = agent_runs["score_binarized"]
        task_weights = agent_runs[script_params["weighting"]]

        bins = get_logarithmic_bins(all_agents_min_time, all_agents_max_time)

        # Calculate success rates for each bin using numpy's histogram, and weighted by weight column
        weighted_counts_success, _ = np.histogram(
            times[successes == 1],
            bins=bins,
            weights=task_weights[successes == 1],
        )
        weighted_counts_total, _ = np.histogram(times, bins=bins, weights=task_weights)

        # Avoid division by zero
        success_rates = np.zeros_like(weighted_counts_total, dtype=float)
        mask = weighted_counts_total > 0
        success_rates[mask] = (
            weighted_counts_success[mask] / weighted_counts_total[mask]
        )

        # Plot histogram bars
        width = np.diff(bins)
        centers = bins[:-1]
        axes[ax_idx].bar(
            centers,
            success_rates,
            width=width,
            alpha=alpha,
            color=agent_color,
            align="edge",
            # edgecolor=_darken_color(agent_color, 0.8),
        )

        if plot_type == "normal":
            # Calculate standard errors
            standard_errors = np.zeros_like(success_rates)
            for i in range(len(bins) - 1):
                if mask[i]:
                    bin_mask = (times >= bins[i]) & (times < bins[i + 1])
                    weights_in_bin = task_weights[bin_mask]
                    p = success_rates[i]

                    # Calculate effective sample size for weighted data
                    n_eff = np.sum(weights_in_bin) ** 2 / np.sum(weights_in_bin**2)

                    # Standard error for weighted binary data
                    if n_eff > 0:
                        variance = (p * (1 - p)) / n_eff
                        if variance > 0:  # Add check for positive variance
                            standard_errors[i] = np.sqrt(variance)

            # Plot error bars for bins with data
            axes[ax_idx].errorbar(
                centers[mask] + width[mask] / 2,  # Center the error bars
                success_rates[mask],
                yerr=2
                * standard_errors[
                    mask
                ],  # 2 standard errors for 95% confidence interval
                fmt="o",
                color=_darken_color(agent_color, 0.65),
                alpha=0.9,
                markersize=5,  # Use fixed marker size
                capsize=3,
                label="Empirical success\nrates w/ ± 2SE",
            )

        # Plot logistic curve using exact same parameters as plot_logistic_regression
        min_x = max(all_agents_min_time, 1 / 6)

        x = np.logspace(np.log(min_x), np.log(32), 1000)
        y = expit(
            agent_info.iloc[0]["coefficient"] * np.log2(x)
            + agent_info.iloc[0]["intercept"]
        )
        line_color = _darken_color(agent_color, 0.5)
        if plot_type == "overlaid":
            line_color = _darken_color(agent_color, 0.3)
        axes[ax_idx].plot(
            x,
            y,
            color=line_color,
            label="Fitted curve",
            linewidth=3,
            alpha=1,
        )

        # Find the x value where the curve crosses 0.5

        p50_line_x = x[np.argmin(np.abs(y - 0.5))]

        if p50_line_x >= all_agents_min_time and p50_line_x <= all_agents_max_time:
            axes[ax_idx].axvline(
                p50_line_x,
                color=line_color,
                linestyle="dashed",
                linewidth=2.5,
                ymax=0.5,
            )
            # add a point at the p50 line, on the y axis
            axes[ax_idx].plot(
                p50_line_x,
                -0.05,
                "x",
                color=agent_color,
                linewidth=2,
                markersize=15,
                alpha=1,
                markeredgewidth=3,
            )
            if script_params["annotate_p50"]:
                axes[ax_idx].annotate(
                    f"Time Horizon:\n{plots.format_time_label(p50_line_x * 60)}",
                    (40 * 60, 0.5),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="right",
                    fontsize=12,
                    va="bottom",
                    color=_darken_color(agent_color, 0.3),
                )
        else:
            if p50_line_x < all_agents_min_time:
                if script_params["annotate_p50"]:
                    axes[ax_idx].annotate(
                        f"Time horizon:\n< {plots.format_time_label(all_agents_min_time * 60)}",
                        (40 * 60, 0.5),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="right",
                        color=_darken_color(agent_color, 0.3),
                        va="bottom",
                        fontsize=12,
                    )
            else:
                pass  # (some of the crap models have long time horizons because of poor fits)

        axes[ax_idx].axhline(
            0.5,
            linestyle="dotted",
            alpha=0.3,
            color="black",
            xmin=0,
            xmax=1,
        )

        plots.log_x_axis(axes[ax_idx])
        # Show only every other x tick if there are more than 10 ticks
        xticks = axes[ax_idx].get_xticks()
        if len(xticks) > 12:
            axes[ax_idx].set_xticks(xticks[::2])

        if plot_type == "normal":
            # Customize subplot
            axes[ax_idx].tick_params(axis="x", rotation=45)
            # Use rename_labels if provided, otherwise use agent name as-is
            display_name = script_params.get("rename_labels", {}).get(agent, agent)
            axes[ax_idx].set_title(
                f"{display_name}",
                fontsize=plot_params["ax_label_fontsize"],
            )
        # if last row, add xlabel
        if idx >= len(axes) - n_cols:
            axes[ax_idx].set_xlabel(
                "Task length (human time)",
                fontsize=plot_params["ax_label_fontsize"],
            )
        if idx % n_cols == 0:
            axes[ax_idx].set_ylabel(
                "Success Probability",
                fontsize=plot_params["ax_label_fontsize"],
            )

        axes[ax_idx].grid(True, alpha=0.15)
        # if last row, plot legend
        if idx % n_cols == n_cols - 1 and idx <= n_rows and plot_type == "normal":
            axes[ax_idx].legend(loc="upper right")

        axes[ax_idx].set_ylim(-0.05, 1.05)

        if plot_type == "overlaid" and idx == 0:
            # Add colorbar showing date range

            cbar = plt.colorbar(sm, ax=axes[ax_idx])
            cbar.ax.set_ylabel(
                "Release Date", fontsize=plot_params["ax_label_fontsize"] - 2
            )

            cbar.ax.yaxis.set_major_formatter(
                FuncFormatter(lambda x, _: pd.Timestamp(x, unit="s").strftime("%Y-%m"))
            )
            cbar.ax.tick_params(
                labelsize=plot_params["ax_label_fontsize"] - 4, rotation=45
            )

    _remove_empty_subplots(agent_summaries, axes, list(focus_agents))

    fig.tight_layout()

    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True)
    fig.savefig(output_file)
    logging.info(f"Saved logistic regression with histogram plots to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-runs-file", type=pathlib.Path, required=True)
    parser.add_argument("--output-file", type=pathlib.Path, required=True)
    parser.add_argument("--plot-format", type=str, default="png")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--script-parameter-group", type=str, required=True)
    parser.add_argument("--params-file", type=pathlib.Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    params = dvc.api.params_show(stages="plot_individual_histograms", deps=True)
    fig_params = params["figs"]["plot_individual_histograms"][
        args.script_parameter_group
    ]
    agent_summaries = pd.read_csv(fig_params["logistic_file"])

    logging.info("Loaded input data")
    all_runs = pd.read_json(args.all_runs_file, lines=True)
    all_runs = _remove_excluded_task_groups(all_runs, fig_params)

    plot_logistic_regression_on_histogram(
        params["plots"],
        agent_summaries,
        all_runs,
        fig_params["include_agents"],
        args.output_file,
        fig_params,
    )


if __name__ == "__main__":
    main()
