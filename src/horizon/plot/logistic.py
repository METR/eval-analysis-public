from __future__ import annotations

import argparse
import logging
import pathlib
from typing import Any, Callable

import dvc.api
import matplotlib.axes
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yaml
from matplotlib import markers
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.dates import date2num, num2date
from matplotlib.figure import Figure
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression
from typing_extensions import Literal

from horizon.utils import plots


def _get_title(script_params: plots.ScriptParams, success_percent: int) -> str:
    # Get included task groups
    if "title" in script_params:
        return script_params["title"]
    task_group_names = ["HCAST", "SWAA", "RE-Bench"]
    included_task_groups = []
    for name in task_group_names:
        if name not in script_params["exclude"]:
            included_task_groups.append(name)

    # Make title
    task_groups_string = " + ".join(included_task_groups)
    title = f"{success_percent}% Time Horizon for {task_groups_string} Tasks"
    return title


def _remove_excluded_task_groups(
    all_runs: pd.DataFrame, script_params: plots.ScriptParams
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


def plot_task_distribution(
    ax: matplotlib.axes.Axes,
    runs_df: pd.DataFrame,
    plot_params: plots.PlotParams,
    weight_key: str,
) -> None:
    """Plot a vertical histogram of the human time estimates for each run."""

    # Because we're plotting the distribution of tasks, equal_task_weight is equivalent to no weight
    use_weighting = weight_key == "invsqrt_task_weight"

    data = runs_df.groupby("task_id")["human_minutes"].first().to_numpy()
    # Make sure we use the same size bins regardless of the range we're plotting
    log_bins = np.arange(np.log10(1 / 60), np.log10(data.max()), 0.2)
    bins = (10**log_bins).tolist()

    if use_weighting:
        # TODO fails if agents are run different numbers of times
        weights = runs_df.groupby("task_id")[weight_key].sum().to_numpy()
        # Multiply data by total weight to get the weighted number of tasks
        ax.hist(
            data,
            bins=bins,  # type: ignore
            weights=weights,
            orientation="horizontal",
            **plot_params["task_distribution_styling"]["hist"],
        )  # type: ignore
        ax.set_xlabel(
            "Number of tasks\n(Weighted)",
            fontsize=plot_params["ax_label_fontsize"],
            labelpad=plot_params["xlabelpad"],
        )
        ax.set_xticks([])  # With weighting, absolute number of runs isn't meaningful
    else:
        ax.hist(
            data,
            bins=bins,  # type: ignore
            orientation="horizontal",
            **plot_params["task_distribution_styling"]["hist"],
        )  # type: ignore
        ax.set_xlabel(
            "Number of tasks",
            fontsize=plot_params["ax_label_fontsize"],
            labelpad=plot_params["xlabelpad"],
        )

    ax.grid(**plot_params["task_distribution_styling"]["grid"])

    ax.set_yscale("log")
    ax.set_yticks([])  # y ticks will be shown in main plot

    ax.set_title(
        "Task Distribution",
        fontsize=plot_params["title_fontsize"],
        pad=plot_params["xlabelpad"],
    )


def setup_fig(include_horizon_graph: bool) -> tuple[plt.Figure, Axes, Axes | None]:  # type: ignore
    if include_horizon_graph:
        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(1, 5, wspace=0.5)
        ax = fig.add_subplot(gs[0, :4])
        ax_hist = fig.add_subplot(gs[0, 4], sharey=ax)  # Share y axis with main plot
        ax_hist.tick_params(axis="y", which="both", left=False, labelleft=False)
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax_hist = None
    return fig, ax, ax_hist


def plot_horizon_graph(
    plot_params: plots.PlotParams,
    all_agent_summaries: pd.DataFrame,
    runs_df: pd.DataFrame,
    release_dates: dict[str, str],
    lower_y_lim: float,
    x_lim_start: str,
    x_lim_end: str,
    subtitle: str,
    title: str,
    weight_key: str,
    exclude_agents: list[str],
    success_percent: int,
    script_params: plots.ScriptParams,
    trendlines: list[plots.TrendlineParams] | None = None,
    upper_y_lim: float | None = None,
    include_task_distribution: str = "none",
    fig: Figure | None = None,
    y_scale: Literal["log", "linear"] = "log",
    confidence_level: float = 0.95,
    marker_override: str | None = None,
) -> None:
    plot_style = plot_params["scatter_styling"]
    agent_style = plot_params["agent_styling"]
    agent_summaries = all_agent_summaries[
        pd.to_datetime(all_agent_summaries["release_date"])
        >= (pd.Timestamp(x_lim_start) - pd.Timedelta(days=365))
    ].copy()
    assert isinstance(agent_summaries, pd.DataFrame)

    if trendlines is None:
        trendlines = []

    fig_created_here: bool = fig is None

    if fig_created_here:
        fig, ax, ax_hist = setup_fig(include_task_distribution != "none")
    else:
        ax = fig.axes[0]
        ax_hist = None

    # Add watermark only if this function created the figure
    # Otherwise, the bootstrap_ci stage will have two watermarks
    if fig_created_here and script_params.get("show_watermark", False):
        parent_dir = pathlib.Path(__file__).parent.parent.parent
        plots.add_watermark(
            fig,
            parent_dir / pathlib.Path("metr-logo.svg"),
            legend_on_right=script_params.get("legend_on_right", False),
        )

    ax.set_ylim(lower_y_lim, upper_y_lim)

    y = agent_summaries[f"p{success_percent}"]
    y_clipped = y.clip(
        # np.finfo(float).eps, np.inf
        lower_y_lim * 1,
        np.inf,
    )  # clip because log scale makes 0 -> -inf

    low_q = (1 - confidence_level) / 2
    high_q = 1 - low_q
    y_low = agent_summaries[f"p{success_percent}q{low_q:.3f}"]
    y_high = agent_summaries[f"p{success_percent}q{high_q:.3f}"]

    yerr = np.array([y - y_low, y_high - y])
    yerr = np.clip(yerr, 0, np.inf)

    legend_labels = []
    legend_handles = []

    for i, agent in enumerate(agent_summaries["agent"]):
        if y_clipped.iloc[i] <= lower_y_lim or y_clipped.iloc[i] >= upper_y_lim:
            continue
        if not script_params.get("hide_error_bars", False):
            ax.errorbar(
                agent_summaries["release_date"].iloc[i],
                y_clipped.iloc[i],
                yerr=[[yerr[0, i]], [yerr[1, i]]],
                **plot_style["error_bar"],
            )
        ax.grid(**plot_style["grid"])
        if marker_override is not None:
            marker = marker_override
        else:
            marker = agent_style[agent]["marker"]
        scatter_handle = ax.scatter(
            agent_summaries["release_date"].iloc[i],
            y_clipped.iloc[i],
            color=agent_style[agent]["lab_color"],
            marker=marker,
            label=agent,
            **plot_style["scatter"],
        )

        legend_labels.append(agent)
        legend_handles.append(scatter_handle)

    # Add arrows for out-of-range points
    mask_out_range = y_clipped != y
    logging.info(f"masking out {mask_out_range.sum()} points")
    ax.scatter(
        agent_summaries.loc[mask_out_range, "release_date"],
        [lower_y_lim * 1.2] * mask_out_range.sum(),  # Place at bottom of visible range
        marker=markers.CARETDOWN,  # type: ignore
        color="grey",
        zorder=10,
        s=150,  # Increase marker size
    )

    annotations = []

    for trendline in trendlines:
        if trendline["fit_type"] == "linear":
            log_scale = False
        elif trendline["fit_type"] == "exponential":
            log_scale = True
        else:
            raise ValueError(f"Invalid fit type: {trendline['fit_type']}")

        if trendline["data_file"] is not None:
            data_file = trendline["data_file"]
            data = pd.read_csv(data_file)
        else:
            data = agent_summaries

        trendline_exclude_agents = trendline.get("exclude_agents", exclude_agents)
        data = _process_agent_summaries(
            trendline_exclude_agents,
            data,
            release_dates,
            after_date=trendline["after_date"],
        )

        print(f"fitting on {len(data)} models")
        trendline_success_percent = trendline.get("success_percent", success_percent)
        reg, score = fit_trendline(
            data[f"p{trendline_success_percent}"],
            pd.to_datetime(data["release_date"]),
            log_scale=log_scale,
        )
        dashed_outside = (data["release_date"].min(), data["release_date"].max())
        annotations.append(
            plot_trendline(
                ax,
                plot_params,
                trendline_params=trendline,
                dashed_outside=dashed_outside,
                reg=reg,
                score=score,
                log_scale=log_scale,
            )
        )

    if include_task_distribution != "none":
        assert ax_hist is not None
        plot_task_distribution(ax_hist, runs_df, plot_params, weight_key)

    if include_task_distribution == "full":
        assert ax_hist is not None
        assert ax is not None
        # y limits are determined by the histogram
        hist_low, hist_high = ax_hist.get_ylim()
        scat_low, scat_high = ax.get_ylim()
        ax.set_ylim(min(hist_low, scat_low), max(hist_high, scat_high))

    elif include_task_distribution == "clipped":
        assert ax_hist is not None
        assert ax is not None
        # y limits are determined by the main plot
        ax_hist.set_ylim(ax.get_ylim())

    plots.make_y_axis(ax, scale=y_scale, script_params=script_params)
    start_year = pd.Timestamp(x_lim_start).year
    end_year = pd.Timestamp(x_lim_end).year + 1
    xticks_skip = script_params.get("xticks_skip", 1) if script_params else 1
    plots.make_quarterly_xticks(ax, start_year, end_year, skip=xticks_skip)

    # Add subticks at regular month intervals if param is present
    months_between_ticks = (
        script_params.get("months_between_ticks", 0) if script_params else 0
    )
    if months_between_ticks > 0:
        assert xticks_skip == 1, "xticks_skip must be 1 when using months_between_ticks"
        plots.add_monthly_minor_ticks(
            ax,
            start_year,
            end_year,
            months_between_ticks,
            pd.Timestamp(x_lim_start),
            pd.Timestamp(x_lim_end),
        )

    ax.set_xlim(
        float(mdates.date2num(pd.Timestamp(x_lim_start))),
        float(mdates.date2num(pd.Timestamp(x_lim_end))),
    )

    ax.set_xlabel(
        script_params.get("xlabel", "Model release date")
        if script_params
        else "Model release date",
        fontsize=script_params.get(
            "ax_label_fontsize", plot_params["ax_label_fontsize"]
        )
        if script_params
        else plot_params["ax_label_fontsize"],
        labelpad=plot_params["xlabelpad"],
    )
    if script_params["show_y_label"]:
        ax.set_ylabel(
            script_params.get(
                "ylabel",
                f"Task time (for humans) that model completes with \n{success_percent}% success rate",
            ),
            fontsize=script_params.get(
                "ax_label_fontsize", plot_params["ax_label_fontsize"]
            ),
            labelpad=plot_params["ylabelpad"],
        )
    kwargs = {}
    if script_params.get("title_location", None) == "left":
        kwargs["x"] = -0.1
        kwargs["loc"] = "left"
    ax.set_title(
        title,
        fontsize=script_params.get("title_fontsize", plot_params["title_fontsize"])
        if script_params
        else plot_params["title_fontsize"],
        pad=3 * plot_params["xlabelpad"],
        **kwargs,
    )
    if subtitle:
        plt.suptitle(
            subtitle,
            y=0.93,
            x=0.1722
            if script_params.get("suptitle_location", "center") == "left"
            else 0.51,
            fontsize=plot_params["suptitle_fontsize"],
        )

    # The graph is too busy if we have both trendlines and legend
    # if not trendlines:
    # Only consider agents that are present in both legend_order and legend_labels
    available_agents = [
        agent for agent in plot_params["legend_order"] if agent in legend_labels
    ]
    # Sort handles and labels based on the filtered order
    sorted_pairs = sorted(
        zip(legend_handles, legend_labels),
        key=lambda pair: available_agents.index(pair[1]),
    )
    legend_handles, legend_labels = zip(*sorted_pairs)

    # Optionally place the legend on the right side of the figure
    # This is useful for long lists of models which overlap with the main plot too much
    legend_on_right = script_params.get("legend_on_right", False)

    if legend_on_right:
        legend_loc = "center left"
        legend_bbox = (1.02, 0.5)
    else:
        legend_loc = "best"
        legend_bbox = None

    ax.legend(
        legend_handles,
        legend_labels,
        loc=legend_loc,
        bbox_to_anchor=legend_bbox,
        fontsize=script_params.get("legend_fontsize", 12) if script_params else 12,
    )

    # Lay the annotations we collected earlier, ensuring they don't overlap
    padding = 10
    line_height = plot_params["annotation_fontsize"]
    bbox = ax.get_window_extent()
    y = (
        (bbox.y1 - bbox.y0) * 72 / fig.dpi
    )  # start at top-left corner of the plot, in axes points
    y = 0
    x = (bbox.x1 - bbox.x0) * 72 / fig.dpi
    for a in annotations:
        if a is not None:
            assert isinstance(a, dict)
            ax.annotate(
                xy=(x - padding, y + padding),
                xycoords="axes points",
                ha="right",
                va="bottom",
                **a,
            )
            n_lines = len(a["text"].split("\n"))
            # next annotation will go above this one
            y += line_height * n_lines + padding


def fit_trendline(
    agent_horizons: pd.Series[float],
    release_dates: pd.Series[pd.Timestamp],
    log_scale: bool = False,
) -> tuple[LinearRegression, float]:
    """Fit a trendline showing the relationship between release date and time horizon.

    Args:
        agent_horizons: Series containing the time horizons for each agent
        release_dates: Series containing the release dates for each agent
        log_scale: Whether to fit in log space (exponential fit) or linear space

    Returns:
        A tuple containing the fitted LinearRegression model and the R^2 score
    """
    # Convert dates to numeric format for regression
    X = np.array([date2num(d) for d in release_dates]).reshape(-1, 1)

    y_raw = agent_horizons.clip(1e-3, np.inf)
    y = np.log(y_raw) if log_scale else y_raw

    # Fit the regression model
    reg = LinearRegression().fit(X, y)

    score = float(reg.score(X, y))

    return reg, score


class FitFunctionWrapper:
    def __init__(self, func: Callable[..., NDArray[Any]], params: list[Any]):
        self.func = func
        self.params = params

    def predict(self, x: NDArray[Any]) -> NDArray[Any]:
        return self.func(x, *self.params)


def plot_trendline(
    ax: Axes,
    plot_params: plots.PlotParams,
    trendline_params: plots.TrendlineParams,
    dashed_outside: tuple[pd.Timestamp, pd.Timestamp] | None,
    reg: LinearRegression | FitFunctionWrapper,
    score: float,
    log_scale: bool = False,
    plot_kwargs: dict[str, Any] = {},
) -> dict[str, Any] | None:
    """Plot a trendline showing the relationship between release date and time horizon."""
    trendline_styling = plot_params["performance_over_time_trendline_styling"]

    # Extract values from trendline_params
    after = trendline_params["after_date"]
    fit_color = trendline_params.get("color", None)
    line_start_date = trendline_params.get("line_start_date", None)
    line_end_date = trendline_params["line_end_date"]
    display_r_squared = trendline_params.get("display_r_squared", False)
    caption = trendline_params.get("caption", None)
    styling = trendline_params.get("styling", None)
    skip_annotation = trendline_params.get("skip_annotation", False)
    fit_type = trendline_params["fit_type"]
    display_after_date = trendline_params.get("display_after_date", True)

    # trendline goes to the end of the x-axis
    start_date = (
        pd.Timestamp(after)
        if line_start_date is None
        else pd.Timestamp(line_start_date)
    )
    end_date = pd.Timestamp(line_end_date)

    if fit_type == "auto":
        fit_type = "exponential" if log_scale else "linear"

    if fit_color is None:
        fit_color = trendline_styling[fit_type]["line"]["color"]

    fit_styling = trendline_styling[fit_type]
    fit_styling["line"]["color"] = fit_color
    # Plot trendline
    pk = {
        **fit_styling["line"],
    } | plot_kwargs
    if styling is not None:
        pk.update(styling)

    x_range = np.linspace(date2num(start_date), date2num(end_date), 120)

    if dashed_outside is not None:
        # ensure that we have datapoints at the dashed/undashed boundaries so that the boundaries are rendered accurately
        x_range = np.unique(
            np.concatenate(
                [
                    x_range,
                    [
                        date2num(dashed_outside[0]) - 1,
                        date2num(dashed_outside[0]),
                        date2num(dashed_outside[1]),
                        date2num(dashed_outside[1]) + 1,
                    ],
                ]
            )
        )

    y_pred = reg.predict(x_range.reshape(-1, 1))

    x_dates = np.array(num2date(x_range))
    y_values = np.exp(y_pred) if log_scale else y_pred  # Convert back from log scale

    if dashed_outside is None:
        ax.plot(x_dates, y_values, **pk)
    else:
        dashed_masks = [
            date2num(x_dates) < date2num(dashed_outside[0]),
            date2num(x_dates) > date2num(dashed_outside[1]),
        ]

        undashed_mask = np.logical_not(np.logical_or.reduce(dashed_masks))

        for mask in dashed_masks:
            ax.plot(
                x_dates[mask],
                y_values[mask],
                **(pk | {"linestyle": "dashed", "alpha": 0.4}),
            )
        ax.plot(x_dates[undashed_mask], y_values[undashed_mask], **pk)

    if caption is None:
        caption = ""

    if skip_annotation:
        return None
    annotation = caption
    if fit_type == "exponential":
        assert isinstance(reg, LinearRegression)
        doubling_time = 1 / reg.coef_[0] * np.log(2)
        annotation += f"\nDoubling time: {doubling_time:.0f} days"
    if display_after_date:
        annotation += "\n" + ("All data" if after == "0000-00-00" else f"{after}+ data")
    if display_r_squared:
        annotation += f"\nR²: {score:.2f}"

    annotation_styling = {
        "color": fit_color,
        "fontsize": plot_params["annotation_fontsize"],
    }
    annotation_styling["color"] = fit_color
    return dict(
        text=annotation,
        transform=ax.get_xaxis_transform(),
        alpha=1,
        **annotation_styling,  # type: ignore
    )


def _process_agent_summaries(
    exclude_agents: list[str] | None,
    agent_summaries: pd.DataFrame,
    release_dates: Any,
    after_date: str | None = None,
    before_date: str | None = None,
) -> pd.DataFrame:
    agent_summaries["release_date"] = agent_summaries["agent"].map(
        release_dates["date"]
    )
    agent_summaries = agent_summaries[agent_summaries["agent"] != "human"]
    if exclude_agents is not None:
        agent_summaries = agent_summaries[
            ~agent_summaries["agent"].isin(exclude_agents)
        ]
    if after_date is not None:
        agent_summaries = agent_summaries[
            agent_summaries["release_date"] >= pd.Timestamp(after_date).date()
        ]
    if before_date is not None:
        agent_summaries = agent_summaries[
            agent_summaries["release_date"] <= pd.Timestamp(before_date).date()
        ]
    return agent_summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=pathlib.Path, required=True)
    parser.add_argument("--output-file", type=pathlib.Path, required=True)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--runs-file", type=pathlib.Path, required=False)
    parser.add_argument("--release-dates", type=pathlib.Path, required=False)
    parser.add_argument("--script-parameter-group", type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    params = dvc.api.params_show(stages="plot_logistic_regression", deps=True)
    fig_params = params["figs"]["plot_logistic_regression"][args.script_parameter_group]

    agent_summaries = pd.read_csv(args.input_file)
    release_dates = yaml.safe_load(args.release_dates.read_text())
    agent_summaries = _process_agent_summaries(
        fig_params["exclude_agents"], agent_summaries, release_dates
    )

    logging.info("Loaded input data")

    runs_df = pd.read_json(args.runs_file, lines=True)
    runs_df = _remove_excluded_task_groups(runs_df, fig_params)

    title = _get_title(fig_params, fig_params.get("success_percent", 50))
    subtitle = fig_params.get("subtitle", "")

    plot_horizon_graph(
        plot_params=params["plots"],
        all_agent_summaries=agent_summaries,
        title=title,
        release_dates=release_dates,
        runs_df=runs_df,
        subtitle=subtitle,
        lower_y_lim=fig_params["lower_y_lim"],
        upper_y_lim=fig_params["upper_y_lim"],
        x_lim_start=fig_params["x_lim_start"],
        x_lim_end=fig_params["x_lim_end"],
        include_task_distribution=fig_params["include_task_distribution"],
        weight_key=fig_params["weighting"],
        trendlines=fig_params["trendlines"],
        exclude_agents=fig_params["exclude_agents"],
        success_percent=fig_params.get("success_percent", 50),
        script_params=fig_params,
    )

    plots.save_or_open_plot(args.output_file, params["plot_format"])


if __name__ == "__main__":
    main()
