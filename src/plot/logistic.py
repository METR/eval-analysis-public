from __future__ import annotations

import argparse
import logging
import pathlib
from typing import Any, cast

import matplotlib.axes
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.dates import date2num, num2date
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.optimize import Bounds, curve_fit
from sklearn.linear_model import LinearRegression
from typing_extensions import Literal, TypedDict

import src.utils.plots


class TrendlineStyling(TypedDict):
    linewidth: float
    alpha: float
    linestyle: str


class TrendlineParams(TypedDict):
    fit_type: Literal["auto", "default", "exponential", "linear"]
    after_date: str
    color: str
    line_start_date: str | None
    line_end_date: str | None
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
    title: str
    subtitle: str
    weighting: str
    include_task_distribution: str
    weight_key: str | None
    trendlines: list[TrendlineParams]
    exclude_agents: list[str]


def _get_title(script_params: ScriptParams) -> str:
    # Get included task groups
    task_group_names = ["General Autonomy", "SWAA", "RE-Bench"]
    included_task_groups = []
    for name in task_group_names:
        if name not in script_params["exclude"]:
            included_task_groups.append(name)

    # Make title
    task_groups_string = "+ ".join(included_task_groups)
    title = f"Time Horizon for {task_groups_string} Tasks"
    return title


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


def plot_task_distribution(
    ax: matplotlib.axes.Axes,
    runs_df: pd.DataFrame,
    plot_params: src.utils.plots.PlotParams,
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
    plot_params: src.utils.plots.PlotParams,
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
    trendlines: list[TrendlineParams] | None = None,
    upper_y_lim: float | None = None,
    include_task_distribution: str = "none",
    fig: Figure | None = None,
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

    if fig is None:
        fig, ax, ax_hist = setup_fig(include_task_distribution != "none")
    else:
        ax = fig.axes[0]
        ax_hist = None

    ax.set_ylim(lower_y_lim, upper_y_lim)

    agent_summaries["50%_clipped"] = agent_summaries["50%"].clip(
        # np.finfo(float).eps, np.inf
        lower_y_lim * 1.5,
        np.inf,
    )  # clip because log scale makes 0 -> -inf

    agent_summaries.loc[:, "50_low"].clip(1 / 60, np.inf)
    agent_summaries.loc[:, "50_high"].clip(1 / 60, np.inf)
    y = agent_summaries["50%"]
    yerr = np.array(
        [y - agent_summaries.loc[:, "50_low"], agent_summaries.loc[:, "50_high"] - y]
    )
    yerr = np.clip(yerr, 0, np.inf)
    legend_labels = []
    legend_handles = []

    for i, agent in enumerate(agent_summaries["agent"]):
        ax.errorbar(
            agent_summaries["release_date"].iloc[i],
            agent_summaries["50%_clipped"].iloc[i],
            yerr=[[yerr[0, i]], [yerr[1, i]]],
            **plot_style["error_bar"],
        )
        ax.grid(**plot_style["grid"])
        scatter_handle = ax.scatter(
            agent_summaries["release_date"].iloc[i],
            agent_summaries["50%_clipped"].iloc[i],
            color=agent_style[agent]["lab_color"],
            marker=agent_style[agent]["marker"],
            label=agent,
            **plot_style["scatter"],
        )

        # If it is actually in the bounds of the plot, add to legend
        if (
            agent_summaries["50%_clipped"].iloc[i] > lower_y_lim
            and agent_summaries["50%_clipped"].iloc[i] < upper_y_lim
        ):
            legend_labels.append(agent)
            legend_handles.append(scatter_handle)

    # Add arrows for out-of-range points
    mask_out_range = agent_summaries["50%_clipped"] != y
    logging.info(f"masking out {mask_out_range.sum()} points")
    ax.scatter(
        agent_summaries.loc[mask_out_range, "release_date"],
        [lower_y_lim * 1.2] * mask_out_range.sum(),  # Place at bottom of visible range
        marker="v",  # type: ignore
        color="black",
    )

    annotations = []
    # color_cycle = ["blue", "red", "green"]
    # if trendlines:
    #     if pd.Timestamp(after_date) < pd.Timestamp("2023-01-01"):
    #         annotations.append(
    #             plot_trendline(
    #                 ax,
    #                 agent_summaries,
    #                 plot_params,
    #                 fit_type="auto",
    #                 after="2024-01-01",
    #                 fit_color=color_cycle[1],
    #                 log_scale=True,
    #                 line_start_date="2023-07-01",
    #                 line_end_date="2025-04-01",
    #             )
    #         )
    #     annotations.append(
    #         plot_trendline(
    #             ax,
    #             agent_summaries,
    #             plot_params,
    #             fit_type="auto",
    #             after=after_date,
    #             fit_color=color_cycle[0],
    #             log_scale=True,
    #             line_start_date="2023-07-01",
    #             line_end_date="2025-04-01",
    #         )
    #     )
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
            data = _process_agent_summaries(exclude_agents, data, release_dates)
        else:
            data = agent_summaries

        reg, score = fit_trendline(data, trendline["after_date"], log_scale)
        annotations.append(
            plot_trendline(
                ax,
                data,
                plot_params,
                fit_type=trendline["fit_type"],
                after=trendline["after_date"],
                reg=reg,
                score=score,
                log_scale=log_scale,
                fit_color=trendline["color"],
                line_start_date=trendline["line_start_date"],
                line_end_date=trendline["line_end_date"],
                display_r_squared=trendline["display_r_squared"],
                caption=trendline["caption"],
                styling=trendline["styling"],
                skip_annotation=trendline["skip_annotation"],
            )
        )

        # if pd.Timestamp(after_date) < pd.Timestamp("2023-01-01"):
        #     annotations.append(
        #         plot_trendline(
        #             ax,
        #             agent_summaries,
        #             plot_params,
        #             fit_type="auto",
        #             after=after_date,
        #             log_scale=False,
        #         )
        #     )
        # annotations.append(
        #     plot_trendline_hyperbolic(ax, agent_summaries, after=after_date)
        # )

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

    src.utils.plots.log_y_axis(ax)
    ax.set_xlim(
        float(mdates.date2num(pd.Timestamp(x_lim_start))),
        float(mdates.date2num(pd.Timestamp(x_lim_end))),
    )
    start_year = pd.Timestamp(num2date(ax.get_xlim()[0])).year + 1
    end_year = pd.Timestamp(num2date(ax.get_xlim()[1])).year + 1

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

    ax.set_xticks(major_ticks)
    ax.set_xticklabels([x.strftime("%Y-%m") for x in major_ticks])
    ax.set_xticks(minor_ticks, minor=True)

    ax.set_xlabel(
        "Release Date",
        fontsize=plot_params["ax_label_fontsize"],
        labelpad=plot_params["xlabelpad"],
    )
    ax.set_ylabel(
        "Human time-to-complete @\n50% chance of model success",
        fontsize=plot_params["ax_label_fontsize"],
        labelpad=plot_params["ylabelpad"],
    )
    ax.set_title(
        f"{title}\n{subtitle}",
        fontsize=plot_params["title_fontsize"],
        pad=plot_params["xlabelpad"],
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

    ax.legend(
        legend_handles,
        legend_labels,
        loc="best",
        fontsize=10,
    )

    # Lay the annotations we collected earlier, ensuring they don't overlap
    padding = 10
    line_height = 12
    bbox = ax.get_window_extent()
    y = (
        (bbox.y1 - bbox.y0) * 72 / fig.dpi
    )  # start at top-left corner of the plot, in axes points
    y = 0
    x = (bbox.x1 - bbox.x0) * 72 / fig.dpi
    annotations = [a for a in annotations if a is not None]
    for a in annotations:
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
    agent_summaries: pd.DataFrame,
    after: str,
    log_scale: bool = False,
    method: str = "OLS",
) -> tuple[LinearRegression, float]:
    """Plot a trendline showing the relationship between release date and time horizon."""
    agent_summaries = agent_summaries.sort_values("release_date")
    mask = agent_summaries["release_date"] >= pd.Timestamp(after).date()

    dates = pd.to_datetime(agent_summaries["release_date"])
    X = np.array([date2num(d) for d in dates]).reshape(-1, 1)
    y_raw = agent_summaries["50%"].clip(1e-3, np.inf)
    y = np.log(y_raw) if log_scale else y_raw

    X = X[mask]
    y = y[mask]
    if method == "WLS":
        spread = np.log(agent_summaries["50_high"]) - np.log(agent_summaries["50_low"])
        weights = 1 / (spread**2 + 1e-6)
        # replace nan with 0
        weights = np.nan_to_num(weights)[mask]
    else:
        weights = None
    reg = LinearRegression().fit(X, y, sample_weight=weights)

    score = float(reg.score(X, y))

    return reg, score


def plot_trendline(
    ax: Axes,
    agent_summaries: pd.DataFrame,
    plot_params: src.utils.plots.PlotParams,
    after: str,
    reg: LinearRegression,
    score: float,
    log_scale: bool = False,
    method: str = "OLS",
    annotate: bool = True,
    fit_color: str | None = None,
    fit_type: Literal["auto", "default", "exponential", "linear"] = "default",
    plot_kwargs: dict[str, Any] = {},
    line_start_date: str | None = None,
    line_end_date: str | None = None,
    display_r_squared: bool = True,
    caption: str | None = None,
    styling: TrendlineStyling | None = None,
    skip_annotation: bool = False,
) -> dict[str, Any] | float | None:
    """Plot a trendline showing the relationship between release date and time horizon."""
    trendline_styling = plot_params["performance_over_time_trendline_styling"]

    # trendline goes to the end of the x-axis
    start_date = (
        pd.Timestamp(after)
        if line_start_date is None
        else pd.Timestamp(line_start_date)
    )
    end_date = (
        pd.Timestamp(after) if line_end_date is None else pd.Timestamp(line_end_date)
    )
    x_range = np.linspace(date2num(start_date), date2num(end_date), 20)
    y_pred = reg.predict(x_range.reshape(-1, 1))

    x_dates = [num2date(x) for x in x_range]
    y_values = np.exp(y_pred) if log_scale else y_pred  # Convert back from log scale

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
    ax.plot(x_dates, y_values, **pk)

    doubling_time = 1 / reg.coef_[0] * np.log(2) if log_scale else None
    if not annotate:
        return doubling_time

    if caption is None:
        caption = f"{fit_type.title()} fit"

    if skip_annotation:
        return None
    annotation = caption
    if display_r_squared:
        annotation += f"\nr^2: {score:.2f}"
    if log_scale and doubling_time is not None:
        annotation += f"\n(Doubling time: {doubling_time:.0f} days)"
        annotation += "\n" + ("All data" if after == "0000-00-00" else f"{after}+ data")
    else:
        annotation += f"\n(Rate: +{reg.coef_[0]*365:.0f} minutes per year)"

    annotation_styling = trendline_styling[fit_type]["annotation"]
    annotation_styling["color"] = fit_color
    return dict(
        text=annotation,
        transform=ax.get_xaxis_transform(),
        alpha=1,
        **annotation_styling,  # type: ignore
    )


def plot_trendline_hyperbolic(
    ax: matplotlib.axes.Axes, agent_summaries: pd.DataFrame, after: str
) -> dict[str, Any]:
    """Plot a trendline showing the relationship between release date and time horizon."""

    def hyperbolic_func(
        x: NDArray[Any], a: NDArray[Any], t_agi: NDArray[Any]
    ) -> NDArray[Any]:
        """
        Tries to fit the y value in log space
        """
        return np.log(a / (t_agi - x))

    log_scale = True
    agent_summaries = agent_summaries.sort_values("release_date")
    mask = agent_summaries["release_date"] >= pd.Timestamp(after).date()
    # Convert release dates to numeric (days since first release)
    dates = pd.to_datetime(agent_summaries["release_date"])
    first_release = min(dates)
    days_since_release = (dates - first_release).dt.days

    # Fit linear regression
    X = (cast(NDArray[Any], days_since_release.values)).reshape(-1, 1)
    y_raw = agent_summaries["50%"].clip(np.finfo(float).eps.astype(float), np.inf)
    y = np.log(y_raw) if log_scale else y_raw
    X = X[mask]
    y = y[mask]

    bounds = Bounds([0, max(X)[0]], [np.inf, np.inf])  # type: ignore[reportArgumentType]
    params, _ = curve_fit(
        hyperbolic_func, X.flatten(), y, bounds=bounds, p0=[5, 365 * 5.0]
    )

    # get r^2
    residuals = y - hyperbolic_func(X.flatten(), *params)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)  # type: ignore[reportOperatorIssue]
    r_squared = 1 - (ss_res / ss_tot)

    # Generate points for trendline, 20 equally spaced
    # x_range = np.array([min(X)[0], max(X)[0]])
    x_range = np.linspace(min(X)[0], max(X)[0], 20)
    y_pred = hyperbolic_func(x_range, params[0], params[1])

    # Convert back to datetime for plotting
    x_dates = [first_release + pd.Timedelta(days=x) for x in x_range]
    y_values = np.exp(y_pred) if log_scale else y_pred  # Convert back from log scale

    # print(f"x_dates: {x_dates}")
    # print(f"y_values: {y_values}")

    fit_color = "green"

    # Plot trendline
    ax.plot(x_dates, y_values, "--", color=fit_color, alpha=0.5)  # type: ignore[reportArgumentType]
    fit_type = "Hyperbolic"
    annotation = f"{fit_type} fit\nr^2: {r_squared:.2f}"
    t_agi = params[1]
    # convert to date
    t_agi = first_release + pd.Timedelta(days=t_agi)
    annotation += f"\n(T_agi: {t_agi:%Y-%m-%d})"

    return dict(
        text=annotation,
        color=fit_color,
        transform=ax.get_xaxis_transform(),
    )


def _process_agent_summaries(
    exclude_agents: list[str] | None, agent_summaries: pd.DataFrame, release_dates: Any
) -> pd.DataFrame:
    agent_summaries["release_date"] = agent_summaries["agent"].map(
        release_dates["date"]
    )
    if exclude_agents is not None:
        agent_summaries = agent_summaries[
            ~agent_summaries["agent"].isin(exclude_agents)
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
    parser.add_argument("--params-file", type=pathlib.Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    params = yaml.safe_load(args.params_file.read_text())
    fig_params = params["figs"]["plot_logistic_regression"][args.script_parameter_group]

    agent_summaries = pd.read_csv(args.input_file)
    release_dates = yaml.safe_load(args.release_dates.read_text())
    agent_summaries = _process_agent_summaries(
        fig_params["exclude_agents"], agent_summaries, release_dates
    )

    logging.info("Loaded input data")

    possible_weightings = params["weighting"]
    for possible_weighting in possible_weightings:
        if possible_weighting["weight_col"] == fig_params["weighting"]:
            weighting = possible_weighting
            break
    else:
        raise ValueError(f"Weighting {fig_params['weighting']} not found")
    subtitle = weighting["graph_snippet"]

    runs_df = pd.read_json(args.runs_file, lines=True)
    runs_df = _remove_excluded_task_groups(runs_df, fig_params)

    title = _get_title(fig_params)

    plot_horizon_graph(
        params["plots"],
        agent_summaries,
        title=title,
        release_dates=release_dates,
        runs_df=runs_df,
        subtitle=subtitle,
        lower_y_lim=fig_params["lower_y_lim"],
        upper_y_lim=fig_params["upper_y_lim"],
        x_lim_start=fig_params["x_lim_start"],
        x_lim_end=fig_params["x_lim_end"],
        include_task_distribution=fig_params["include_task_distribution"],
        weight_key=weighting["weight_col"],
        trendlines=fig_params["trendlines"],
        exclude_agents=fig_params["exclude_agents"],
    )

    src.utils.plots.save_or_open_plot(args.output_file, params["plot_format"])


if __name__ == "__main__":
    main()
    main()
