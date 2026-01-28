from __future__ import annotations

import argparse
import logging
import pathlib
from typing import Any

import dvc.api
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yaml
from matplotlib import markers
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.dates import date2num
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from typing_extensions import Literal

from horizon.plot.logistic import (
    FitFunctionWrapper,
    plot_trendline,
)
from horizon.utils import plots


def _get_title(script_params: plots.ScriptParams, success_percent: int) -> str:
    if "title" in script_params:
        return script_params["title"]
    task_group_names = ["HCAST", "SWAA", "RE-Bench"]
    included_task_groups = []
    for name in task_group_names:
        if name not in script_params["exclude"]:
            included_task_groups.append(name)

    task_groups_string = " + ".join(included_task_groups)
    title = f"{success_percent}% Time Horizon for {task_groups_string} Tasks"
    return title


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
    trendlines: list[plots.TrendlineParams] | None = None,
    upper_y_lim: float | None = None,
    legend: bool = True,
    include_task_distribution: str = "none",
    fig: Figure | None = None,
    y_scale: Literal["log", "linear"] = "log",
    confidence_level: float = 0.95,
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

    fig, ax, ax_hist = setup_fig(False)

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
        ax.errorbar(
            agent_summaries["release_date"].iloc[i],
            y_clipped.iloc[i],
            yerr=[[yerr[0, i]], [yerr[1, i]]],
            **plot_style["error_bar"],
        )
        ax.grid(**plot_style["grid"])
        scatter_handle = ax.scatter(
            agent_summaries["release_date"].iloc[i],
            y_clipped.iloc[i],
            color=agent_style[agent]["lab_color"],
            marker=agent_style[agent]["marker"],
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
        assert (
            "data_file" not in trendline
        ), "data_file is not supported for alternative fits"
        data = agent_summaries
        print(f"fitting on {len(data)} models")
        y, x = data[f"p{success_percent}"], pd.to_datetime(data["release_date"])
        if trendline["fit_type"] == "linear":
            reg, score = fit_trendline(y, x, log_scale=False)
        elif trendline["fit_type"] == "exponential":
            reg, score = fit_trendline(y, x, log_scale=True)
        elif trendline["fit_type"] == "hyperbolic":
            reg, score = fit_trendline_hyperbolic(y, x)
        else:
            raise ValueError(f"Invalid fit type: {trendline['fit_type']}")

        dashed_outside = (data["release_date"].min(), data["release_date"].max())
        annotations.append(
            plot_trendline(
                ax,
                plot_params,
                trendline_params=trendline,
                dashed_outside=dashed_outside,
                reg=reg,
                score=score,
                log_scale=trendline["fit_type"] != "linear",
            )
        )

    plots.make_y_axis(ax, scale=y_scale)
    start_year = pd.Timestamp(x_lim_start).year
    end_year = pd.Timestamp(x_lim_end).year + 1
    plots.make_quarterly_xticks(ax, start_year, end_year)

    ax.set_xlim(
        float(mdates.date2num(pd.Timestamp(x_lim_start))),
        float(mdates.date2num(pd.Timestamp(x_lim_end))),
    )

    ax.set_xlabel(
        "Date model was released",
        fontsize=plot_params["ax_label_fontsize"],
        labelpad=plot_params["xlabelpad"],
    )
    ax.set_ylabel(
        f"Task time (for humans) that model completes with \n{success_percent}% success rate",
        fontsize=plot_params["ax_label_fontsize"],
        labelpad=plot_params["ylabelpad"],
    )
    ax.set_title(
        title,
        fontsize=plot_params["title_fontsize"],
        pad=3 * plot_params["xlabelpad"],
    )
    plt.suptitle(subtitle, y=0.93, x=0.51, fontsize=plot_params["suptitle_fontsize"])

    available_agents = [
        agent for agent in plot_params["legend_order"] if agent in legend_labels
    ]

    sorted_pairs = sorted(
        zip(legend_handles, legend_labels),
        key=lambda pair: available_agents.index(pair[1]),
    )
    legend_handles, legend_labels = zip(*sorted_pairs)

    if legend:
        print(f"legend_handles: {legend_handles}")
        ax.legend(
            legend_handles,
            legend_labels,
            loc="best",
            fontsize=10,
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
    X = np.array([date2num(d) for d in release_dates]).reshape(-1, 1)

    y_raw = agent_horizons.clip(1e-3, np.inf)
    y = np.log(y_raw) if log_scale else y_raw

    reg = LinearRegression().fit(X, y)

    score = float(reg.score(X, y))

    return reg, score


def r2_score(y_pred: NDArray[Any], y_true: NDArray[Any]) -> float:
    residuals = y_true - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def fit_trendline_hyperbolic(
    agent_horizons: pd.Series[float],
    release_dates: pd.Series[pd.Timestamp],
) -> tuple[FitFunctionWrapper, float]:
    """Fit a hyperbolic trendline showing the relationship between release date and time horizon.

    Args:
        agent_horizons: Series containing the time horizons for each agent
        release_dates: Series containing the release dates for each agent

    Returns:
        A tuple containing the fitted FitFunctionWrapper model and the R^2 score
    """

    def hyperbolic_func(
        x: NDArray[Any], a: NDArray[Any], t_agi: NDArray[Any]
    ) -> NDArray[Any]:
        """Hyperbolic function for fitting in log space"""
        return np.log(a / (t_agi - x))

    X = np.array([date2num(d) for d in release_dates]).reshape(-1, 1)
    y = np.log(agent_horizons.clip(lower=np.finfo(float).eps))

    bounds = ([0, X.max()], [np.inf, np.inf])
    params, _ = curve_fit(
        hyperbolic_func, X.flatten(), y, bounds=bounds, p0=[5, max(X.flatten()) + 10]
    )

    model = FitFunctionWrapper(hyperbolic_func, list(params))

    y_pred = model.predict(X.flatten())
    print(f"y_pred: {y_pred}")
    print(f"y: {y}")
    r_squared = r2_score(y_pred, y)
    return model, float(r_squared)


def _process_agent_summaries(
    exclude_agents: list[str] | None,
    agent_summaries: pd.DataFrame,
    release_dates: Any,
    after_date: str | None = None,
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
    params = dvc.api.params_show(stages="plot_horizon_alternative_fits", deps=True)
    fig_params = params["figs"]["plot_horizon_alternative_fits"][
        args.script_parameter_group
    ]

    agent_summaries = pd.read_csv(args.input_file)
    release_dates = yaml.safe_load(args.release_dates.read_text())
    agent_summaries = _process_agent_summaries(
        fig_params["exclude_agents"], agent_summaries, release_dates
    )

    logging.info("Loaded input data")

    runs_df = pd.read_json(args.runs_file, lines=True)

    title = _get_title(fig_params, fig_params.get("success_percent", 50))
    subtitle = fig_params.get("subtitle", "")

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
        weight_key=fig_params["weighting"],
        trendlines=fig_params["trendlines"],
        exclude_agents=fig_params["exclude_agents"],
        success_percent=fig_params.get("success_percent", 50),
        legend=False,
    )

    plots.save_or_open_plot(args.output_file, params["plot_format"])


if __name__ == "__main__":
    main()
