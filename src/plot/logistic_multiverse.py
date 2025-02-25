import argparse
import itertools
import logging
import pathlib
from dataclasses import dataclass
from typing import Any

import dvc.api
import matplotlib.axes
import matplotlib.figure
import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm

import src.utils.plots
from src.plot.logistic import fit_trendline, plot_horizon_graph, plot_trendline
from src.wrangle.logistic import run_logistic_regressions


@dataclass
class MultiverseRecord:
    weighting: str
    regularization: float
    after_date: str
    method: str
    doubling_time: float


def record_metrics(records: list[dict[str, Any]], metrics_file: pathlib.Path) -> None:
    df = pd.DataFrame(records)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "overall": {
            "10%": float(df["doubling_time"].quantile(0.1)),
            "50%": float(df["doubling_time"].quantile(0.5)),
            "90%": float(df["doubling_time"].quantile(0.9)),
        },
        "marginal_medians": {
            "weighting": {
                weight: float(df[df["weighting"] == weight]["doubling_time"].median())
                for weight in df["weighting"].unique()
            },
            "regularization": {
                str(reg): float(
                    df[df["regularization"] == reg]["doubling_time"].median()
                )
                for reg in df["regularization"].unique()
            },
            "after_date": {
                date: float(df[df["after_date"] == date]["doubling_time"].median())
                for date in df["after_date"].unique()
            },
            "method": {
                method: float(df[df["method"] == method]["doubling_time"].median())
                for method in df["method"].unique()
            },
        },
    }
    metrics_file.write_text(yaml.dump(out))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-file", type=str, required=True)
    parser.add_argument("--release-dates", type=pathlib.Path, required=True)
    parser.add_argument("--temp-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-file", type=pathlib.Path, required=True)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--output-metrics-file", type=pathlib.Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    params = dvc.api.params_show(stages="plot_logistic_multiverse")
    fig_params = params["figs"]["plot_logistic_multiverse"]

    weightings = fig_params["weightings"]
    regularizations = fig_params["regularizations"]
    fig, axs = plt.subplots(1, 2, width_ratios=[6, 1], figsize=(10, 7))
    ax0: matplotlib.axes.Axes = axs[0]
    ax1: matplotlib.axes.Axes = axs[1]
    records = []

    runs = pd.read_json(args.runs_file, lines=True)
    runs.rename(columns={"alias": "agent"}, inplace=True)
    runs = runs[runs["agent"] != "human"]

    logging.info("Loaded input data")

    first_line = True
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Weightings ({len(weightings)}): {weightings}")
    print(f"Regularizations ({len(regularizations)}): {regularizations}")
    for weighting, regularization in tqdm(
        itertools.product(weightings, regularizations)
    ):
        agent_summaries = run_logistic_regressions(
            runs,
            args.release_dates,
            weighting,
            float(regularization),
            exclude_task_sources=None,
        )

        agent_summaries["release_date"] = pd.to_datetime(
            agent_summaries["release_date"]
        ).dt.date

        agent_summaries = agent_summaries[
            ~agent_summaries["agent"].isin(fig_params["exclude_agents"])
        ]
        if first_line:
            first_line = False
            plot_horizon_graph(
                plot_params=params["plots"],
                all_agent_summaries=agent_summaries,
                runs_df=runs,
                release_dates=args.release_dates,
                lower_y_lim=0.5 / 60,
                x_lim_start="2018-12-01",
                x_lim_end="2025-01-01",
                subtitle="",
                title="",
                weight_key=weighting,
                exclude_agents=fig_params["exclude_agents"],
                trendlines=None,
                upper_y_lim=8 * 60,
                include_task_distribution="none",
                fig=fig,
            )

        print(f"Plotting trendline for {weighting} {regularization}")
        reg, score = fit_trendline(
            agent_summaries=agent_summaries,
            after="2019-01-01",
            log_scale=True,
            method="OLS",
        )
        doubling_time = np.log(2) / reg.coef_[0]

        plot_trendline(
            ax=ax0,
            agent_summaries=agent_summaries,
            plot_params=params["plots"],
            after="2019-01-01",
            reg=reg,
            score=score,
            line_end_date="2025-01-01",
            log_scale=True,
            annotate=False,
            fit_type="exponential",
            fit_color="blue",
            method="OLS",
        )

        records.append(
            MultiverseRecord(
                weighting=weighting,
                regularization=regularization,
                after_date="2019-01-01",
                method="OLS",
                doubling_time=doubling_time,
            )
        )

    print(f"Records: {records}")

    doubling_times = [record.doubling_time for record in records]
    print(f"Doubling times: {doubling_times}")
    ax1.boxplot(doubling_times, vert=True)
    ax1.set_xticklabels(["Doubling times\n(days)"])
    ax1.set_ylim(0, 365)

    fig.suptitle(
        f"Multiverse analysis of time horizon, {len(records)} fits\nParameters: weighting, regularization"
    )

    # Annotate top left with range
    ax0.annotate(
        f"Range: {min(doubling_times):.0f} - {max(doubling_times):.0f} days",
        (0.05, 0.9),
        xycoords="axes fraction",
        fontsize=params["plots"]["performance_over_time_trendline_styling"]["default"][
            "annotation"
        ]["fontsize"],
    )

    record_metrics(records, args.output_metrics_file)
    logging.info(f"Saved metrics to {args.output_metrics_file}")

    src.utils.plots.save_or_open_plot(args.output_file, params["plot_format"])


if __name__ == "__main__":
    main()
