import argparse
import logging
import pathlib
from typing import Any, cast

import dvc.api
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def get_time_bucket(human_minutes: float) -> str | None:
    if 1 <= human_minutes < 4:
        return "1-4 min"
    elif 4 <= human_minutes < 15:
        return "4-15 min"
    elif 15 <= human_minutes < 60:
        return "15-60 min"
    elif 60 <= human_minutes < 240:
        return "1-4 hr"
    elif human_minutes >= 240:
        return "4hr+"
    else:
        return None


def _format_token_label(token_value: float) -> str:
    if token_value >= 1000000:
        return f"{int(token_value / 1000000)}M"
    else:
        return f"{int(token_value / 1000)}k"


def format_duration(minutes: float) -> str:
    if minutes < 60:
        return f"{int(minutes)}min"
    else:
        hours = int(minutes // 60)
        remaining_minutes = int(minutes % 60)
        return f"{hours}hr{remaining_minutes}min"


def _title_with_warning(title: str, warning: str | None) -> str:
    return f"{title}\n{warning}" if warning else title


def get_token_buckets(
    token_range: tuple[float, float], reference_lines: list[float]
) -> list[tuple[float, float, str]]:
    boundaries = [token_range[0]] + sorted(reference_lines) + [token_range[1]]
    buckets = []
    for i in range(len(boundaries) - 1):
        low = boundaries[i]
        high = boundaries[i + 1]
        is_last = i == len(boundaries) - 2

        if is_last:
            label = f">{_format_token_label(low)}"
        else:
            label = f"{_format_token_label(low)}-{_format_token_label(high)}"

        buckets.append((low, high, label))
    return buckets


def add_token_bucket(
    df: pd.DataFrame, token_buckets: list[tuple[float, float, str]]
) -> pd.DataFrame:
    df = df.copy()
    bins = [low for low, _, _ in token_buckets]
    labels = [label for _, _, label in token_buckets]

    last_bucket_high = token_buckets[-1][1]
    max_tokens = df["tokens_count"].max() if len(df) > 0 else last_bucket_high
    bins.append(max(max_tokens * 2, last_bucket_high * 2, 1e10))

    df["token_bucket"] = pd.cut(
        df["tokens_count"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )
    return df


def load_and_prepare_data(input_file: pathlib.Path) -> pd.DataFrame:
    df = pd.read_json(input_file, lines=True, orient="records", convert_dates=False)
    df = df[df["task_source"] != "SWAA"]
    df = df[df["tokens_count"].notna()]
    if len(df) == 0:
        raise ValueError("No runs with tokens_count found")

    df["time_bucket"] = df["human_minutes"].apply(get_time_bucket)
    df = df[df["time_bucket"].notna()]
    df = df.dropna(subset=["alias", "model"])

    if df.empty:
        raise ValueError("No runs remaining after filtering for aliases and models")

    return df


def _draw_reference_lines(
    ax: Axes,
    reference_lines: list[float],
    token_range: tuple[float, float],
    y_axis_max: float,
) -> None:
    for ref_line in reference_lines:
        if token_range[0] <= ref_line <= token_range[1]:
            ax.axvline(
                ref_line,
                color="black",
                linestyle="dashed",
                linewidth=0.8,
                alpha=0.5,
            )
            label = _format_token_label(ref_line)
            ax.text(
                ref_line,
                y_axis_max * 0.95,
                label,
                rotation=90,
                ha="right",
                va="top",
                fontsize=8,
                alpha=0.7,
            )


def _plot_token_histogram_subplot(
    data_df: pd.DataFrame,
    ax: Axes,
    token_bin_edges: NDArray[np.float64],
    token_range: tuple[float, float],
    y_axis_max: float,
    reference_lines: list[float],
    width_factor: float,
    title: str,
) -> None:
    successful = data_df[data_df["score_binarized"] == 1]["tokens_count"]
    unsuccessful = data_df[data_df["score_binarized"] == 0]["tokens_count"]

    counts_success, _ = np.histogram(successful, bins=token_bin_edges)
    counts_failure, _ = np.histogram(unsuccessful, bins=token_bin_edges)

    bin_centers = (token_bin_edges[:-1] + token_bin_edges[1:]) / 2
    width = (token_bin_edges[1:] - token_bin_edges[:-1]) * width_factor

    ax.bar(
        bin_centers,
        counts_success,
        width=width,
        color="green",
        label="Successful",
        align="center",
    )
    ax.bar(
        bin_centers,
        counts_failure,
        width=width,
        bottom=counts_success,
        color="red",
        label="Unsuccessful",
        align="center",
    )

    _draw_reference_lines(ax, reference_lines, token_range, y_axis_max)

    ax.set_xscale("log")
    ax.set_xlim(token_range)
    ax.set_ylim(0, y_axis_max)
    ax.set_xlabel("Tokens per run")
    ax.set_ylabel("Run count")
    ax.set_title(title)


def _add_figure_legend(fig: Figure, axes: Any) -> None:
    axes_flat = (
        axes.flatten()
        if isinstance(axes, np.ndarray) and axes.ndim > 1
        else (axes if isinstance(axes, np.ndarray) else [axes])
    )
    for ax in axes_flat:
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            fig.legend(handles, labels, loc="lower right", ncol=2)
            break


def _save_fig(fig: Figure, output_file: pathlib.Path) -> None:
    fig.savefig(output_file, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved plot to {output_file}")


def load_release_dates(release_dates_file: pathlib.Path) -> dict[str, str]:
    with open(release_dates_file, "r") as f:
        return yaml.safe_load(f)["date"]


def get_recent_models(
    df: pd.DataFrame, release_dates: dict[str, str]
) -> list[tuple[str, str]]:
    alias_to_model = df.groupby("alias")["model"].first().dropna().to_dict()

    aliases_with_dates = [
        (alias, release_dates[alias], alias_to_model[alias])
        for alias in alias_to_model.keys()
        if alias in release_dates
    ]
    aliases_with_dates.sort(key=lambda x: x[1], reverse=True)
    return [(alias, model) for alias, _, model in aliases_with_dates[:6]]


def plot_recent_models(
    df: pd.DataFrame,
    models: list[tuple[str, str]],
    token_range: tuple[float, float],
    token_bin_edges: NDArray[np.float64],
    y_axis_max: float,
    reference_lines: list[float],
    width_factor: float,
    aggregate_warning: str | None = None,
) -> Figure:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes_flat = axes.flatten()

    for idx, (alias, model) in enumerate(models):
        ax = axes_flat[idx]
        model_df = df[df["model"] == model]
        _plot_token_histogram_subplot(
            model_df,
            ax,
            token_bin_edges,
            token_range,
            y_axis_max,
            reference_lines,
            width_factor,
            alias,
        )

    _add_figure_legend(fig, axes_flat)
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    fig.suptitle(
        _title_with_warning(
            "Token histograms: 6 most recent models", aggregate_warning
        ),
        y=0.98,
    )
    return fig


def plot_single_model(
    model_df: pd.DataFrame,
    alias: str,
    time_buckets: list[str],
    token_range: tuple[float, float],
    token_bin_edges: NDArray[np.float64],
    y_axis_max: float,
    reference_lines: list[float],
    width_factor: float,
    aggregate_warning: str | None = None,
) -> Figure:
    if len(time_buckets) > 6:
        raise ValueError("time_buckets must have at most 6 elements for 2x3 grid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for idx, bucket in enumerate(time_buckets):
        ax = axes[idx // 3, idx % 3]
        bucket_df = model_df[model_df["time_bucket"] == bucket]

        if len(bucket_df) == 0:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(f"Human task length: {bucket}")
            continue

        _plot_token_histogram_subplot(
            bucket_df,
            ax,
            token_bin_edges,
            token_range,
            y_axis_max,
            reference_lines,
            width_factor,
            f"Human task length: {bucket}",
        )

    for idx in range(len(time_buckets), 6):
        axes[idx // 3, idx % 3].axis("off")

    _add_figure_legend(fig, axes)

    plt.tight_layout(rect=(0, 0, 1, 0.92))

    fig.suptitle(
        _title_with_warning(
            f"{alias}: Token histograms by human time bucket", aggregate_warning
        ),
        y=0.98,
    )

    return fig


def plot_task_heatmap(
    df: pd.DataFrame,
    token_buckets: list[tuple[float, float, str]],
    target_alias: str | None = None,
) -> Figure:
    if target_alias:
        df = df[df["alias"] == target_alias]

    df = add_token_bucket(df, token_buckets)
    df = df[df["token_bucket"].notna()]

    heatmap_df = (
        df.groupby(["task_id", "token_bucket"])["score_binarized"]
        .mean()
        .unstack(fill_value=None)
    )
    heatmap_df = heatmap_df.dropna(how="all")

    bucket_labels = [label for _, _, label in token_buckets]
    heatmap_df = heatmap_df[[col for col in bucket_labels if col in heatmap_df.columns]]

    task_lengths = df.groupby("task_id")["human_minutes"].first()
    task_lengths = task_lengths[task_lengths.index.isin(heatmap_df.index)].sort_values()
    heatmap_df = heatmap_df.reindex(task_lengths.index)

    y_labels = [
        f"{task_id} ({format_duration(float(task_lengths.loc[task_id]))})"
        for task_id in heatmap_df.index
    ]

    fig, ax = plt.subplots(
        figsize=(max(12, len(token_buckets) * 1.5), max(8, len(heatmap_df) * 0.3))
    )
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        cbar=False,
        ax=ax,
        yticklabels=y_labels,
    )
    ax.set_xlabel("Token bucket")
    ax.set_ylabel("Task ID")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.suptitle(f"Task-level success rates for {target_alias}")
    return fig


def plot_time_bucket_heatmap(
    df: pd.DataFrame,
    token_buckets: list[tuple[float, float, str]],
    time_buckets: list[str],
    target_alias: str | None = None,
    aggregate_warning: str | None = None,
) -> Figure:
    if target_alias:
        df = df[df["alias"] == target_alias]

    df = add_token_bucket(df, token_buckets)
    df = df[df["token_bucket"].notna()]

    heatmap_data = []
    run_counts_data = []
    bucket_labels = [label for _, _, label in token_buckets]
    for time_bucket in time_buckets:
        time_df = df[df["time_bucket"] == time_bucket]
        row: dict[str, str | float] = {"time_bucket": time_bucket}
        run_counts_row: dict[str, str | float] = {"time_bucket": time_bucket}
        for bucket_label in bucket_labels:
            bucket_df = time_df[time_df["token_bucket"] == bucket_label]
            if len(bucket_df) > 0:
                row[bucket_label] = bucket_df["score_binarized"].mean()
                run_counts_row[bucket_label] = float(len(bucket_df))
        heatmap_data.append(row)
        run_counts_data.append(run_counts_row)

    heatmap_df = pd.DataFrame(heatmap_data).set_index("time_bucket")
    run_counts_df = (
        pd.DataFrame(run_counts_data)
        .set_index("time_bucket")
        .apply(pd.to_numeric, errors="coerce")
    )
    heatmap_df = heatmap_df.dropna(how="all")
    heatmap_df = heatmap_df[[col for col in bucket_labels if col in heatmap_df.columns]]

    annotation_df = heatmap_df.copy().astype(str)
    for col in annotation_df.columns:
        for idx in annotation_df.index:
            if pd.notna(heatmap_df.loc[idx, col]):
                rate = cast(float, heatmap_df.loc[idx, col])
                if (
                    idx in run_counts_df.index
                    and col in run_counts_df.columns
                    and pd.notna(run_counts_df.loc[idx, col])
                ):
                    count_val = cast(float, run_counts_df.loc[idx, col])
                    count_int = int(count_val)
                else:
                    count_int = 0
                annotation_df.loc[idx, col] = f"{rate:.2f}\n({count_int} runs)"

    fig, ax = plt.subplots(
        figsize=(max(12, len(token_buckets) * 1.5), max(6, len(time_buckets) * 0.8))
    )
    sns.heatmap(
        heatmap_df,
        annot=annotation_df,
        fmt="",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Success Rate"},
        ax=ax,
    )
    ax.set_xlabel("Token bucket")
    ax.set_ylabel("Time bucket")
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    subject = target_alias or "all aliases"
    fig.suptitle(
        _title_with_warning(
            f"Time bucket success rates for {subject}", aggregate_warning
        ),
        y=0.98,
    )
    return fig


def calculate_metrics(
    df: pd.DataFrame,
    token_buckets: list[tuple[float, float, str]],
    time_buckets: list[str],
    target_alias: str | None = None,
) -> dict[str, Any]:
    if target_alias:
        df = df[df["alias"] == target_alias]

    df = add_token_bucket(df, token_buckets)
    if df.empty:
        raise ValueError("No runs remaining after filtering for token buckets")

    return {
        "average_tokens_all_tasks": float(round(df["tokens_count"].mean(), 3)),
        "average_tokens_by_time_bucket": {
            time_bucket: float(
                round(df[df["time_bucket"] == time_bucket]["tokens_count"].mean(), 3)
            )
            for time_bucket in time_buckets
            if len(df[df["time_bucket"] == time_bucket]) > 0
        },
        "average_success_rate_by_token_bucket": {
            bucket_label: float(
                round(
                    df[df["token_bucket"] == bucket_label]["score_binarized"].mean(), 3
                )
            )
            for _, _, bucket_label in token_buckets
            if len(df[df["token_bucket"] == bucket_label]) > 0
        },
    }


def plot_token_usage_distribution(
    input_file: pathlib.Path,
    output_dir: pathlib.Path,
    plot_format: str,
    params: dict[str, Any],
    release_dates_file: pathlib.Path,
) -> None:
    df = load_and_prepare_data(input_file)

    token_range_param = params.get("token_range", [1e5, 1e8])
    token_range = (float(token_range_param[0]), float(token_range_param[1]))
    y_axis_max = float(params.get("y_axis_max", 70))
    n_bins = int(params.get("n_bins", 50))
    reference_lines = params.get(
        "reference_lines", [125000, 250000, 500000, 1000000, 2000000, 5000000, 10000000]
    )
    width_factor = params.get("width_factor", 0.8)
    target_alias = params.get("alias")
    if target_alias is None:
        raise ValueError("alias parameter is required")
    time_buckets = params.get(
        "time_buckets", ["1-4 min", "4-15 min", "15-60 min", "1-4 hr", "4hr+"]
    )
    aggregate_warning = params.get(
        "aggregate_warning",
        "Warning: refer to task-level plots for per-task detail.",
    )

    token_bin_edges = np.logspace(
        np.log10(token_range[0]),
        np.log10(token_range[1]),
        n_bins + 1,
    )
    token_buckets = get_token_buckets(token_range, reference_lines)

    output_dir.mkdir(parents=True, exist_ok=True)

    release_dates = load_release_dates(release_dates_file)
    recent_models = get_recent_models(df, release_dates)
    logger.info(
        f"Generating recent models overview for: {[alias for alias, _ in recent_models]}"
    )
    _save_fig(
        plot_recent_models(
            df,
            recent_models,
            token_range,
            token_bin_edges,
            y_axis_max,
            reference_lines,
            width_factor,
            aggregate_warning=aggregate_warning,
        ),
        output_dir / f"token_histograms_recent.{plot_format}",
    )

    alias_df = df[df["alias"] == target_alias]
    logger.info(
        "Generating single model plot for %s (%d runs)",
        target_alias,
        len(alias_df),
    )
    _save_fig(
        plot_single_model(
            model_df=alias_df,
            alias=target_alias,
            time_buckets=time_buckets,
            token_range=token_range,
            token_bin_edges=token_bin_edges,
            y_axis_max=y_axis_max,
            reference_lines=reference_lines,
            width_factor=width_factor,
            aggregate_warning=aggregate_warning,
        ),
        output_dir / f"token_histograms.{plot_format}",
    )

    logger.info("Generating task-level heatmap")
    _save_fig(
        plot_task_heatmap(df, token_buckets, target_alias),
        output_dir / f"task_heatmap.{plot_format}",
    )

    logger.info("Generating time-bucket heatmap")
    _save_fig(
        plot_time_bucket_heatmap(
            df,
            token_buckets,
            time_buckets,
            target_alias,
            aggregate_warning=aggregate_warning,
        ),
        output_dir / f"time_bucket_heatmap.{plot_format}",
    )

    logger.info("Calculating metrics")
    metrics = calculate_metrics(df, token_buckets, time_buckets, target_alias)
    metrics_dir = output_dir.parent.parent / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = metrics_dir / "token_usage_metrics.yaml"
    with open(metrics_file, "w") as f:
        yaml.dump(metrics, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Saved metrics to {metrics_file}")


def main(
    input_file: pathlib.Path,
    output_dir: pathlib.Path,
    log_level: str,
    release_dates_file: pathlib.Path,
) -> None:
    logging.basicConfig(level=log_level.upper())

    params = dvc.api.params_show(stages="plot_token_usage_distribution", deps=True)
    plot_params_config = params["figs"]["plot_token_usage_distribution"]
    plot_format = params.get("plot_format", "png")

    plot_token_usage_distribution(
        input_file=input_file,
        output_dir=output_dir,
        plot_format=plot_format,
        params=plot_params_config,
        release_dates_file=release_dates_file,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--log-level", type=str, required=True)
    parser.add_argument("--release-dates-file", type=pathlib.Path, required=True)
    args = parser.parse_args()

    main(**vars(args))
