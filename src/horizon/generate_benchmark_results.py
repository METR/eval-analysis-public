"""
Generate benchmark result table in YAML format. This is used by Epoch
for their dashboard.

Example output:
benchmark_name: METR-Horizon-v1
version: 012346789abcdef

results:
  claude-3-7-sonnet-20250219:
    agents:
      agent-1:
        p50_horizon_length:
          estimate: float
          ci_low: float
          ci_high: float
        p80_horizon_length:
          estimate: float
          ci_low: float
          ci_high: float
        usage:
          usd: float
          tokens: float
          working_time: float
      agent-2: ...
    release_date: str
  ...


Notes:
- `working_time` is the time taken by the agent to complete one run on each task
- This does not support multiple scaffolds per alias
"""

import argparse
import logging
import pathlib
from collections import defaultdict
from typing import Any

import pandas as pd
import yaml

from horizon.compute_trendline_ci import get_sota_agents
from horizon.plot.bootstrap_ci import (
    DoublingTimeStats,
    compute_bootstrap_confidence_region,
)


def defaultdict_to_dict(d: defaultdict | dict) -> dict:  # type: ignore
    if isinstance(d, defaultdict) or isinstance(d, dict):
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d


def _get_trend_stats(
    agent_summaries: pd.DataFrame,
    bootstrap_results: pd.DataFrame,
    release_dates: dict[str, str],
    after_date: str,
    before_date: str,
) -> DoublingTimeStats:
    sota_agents = get_sota_agents(
        agent_summaries, release_dates, after_date, before_date
    )
    agent_summaries_for_fitting = agent_summaries[
        agent_summaries["agent"].isin(sota_agents)
    ]
    assert len(agent_summaries_for_fitting) == len(sota_agents)
    bootstrap_results_for_fitting = bootstrap_results[
        [f"{agent}_p50" for agent in sota_agents]
    ]
    stats, _, _, _ = compute_bootstrap_confidence_region(
        agent_summaries=agent_summaries_for_fitting,
        bootstrap_results=bootstrap_results_for_fitting,
        release_dates={"date": release_dates},  # type: ignore
        after_date=after_date,
        max_date=pd.to_datetime(before_date),
        confidence_level=0.95,
    )
    return stats


def _get_all_trend_stats(
    agent_summaries: pd.DataFrame,
    bootstrap_results: pd.DataFrame,
    release_dates: dict[str, str],
) -> dict[str, dict[str, float]]:
    all_time_stats = _get_trend_stats(
        agent_summaries,
        bootstrap_results,
        release_dates,
        "2019-01-01",
        "2030-01-01",
    )

    from_2023_on_stats = _get_trend_stats(
        agent_summaries,
        bootstrap_results,
        release_dates,
        "2023-01-01",
        "2030-01-01",
    )

    return {
        "all_time": {
            "point_estimate": round(all_time_stats.point_estimate, 3),
            "ci_low": round(all_time_stats.ci_lower, 3),
            "ci_high": round(all_time_stats.ci_upper, 3),
        },
        "from_2023_on": {
            "point_estimate": round(from_2023_on_stats.point_estimate, 3),
            "ci_low": round(from_2023_on_stats.ci_lower, 3),
            "ci_high": round(from_2023_on_stats.ci_upper, 3),
        },
    }


def generate_benchmark_metrics(
    df_runs: pd.DataFrame,
    df_summaries: pd.DataFrame,
    df_bootstrap_results: pd.DataFrame,
    release_dates: dict[str, str],
    benchmark_name: str,
    long_tasks_version: str,
    swaa_version: str,
    logger: logging.Logger,
    include_transcript_links: bool = False,
    benchmark_results_to_stitch: dict[str, Any] | None = None,
    model_names_to_stitch: list[str] | None = None,
) -> dict[str, Any]:
    results = dict()
    dated_results = defaultdict(set)

    df_runs["duration_minutes"] = (df_runs["completed_at"] - df_runs["started_at"]) / (
        60 * 1000
    )

    summary_column_map = {
        "p50_horizon_length": {
            "estimate": "p50",
            "ci_low": "p50q0.025",
            "ci_high": "p50q0.975",
        },
        "p80_horizon_length": {
            "estimate": "p80",
            "ci_low": "p80q0.025",
            "ci_high": "p80q0.975",
        },
    }

    agents = set(df_runs["alias"].unique()) - {"human"}

    for agent in agents:
        agent_df = df_runs[df_runs["alias"] == agent]
        assert (
            agent_df["model"].nunique() == 1
        ), f"Multiple models in runs for alias {agent}: {agent_df['model'].unique()}"

        model = agent_df["model"].iloc[0]

        agent_result = defaultdict(dict)

        # data from agent summaries
        agent_summary = df_summaries[df_summaries["agent"] == agent].iloc[0]
        for metric, metric_summary_column_map in summary_column_map.items():
            for metric_summary_key, df_col in metric_summary_column_map.items():
                agent_result[metric][metric_summary_key] = float(agent_summary[df_col])
        agent_result["average_score"]["estimate"] = float(agent_summary["average"])

        # data from runs
        agent_result["usage"]["working_time"] = (
            agent_df.groupby("task_id")["duration_minutes"].mean().sum().item()
        )
        agent_result["usage"]["usd"] = (
            agent_df.groupby("task_id")["generation_cost"].mean().sum().item()
        )

        transcript_links = []
        if include_transcript_links and agent != "GPT-2":
            for _, run_id in agent_df[agent_df["task_source"] != "SWAA"][
                "run_id"
            ].items():
                if run_id.startswith("mp4-server_"):
                    run_id = run_id.replace("mp4-server_", "")
                url = f"https://transcripts.metr.org/run/#{run_id}/"
                transcript_links.append(url)
            agent_result["links"] = {"transcripts": transcript_links}

        if model not in results:
            results[model] = {}

        results[model]["metrics"] = agent_result
        results[model]["release_date"] = release_dates[agent]
        results[model]["scaffolds"] = list(agent_df["scaffold"].unique())
        results[model]["benchmark_name"] = benchmark_name
        dated_results[release_dates[agent]].add(
            (
                model,
                agent_result["p50_horizon_length"]["estimate"],
            )
        )

    if benchmark_results_to_stitch:
        assert (
            model_names_to_stitch
        ), "model_names_to_stitch is required when stitching benchmark results"
        old_results = benchmark_results_to_stitch.get("results", {})
        for model_name in model_names_to_stitch:
            assert (
                model_name in old_results
            ), f"Model {model_name} not found in benchmark results to stitch"
            assert (
                model_name not in results
            ), f"Model {model_name} already exists in current results"
            results[model_name] = old_results[model_name].copy()
            dated_results[results[model_name]["release_date"]].add(
                (
                    model_name,
                    results[model_name]["metrics"]["p50_horizon_length"]["estimate"],
                )
            )

    highest_horizon_so_far = float("-inf")
    for release_date, results_on_date in sorted(dated_results.items()):
        highest_horizon_so_far = max(
            highest_horizon_so_far, max(horizon for model, horizon in results_on_date)
        )
        for model, horizon in results_on_date:
            if horizon < highest_horizon_so_far:
                results[model]["metrics"]["is_sota"] = False
            else:
                results[model]["metrics"]["is_sota"] = True

    results = defaultdict_to_dict(results)

    doubling_time_stats = _get_all_trend_stats(
        df_summaries, df_bootstrap_results, release_dates
    )

    return {
        "benchmark_name": benchmark_name,
        "long_tasks_version": long_tasks_version,
        "swaa_version": swaa_version,
        "results": results,
        "doubling_time_in_days": doubling_time_stats,
    }


def main(
    runs_file: pathlib.Path,
    agent_summaries_file: pathlib.Path,
    bootstrap_results_file: pathlib.Path,
    release_dates_file: pathlib.Path,
    output_metrics_file: pathlib.Path,
    include_transcript_links: bool,
    benchmark_name: str,
    benchmark_long_tasks_version: str,
    benchmark_swaa_version: str,
    benchmark_results_to_stitch: pathlib.Path | None = None,
    model_names_to_stitch: list[str] | None = None,
) -> None:
    df_runs = pd.read_json(runs_file, lines=True, orient="records", convert_dates=False)
    assert "scaffold" in df_runs.columns, "scaffold column is required"

    df_agent_summaries = pd.read_csv(agent_summaries_file)

    release_dates = yaml.safe_load(release_dates_file.read_text())["date"]

    df_bootstrap_results = pd.read_csv(bootstrap_results_file)

    old_benchmark_results = None
    if benchmark_results_to_stitch:
        old_benchmark_results = yaml.safe_load(benchmark_results_to_stitch.read_text())

    output_metrics_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level="INFO",
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    metrics = generate_benchmark_metrics(
        df_runs,
        df_agent_summaries,
        df_bootstrap_results,
        release_dates,
        benchmark_name,
        benchmark_long_tasks_version,
        benchmark_swaa_version,
        logger,
        include_transcript_links=include_transcript_links,
        benchmark_results_to_stitch=old_benchmark_results,
        model_names_to_stitch=model_names_to_stitch,
    )

    with open(output_metrics_file, "w") as f:
        yaml.dump(metrics, f, default_flow_style=False, sort_keys=True)

    logger.info(f"Wrote metrics to {output_metrics_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-file",
        type=pathlib.Path,
        required=True,
        help="Input JSONL file with normalized runs",
    )
    parser.add_argument(
        "--agent-summaries-file",
        type=pathlib.Path,
        help="Input JSONL file with agent summaries",
    )
    parser.add_argument(
        "--bootstrap-results-file",
        type=pathlib.Path,
        help="Input CSV file with bootstrap results",
    )
    parser.add_argument(
        "--release-dates-file",
        type=pathlib.Path,
        required=True,
        help="Input YAML file with a release date for each model",
    )
    parser.add_argument(
        "--output-metrics-file",
        type=pathlib.Path,
        help="Output YAML file with benchmark results",
    )
    parser.add_argument(
        "--include-transcript-links",
        action="store_true",
        help="Include transcript links in the output",
    )
    parser.add_argument(
        "--benchmark-name",
        type=str,
        required=True,
        help="Benchmark name (e.g., METR-Horizon-v1)",
    )
    parser.add_argument(
        "--benchmark-long-tasks-version",
        type=str,
        required=True,
        help="Commit hash for the long tasks manifest file",
    )
    parser.add_argument(
        "--benchmark-swaa-version",
        type=str,
        required=True,
        help="Commit hash for the SWAA manifest file",
    )
    parser.add_argument(
        "--benchmark-results-to-stitch",
        type=pathlib.Path,
        default=None,
        help="Optional path to benchmark_results.yaml to stitch models from",
    )
    parser.add_argument(
        "--model-names-to-stitch",
        type=str,
        default=None,
        help="Comma-separated list of model names to stitch from the old benchmark",
    )
    args = parser.parse_args()

    model_names_to_stitch = None
    if args.model_names_to_stitch:
        model_names_to_stitch = [
            name.strip() for name in args.model_names_to_stitch.split(",")
        ]

    main(
        runs_file=args.runs_file,
        agent_summaries_file=args.agent_summaries_file,
        bootstrap_results_file=args.bootstrap_results_file,
        release_dates_file=args.release_dates_file,
        output_metrics_file=args.output_metrics_file,
        include_transcript_links=args.include_transcript_links,
        benchmark_name=args.benchmark_name,
        benchmark_long_tasks_version=args.benchmark_long_tasks_version,
        benchmark_swaa_version=args.benchmark_swaa_version,
        benchmark_results_to_stitch=args.benchmark_results_to_stitch,
        model_names_to_stitch=model_names_to_stitch,
    )
