"""Calculate baseline statistics from run data."""

import argparse
import json
import logging
import pathlib
from dataclasses import asdict, dataclass
from typing import Any, Dict

import yaml


@dataclass
class SourceStats:
    """Statistics for a specific task source."""

    num_runs_all_agents: int
    num_unique_tasks: int
    num_tasks_with_baselines: int
    num_tasks_with_estimates: int
    num_human_runs: int


@dataclass
class BaselineStats:
    """Overall baseline statistics."""

    total_runs: int
    unique_tasks: int
    source_stats: Dict[str, SourceStats]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        result: Dict[str, Any] = {
            "total_runs": self.total_runs,
            "unique_tasks": self.unique_tasks,
        }
        for source, stats in self.source_stats.items():
            result[source] = asdict(stats)
        return result


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        type=pathlib.Path,
        required=True,
        help="Path to input JSONL file containing run data",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def _calculate_statistics(runs: list[dict[str, str | int]]) -> BaselineStats:
    """Calculate baseline statistics from run data.

    Args:
        runs: List of run dictionaries from JSONL file

    Returns:
        Dictionary containing calculated statistics
    """

    if len(runs) == 0:
        return BaselineStats(
            total_runs=0,
            unique_tasks=0,
            source_stats={},
        )

    # Get all unique task sources and ensure they're strings
    task_sources = {
        str(run.get("task_source"))
        for run in runs
        if run.get("task_source") is not None
    }

    source_stats: Dict[str, SourceStats] = {}

    # Add stats for each task source
    for source in task_sources:
        source_runs = [run for run in runs if str(run.get("task_source")) == source]
        source_stats[source] = SourceStats(
            num_runs_all_agents=len(source_runs),
            num_human_runs=len(
                [run for run in source_runs if run.get("agent") == "human"]
            ),
            num_unique_tasks=len({run["task_id"] for run in source_runs}),
            num_tasks_with_baselines=len(
                {
                    run["task_id"]
                    for run in source_runs
                    if run.get("human_source") == "baseline"
                }
            ),
            num_tasks_with_estimates=len(
                {
                    run["task_id"]
                    for run in source_runs
                    if run.get("human_source") == "estimate"
                }
            ),
        )

        if (
            source_stats[source].num_unique_tasks
            != source_stats[source].num_tasks_with_baselines
            + source_stats[source].num_tasks_with_estimates
        ):
            raise ValueError(
                f"Unique tasks ({source_stats[source].num_unique_tasks}) do not match sum of baselined and estimated tasks ({source_stats[source].num_tasks_with_baselines} + {source_stats[source].num_tasks_with_estimates})"
            )

    if not task_sources:
        raise ValueError(
            f"Source is None for runs: {[run['run_id'] for run in runs if run.get('task_source') is None]}"
        )

    return BaselineStats(
        total_runs=len(runs),
        unique_tasks=len({run.get("task_id") for run in runs}),
        source_stats=source_stats,
    )


def main() -> None:
    """Calculate baseline statistics from run data."""
    args = _parse_args()
    logging.basicConfig(level=args.log_level)

    # Read runs from JSONL file
    runs = []

    with open(args.input_file) as f:
        for line in f:
            runs.append(json.loads(line))

    # Calculate statistics
    stats = _calculate_statistics(runs)

    # Write statistics to YAML file
    metrics_dir = pathlib.Path("metrics")
    metrics_dir.mkdir(exist_ok=True)

    with open(metrics_dir / "baseline_statistics.yaml", "w") as f:
        yaml.safe_dump(stats.to_dict(), f, sort_keys=True)


if __name__ == "__main__":
    main()
