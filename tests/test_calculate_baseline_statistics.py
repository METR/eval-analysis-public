"""Tests for calculate_baseline_statistics module."""

import pytest

from src.calculate_baseline_statistics import (
    BaselineStats,
    SourceStats,
    _calculate_statistics,
)


@pytest.mark.parametrize(
    "runs,expected",
    [
        # Empty list case
        (
            [],
            BaselineStats(
                total_runs=0,
                unique_tasks=0,
                source_stats={},
            ),
        ),
        # Single baseline task
        (
            [
                {
                    "task_id": "task1",
                    "human_source": "baseline",
                    "task_source": "source1",
                }
            ],
            BaselineStats(
                total_runs=1,
                unique_tasks=1,
                source_stats={
                    "source1": SourceStats(
                        num_runs_all_agents=1,
                        num_unique_tasks=1,
                        num_tasks_with_baselines=1,
                        num_tasks_with_estimates=0,
                        num_human_runs=0,
                    ),
                },
            ),
        ),
        # Single estimate task
        (
            [
                {
                    "task_id": "task1",
                    "human_source": "estimate",
                    "task_source": "source1",
                }
            ],
            BaselineStats(
                total_runs=1,
                unique_tasks=1,
                source_stats={
                    "source1": SourceStats(
                        num_runs_all_agents=1,
                        num_unique_tasks=1,
                        num_tasks_with_baselines=0,
                        num_tasks_with_estimates=1,
                        num_human_runs=0,
                    ),
                },
            ),
        ),
        # Multiple tasks with different sources
        (
            [
                {
                    "task_id": "task1",
                    "human_source": "baseline",
                    "task_source": "source1",
                },
                {
                    "task_id": "task2",
                    "human_source": "estimate",
                    "task_source": "source2",
                },
                {
                    "task_id": "task3",
                    "human_source": "baseline",
                    "task_source": "source1",
                },
            ],
            BaselineStats(
                total_runs=3,
                unique_tasks=3,
                source_stats={
                    "source1": SourceStats(
                        num_runs_all_agents=2,
                        num_unique_tasks=2,
                        num_tasks_with_baselines=2,
                        num_tasks_with_estimates=0,
                        num_human_runs=0,
                    ),
                    "source2": SourceStats(
                        num_runs_all_agents=1,
                        num_unique_tasks=1,
                        num_tasks_with_baselines=0,
                        num_tasks_with_estimates=1,
                        num_human_runs=0,
                    ),
                },
            ),
        ),
    ],
)
def test_calculate_statistics(
    runs: list[dict[str, str | int]], expected: BaselineStats
) -> None:
    """Test _calculate_statistics function with various input scenarios."""
    assert _calculate_statistics(runs) == expected
