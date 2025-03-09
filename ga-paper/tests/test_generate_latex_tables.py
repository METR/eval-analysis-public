import pathlib
from typing import Any, Dict
from unittest.mock import patch

import pytest

from src.generate_latex_tables import (
    _calculate_avg_time,
    _clean_family_summary_for_latex,
    _clean_text_for_latex,
    _format_time,
    _get_contamination_counts,
    _get_expertise_counts,
    _get_qualitative_label_stats,
    _get_subdomain_stats,
)


class TestTimeCalculations:
    def test_format_time_minutes(self) -> None:
        """Test that _format_time formats minutes correctly."""
        assert _format_time(30) == "30m"
        assert _format_time(59) == "59m"
        assert _format_time(1) == "1m"

    def test_format_time_hours(self) -> None:
        """Test that _format_time formats hours correctly."""
        assert _format_time(60) == "1.0h"
        assert _format_time(90) == "1.5h"
        assert _format_time(120) == "2.0h"
        assert _format_time(64.1721) == "1.1h"

    def test_calculate_avg_time_empty(self) -> None:
        """Test _calculate_avg_time with no matching tasks."""
        family_key = "test_family"
        task_difficulty_data = {
            "other_family/task1": {"minutes": "10"},
        }
        blueprint_tasks = ["task1"]

        avg_time, source = _calculate_avg_time(
            family_key, task_difficulty_data, blueprint_tasks
        )
        assert avg_time == 0.0
        assert source == "N/A"

    def test_calculate_avg_time_with_data(self) -> None:
        """Test _calculate_avg_time with matching tasks."""
        family_key = "test_family"
        task_difficulty_data = {
            "test_family/task1": {"minutes": "10", "source": "estimate"},
            "test_family/task2": {"minutes": "20", "source": "estimate"},
            "test_family/task3": {"minutes": "30", "source": "baseline"},
            "other_family/task1": {"minutes": "100"},
        }
        blueprint_tasks = ["task1", "task2", "task3"]

        avg_time, source = _calculate_avg_time(
            family_key, task_difficulty_data, blueprint_tasks
        )
        assert avg_time == 20.0  # (10 + 20 + 30) / 3
        assert source == "estimate"  # Most common source

    def test_calculate_avg_time_source_priority(self) -> None:
        """Test that _calculate_avg_time breaks ties in favor of 'estimate'."""
        family_key = "test_family"

        # Test tie-breaking with equal counts
        task_difficulty_data = {
            "test_family/task1": {"minutes": "10", "source": "baseline"},
            "test_family/task2": {"minutes": "20", "source": "estimate"},
        }
        blueprint_tasks = ["task1", "task2"]

        avg_time, source = _calculate_avg_time(
            family_key, task_difficulty_data, blueprint_tasks
        )
        assert avg_time == 15.0
        # Should break tie (baseline: 1, estimate: 1) in favor of estimate
        assert source == "estimate"  # Tie broken in favor of estimate


class TestTextFormatting:
    def test_clean_text_for_latex(self) -> None:
        """Test that _clean_text_for_latex escapes LaTeX special characters."""
        test_text = "Text with special chars: & % $ # _ { } ~ ^ < >"
        expected = r"Text with special chars: \& \% \$ \# \_ \{ \} \~ \^ \textless \textgreater"
        assert _clean_text_for_latex(test_text) == expected

    def test_clean_family_summary_for_latex(self) -> None:
        """Test that _clean_family_summary_for_latex processes markdown correctly."""
        # Test with markdown links
        test_with_links = "This is a [link](https://example.com) in text."
        expected_links = "This is a link in text."
        assert _clean_family_summary_for_latex(test_with_links) == expected_links

        # Test with code blocks (should remove them)
        test_with_code = "Text before\n```\ncode block\n```\nText after"
        expected_code = "Text before\n\nText after"
        assert _clean_family_summary_for_latex(test_with_code) == expected_code

        # Test with long text (should truncate)
        long_text = "First paragraph.\n\n" + "X" * 600
        assert _clean_family_summary_for_latex(long_text).endswith("...")
        assert len(_clean_family_summary_for_latex(long_text)) < len(long_text)


class TestStatisticsCalculation:
    @pytest.fixture
    def mock_blueprint(self) -> Dict[str, Any]:
        return {
            "name": "Eval Test Set",
            "task_families": {
                "family0": {
                    "include_source": True,
                    "version": "0.1.2",
                    "tasks": ["task1", "task2"],
                },
                "family1": {
                    "include_source": True,
                    "version": "0.1.3",
                    "tasks": ["task1", "task2"],
                },
                "family2": {
                    "include_source": True,
                    "version": "0.2.4",
                    "tasks": ["task1"],
                },
                "family3": {
                    "include_source": True,
                    "version": "2.1.5",
                    "tasks": ["task1", "task2", "task3"],
                },
            },
        }

    @pytest.fixture
    def mock_labels(self) -> Dict[str, Dict[str, Any]]:
        return {
            "family0": {
                "adversarial": False,
                "category": "MLE training/finetuning",
                "expertise_level": "1-3 years of experience",
                "exploration_experimentation": True,
                "managing_resources": True,
                "novelty_required": True,
                "realistic": True,
                "scary_ara": False,
                "situational_awareness_~required": False,
                "toy_puzzle": False,
                "working_on_existing_code": True,
            },
            "family1": {
                "category": "MLE training/finetuning",
                "expertise_level": "1-3 years of experience",
                "adversarial": False,
                "exploration_experimentation": True,
                "managing_resources": False,
                "novelty_required": True,
                "realistic": True,
                "scary_ara": False,
                "situational_awareness_~required": False,
                "toy_puzzle": False,
                "working_on_existing_code": True,
            },
            "family2": {
                "category": "SWE debugging",
                "expertise_level": "<1 year of experience",
                "adversarial": True,
                "exploration_experimentation": False,
                "managing_resources": True,
                "novelty_required": False,
                "realistic": True,
                "scary_ara": False,
                "situational_awareness_~required": True,
                "toy_puzzle": False,
                "working_on_existing_code": True,
            },
            "family3": {
                "category": "Cybersecurity CTF",
                "expertise_level": ">3 years of experience",
                "adversarial": True,
                "exploration_experimentation": True,
                "managing_resources": False,
                "novelty_required": True,
                "realistic": False,
                "scary_ara": True,
                "situational_awareness_~required": True,
                "toy_puzzle": True,
                "working_on_existing_code": False,
            },
        }

    def test_get_qualitative_label_stats(
        self, mock_blueprint: Dict[str, Any], mock_labels: Dict[str, Dict[str, Any]]
    ) -> None:
        """Test _get_qualitative_label_stats counts boolean labels correctly."""
        with patch("src.generate_latex_tables._load_yaml") as mock_load:
            mock_load.side_effect = [mock_labels, mock_blueprint]

            label_counts, total_families = _get_qualitative_label_stats(
                pathlib.Path("labels.yaml"),
                pathlib.Path("blueprint.yaml"),
            )

            assert total_families == 4
            assert label_counts["realistic"] == 3
            assert label_counts["adversarial"] == 2
            assert label_counts["working_on_existing_code"] == 3

    def test_get_subdomain_stats(
        self, mock_blueprint: Dict[str, Any], mock_labels: Dict[str, Dict[str, Any]]
    ) -> None:
        """Test _get_subdomain_stats groups categories correctly."""
        with patch("src.generate_latex_tables._load_yaml") as mock_load:
            mock_load.side_effect = [mock_labels, mock_blueprint]

            category_counts, labeled_families = _get_subdomain_stats(
                pathlib.Path("labels.yaml"),
                pathlib.Path("blueprint.yaml"),
            )

            assert labeled_families == 4
            assert category_counts["MLE"]["MLE training/finetuning"] == 2
            assert category_counts["SWE"]["SWE debugging"] == 1
            assert category_counts["Cybersecurity"]["Cybersecurity CTF"] == 1

    def test_get_expertise_counts(
        self, mock_blueprint: Dict[str, Any], mock_labels: Dict[str, Dict[str, Any]]
    ) -> None:
        """Test _get_expertise_counts tallies expertise levels correctly."""
        with patch("src.generate_latex_tables._load_yaml") as mock_load:
            mock_load.side_effect = [mock_labels, mock_blueprint]

            expertise_counts = _get_expertise_counts(
                pathlib.Path("labels.yaml"),
                pathlib.Path("blueprint.yaml"),
            )

            assert expertise_counts["1-3 years of experience"] == 2
            assert expertise_counts["<1 year of experience"] == 1
            assert expertise_counts[">3 years of experience"] == 1

    def test_get_contamination_counts(self, mock_blueprint: Dict[str, Any]) -> None:
        """Test _get_contamination_counts tallies contamination levels correctly."""
        with patch("src.generate_latex_tables._load_yaml", return_value=mock_blueprint):
            with patch(
                "src.generate_latex_tables._get_contamination_level"
            ) as mock_get_level:
                # Set up mock to return different levels for different families
                def side_effect(path: pathlib.Path) -> str:
                    family = str(path).split("/")[-2]  # Extract family from path
                    if family == "family0":
                        return "fully_private"
                    elif family == "family1":
                        return "fully_private"
                    elif family == "family2":
                        return "public_problem"
                    else:
                        return "public_solution"

                mock_get_level.side_effect = side_effect

                contamination_counts = _get_contamination_counts(
                    pathlib.Path("blueprint.yaml"),
                    pathlib.Path("task_repo_dir"),
                )

                # Check that the counts match the number of tasks in each family
                assert contamination_counts["fully_private"] == (
                    2,
                    4,
                )  # 2 families, 4 tasks
                assert contamination_counts["public_problem"] == (
                    1,
                    1,
                )  # 1 family, 1 task
                assert contamination_counts["public_solution"] == (
                    1,
                    3,
                )  # 1 family, 3 tasks
