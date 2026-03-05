from __future__ import annotations

import logging
import pathlib

import pandas as pd

from horizon.compute_task_weights import add_task_weight_columns

logger = logging.getLogger(__name__)


def load_runs_with_additional_files(
    main_runs_file: pathlib.Path,
    additional_runs_files: list[str],
    *,
    convert_dates: bool = True,
) -> pd.DataFrame:
    """Load runs from a main file and optionally combine with additional runs files.

    Expects JSONL files in the format output by the filter_runs_for_reports stage
    (after compute_task_weights has added weight columns).

    When additional files are provided, the data is concatenated and task weights
    are recomputed since per-file weights don't sum correctly when combined.

    Args:
        main_runs_file: Path to the main runs JSONL file.
        additional_runs_files: List of paths to additional runs JSONL files.
        convert_dates: Whether to convert date columns (passed to pd.read_json).

    Returns:
        Combined DataFrame with recomputed weights if additional files were provided.
    """
    data = pd.read_json(
        main_runs_file, lines=True, orient="records", convert_dates=convert_dates
    )
    logger.info(f"Loaded {len(data)} runs from {main_runs_file}")

    if additional_runs_files:
        additional_dfs = []
        for f in additional_runs_files:
            additional_df = pd.read_json(
                f, lines=True, orient="records", convert_dates=convert_dates
            )
            logger.info(f"Loaded {len(additional_df)} additional runs from {f}")
            additional_dfs.append(additional_df)
        data = pd.concat([data, *additional_dfs], ignore_index=True)
        # Recompute weights since per-file weights don't sum correctly when combined
        data = data.drop(columns=["equal_task_weight", "invsqrt_task_weight"])
        data = add_task_weight_columns(data)
        logger.info(f"Total {len(data)} runs after combining additional files")

    return data
