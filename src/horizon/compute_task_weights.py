from __future__ import annotations

import argparse
import logging
import pathlib

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger("eval_pipeline.compute_task_weights")


def compute_sample_weights(
    df_agent: pd.DataFrame,
) -> pd.DataFrame:
    """
    Input: df_agent, runs for a single agent
    Output: two weighting columns
        - equal_task_weight: each run is 1 / n_run_in_task
        - invsqrt_task_weight: each run is 1 / (n_run_in_task * sqrt(n_task_in_family))
    """
    df_agent_tasks = (
        df_agent.groupby("task_id")
        .agg(
            {
                "task_family": "first",
                "run_id": "count",
            }
        )
        .rename(columns={"run_id": "num_runs_in_task"})
    )

    nans = df_agent.score_binarized.isna().sum()
    assert nans == 0, f"NaN scores in df_agent: {nans}, {df_agent.run_id}"

    equal_task_weight = 1 / df_agent_tasks["num_runs_in_task"]
    equal_task_weight = (
        df_agent["task_id"].map(equal_task_weight).rename("equal_task_weight")
    )
    assert np.allclose(equal_task_weight.sum(), df_agent["task_id"].nunique())

    family_sizes = (
        df_agent_tasks.reset_index().groupby("task_family")["task_id"].count()
    )
    invsqrt_tasks_in_family = 1 / np.sqrt(df_agent["task_family"].map(family_sizes))
    invsqrt_weight = (equal_task_weight * invsqrt_tasks_in_family).rename(
        "invsqrt_task_weight"
    )
    assert np.allclose(invsqrt_weight.sum(), np.sqrt(family_sizes).sum())

    equal_task_weight = equal_task_weight / equal_task_weight.sum()
    invsqrt_weight = invsqrt_weight / invsqrt_weight.sum()

    assert len(equal_task_weight) == len(invsqrt_weight) == len(df_agent)

    return pd.concat([equal_task_weight, invsqrt_weight], axis=1)


def add_task_weight_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Make sure all models have an alias
    null_mask = df["alias"].isna()
    assert not null_mask.any(), "There are runs with null aliases"

    df_sample_weights = pd.concat(
        [
            compute_sample_weights(df_agent)
            for _, df_agent in df.groupby("alias", as_index=False)
        ],
        ignore_index=False,
    )
    df = df.join(df_sample_weights)
    return df


def _write_metrics(
    df: pd.DataFrame, output_metrics_file: pathlib.Path | None = None
) -> None:
    if output_metrics_file is None:
        return

    avg_task_family_weights = (
        df.groupby(["alias", "task_family"])[
            ["invsqrt_task_weight", "equal_task_weight"]
        ]
        .sum()
        .groupby("task_family")
        .mean()
        .round(4)
        .sort_values(["invsqrt_task_weight", "task_family"], ascending=[False, True])
    )

    n_tasks_in_family = (
        df.groupby("task_family")["task_id"].nunique().rename("n_tasks_in_family")
    )
    # Assert that weights sum to 1.0 per agent
    for alias, agent_weights in df.groupby("alias"):
        for column in ["invsqrt_task_weight", "equal_task_weight"]:
            assert np.allclose(
                agent_weights[column].sum(), 1.0
            ), f"{column} weights for {alias} do not sum to 1.0"

    metrics = avg_task_family_weights.join(n_tasks_in_family).to_dict(orient="index")
    output_metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_metrics_file, "w") as f:
        yaml.dump(metrics, f, sort_keys=False)
    logger.info(f"Wrote metrics file to {output_metrics_file}")


def main(
    full_data_file: pathlib.Path,
    output_file: pathlib.Path,
    output_metrics_file: pathlib.Path | None = None,
) -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s: %(message)s",
        level=logging.INFO,
    )
    df = pd.read_json(full_data_file, lines=True, orient="records", convert_dates=False)
    old_columns = set(df.columns)
    df = add_task_weight_columns(df)
    new_columns = set(df.columns) - old_columns

    # Check that all the newly added columns sum to 1.0, per agent
    for alias, agent_weights in df.groupby("alias"):
        for column in new_columns:
            assert np.allclose(
                agent_weights[column].sum(), 1.0
            ), f"{column} weights for {alias} do not sum to 1.0"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_file, index=False, orient="records", lines=True)

    _write_metrics(df, output_metrics_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-data-file", type=pathlib.Path, required=True)
    parser.add_argument("--output-file", type=pathlib.Path, required=True)
    parser.add_argument("--output-metrics-file", type=pathlib.Path, default=None)
    args = parser.parse_args()
    main(**vars(args))
