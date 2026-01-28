import argparse
import pathlib

import dvc.api
import joblib
import numpy as np
import pandas as pd

from horizon.wrangle.logistic import (
    WrangleParams,
    agent_regression,
)
from horizon.wrangle.score_over_resource import (
    handle_human_agent,
)


def add_best_model_at_each_task(
    df: pd.DataFrame, score_column: str, exclude_agents: list[str] | None = None
) -> pd.DataFrame:
    original_df = df.copy()

    if exclude_agents:
        df = df[~df["alias"].isin(exclude_agents)]
    best_model_rows = []
    for task_id in df["task_id"].unique():
        model_rows = df[
            (df["task_id"] == task_id)
            & (~df["alias"].str.contains("human", case=False))
        ]
        if model_rows.empty:
            raise ValueError(f"No model data for task {task_id}")
        mean_scores = model_rows.groupby("alias")[score_column].mean()
        best_model_for_task = mean_scores.idxmax()
        best_model_rows_for_task = model_rows[
            model_rows["alias"] == best_model_for_task
        ].copy()
        best_model_rows_for_task = best_model_rows_for_task.assign(
            alias="Best Model for Each Task"
        )
        best_model_rows.extend(best_model_rows_for_task.to_dict("records"))
    best_model_df = pd.DataFrame(best_model_rows)

    df = pd.concat([original_df, best_model_df])
    return df


def _process_one_agent_cost_combination(
    agent_alias: str,
    cost_value: float,
    agent_specific_df: pd.DataFrame,
    usage_column: str,
    score_column: str,
    wrangle_params: WrangleParams,
) -> dict[str, float] | None:
    agent_at_cost_df = agent_specific_df.copy()

    agent_at_cost_df[score_column] = agent_at_cost_df.apply(
        lambda row: row[score_column] if row[usage_column] <= cost_value else 0,
        axis=1,
    )

    weights = np.asarray(agent_at_cost_df[wrangle_params["weighting"]].values)
    weights = weights / np.sum(weights)

    agent_regression_results = agent_regression(
        x=agent_at_cost_df["human_minutes"].values,  # type: ignore
        y=agent_at_cost_df[score_column].values,  # type: ignore
        weights=weights,  # type: ignore
        agent_name=agent_alias,
        regularization=wrangle_params["regularization"],
        success_percents=wrangle_params["success_percents"],
        confidence_level=wrangle_params["confidence_level"],
        ensure_weights_sum_to_1=True,
    )

    result_dict = agent_regression_results.to_dict()
    result_dict["alias"] = agent_alias
    result_dict[usage_column] = cost_value
    return result_dict


def process_data(
    df: pd.DataFrame,
    usage_column: str,
    score_column: str,
    wrangle_params: WrangleParams,
) -> pd.DataFrame:
    tasks = []
    for agent_alias in df["alias"].unique():
        agent_df_slice = df[df["alias"] == agent_alias]
        for cost in sorted(agent_df_slice[usage_column].unique()):
            tasks.append(
                joblib.delayed(_process_one_agent_cost_combination)(
                    agent_alias=agent_alias,
                    cost_value=cost,
                    agent_specific_df=agent_df_slice,
                    usage_column=usage_column,
                    score_column=score_column,
                    wrangle_params=wrangle_params,
                )
            )

    results_list = joblib.Parallel(n_jobs=-1, verbose=1)(tasks)
    return pd.DataFrame([res for res in results_list if res is not None])


def main(
    runs_file: pathlib.Path,
    output_file: pathlib.Path,
    include_human: bool,
    score_column: str,
    wrangle_params: str,
    x: str,
    include_best_model: bool,
    fig_name: str,
) -> None:
    df = pd.read_json(runs_file, lines=True, orient="records")
    df = df[df["task_source"].isin(["HCAST", "RE-Bench"])]

    params = dvc.api.params_show(
        stages="wrangle_score_over_resource_horizon", deps=True
    )
    fig_params = params["figs"]["wrangle_score_over_resource_horizon"][fig_name]
    exclude_agent = fig_params["exclude_agents"]
    wrangle_params_dict = WrangleParams(
        **params["figs"]["wrangle_logistic"][wrangle_params]
    )

    df = handle_human_agent(df, include_human, score_column, x)
    if include_best_model:
        df = add_best_model_at_each_task(df, score_column, exclude_agents=exclude_agent)
    if exclude_agent:
        df = df[~df["alias"].isin(exclude_agent)]
    df.loc[df[score_column].isna(), score_column] = 0

    if x == "tokens_count":
        assert "tokens_count" in df.columns, "tokens_count column missing"
        df = df[df["tokens_count"].notna()]
        df = df[df["tokens_count"] > 0]
        assert not df.empty, "No rows remain after filtering for tokens_count"

    df = process_data(
        df=df,
        usage_column=x,
        score_column=score_column,
        wrangle_params=wrangle_params_dict,
    )
    pathlib.Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_file, orient="records", lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-file",
        type=pathlib.Path,
        required=True,
        help="Path to the runs.jsonl file",
    )
    parser.add_argument(
        "--output-file",
        type=pathlib.Path,
        help="Path to save the output figure",
        required=True,
    )
    parser.add_argument(
        "--include-human",
        action="store_true",
        help="Whether to include the human agent in the data",
    )
    parser.add_argument(
        "--score-column",
        type=str,
        required=True,
        help="Column to use for scoring. E.g. score_binarized",
    )
    parser.add_argument(
        "--include-best-model",
        action="store_true",
        help="Whether to include the best model at each task in the data",
    )
    parser.add_argument(
        "--wrangle-params",
        type=str,
        required=True,
        help="Name of wrangle params set to use. E.g. headline, ga_rebench",
    )
    parser.add_argument(
        "--x",
        required=True,
        type=str,
        help="Column to use for x-axis. E.g. generation_cost or action_count",
    )
    parser.add_argument(
        "--fig-name",
        type=str,
        required=True,
        help="Name of the fig params (e.g. generation_cost, tokens_count)",
    )
    args = parser.parse_args()
    main(**vars(args))
