"""
Generate model/task result tables in LaTeX and YAML formats. This is used by Epoch
for their dashboard.

This script provides two main functionalities:
1. Generate a LaTeX table showing average success rates by model and task source
2. Generate detailed YAML files with comprehensive task performance metrics

Example CLI usage:
    # Generate only LaTeX table
    python -m src.plot.generate_model_task_table \
        --input-file data/processed/normalized_runs.jsonl \
        --output-file output/model_task_table.tex

    # Generate both LaTeX table and YAML metrics files
    python -m src.plot.generate_model_task_table \
        --input-file data/processed/normalized_runs.jsonl \
        --output-file output/model_task_table.tex \
        --output-metrics-dir output/task_results
"""

import argparse
import pathlib
from collections import defaultdict
from typing import Any

import dvc.api
import pandas as pd
import yaml


def generate_latex_table(df: pd.DataFrame, fig_params: dict[str, Any]) -> str:
    """Generate a LaTeX table showing average score_binarized by model and task source."""
    df = df[df["alias"].isin(fig_params["include_agents"])]
    # Weight tasks equally
    df_agg = df.groupby(["alias", "task_id"]).agg(
        {
            "score_binarized": "mean",
            "task_source": "first",
        }
    )
    df_agg = df_agg.rename(columns={"task_source": "Task Source"})
    pivot = pd.pivot_table(
        df_agg,
        values="score_binarized",
        index="alias",
        columns="Task Source",
        aggfunc="mean",
        fill_value=0,
    )

    # Sort index alphabetically
    pivot = pivot.sort_index().astype(object)

    pivot.loc["GPT-2"] = ["-", "-", pivot.loc["GPT-2", "SWAA"]]
    pivot.loc["o3", "SWAA"] = "-"
    pivot.loc["o4-mini", "SWAA"] = "-"
    # Drop column name from index so it doesn't print
    pivot.index.name = None

    # Convert to LaTeX with specific formatting
    latex_table = pivot.to_latex(
        float_format=lambda x: f"{x:.2f}",
        bold_rows=True,
        caption="Average Success Rate by Model and Task Source",
        label="tab:model_task_success",
        position="htbp",
        header=["HCAST", "RE-Bench", "SWAA"],
        columns=["HCAST", "RE-Bench", "SWAA"],
        index_names=False,
    )

    # Add centering command after the first line of the LaTeX table
    latex_table_lines = latex_table.split("\n")
    latex_table_lines.insert(1, "\\centering")
    latex_table = "\n".join(latex_table_lines)

    return latex_table


def defaultdict_to_dict(d: defaultdict | dict) -> dict:  # type: ignore
    if isinstance(d, defaultdict) or isinstance(d, dict):
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d


def generate_task_metrics(
    task_id: str, task_df: pd.DataFrame, params: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """
    Generate detailed task result information.

    Returns a dictionary with task results for each task ID, containing model
    and agent performance metrics including costs, time taken, token information,
    version, and completion percentage.

    Fields included for each task:
    - Model name
    - Agent name
    - Task name and ID
    - Costs (normalized by number of runs)
    - Time the agent took
    - Token costs (in USD and token count)
    - Version/hash of the benchmark or task
    - Success rate (with confidence intervals if available)
    - Number of runs
    - Optional fields like horizon_length and links to transcripts
    """
    versions = task_df["task_version"].unique()
    versions = [v for v in versions if v is not None]
    if len(versions) > 1:
        major_versions = [v.split(".")[0] for v in versions]
        if len(set(major_versions)) > 1:
            raise ValueError(f"Task {task_id} has multiple versions: {versions}")
    version = versions[0] if versions else None

    task_data = {
        "task_id": task_id,
        "version": version,
        "results": defaultdict(lambda: defaultdict(dict)),
    }

    for (model, scaffold), agent_df in task_df.groupby(["model", "scaffold"]):
        num_runs = len(agent_df)
        avg_score = agent_df["score_binarized"].mean()

        duration_minutes = (
            (agent_df["completed_at"] - agent_df["started_at"]).mean()
        ) / (60 * 1000)

        generation_cost = (
            agent_df["generation_cost"].sum() / num_runs
            if "generation_cost" in agent_df.columns
            else None
        )
        human_cost = (
            agent_df["human_cost"].iloc[0] if "human_cost" in agent_df.columns else None
        )

        ci_low, ci_high = None, None
        if num_runs >= 3:
            # Simple approximation of 95% confidence interval
            std_dev = agent_df["score_binarized"].std()
            margin = 1.96 * std_dev / (num_runs**0.5)
            ci_low = max(0.0, avg_score - margin)
            ci_high = min(1.0, avg_score + margin)

        time_limit_minutes = (
            agent_df["time_limit"].iloc[0] / 60
            if "time_limit" in agent_df.columns
            else None
        )

        # Build agent result data
        agent_result = {
            "num_runs": num_runs,
            "success_probability": {"estimate": float(avg_score)},
        }

        if time_limit_minutes is not None:
            agent_result["time_limit_minutes"] = float(time_limit_minutes)

        if duration_minutes is not None:
            agent_result["duration_minutes"] = float(duration_minutes)

        cost_data = {}
        if generation_cost is not None:
            cost_data["generation_usd"] = float(generation_cost)
        if human_cost is not None:
            cost_data["human_equivalent_usd"] = float(human_cost)
        if cost_data:
            agent_result["cost"] = cost_data

        if ci_low is not None and ci_high is not None:
            agent_result["success_probability"]["ci_low"] = float(ci_low)
            agent_result["success_probability"]["ci_high"] = float(ci_high)

        transcript_links = []

        for _, run_id in agent_df["run_id"].items():
            # SWAA runs are stored as ints, and don't have transcripts
            if isinstance(run_id, str):
                if run_id.startswith("mp4-server_"):
                    run_id = run_id.replace("mp4-server_", "")
                url = f"https://transcripts.metr.org/run/#{run_id}/"
                transcript_links.append(url)

        if transcript_links:
            agent_result["links"] = {"transcripts": transcript_links}

        task_data["results"][model][scaffold] = agent_result

    return defaultdict_to_dict(task_data)


def main(
    input_file: pathlib.Path,
    output_file: pathlib.Path,
    output_metrics_dir: pathlib.Path,
) -> None:
    params = dvc.api.params_show(stages="generate_model_task_table")
    fig_params = params["figs"]["generate_model_task_table"]

    df = pd.read_json(input_file, lines=True, orient="records", convert_dates=False)
    assert "scaffold" in df.columns, "scaffold column is required"
    latex_table = generate_latex_table(df, fig_params)
    with open(output_file, "w") as f:
        f.write(latex_table)

    if not output_metrics_dir:
        return

    output_metrics_dir.mkdir(parents=True, exist_ok=True)

    for task_id, task_df in df.groupby("task_id"):
        task_id_str = str(task_id)
        task_metrics = generate_task_metrics(task_id_str, task_df, params)

        task_parts = task_id_str.split("/")
        task_family = task_parts[0]
        task_dir = output_metrics_dir / task_family
        task_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_metrics_dir / f"{task_id}.yaml"

        with open(output_file, "w") as f:
            yaml.dump(task_metrics, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        type=pathlib.Path,
        required=True,
        help="Input JSONL file with normalized runs",
    )
    parser.add_argument(
        "--output-file",
        type=pathlib.Path,
        required=True,
        help="Output LaTeX file",
    )
    parser.add_argument(
        "--output-metrics-dir",
        type=pathlib.Path,
        help="Output directory for YAML metrics files",
    )
    args = parser.parse_args()

    main(
        input_file=args.input_file,
        output_file=args.output_file,
        output_metrics_dir=args.output_metrics_dir,
    )
