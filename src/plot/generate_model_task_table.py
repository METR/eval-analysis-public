import argparse
import pathlib
from typing import Any

import dvc.api
import pandas as pd


def generate_latex_table(
    input_file: pathlib.Path, output_file: pathlib.Path, fig_params: dict[str, Any]
) -> None:
    """Generate a LaTeX table showing average score_binarized by model and task source."""
    # Read the normalized runs
    df = pd.read_json(input_file, lines=True, orient="records", convert_dates=False)
    df = df[df["alias"].isin(fig_params["include_agents"])]
    # Weight tasks equally
    df_agg = df.groupby(["alias", "task_id"]).agg(
        {
            "score_binarized": "mean",
            "task_source": "first",
        }
    )
    pivot = pd.pivot_table(
        df_agg,
        values="score_binarized",
        index="alias",
        columns="task_source",
        aggfunc="mean",
        fill_value=0,
    )

    # Sort index alphabetically
    pivot = pivot.sort_index()

    # Convert to LaTeX with specific formatting
    latex_table = pivot.to_latex(
        float_format=lambda x: f"{x:.3f}",
        bold_rows=True,
        caption="Average Success Rate by Model and Task Source",
        label="tab:model_task_success",
        position="htbp",
    )

    # Write to file
    with open(output_file, "w") as f:
        f.write(latex_table)


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
    args = parser.parse_args()

    params = dvc.api.params_show(stages="generate_model_task_table")
    fig_params = params["figs"]["generate_model_task_table"]
    generate_latex_table(args.input_file, args.output_file, fig_params)
