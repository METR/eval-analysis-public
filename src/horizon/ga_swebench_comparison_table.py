import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_and_process_csv(file_path: Path) -> pd.DataFrame:
    """Read CSV and extract the 50% time horizon value."""
    df = pd.read_csv(file_path)
    # Convert 50% column to minutes
    df["time_horizon"] = df["p50"]
    return df[["agent", "time_horizon"]]


def create_latex_table(headline_file: Path, swebench_file: Path) -> str:
    """Create LaTeX table comparing time horizons between GA and SWE-Bench."""
    # Read data
    ga_df = read_and_process_csv(headline_file)
    swe_df = read_and_process_csv(swebench_file)

    # Merge dataframes
    merged_df = pd.merge(ga_df, swe_df, on="agent", suffixes=("_ga", "_swe"))

    # Calculate ratio
    merged_df["ratio"] = merged_df["time_horizon_ga"] / merged_df["time_horizon_swe"]

    # Calculate geometric means
    geo_mean_ga = np.exp(np.mean(np.log(merged_df["time_horizon_ga"])))
    geo_mean_swe = np.exp(np.mean(np.log(merged_df["time_horizon_swe"])))
    geo_mean_ratio = geo_mean_ga / geo_mean_swe

    # Sort by GA time horizon
    merged_df = merged_df.sort_values("time_horizon_ga")

    # Start building LaTeX table
    latex_table = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"\textbf{Model} & \textbf{\begin{tabular}[c]{@{}c@{}}GA Task Suite\\Model Time\\Horizon\end{tabular}} & \textbf{\begin{tabular}[c]{@{}c@{}}SWE-Bench Verified\\Model Time\\Horizon\end{tabular}} & \textbf{\begin{tabular}[c]{@{}c@{}}GA / SWE-Bench\\Verified Model\\Time Horizon Factor\end{tabular}} \\",
        r"\hline",
    ]

    # Add rows
    for _, row in merged_df.iterrows():
        model_name = row["agent"]
        ga_time = f"{row['time_horizon_ga']:.2f}"
        swe_time = f"{row['time_horizon_swe']:.2f}"
        ratio = f"{row['ratio']:.1f}x"

        latex_row = f"{model_name} & {ga_time} min & {swe_time} min & {ratio} \\\\"
        latex_table.append(latex_row)

    # Add geometric mean row
    latex_table.extend(
        [
            r"\hline",
            f"\\textbf{{Geometric Mean}} & {geo_mean_ga:.2f} min & {geo_mean_swe:.2f} min & {geo_mean_ratio:.1f}x \\\\",
            r"\hline",
            r"\end{tabular}",
            r"\caption{The time horizon of less capable models is substantially longer on our tasks than on SWE-bench Verified.}",
            r"\label{tab:swe-bench-vs-other-tasks}",
            r"\end{table}",
        ]
    )

    return "\n".join(latex_table)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headline-fits-file", type=str, required=True)
    parser.add_argument("--swebench-fits-file", type=str, required=True)
    parser.add_argument("--output-table-file", type=str, required=True)
    args = parser.parse_args()

    latex_table = create_latex_table(
        Path(args.headline_fits_file), Path(args.swebench_fits_file)
    )
    with open(args.output_table_file, "w") as f:
        f.write(latex_table)

    logger.info(f"LaTeX table saved to {args.output_table_file}")


if __name__ == "__main__":
    main()
