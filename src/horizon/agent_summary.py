import argparse
import logging
import pathlib
from typing import Any, Sequence

import dvc.api
import pandas as pd
import yaml


def _write_metrics_file(
    averages_by_agent_and_task: pd.DataFrame, output_metrics_file: pathlib.Path
) -> None:
    # FlowList used to reduce unnecessary lines of code in resulting metrics file
    class FlowList(list[str]):
        pass

    def _represent_flow_list(dumper: Any, data: Any) -> Any:
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

    yaml.add_representer(FlowList, _represent_flow_list)

    success_rates_by_agent = {}
    for agent in averages_by_agent_and_task.columns:
        agent_data = averages_by_agent_and_task[agent].dropna()

        always_succeeds = FlowList(sorted(agent_data[agent_data == 1.0].index.tolist()))
        always_fails = FlowList(sorted(agent_data[agent_data == 0.0].index.tolist()))
        partial_success = agent_data[(agent_data > 0.0) & (agent_data < 1.0)].to_dict()

        success_rates_by_agent[agent] = {
            "always_succeeds": always_succeeds,
            "always_fails": always_fails,
            "partial_success": partial_success,
        }

    metrics = {"success_rates_by_agent": success_rates_by_agent}
    output_metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_metrics_file, "w") as f:
        yaml.dump(metrics, f, sort_keys=False, width=float("inf"))
    logging.info(f"Wrote metrics file to {output_metrics_file}")


def generate_agent_summary(
    runs: pd.DataFrame, output_file: pathlib.Path, focus_agents: Sequence[str]
) -> pd.DataFrame:
    runs = runs[runs["alias"].isin(focus_agents)]
    task_agent_counts = pd.crosstab(
        runs["task_id"], runs["alias"], margins=True, margins_name="Total"
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    task_agent_counts.to_csv(output_file, index=True)

    nonzero_counts = pd.crosstab(
        runs[runs["score_binarized"] > 0]["task_id"],
        runs[runs["score_binarized"] > 0]["alias"],
        margins=True,
        margins_name="Total",
    )

    nonzero_counts.to_csv(str(output_file).replace(".csv", "_nonzero.csv"), index=True)

    averages_by_agent_and_task = pd.pivot_table(
        runs,
        values="score_binarized",
        index="task_id",
        columns="alias",
        aggfunc=lambda x: round(x.mean(), 3),
    )
    averages_by_agent_and_task.to_csv(
        str(output_file).replace(".csv", "_averages.csv"), index=True
    )

    return averages_by_agent_and_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=pathlib.Path, required=True)
    parser.add_argument("--output-file-prefix", type=pathlib.Path, required=True)
    parser.add_argument("--output-metrics-file", type=pathlib.Path, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    runs = pd.read_json(
        args.input_file, lines=True, orient="records", convert_dates=False
    )
    logging.info("Loaded input data")

    focus_agents = dvc.api.params_show(stages="generate_agent_summary")[
        "agent_summary"
    ]["agents"]

    averages_by_agent_and_task = generate_agent_summary(
        runs, args.output_file_prefix, focus_agents
    )

    if args.output_metrics_file is not None:
        _write_metrics_file(averages_by_agent_and_task, args.output_metrics_file)


if __name__ == "__main__":
    main()
