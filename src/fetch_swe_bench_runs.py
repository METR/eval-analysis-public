import argparse
import csv
import json
import os
import subprocess
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd
from datasets import load_dataset

SWE_BENCH_REPO_DIR = "data/swe_bench_results"  # TODO: move to data
SWE_BENCH_REPO_URL = "https://github.com/swe-bench/experiments.git"


def clone_experiments_repo() -> None:
    """s
    Clones the swe-bench experiments repo into local_dir if not already cloned. Otherwise pull latest changes.
    """

    if not os.path.exists(SWE_BENCH_REPO_DIR):
        print(f"Cloning {SWE_BENCH_REPO_URL} into {SWE_BENCH_REPO_DIR}...")
        subprocess.run(
            f"git clone --depth 1 {SWE_BENCH_REPO_URL} {SWE_BENCH_REPO_DIR}",
            shell=True,
            check=True,
        )
    else:
        print(
            f"Repository already exists at {SWE_BENCH_REPO_DIR}. Pulling latest changes..."
        )
        subprocess.run(f"cd {SWE_BENCH_REPO_DIR} && git pull", shell=True, check=True)


def convert_time_estimate_to_minutes(time_str: str) -> float:
    def geometric_mean(x: float, y: float) -> float:
        return (x * y) ** 0.5

    if time_str == "<15 min fix":
        return geometric_mean(1, 15)
    elif time_str == "15 min - 1 hour":
        return geometric_mean(15, 60)
    elif time_str == "1-4 hours":
        return geometric_mean(60, 240)
    elif time_str == ">4 hours":
        return geometric_mean(240, 960)
    else:
        raise ValueError(f"Unknown time format: {time_str}")


def get_time_estimates(annotations_file: str) -> Dict[str, float]:
    """
    Loads time estimates from the provided CSV file for verified tasks.
    """
    if not os.path.exists(annotations_file):
        raise FileNotFoundError(f"Could not find {annotations_file}")

    # Load the verified dataset to filter tasks
    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    verified_instance_ids = {ex["instance_id"] for ex in list(dataset)}

    time_estimates: Dict[str, float] = {}
    with open(annotations_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            instance_id = row["instance_id"]
            if instance_id in verified_instance_ids:
                difficulty_bucket = row["difficulty"]
                time_estimates[instance_id] = convert_time_estimate_to_minutes(
                    difficulty_bucket
                )
    assert (
        len(time_estimates) == 500
    ), f"SWE-bench Verified has 500 tasks, but got {len(time_estimates)}"
    return time_estimates


def get_model_results(
    models: Dict[str, str],
    time_estimates: Dict[str, float],
) -> Dict[str, Dict[str, List[str]]]:
    """
    Loads model results given a local experiments repo directory, a dictionary of model names to model IDs,
    and time estimates. Returns a dictionary with solved and not solved tasks per model.
    """
    model_solutions = OrderedDict()
    for model_name, model_id in models.items():
        results_path = os.path.join(
            SWE_BENCH_REPO_DIR,
            "evaluation",
            "verified",
            model_id,
            "results",
            "results.json",
        )
        assert os.path.exists(results_path), f"results.json not found for {model_id}"

        with open(results_path, "r") as f:
            data = json.load(f)
        print(model_name, len(data.get("no_generation", [])))
        solved_tasks = set(data.get("resolved", []))
        all_tasks = set(time_estimates.keys())
        not_solved = all_tasks - solved_tasks

        assert (
            len(all_tasks) == 500
        ), f"Expected 500 tasks, got {len(all_tasks)}, in {results_path}"

        model_solutions[model_name] = {
            "solved": list(solved_tasks),
            "not_solved": list(not_solved),
        }

    return model_solutions


def main() -> None:
    """
    Converts SWE-Bench Verified results to a jsonl file of runs that we can process in the rest of our
    pipeline.

    A few notes:
    1. We use the swe-bench verified dataset, which has 500 tasks.
    2. We use the ensembled annotations for SWE-Bench verified, downloaded from https://openai.com/index/introducing-swe-bench-verified/.
    3. We use the time estimates from the ensembled annotations to convert the time estimates to minutes, using a geometric mean.
       Note that the time estimate for the end of the highest time bucket is 960 minutes, which is a somewhat arbitrary choice.
    4. We assume that all tasks from the same repo are from the same task family
    """
    parser = argparse.ArgumentParser(
        description="Process SWE Bench data and generate runs files."
    )
    parser.add_argument(
        "--annotations",
        type=str,
        help="Path to the ensembled annotations for SWE-Bench verified, downloaded from https://openai.com/index/introducing-swe-bench-verified/.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Path to output swe_bench_runs jsonl file.",
    )
    args = parser.parse_args()

    clone_experiments_repo()

    # Define models mapping. Note that these are all the models I could personally find in the
    # SWE-Bench experiments repo.
    # TODO: SOMEONE WITH MODELS KNOWLEDGE SHOULD CHECK THESE! I DO NOT KNOW THE MODELS VERY WELL
    # AND THE DATES ARE A BIT HARD TO PARSE. MAKE SURE TO CHECK THE REPO FOR THE CORRECT MODEL ID
    # FOR EACH MODEL.
    models = {
        "Claude 3 Opus": "20240402_sweagent_claude3opus",
        "Claude 3.5 Sonnet (Old)": "20241029_epam-ai-run-claude-3-5-sonnet",
        "Claude 3.5 Sonnet (New)": "20241029_OpenHands-CodeAct-2.1-sonnet-20241022",
        "o1": "20250117_wandb_programmer_o1_crosscheck5",
        "GPT-4o": "20241028_agentless-1.5_gpt4o",
        "GPT-4 1106": "20240402_sweagent_gpt4",
    }

    time_estimates = get_time_estimates(args.annotations)

    model_solutions = get_model_results(models, time_estimates)

    task_ids = list(time_estimates.keys())
    assert len(task_ids) == 500, f"Expected 500 tasks, got {len(task_ids)}"

    # Count number of tasks per family
    task_family_counts = defaultdict(int)
    for task_id in task_ids:
        family = task_id.split("__")[0]
        task_family_counts[family] += 1
        assert family in [
            "astropy",
            "django",
            "pydata",
            "pallets",
            "psf",
            "pytest-dev",
            "scikit-learn",
            "sphinx-doc",
            "sympy",
            "matplotlib",
            "mwaskom",
            "pylint-dev",
        ], f"Unknown task family: {family}"

    aliases = list(model_solutions.keys())

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for alias in aliases:
            rows = []
            not_solved_set = model_solutions[alias]["not_solved"]
            solved_set = model_solutions[alias]["solved"]

            for task_id in task_ids:
                human_minutes = time_estimates[task_id]

                # Make sure we only get one
                assert bool(task_id in not_solved_set) != bool(
                    task_id in solved_set
                ), f"Task {task_id} is in both not_solved and solved"
                score = 0 if task_id in not_solved_set else 1

                task_family = task_id.split("__")[0]
                assert (
                    task_family in task_family_counts
                ), f"Task family {task_family} not found in task_family_counts"

                invsqrt_task_weight = 1 / (task_family_counts[task_family] ** 0.5)

                row = {
                    "task_id": task_id,
                    "task_family": task_id.split("__")[0],
                    "human_minutes": human_minutes,
                    "score_cont": score,
                    "score_binarized": score,
                    "alias": alias,
                    "equal_task_weight": 1 / len(task_ids),
                    "invsqrt_task_weight": invsqrt_task_weight,
                }
                rows.append(row)

            df = pd.DataFrame(rows)
            df["equal_task_weight"] = df["equal_task_weight"].astype(float)
            df["invsqrt_task_weight"] = df["invsqrt_task_weight"].astype(float)
            df["equal_task_weight"] = (
                df["equal_task_weight"] / df["equal_task_weight"].sum()
            )
            df["invsqrt_task_weight"] = (
                df["invsqrt_task_weight"] / df["invsqrt_task_weight"].sum()
            )
            assert (
                abs(df["equal_task_weight"].sum() - 1) < 1e-6
            ), "equal_task_weight sum is not close to 1"
            assert (
                abs(df["invsqrt_task_weight"].sum() - 1) < 1e-6
            ), "invsqrt_task_weight sum is not close to 1"
            # write each row as a json object
            df["task_source"] = "swe_bench"
            for _, row in df.iterrows():
                f.write(json.dumps(row.to_dict()) + "\n")


if __name__ == "__main__":
    main()
