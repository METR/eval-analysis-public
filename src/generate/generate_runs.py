"""
Generate synthetic run data for NAME (Novelty Achievement MeasurE) metric.

Combines synthetic tasks with novice learning simulation and AI zero-shot simulation
to produce a complete dataset matching the all_runs.jsonl schema.
"""

import json
import pathlib
import yaml
from typing import List, Dict, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime

from src.simulate.novice_learning import (
    NoviceLearningSimulator,
    AIZeroShotSimulator,
    LearningParameters,
    create_ai_model
)
from src.generate.synthetic_tasks import TaskDefinition


@dataclass
class RunRecord:
    """Single run record matching all_runs.jsonl schema."""

    # Core identifiers
    task_id: str
    run_id: str
    alias: str  # "human_novice" or model name

    # Score fields
    score_cont: float
    score_binarized: int

    # NAME-specific fields
    attempt_number: int
    prior_attempts: int
    learner_type: str  # human_novice | ai_zero_shot | ai_few_shot
    examples_seen: int
    feedback_type: str
    feedback_received: str

    # Task difficulty metadata
    first_attempt_success_rate: float
    learning_rate_lambda: float
    max_attempts: int

    # Compatibility fields (for integration with existing pipeline)
    task_family: str
    task_source: str  # "NAME_synthetic"
    human_minutes: float  # Placeholder (not meaningful for NAME)
    human_score: float  # Placeholder
    human_source: str  # "synthetic"
    generation_cost: float  # Placeholder
    human_cost: float  # Placeholder
    equal_task_weight: float
    invsqrt_task_weight: float

    # Placeholder for future expansion
    quality_score: Optional[float]
    task_domain: str
    verification_type: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "run_id": self.run_id,
            "alias": self.alias,
            "score_cont": self.score_cont,
            "score_binarized": self.score_binarized,
            "attempt_number": self.attempt_number,
            "prior_attempts": self.prior_attempts,
            "learner_type": self.learner_type,
            "examples_seen": self.examples_seen,
            "feedback_type": self.feedback_type,
            "feedback_received": self.feedback_received,
            "first_attempt_success_rate": self.first_attempt_success_rate,
            "learning_rate_lambda": self.learning_rate_lambda,
            "max_attempts": self.max_attempts,
            "task_family": self.task_family,
            "task_source": self.task_source,
            "human_minutes": self.human_minutes,
            "human_score": self.human_score,
            "human_source": self.human_source,
            "generation_cost": self.generation_cost,
            "human_cost": self.human_cost,
            "equal_task_weight": self.equal_task_weight,
            "invsqrt_task_weight": self.invsqrt_task_weight,
            "quality_score": self.quality_score,
            "task_domain": self.task_domain,
            "verification_type": self.verification_type
        }


class SyntheticRunGenerator:
    """Generate synthetic run data combining tasks, human learners, and AI models."""

    def __init__(
        self,
        tasks: List[TaskDefinition],
        n_novices_per_task: int = 50,
        max_attempts: int = 10,
        ai_models: Optional[List[str]] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize run generator.

        Args:
            tasks: List of TaskDefinition objects
            n_novices_per_task: Number of novice learners per task
            max_attempts: Maximum attempts per learner
            ai_models: List of AI model names to simulate
            random_seed: Random seed for reproducibility
        """
        self.tasks = tasks
        self.n_novices_per_task = n_novices_per_task
        self.max_attempts = max_attempts
        self.ai_models = ai_models or ["baseline-weak", "baseline-medium", "baseline-strong"]
        self.rng = np.random.default_rng(random_seed)

        # Calculate task weights (for compatibility)
        self._calculate_task_weights()

    def _calculate_task_weights(self):
        """Calculate task weights for diversity weighting."""
        n_tasks = len(self.tasks)

        # Equal weight
        equal_weight = 1.0 / n_tasks

        # Inverse sqrt weight (from task family diversity)
        family_counts = {}
        for task in self.tasks:
            family = task.task_family
            family_counts[family] = family_counts.get(family, 0) + 1

        self.task_weights = {}
        for task in self.tasks:
            family_size = family_counts[task.task_family]
            invsqrt_weight = 1.0 / np.sqrt(family_size) / len(set(t.task_family for t in self.tasks))

            self.task_weights[task.task_id] = {
                "equal": equal_weight,
                "invsqrt": invsqrt_weight
            }

    def _generate_human_novice_runs(
        self,
        task: TaskDefinition
    ) -> List[RunRecord]:
        """
        Generate novice learning runs for a single task.

        Args:
            task: TaskDefinition

        Returns:
            List of RunRecords for all novice attempts
        """
        runs = []

        # Create learning simulator for this task
        params = LearningParameters.from_task_difficulty(
            difficulty=task.first_attempt_success_rate,
            lambda_learning=task.learning_rate_lambda
        )
        simulator = NoviceLearningSimulator(params, random_seed=None)  # Different seed per task

        # Simulate multiple novices
        for novice_id in range(self.n_novices_per_task):
            trajectory = simulator.simulate_learner_trajectory(
                max_attempts=self.max_attempts,
                stop_on_success=False
            )

            # Convert to RunRecords
            for attempt_data in trajectory:
                attempt_num = attempt_data["attempt_number"]
                success = attempt_data["success"]

                run_id = f"novice_{novice_id:04d}_attempt_{attempt_num:02d}_{task.task_id.replace('/', '_')}"

                run = RunRecord(
                    task_id=task.task_id,
                    run_id=run_id,
                    alias="human_novice",
                    score_cont=1.0 if success else 0.0,
                    score_binarized=1 if success else 0,
                    attempt_number=attempt_num,
                    prior_attempts=attempt_num - 1,
                    learner_type="human_novice",
                    examples_seen=0,
                    feedback_type="binary",
                    feedback_received="success" if success else "failure",
                    first_attempt_success_rate=task.first_attempt_success_rate,
                    learning_rate_lambda=task.learning_rate_lambda,
                    max_attempts=task.max_attempts,
                    task_family=task.task_family,
                    task_source="NAME_synthetic",
                    human_minutes=0.0,  # Not meaningful for NAME
                    human_score=1.0,
                    human_source="synthetic",
                    generation_cost=0.0,
                    human_cost=0.0,
                    equal_task_weight=self.task_weights[task.task_id]["equal"],
                    invsqrt_task_weight=self.task_weights[task.task_id]["invsqrt"],
                    quality_score=None,
                    task_domain=task.domain,
                    verification_type=task.verification_type
                )

                runs.append(run)

        return runs

    def _generate_ai_zero_shot_runs(
        self,
        task: TaskDefinition,
        model_name: str,
        n_attempts: int = 20
    ) -> List[RunRecord]:
        """
        Generate AI zero-shot runs for a single task.

        Args:
            task: TaskDefinition
            model_name: Name of AI model
            n_attempts: Number of zero-shot attempts (for statistical reliability)

        Returns:
            List of RunRecords for AI attempts
        """
        runs = []

        # Create AI simulator
        ai = create_ai_model(model_name, random_seed=None)

        # Generate multiple attempts
        for attempt_idx in range(n_attempts):
            success = ai.simulate_attempt(task.first_attempt_success_rate)

            run_id = f"{model_name}_attempt_{attempt_idx:03d}_{task.task_id.replace('/', '_')}"

            run = RunRecord(
                task_id=task.task_id,
                run_id=run_id,
                alias=model_name,
                score_cont=1.0 if success else 0.0,
                score_binarized=1 if success else 0,
                attempt_number=1,  # Always 1 for zero-shot
                prior_attempts=0,
                learner_type="ai_zero_shot",
                examples_seen=0,
                feedback_type="none",
                feedback_received="",
                first_attempt_success_rate=task.first_attempt_success_rate,
                learning_rate_lambda=task.learning_rate_lambda,
                max_attempts=1,  # Zero-shot = single attempt conceptually
                task_family=task.task_family,
                task_source="NAME_synthetic",
                human_minutes=0.0,
                human_score=1.0,
                human_source="synthetic",
                generation_cost=0.0,
                human_cost=0.0,
                equal_task_weight=self.task_weights[task.task_id]["equal"],
                invsqrt_task_weight=self.task_weights[task.task_id]["invsqrt"],
                quality_score=None,
                task_domain=task.domain,
                verification_type=task.verification_type
            )

            runs.append(run)

        return runs

    def generate_all_runs(self) -> List[RunRecord]:
        """
        Generate complete dataset: novice learning + AI zero-shot for all tasks.

        Returns:
            List of all RunRecords
        """
        all_runs = []

        print(f"Generating runs for {len(self.tasks)} tasks...")
        print(f"  - {self.n_novices_per_task} novices × {self.max_attempts} attempts per task")
        print(f"  - {len(self.ai_models)} AI models × 20 attempts per task")

        for idx, task in enumerate(self.tasks):
            if (idx + 1) % 20 == 0:
                print(f"  Progress: {idx + 1}/{len(self.tasks)} tasks")

            # Generate novice runs
            novice_runs = self._generate_human_novice_runs(task)
            all_runs.extend(novice_runs)

            # Generate AI runs
            for model_name in self.ai_models:
                ai_runs = self._generate_ai_zero_shot_runs(task, model_name, n_attempts=20)
                all_runs.extend(ai_runs)

        print(f"\nGenerated {len(all_runs)} total runs")
        return all_runs

    def save_runs(self, runs: List[RunRecord], output_path: str):
        """
        Save runs to JSONL file (one JSON object per line).

        Args:
            runs: List of RunRecords
            output_path: Path to output JSONL file
        """
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for run in runs:
                json.dump(run.to_dict(), f)
                f.write('\n')

    def get_summary_statistics(self, runs: List[RunRecord]) -> dict:
        """
        Calculate summary statistics for generated runs.

        Args:
            runs: List of RunRecords

        Returns:
            Dictionary of summary statistics
        """
        # Convert to DataFrame for analysis
        df = pd.DataFrame([run.to_dict() for run in runs])

        # Split by learner type
        df_novice = df[df["learner_type"] == "human_novice"]
        df_ai = df[df["learner_type"] == "ai_zero_shot"]

        # Novice statistics
        novice_stats = {
            "total_runs": len(df_novice),
            "unique_learners": df_novice["run_id"].apply(lambda x: x.split("_")[1]).nunique(),
            "mean_success_rate": df_novice["score_binarized"].mean(),
            "first_attempt_success_rate": df_novice[df_novice["attempt_number"] == 1]["score_binarized"].mean()
        }

        # AI statistics
        ai_stats = {}
        for model in self.ai_models:
            df_model = df_ai[df_ai["alias"] == model]
            ai_stats[model] = {
                "total_runs": len(df_model),
                "mean_success_rate": df_model["score_binarized"].mean()
            }

        return {
            "total_runs": len(runs),
            "n_tasks": df["task_id"].nunique(),
            "novice_stats": novice_stats,
            "ai_stats": ai_stats,
            "runs_by_domain": df.groupby("task_domain")["run_id"].count().to_dict()
        }


# Example usage and CLI
if __name__ == "__main__":
    import sys

    # Load tasks from YAML
    task_file = "data/external/name_tasks.yaml"
    output_file = "data/external/name_runs.jsonl"

    # Parse command line args if provided
    if len(sys.argv) > 1:
        task_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    # Load tasks
    print(f"Loading tasks from: {task_file}")
    with open(task_file, 'r') as f:
        tasks_data = yaml.safe_load(f)

    tasks = [TaskDefinition(**task_dict) for task_dict in tasks_data["tasks"]]
    print(f"Loaded {len(tasks)} tasks")

    # Generate runs
    generator = SyntheticRunGenerator(
        tasks=tasks,
        n_novices_per_task=50,
        max_attempts=10,
        ai_models=["baseline-weak", "baseline-medium", "baseline-strong"],
        random_seed=42
    )

    runs = generator.generate_all_runs()

    # Print summary
    summary = generator.get_summary_statistics(runs)
    print("\nRun Generation Summary:")
    print(f"  Total runs: {summary['total_runs']}")
    print(f"  Tasks: {summary['n_tasks']}")
    print(f"\n  Novice learners:")
    print(f"    Total runs: {summary['novice_stats']['total_runs']}")
    print(f"    Unique learners: {summary['novice_stats']['unique_learners']}")
    print(f"    Overall success rate: {summary['novice_stats']['mean_success_rate']:.3f}")
    print(f"    First-attempt success rate: {summary['novice_stats']['first_attempt_success_rate']:.3f}")
    print(f"\n  AI models:")
    for model, stats in summary['ai_stats'].items():
        print(f"    {model}:")
        print(f"      Runs: {stats['total_runs']}")
        print(f"      Success rate: {stats['mean_success_rate']:.3f}")

    print(f"\n  Runs by domain:")
    for domain, count in summary['runs_by_domain'].items():
        print(f"    {domain}: {count}")

    # Save to file
    generator.save_runs(runs, output_file)
    print(f"\nRuns saved to: {output_file}")
