"""
Synthetic task generator for NAME (Novelty Achievement MeasurE) metric.

Generates synthetic tasks with calibrated difficulty levels across multiple domains.
"""

import yaml
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import numpy as np


# Task domains for MVP
# Mix of verifiable and non-verifiable tasks
TASK_DOMAINS = [
    "logic_puzzle",       # verifiable
    "code_completion",    # verifiable
    "pattern_recognition",# verifiable
    "creative_writing",   # non-verifiable
    "communication_tasks" # non-verifiable (replaces data_analysis)
]


@dataclass
class TaskDefinition:
    """Definition of a synthetic task."""

    task_id: str
    task_family: str  # domain name
    domain: str
    description: str
    first_attempt_success_rate: float  # Difficulty: novice first-attempt success
    learning_rate_lambda: float  # Learning speed parameter
    max_attempts: int  # Maximum attempts allowed
    verification_type: str  # automated | llm_judge | expert_review
    is_verifiable: bool  # True if automated scoring possible

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


class SyntheticTaskGenerator:
    """
    Generate synthetic tasks with parameterized difficulty.

    Tasks span a range of difficulty levels (first-attempt success rates)
    and cover multiple domains.
    """

    def __init__(
        self,
        n_tasks: int = 100,
        difficulty_range: tuple = (0.05, 0.8),
        domains: Optional[List[str]] = None,
        default_lambda: float = 0.4,
        max_attempts: int = 10,
        random_seed: Optional[int] = None
    ):
        """
        Initialize task generator.

        Args:
            n_tasks: Total number of tasks to generate
            difficulty_range: (min, max) first-attempt success rates
            domains: List of task domains (default: TASK_DOMAINS)
            default_lambda: Default learning rate parameter
            max_attempts: Max attempts per task
            random_seed: Random seed for reproducibility
        """
        self.n_tasks = n_tasks
        self.difficulty_range = difficulty_range
        self.domains = domains or TASK_DOMAINS
        self.default_lambda = default_lambda
        self.max_attempts = max_attempts
        self.rng = np.random.default_rng(random_seed)

        # Map domains to verification types
        self.domain_verification = {
            "logic_puzzle": ("automated", True),
            "code_completion": ("automated", True),
            "pattern_recognition": ("automated", True),
            "creative_writing": ("llm_judge", False),
            "communication_tasks": ("llm_judge", False)
        }

    def _generate_difficulty_levels(self) -> np.ndarray:
        """
        Generate difficulty levels (first-attempt success rates).

        Distributes tasks across difficulty spectrum with slight bias
        toward mid-range difficulties (more realistic).

        Returns:
            Array of difficulty levels
        """
        # Use beta distribution for slightly more realistic distribution
        # Most tasks are mid-difficulty, fewer very easy or very hard
        min_diff, max_diff = self.difficulty_range

        # Beta(2, 2) gives bell curve centered at 0.5
        beta_samples = self.rng.beta(2, 2, size=self.n_tasks)

        # Scale to difficulty range
        difficulties = min_diff + (max_diff - min_diff) * beta_samples

        return difficulties

    def _sample_learning_rate(self, difficulty: float) -> float:
        """
        Sample learning rate with some variation based on difficulty.

        Harder tasks tend to have slightly slower learning (lower lambda),
        but there's significant variation.

        Args:
            difficulty: First-attempt success rate

        Returns:
            Learning rate lambda
        """
        # Base learning rate inversely related to difficulty (harder = slower learning)
        base_lambda = self.default_lambda * (0.7 + 0.6 * difficulty)

        # Add noise
        noise = self.rng.normal(0, 0.05)
        lambda_value = base_lambda + noise

        # Clip to reasonable range [0.2, 0.6]
        return np.clip(lambda_value, 0.2, 0.6)

    def _generate_task_description(
        self,
        domain: str,
        difficulty: float,
        task_idx: int
    ) -> str:
        """
        Generate a placeholder task description.

        In real implementation, this would describe actual task content.

        Args:
            domain: Task domain
            difficulty: First-attempt success rate
            task_idx: Task index for unique ID

        Returns:
            Task description string
        """
        difficulty_label = self._difficulty_to_label(difficulty)

        descriptions = {
            "logic_puzzle": f"Solve a {difficulty_label} logic puzzle",
            "code_completion": f"Complete a {difficulty_label} code implementation",
            "pattern_recognition": f"Identify pattern in {difficulty_label} sequence",
            "creative_writing": f"Write a {difficulty_label} creative piece",
            "communication_tasks": f"Draft a {difficulty_label} communication"
        }

        return descriptions.get(domain, f"{difficulty_label} task in {domain}")

    def _difficulty_to_label(self, difficulty: float) -> str:
        """Convert difficulty number to human-readable label."""
        if difficulty < 0.2:
            return "very hard"
        elif difficulty < 0.4:
            return "hard"
        elif difficulty < 0.6:
            return "moderate"
        elif difficulty < 0.75:
            return "easy"
        else:
            return "very easy"

    def generate_task(
        self,
        difficulty: float,
        domain: str,
        task_idx: int
    ) -> TaskDefinition:
        """
        Generate a single task with specified difficulty and domain.

        Args:
            difficulty: First-attempt success rate (0.0-1.0)
            domain: Task domain
            task_idx: Task index for unique ID

        Returns:
            TaskDefinition
        """
        # Generate learning rate
        lambda_value = self._sample_learning_rate(difficulty)

        # Get verification type
        verification_type, is_verifiable = self.domain_verification[domain]

        # Create task ID
        task_id = f"{domain}/task_{task_idx:04d}"

        # Generate description
        description = self._generate_task_description(domain, difficulty, task_idx)

        return TaskDefinition(
            task_id=task_id,
            task_family=domain,
            domain=domain,
            description=description,
            first_attempt_success_rate=float(round(difficulty, 4)),
            learning_rate_lambda=float(round(lambda_value, 4)),
            max_attempts=int(self.max_attempts),
            verification_type=verification_type,
            is_verifiable=bool(is_verifiable)
        )

    def generate_task_suite(self) -> List[TaskDefinition]:
        """
        Generate complete suite of synthetic tasks.

        Tasks are balanced across domains and difficulty levels.

        Returns:
            List of TaskDefinition objects
        """
        tasks = []

        # Generate difficulty levels
        difficulties = self._generate_difficulty_levels()

        # Assign domains (balanced distribution)
        domain_assignments = [
            self.domains[i % len(self.domains)]
            for i in range(self.n_tasks)
        ]

        # Shuffle domain assignments for variety
        self.rng.shuffle(domain_assignments)

        # Generate tasks
        for idx, (difficulty, domain) in enumerate(zip(difficulties, domain_assignments)):
            task = self.generate_task(
                difficulty=difficulty,
                domain=domain,
                task_idx=idx
            )
            tasks.append(task)

        return tasks

    def save_tasks(self, tasks: List[TaskDefinition], output_path: str):
        """
        Save task suite to YAML file.

        Args:
            tasks: List of TaskDefinition objects
            output_path: Path to output YAML file
        """
        tasks_dict = {
            "tasks": [task.to_dict() for task in tasks],
            "metadata": {
                "n_tasks": len(tasks),
                "difficulty_range": list(self.difficulty_range),
                "domains": self.domains,
                "default_lambda": self.default_lambda,
                "max_attempts": self.max_attempts
            }
        }

        with open(output_path, 'w') as f:
            yaml.dump(tasks_dict, f, default_flow_style=False, sort_keys=False)

    def get_summary_statistics(self, tasks: List[TaskDefinition]) -> dict:
        """
        Calculate summary statistics for generated tasks.

        Args:
            tasks: List of TaskDefinition objects

        Returns:
            Dictionary of summary statistics
        """
        difficulties = [t.first_attempt_success_rate for t in tasks]
        lambdas = [t.learning_rate_lambda for t in tasks]

        # Count by domain
        domain_counts = {}
        for domain in self.domains:
            domain_counts[domain] = sum(1 for t in tasks if t.domain == domain)

        # Verification type counts
        verification_counts = {}
        for task in tasks:
            v_type = task.verification_type
            verification_counts[v_type] = verification_counts.get(v_type, 0) + 1

        return {
            "total_tasks": len(tasks),
            "difficulty": {
                "mean": float(np.mean(difficulties)),
                "std": float(np.std(difficulties)),
                "min": float(np.min(difficulties)),
                "max": float(np.max(difficulties)),
                "median": float(np.median(difficulties))
            },
            "learning_rate": {
                "mean": float(np.mean(lambdas)),
                "std": float(np.std(lambdas)),
                "min": float(np.min(lambdas)),
                "max": float(np.max(lambdas))
            },
            "domains": domain_counts,
            "verification_types": verification_counts,
            "verifiable_tasks": sum(1 for t in tasks if t.is_verifiable),
            "non_verifiable_tasks": sum(1 for t in tasks if not t.is_verifiable)
        }


# Example usage and CLI
if __name__ == "__main__":
    import sys
    import pathlib

    # Default parameters
    n_tasks = 100
    output_path = "data/external/name_tasks.yaml"

    # Parse command line args if provided
    if len(sys.argv) > 1:
        n_tasks = int(sys.argv[1])
    if len(sys.argv) > 2:
        output_path = sys.argv[2]

    # Generate tasks
    print(f"Generating {n_tasks} synthetic tasks...")
    generator = SyntheticTaskGenerator(
        n_tasks=n_tasks,
        difficulty_range=(0.05, 0.8),
        default_lambda=0.4,
        max_attempts=10,
        random_seed=42
    )

    tasks = generator.generate_task_suite()

    # Print summary
    summary = generator.get_summary_statistics(tasks)
    print("\nTask Suite Summary:")
    print(f"  Total tasks: {summary['total_tasks']}")
    print(f"  Difficulty range: {summary['difficulty']['min']:.3f} - {summary['difficulty']['max']:.3f}")
    print(f"  Mean difficulty: {summary['difficulty']['mean']:.3f} ± {summary['difficulty']['std']:.3f}")
    print(f"  Mean learning rate: {summary['learning_rate']['mean']:.3f} ± {summary['learning_rate']['std']:.3f}")
    print(f"\n  Tasks by domain:")
    for domain, count in summary['domains'].items():
        print(f"    {domain}: {count}")
    print(f"\n  Verification types:")
    for v_type, count in summary['verification_types'].items():
        print(f"    {v_type}: {count}")
    print(f"\n  Verifiable: {summary['verifiable_tasks']}")
    print(f"  Non-verifiable: {summary['non_verifiable_tasks']}")

    # Save to file
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    generator.save_tasks(tasks, output_path)
    print(f"\nTasks saved to: {output_path}")

    # Show a few example tasks
    print(f"\nExample tasks:")
    for i in [0, len(tasks)//2, -1]:
        task = tasks[i]
        print(f"\n  {task.task_id}")
        print(f"    Domain: {task.domain}")
        print(f"    Difficulty: {task.first_attempt_success_rate}")
        print(f"    Learning rate: {task.learning_rate_lambda}")
        print(f"    Verification: {task.verification_type}")
        print(f"    Description: {task.description}")
