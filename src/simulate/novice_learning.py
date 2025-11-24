"""
Novice learning simulator for NAME (Novelty Achievement MeasurE) metric.

Implements a noisy exponential learning model that captures:
- Semi-random starting point (base_rate + initial noise)
- Turbulent improvement in early attempts
- Steady, shallow improvement after practice
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from dataclasses import dataclass


@dataclass
class LearningParameters:
    """Parameters for the exponential learning model."""

    base_rate: float  # First-attempt success probability (task difficulty)
    learning_gain: float  # Maximum learnable improvement
    lambda_learning: float  # Learning speed parameter
    noise_std_initial: float  # Initial noise standard deviation

    @classmethod
    def from_task_difficulty(
        cls,
        difficulty: float,
        lambda_learning: float = 0.4,
        noise_std_initial: float = 0.1,
        max_learning_gain: float = 0.8
    ) -> "LearningParameters":
        """
        Create learning parameters from task difficulty.

        Args:
            difficulty: First-attempt success rate (0.0-1.0)
            lambda_learning: Learning speed (default: 0.4)
            noise_std_initial: Initial noise level (default: 0.1)
            max_learning_gain: Maximum improvement possible (default: 0.8)

        Returns:
            LearningParameters configured for the task
        """
        # Harder tasks (low base_rate) have more room for improvement
        learning_gain = (1.0 - difficulty) * max_learning_gain

        return cls(
            base_rate=difficulty,
            learning_gain=learning_gain,
            lambda_learning=lambda_learning,
            noise_std_initial=noise_std_initial
        )


class NoviceLearningSimulator:
    """
    Simulate novice learning curves with exponential improvement + noise.

    Learning model:
        P(success | attempt n) = base_rate + learning_gain * (1 - exp(-λn)) + noise(n)

    where noise(n) ~ N(0, σ₀/√n) decreases as learning stabilizes.
    """

    def __init__(
        self,
        params: LearningParameters,
        random_seed: Optional[int] = None
    ):
        """
        Initialize simulator with learning parameters.

        Args:
            params: Learning parameters for the task
            random_seed: Random seed for reproducibility (optional)
        """
        self.params = params
        self.rng = np.random.default_rng(random_seed)

    def probability_success(self, attempt_number: int) -> float:
        """
        Calculate probability of success on a given attempt.

        Args:
            attempt_number: 1-indexed attempt number

        Returns:
            Probability of success (0.0-1.0)
        """
        if attempt_number < 1:
            raise ValueError("Attempt number must be >= 1")

        # Core exponential learning
        learning_term = self.params.learning_gain * (
            1 - np.exp(-self.params.lambda_learning * attempt_number)
        )

        # Decreasing noise (turbulent early, stable later)
        noise_std = self.params.noise_std_initial / np.sqrt(attempt_number)
        noise = self.rng.normal(0, noise_std)

        # Clip to valid probability range
        prob = self.params.base_rate + learning_term + noise
        return np.clip(prob, 0.0, 1.0)

    def simulate_attempt(self, attempt_number: int) -> bool:
        """
        Simulate a single attempt.

        Args:
            attempt_number: 1-indexed attempt number

        Returns:
            True if successful, False otherwise
        """
        prob = self.probability_success(attempt_number)
        return self.rng.random() < prob

    def simulate_learner_trajectory(
        self,
        max_attempts: int = 10,
        stop_on_success: bool = False
    ) -> List[Dict[str, any]]:
        """
        Simulate a single learner's trajectory over multiple attempts.

        Args:
            max_attempts: Maximum number of attempts
            stop_on_success: Stop after first success (default: False)

        Returns:
            List of attempt records with keys: attempt_number, success, probability
        """
        trajectory = []

        for attempt_num in range(1, max_attempts + 1):
            success = self.simulate_attempt(attempt_num)
            prob = self.probability_success(attempt_num)

            trajectory.append({
                "attempt_number": attempt_num,
                "success": success,
                "probability": prob
            })

            if stop_on_success and success:
                break

        return trajectory

    def simulate_multiple_learners(
        self,
        n_learners: int,
        max_attempts: int = 10,
        stop_on_success: bool = False
    ) -> pd.DataFrame:
        """
        Simulate multiple learners attempting the same task.

        Args:
            n_learners: Number of learners to simulate
            max_attempts: Maximum attempts per learner
            stop_on_success: Stop each learner after first success

        Returns:
            DataFrame with columns: learner_id, attempt_number, success, probability
        """
        all_data = []

        for learner_id in range(n_learners):
            trajectory = self.simulate_learner_trajectory(
                max_attempts=max_attempts,
                stop_on_success=stop_on_success
            )

            for attempt_data in trajectory:
                all_data.append({
                    "learner_id": learner_id,
                    **attempt_data
                })

        return pd.DataFrame(all_data)

    def calculate_expected_learning_curve(
        self,
        max_attempts: int = 10
    ) -> pd.DataFrame:
        """
        Calculate expected (noise-free) learning curve.

        Useful for visualization and analysis without stochastic noise.

        Args:
            max_attempts: Number of attempts to calculate

        Returns:
            DataFrame with columns: attempt_number, success_rate
        """
        attempts = np.arange(1, max_attempts + 1)

        # Calculate learning curve without noise
        learning_terms = self.params.learning_gain * (
            1 - np.exp(-self.params.lambda_learning * attempts)
        )
        success_rates = self.params.base_rate + learning_terms
        success_rates = np.clip(success_rates, 0.0, 1.0)

        return pd.DataFrame({
            "attempt_number": attempts,
            "success_rate": success_rates
        })


class AIZeroShotSimulator:
    """
    Simulate AI zero-shot performance on tasks.

    Maps AI capability level to success probability on tasks of varying difficulty.
    """

    def __init__(
        self,
        capability: float,
        steepness: float = 10.0,
        random_seed: Optional[int] = None
    ):
        """
        Initialize AI simulator.

        Args:
            capability: AI's intrinsic capability level (0.0-1.0)
                       Higher capability = can handle harder tasks
            steepness: Steepness of logistic transition (default: 10.0)
            random_seed: Random seed for reproducibility
        """
        if not 0.0 <= capability <= 1.0:
            raise ValueError("Capability must be between 0.0 and 1.0")

        self.capability = capability
        self.steepness = steepness
        self.rng = np.random.default_rng(random_seed)

    def probability_success(self, task_difficulty: float) -> float:
        """
        Calculate probability of zero-shot success on a task.

        Uses logistic function: P(success) = 1 / (1 + exp(-k*(capability - difficulty)))

        Args:
            task_difficulty: Task's first-attempt success rate for novices

        Returns:
            Probability of success (0.0-1.0)
        """
        # Logistic function centered at capability level
        z = self.steepness * (self.capability - task_difficulty)
        prob = 1.0 / (1.0 + np.exp(-z))
        return prob

    def simulate_attempt(self, task_difficulty: float) -> bool:
        """
        Simulate a single zero-shot attempt.

        Args:
            task_difficulty: Task's difficulty level

        Returns:
            True if successful, False otherwise
        """
        prob = self.probability_success(task_difficulty)
        return self.rng.random() < prob

    def simulate_multiple_tasks(
        self,
        task_difficulties: List[float],
        attempts_per_task: int = 1
    ) -> pd.DataFrame:
        """
        Simulate AI attempts on multiple tasks.

        Args:
            task_difficulties: List of task difficulty levels
            attempts_per_task: Number of attempts per task (default: 1 for zero-shot)

        Returns:
            DataFrame with columns: task_idx, attempt_number, task_difficulty,
                                   success, probability
        """
        all_data = []

        for task_idx, difficulty in enumerate(task_difficulties):
            prob = self.probability_success(difficulty)

            for attempt_num in range(1, attempts_per_task + 1):
                success = self.simulate_attempt(difficulty)

                all_data.append({
                    "task_idx": task_idx,
                    "attempt_number": attempt_num,
                    "task_difficulty": difficulty,
                    "success": success,
                    "probability": prob
                })

        return pd.DataFrame(all_data)


def create_ai_model(
    model_name: str,
    random_seed: Optional[int] = None
) -> AIZeroShotSimulator:
    """
    Factory function to create AI models with predefined capability levels.

    Args:
        model_name: Name of the model ("weak", "medium", "strong", or custom)
        random_seed: Random seed for reproducibility

    Returns:
        AIZeroShotSimulator configured for the model
    """
    # Predefined capability levels
    CAPABILITY_MAP = {
        "baseline-weak": 0.3,
        "baseline-medium": 0.5,
        "baseline-strong": 0.7,
    }

    # Allow custom capability specification: "custom-0.65"
    if model_name.startswith("custom-"):
        capability = float(model_name.split("-")[1])
    else:
        capability = CAPABILITY_MAP.get(
            model_name.lower(),
            0.5  # default to medium
        )

    return AIZeroShotSimulator(capability=capability, random_seed=random_seed)


# Example usage
if __name__ == "__main__":
    # Example: Simulate novice learning on a moderately difficult task
    params = LearningParameters.from_task_difficulty(difficulty=0.35)
    simulator = NoviceLearningSimulator(params, random_seed=42)

    # Simulate 50 learners
    df_learning = simulator.simulate_multiple_learners(
        n_learners=50,
        max_attempts=10
    )

    print("Novice Learning Simulation:")
    print(df_learning.groupby("attempt_number")["success"].agg(["mean", "count"]))

    # Example: Simulate AI zero-shot performance
    ai = create_ai_model("baseline-medium", random_seed=42)
    difficulties = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    df_ai = ai.simulate_multiple_tasks(difficulties, attempts_per_task=20)

    print("\nAI Zero-Shot Simulation:")
    print(df_ai.groupby("task_difficulty")["success"].agg(["mean", "count"]))
