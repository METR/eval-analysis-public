"""
NAME (Novelty Achievement MeasurE) metric calculation.

Calculate N-attempt equivalence horizon for AI models by comparing:
- AI zero-shot performance across task difficulties
- Human novice learning curves (attempts → success rate)
"""

import json
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import pathlib


@dataclass
class LearningCurvefit:
    """Fitted learning curve for a single task."""

    task_id: str
    first_attempt_success_rate: float
    fitted_lambda: float  # Learning rate parameter
    fitted_asymptote: float  # Maximum achievable success rate
    n_for_50pct: float  # Number of attempts to reach 50% success
    n_for_80pct: float  # Number of attempts to reach 80% success
    rmse: float  # Root mean squared error of fit
    n_novices: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class NAMEHorizon:
    """N-attempt equivalence horizon for an AI model."""

    model: str
    n_attempt_horizon_50: float  # N-attempt equivalence at 50% AI success
    n_attempt_horizon_80: float  # N-attempt equivalence at 80% AI success
    ai_success_rate_overall: float
    difficulty_at_50pct: float  # Task difficulty where AI reaches 50%
    difficulty_at_80pct: float  # Task difficulty where AI reaches 80%
    n_tasks: int

    def to_dict(self) -> dict:
        return asdict(self)


class NAMEMetricCalculator:
    """Calculate N-attempt equivalence horizon for AI models."""

    def __init__(self, runs_path: str):
        """
        Initialize calculator with run data.

        Args:
            runs_path: Path to name_runs.jsonl file
        """
        self.runs_path = runs_path
        self.df_runs = self._load_runs()
        self.df_novice = self.df_runs[self.df_runs["learner_type"] == "human_novice"]
        self.df_ai = self.df_runs[self.df_runs["learner_type"] == "ai_zero_shot"]

    def _load_runs(self) -> pd.DataFrame:
        """Load runs from JSONL file."""
        runs = []
        with open(self.runs_path, 'r') as f:
            for line in f:
                runs.append(json.loads(line))
        return pd.DataFrame(runs)

    def _exponential_learning_model(
        self,
        n: np.ndarray,
        base_rate: float,
        learning_gain: float,
        lambda_param: float
    ) -> np.ndarray:
        """
        Exponential learning model (without noise).

        P(success | n) = base_rate + learning_gain * (1 - exp(-λn))

        Args:
            n: Attempt numbers
            base_rate: Initial success rate
            learning_gain: Maximum learnable improvement
            lambda_param: Learning speed

        Returns:
            Success probabilities
        """
        return base_rate + learning_gain * (1 - np.exp(-lambda_param * n))

    def fit_task_learning_curve(self, task_id: str) -> Optional[LearningCurvefit]:
        """
        Fit learning curve for a single task using novice data.

        Args:
            task_id: Task identifier

        Returns:
            LearningCurveFit or None if fitting fails
        """
        # Get novice data for this task
        task_data = self.df_novice[self.df_novice["task_id"] == task_id]

        if len(task_data) == 0:
            return None

        # Calculate empirical success rate by attempt
        attempt_success = task_data.groupby("attempt_number")["score_binarized"].agg(["mean", "count"])

        if len(attempt_success) < 3:  # Need at least 3 points to fit
            return None

        attempts = attempt_success.index.values
        success_rates = attempt_success["mean"].values

        # Fit exponential learning model
        try:
            # Initial parameter guesses
            p0 = [
                success_rates[0],  # base_rate
                max(0.5, 1.0 - success_rates[0]),  # learning_gain
                0.4  # lambda
            ]

            # Bounds: base_rate [0,1], learning_gain [0,1], lambda [0.1, 1.0]
            bounds = ([0, 0, 0.1], [1, 1, 1.0])

            popt, _ = curve_fit(
                self._exponential_learning_model,
                attempts,
                success_rates,
                p0=p0,
                bounds=bounds,
                maxfev=10000
            )

            base_rate, learning_gain, lambda_param = popt

            # Calculate predictions and RMSE
            predicted = self._exponential_learning_model(attempts, *popt)
            rmse = np.sqrt(np.mean((success_rates - predicted) ** 2))

            # Calculate asymptote (maximum achievable success rate)
            asymptote = base_rate + learning_gain

            # Calculate N for 50% and 80% success
            # Solve: target = base_rate + learning_gain * (1 - exp(-λN))
            # => N = -log((asymptote - target) / learning_gain) / λ

            def attempts_for_success_rate(target: float) -> float:
                if target <= base_rate:
                    return 1.0  # Already at or above target
                if target >= asymptote:
                    return np.inf  # Can't reach target
                return -np.log((asymptote - target) / learning_gain) / lambda_param

            n_for_50pct = attempts_for_success_rate(0.5)
            n_for_80pct = attempts_for_success_rate(0.8)

            return LearningCurvefit(
                task_id=task_id,
                first_attempt_success_rate=float(success_rates[0]),
                fitted_lambda=float(lambda_param),
                fitted_asymptote=float(asymptote),
                n_for_50pct=float(n_for_50pct),
                n_for_80pct=float(n_for_80pct),
                rmse=float(rmse),
                n_novices=int(task_data["run_id"].apply(lambda x: x.split("_")[1]).nunique())
            )

        except Exception as e:
            print(f"Warning: Could not fit learning curve for {task_id}: {e}")
            return None

    def fit_all_learning_curves(self) -> pd.DataFrame:
        """
        Fit learning curves for all tasks.

        Returns:
            DataFrame of LearningCurveFit objects
        """
        task_ids = self.df_novice["task_id"].unique()

        fits = []
        for task_id in task_ids:
            fit = self.fit_task_learning_curve(task_id)
            if fit is not None:
                fits.append(fit.to_dict())

        return pd.DataFrame(fits)

    def calculate_ai_performance_by_difficulty(self, model: str) -> pd.DataFrame:
        """
        Calculate AI success rate binned by task difficulty.

        Args:
            model: AI model name

        Returns:
            DataFrame with difficulty bins and success rates
        """
        model_data = self.df_ai[self.df_ai["alias"] == model]

        # Group by task and calculate success rate
        task_performance = model_data.groupby("task_id").agg({
            "score_binarized": "mean",
            "first_attempt_success_rate": "first"
        }).reset_index()

        # Create difficulty bins
        difficulty_bins = np.linspace(0, 1, 11)  # 10 bins
        task_performance["difficulty_bin"] = pd.cut(
            task_performance["first_attempt_success_rate"],
            bins=difficulty_bins,
            labels=[f"{difficulty_bins[i]:.1f}-{difficulty_bins[i+1]:.1f}"
                   for i in range(len(difficulty_bins)-1)]
        )

        # Calculate mean success rate per bin
        binned_performance = task_performance.groupby("difficulty_bin").agg({
            "score_binarized": ["mean", "count"],
            "first_attempt_success_rate": "mean"
        }).reset_index()

        binned_performance.columns = ["difficulty_bin", "success_rate", "n_tasks", "mean_difficulty"]

        return binned_performance

    def calculate_n_attempt_horizon(
        self,
        model: str,
        learning_curves_df: pd.DataFrame,
        success_threshold: float = 0.5
    ) -> NAMEHorizon:
        """
        Calculate N-attempt equivalence horizon for an AI model.

        Process:
        1. For each difficulty level, find N where novices reach success_threshold
        2. Find difficulty level where AI reaches success_threshold
        3. Return N at that difficulty level

        Args:
            model: AI model name
            learning_curves_df: DataFrame of fitted learning curves
            success_threshold: Success rate threshold (0.5 or 0.8)

        Returns:
            NAMEHorizon object
        """
        # Get AI performance by task
        model_data = self.df_ai[self.df_ai["alias"] == model]
        ai_task_performance = model_data.groupby("task_id")["score_binarized"].mean()

        # Merge with learning curves
        merged = learning_curves_df.copy()
        merged["ai_success"] = merged["task_id"].map(ai_task_performance)
        merged = merged.dropna()

        # Sort by difficulty
        merged = merged.sort_values("first_attempt_success_rate")

        # Find difficulty where AI reaches each threshold (50% and 80%)
        # Use interpolation
        difficulties = merged["first_attempt_success_rate"].values
        ai_success_rates = merged["ai_success"].values

        # Create interpolation function (AI success vs difficulty)
        if len(difficulties) < 2:
            # Not enough data
            return NAMEHorizon(
                model=model,
                n_attempt_horizon_50=np.nan,
                n_attempt_horizon_80=np.nan,
                ai_success_rate_overall=ai_success_rates.mean() if len(ai_success_rates) > 0 else np.nan,
                difficulty_at_50pct=np.nan,
                difficulty_at_80pct=np.nan,
                n_tasks=len(merged)
            )

        # Interpolate to find difficulty at 50% and 80% AI success
        try:
            interp_func = interp1d(
                ai_success_rates,
                difficulties,
                bounds_error=False,
                fill_value="extrapolate"
            )

            difficulty_at_50 = float(interp_func(0.5))
            difficulty_at_80 = float(interp_func(0.8))

            # Clip to valid range
            difficulty_at_50 = np.clip(difficulty_at_50, 0, 1)
            difficulty_at_80 = np.clip(difficulty_at_80, 0, 1)

            # Find N-attempts at these difficulties
            # Use another interpolation: N vs difficulty
            n_50_values = merged["n_for_50pct"].values
            n_80_values = merged["n_for_80pct"].values

            # Filter out inf values
            valid_50 = np.isfinite(n_50_values)
            valid_80 = np.isfinite(n_80_values)

            if valid_50.sum() >= 2:
                interp_n50 = interp1d(
                    difficulties[valid_50],
                    n_50_values[valid_50],
                    bounds_error=False,
                    fill_value="extrapolate"
                )
                n_attempt_horizon_50 = float(interp_n50(difficulty_at_50))
            else:
                n_attempt_horizon_50 = np.nan

            if valid_80.sum() >= 2:
                interp_n80 = interp1d(
                    difficulties[valid_80],
                    n_80_values[valid_80],
                    bounds_error=False,
                    fill_value="extrapolate"
                )
                n_attempt_horizon_80 = float(interp_n80(difficulty_at_80))
            else:
                n_attempt_horizon_80 = np.nan

        except Exception as e:
            print(f"Warning: Could not calculate horizon for {model}: {e}")
            n_attempt_horizon_50 = np.nan
            n_attempt_horizon_80 = np.nan
            difficulty_at_50 = np.nan
            difficulty_at_80 = np.nan

        return NAMEHorizon(
            model=model,
            n_attempt_horizon_50=n_attempt_horizon_50,
            n_attempt_horizon_80=n_attempt_horizon_80,
            ai_success_rate_overall=float(ai_success_rates.mean()),
            difficulty_at_50pct=difficulty_at_50,
            difficulty_at_80pct=difficulty_at_80,
            n_tasks=len(merged)
        )

    def calculate_all_horizons(self) -> pd.DataFrame:
        """
        Calculate N-attempt horizons for all AI models.

        Returns:
            DataFrame of NAMEHorizon objects
        """
        # First fit all learning curves
        print("Fitting learning curves for all tasks...")
        learning_curves_df = self.fit_all_learning_curves()
        print(f"Successfully fit {len(learning_curves_df)} learning curves")

        # Get all AI models
        ai_models = self.df_ai["alias"].unique()

        horizons = []
        for model in ai_models:
            print(f"Calculating horizon for {model}...")
            horizon = self.calculate_n_attempt_horizon(model, learning_curves_df)
            horizons.append(horizon.to_dict())

        return pd.DataFrame(horizons)

    def save_results(
        self,
        learning_curves_df: pd.DataFrame,
        horizons_df: pd.DataFrame,
        output_dir: str = "data/wrangled"
    ):
        """
        Save learning curves and horizons to CSV files.

        Args:
            learning_curves_df: DataFrame of learning curves
            horizons_df: DataFrame of N-attempt horizons
            output_dir: Output directory
        """
        output_path = pathlib.Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        learning_curves_df.to_csv(output_path / "name_learning_curves.csv", index=False)
        horizons_df.to_csv(output_path / "name_horizons.csv", index=False)

        print(f"\nResults saved to:")
        print(f"  {output_path / 'name_learning_curves.csv'}")
        print(f"  {output_path / 'name_horizons.csv'}")


# Example usage and CLI
if __name__ == "__main__":
    import sys

    # Default parameters
    runs_file = "data/external/name_runs.jsonl"
    output_dir = "data/wrangled"

    # Parse command line args
    if len(sys.argv) > 1:
        runs_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]

    print(f"Loading runs from: {runs_file}")
    calculator = NAMEMetricCalculator(runs_file)

    print(f"\nDataset summary:")
    print(f"  Total runs: {len(calculator.df_runs)}")
    print(f"  Novice runs: {len(calculator.df_novice)}")
    print(f"  AI runs: {len(calculator.df_ai)}")
    print(f"  Tasks: {calculator.df_runs['task_id'].nunique()}")
    print(f"  AI models: {calculator.df_ai['alias'].unique().tolist()}")

    # Fit learning curves
    learning_curves_df = calculator.fit_all_learning_curves()

    print(f"\nLearning curve summary:")
    print(f"  Mean first-attempt success: {learning_curves_df['first_attempt_success_rate'].mean():.3f}")
    print(f"  Mean fitted asymptote: {learning_curves_df['fitted_asymptote'].mean():.3f}")
    print(f"  Mean learning rate (λ): {learning_curves_df['fitted_lambda'].mean():.3f}")
    print(f"  Mean RMSE: {learning_curves_df['rmse'].mean():.4f}")

    # Calculate horizons
    horizons_df = calculator.calculate_all_horizons()

    print(f"\nNAME Horizons:")
    for _, row in horizons_df.iterrows():
        print(f"\n  {row['model']}:")
        print(f"    N-attempt horizon (50%): {row['n_attempt_horizon_50']:.2f} attempts")
        print(f"    N-attempt horizon (80%): {row['n_attempt_horizon_80']:.2f} attempts")
        print(f"    Overall AI success rate: {row['ai_success_rate_overall']:.3f}")
        print(f"    Difficulty at 50% AI success: {row['difficulty_at_50pct']:.3f}")

    # Save results
    calculator.save_results(learning_curves_df, horizons_df, output_dir)
