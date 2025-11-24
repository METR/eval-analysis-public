"""
Visualization tools for NAME (Novelty Achievement MeasurE) metric.

Creates plots for learning curves, AI vs human comparison, and N-attempt horizons.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from typing import Optional, List
import json


class NAMEPlotter:
    """Create visualizations for NAME metric analysis."""

    def __init__(
        self,
        learning_curves_path: str,
        horizons_path: str,
        runs_path: str,
        output_dir: str = "plots"
    ):
        """
        Initialize plotter with data paths.

        Args:
            learning_curves_path: Path to name_learning_curves.csv
            horizons_path: Path to name_horizons.csv
            runs_path: Path to name_runs.jsonl
            output_dir: Directory for output plots
        """
        self.learning_curves_df = pd.read_csv(learning_curves_path)
        self.horizons_df = pd.read_csv(horizons_path)
        self.runs_df = self._load_runs(runs_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Color scheme
        self.colors = {
            "baseline-weak": "#d62728",  # red
            "baseline-medium": "#ff7f0e",  # orange
            "baseline-strong": "#2ca02c",  # green
            "human_novice": "#1f77b4"  # blue
        }

    def _load_runs(self, runs_path: str) -> pd.DataFrame:
        """Load runs from JSONL."""
        runs = []
        with open(runs_path, 'r') as f:
            for line in f:
                runs.append(json.loads(line))
        return pd.DataFrame(runs)

    def plot_learning_curves_sample(
        self,
        n_tasks: int = 6,
        filename: str = "name_learning_curves_sample.png"
    ):
        """
        Plot learning curves for a sample of tasks.

        Args:
            n_tasks: Number of tasks to plot
            filename: Output filename
        """
        # Select tasks with diverse difficulties
        sorted_tasks = self.learning_curves_df.sort_values("first_attempt_success_rate")
        indices = np.linspace(0, len(sorted_tasks) - 1, n_tasks, dtype=int)
        sample_tasks = sorted_tasks.iloc[indices]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, (_, task_data) in enumerate(sample_tasks.iterrows()):
            ax = axes[idx]

            task_id = task_data["task_id"]
            task_runs = self.runs_df[
                (self.runs_df["task_id"] == task_id) &
                (self.runs_df["learner_type"] == "human_novice")
            ]

            # Calculate empirical success rate
            empirical = task_runs.groupby("attempt_number")["score_binarized"].mean()

            # Plot empirical data
            ax.plot(
                empirical.index,
                empirical.values,
                'o-',
                color=self.colors["human_novice"],
                label="Empirical",
                markersize=6,
                linewidth=2
            )

            # Plot fitted curve
            attempts = np.linspace(1, 10, 100)
            base_rate = empirical.values[0]  # First attempt
            lambda_param = task_data["fitted_lambda"]
            asymptote = task_data["fitted_asymptote"]
            learning_gain = asymptote - base_rate

            fitted = base_rate + learning_gain * (1 - np.exp(-lambda_param * attempts))

            ax.plot(
                attempts,
                fitted,
                '--',
                color=self.colors["human_novice"],
                alpha=0.7,
                label="Fitted",
                linewidth=2
            )

            # Styling
            ax.set_xlabel("Attempt Number", fontsize=10)
            ax.set_ylabel("Success Rate", fontsize=10)
            ax.set_title(
                f"{task_id}\n(Difficulty: {task_data['first_attempt_success_rate']:.2f}, λ: {lambda_param:.2f})",
                fontsize=9
            )
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(0.5, 10.5)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        plt.suptitle("Novice Learning Curves (Sample Tasks)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved: {self.output_dir / filename}")

    def plot_ai_vs_human_first_attempt(
        self,
        filename: str = "name_ai_vs_human_first_attempt.png"
    ):
        """
        Scatter plot: Human first-attempt success vs AI zero-shot success.

        Args:
            filename: Output filename
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        ai_models = self.horizons_df["model"].unique()

        for idx, model in enumerate(ai_models):
            ax = axes[idx]

            # Get AI performance by task
            ai_data = self.runs_df[
                (self.runs_df["learner_type"] == "ai_zero_shot") &
                (self.runs_df["alias"] == model)
            ]
            ai_task_success = ai_data.groupby("task_id")["score_binarized"].mean()

            # Merge with human first-attempt
            plot_data = self.learning_curves_df.copy()
            plot_data["ai_success"] = plot_data["task_id"].map(ai_task_success)
            plot_data = plot_data.dropna()

            # Scatter plot
            ax.scatter(
                plot_data["first_attempt_success_rate"],
                plot_data["ai_success"],
                alpha=0.6,
                s=50,
                color=self.colors.get(model, "gray"),
                edgecolors='black',
                linewidth=0.5
            )

            # Diagonal line (y=x)
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1.5, label="y=x")

            # Styling
            ax.set_xlabel("Human First-Attempt Success Rate", fontsize=11)
            ax.set_ylabel(f"{model} Zero-Shot Success Rate", fontsize=11)
            ax.set_title(f"{model}", fontsize=12, fontweight='bold')
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            ax.set_aspect('equal')

        plt.suptitle(
            "AI Zero-Shot vs Human First-Attempt Success",
            fontsize=14,
            fontweight='bold'
        )
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved: {self.output_dir / filename}")

    def plot_n_attempt_horizons(
        self,
        filename: str = "name_n_attempt_horizons.png"
    ):
        """
        Bar chart: N-attempt horizons for each AI model.

        Args:
            filename: Output filename
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        models = self.horizons_df["model"].tolist()
        horizons_50 = self.horizons_df["n_attempt_horizon_50"].tolist()
        horizons_80 = self.horizons_df["n_attempt_horizon_80"].tolist()

        colors_list = [self.colors.get(m, "gray") for m in models]

        # 50% horizon
        ax = axes[0]
        bars = ax.bar(models, horizons_50, color=colors_list, edgecolor='black', linewidth=1.5)
        ax.set_ylabel("N-Attempt Horizon", fontsize=12)
        ax.set_title("N-Attempt Horizon @ 50% AI Success", fontsize=13, fontweight='bold')
        ax.set_ylim(0, max(horizons_50) * 1.2)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.2f}',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )

        # 80% horizon
        ax = axes[1]
        bars = ax.bar(models, horizons_80, color=colors_list, edgecolor='black', linewidth=1.5)
        ax.set_ylabel("N-Attempt Horizon", fontsize=12)
        ax.set_title("N-Attempt Horizon @ 80% AI Success", fontsize=13, fontweight='bold')
        ax.set_ylim(0, max(horizons_80) * 1.2)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.2f}',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )

        plt.suptitle(
            "NAME Metric: N-Attempt Equivalence Horizons",
            fontsize=14,
            fontweight='bold'
        )
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved: {self.output_dir / filename}")

    def plot_difficulty_distribution(
        self,
        filename: str = "name_difficulty_distribution.png"
    ):
        """
        Histogram: Distribution of task difficulties.

        Args:
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        difficulties = self.learning_curves_df["first_attempt_success_rate"]

        ax.hist(
            difficulties,
            bins=20,
            color=self.colors["human_novice"],
            edgecolor='black',
            linewidth=1.2,
            alpha=0.7
        )

        ax.set_xlabel("First-Attempt Success Rate (Task Difficulty)", fontsize=12)
        ax.set_ylabel("Number of Tasks", fontsize=12)
        ax.set_title(
            "Distribution of Task Difficulties",
            fontsize=14,
            fontweight='bold'
        )
        ax.grid(True, alpha=0.3, axis='y')

        # Add statistics
        mean_diff = difficulties.mean()
        median_diff = difficulties.median()
        ax.axvline(mean_diff, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_diff:.3f}')
        ax.axvline(median_diff, color='green', linestyle='--', linewidth=2, label=f'Median: {median_diff:.3f}')
        ax.legend(fontsize=11)

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved: {self.output_dir / filename}")

    def create_all_plots(self):
        """Generate all NAME visualization plots."""
        print("Generating NAME visualizations...")

        self.plot_learning_curves_sample()
        self.plot_ai_vs_human_first_attempt()
        self.plot_n_attempt_horizons()
        self.plot_difficulty_distribution()

        print(f"\nAll plots saved to: {self.output_dir}")


# Example usage and CLI
if __name__ == "__main__":
    import sys

    # Default parameters
    learning_curves_file = "data/wrangled/name_learning_curves.csv"
    horizons_file = "data/wrangled/name_horizons.csv"
    runs_file = "data/external/name_runs.jsonl"
    output_dir = "plots"

    # Parse command line args
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]

    print("Creating NAME visualizations...")
    plotter = NAMEPlotter(
        learning_curves_path=learning_curves_file,
        horizons_path=horizons_file,
        runs_path=runs_file,
        output_dir=output_dir
    )

    plotter.create_all_plots()

    print("\nVisualization complete!")
