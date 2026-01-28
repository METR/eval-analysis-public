import argparse
import logging
import pathlib
from typing import Sequence

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from scipy.special import expit

from horizon.utils import plots


def _calculate_deviance_residuals(observed: float, predicted: float) -> float:
    """Calculate deviance residuals for logistic regression."""
    sign = 1 if observed > predicted else -1
    return sign * np.sqrt(
        -2
        * (
            observed * np.log(predicted + 1e-10)
            + (1 - observed) * np.log(1 - predicted + 1e-10)
        )
    )


def plot_qq_residuals(
    plot_params: plots.PlotParams,
    agent_summaries: pd.DataFrame,
    all_runs: pd.DataFrame,
    focus_agents: Sequence[str],
    output_file: pathlib.Path,
) -> None:
    """Create Q-Q plots of deviance residuals for each agent.

    Args:
        plot_params: Plot parameters including colors
        agent_summaries: DataFrame with agent regression parameters
        all_runs: DataFrame with individual run results
        focus_agents: List of agents to plot
        output_file: Path to save the Q-Q plot figure
    """
    agent_summaries = agent_summaries[agent_summaries["agent"].isin(focus_agents)]
    n_agents = len(focus_agents)
    n_cols = 3
    n_rows = (n_agents + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()  # Flatten to make indexing easier

    for idx, (_, agent_info) in enumerate(agent_summaries.groupby("agent")):
        agent = agent_info.iloc[0]["agent"]
        agent_color = plots.get_agent_color(plot_params=plot_params, agent=agent)

        # Filter runs for this agent
        agent_runs = all_runs[all_runs["alias"] == agent]

        # Calculate predicted probabilities
        times = agent_runs["human_minutes"]
        predicted_probs = expit(
            agent_info.iloc[0]["coefficient"] * np.log2(times)
            + agent_info.iloc[0]["intercept"]
        )

        # Calculate residuals
        residuals = [
            _calculate_deviance_residuals(obs, pred)
            for obs, pred in zip(agent_runs["score_binarized"], predicted_probs)
        ]

        # Create Q-Q plot for this agent
        theoretical_quantiles = np.random.normal(0, 1, len(residuals))
        theoretical_quantiles.sort()
        residuals_sorted = np.sort(residuals)

        axes[idx].scatter(
            theoretical_quantiles, residuals_sorted, color=agent_color, alpha=0.5
        )

        # Add reference line
        min_val = min(min(residuals), -3)
        max_val = max(max(residuals), 3)
        axes[idx].plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5)

        # Customize subplot
        axes[idx].set_title(f"Q-Q Plot: {agent}")
        axes[idx].set_xlabel("Theoretical Quantiles")
        axes[idx].set_ylabel("Sample Quantiles")

    # Remove empty subplots if any
    for idx in range(len(focus_agents), len(axes)):
        axes[idx].remove()

    fig.tight_layout()
    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True)
    fig.savefig(output_file)
    logging.info(f"Saved Q-Q plots to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-runs-file", type=pathlib.Path, required=True)
    parser.add_argument("--output-file", type=pathlib.Path, required=True)
    parser.add_argument("--plot-format", type=str, default="png")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--script-parameter-group", type=str, required=True)
    parser.add_argument("--params-file", type=pathlib.Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    params = yaml.safe_load(open(args.params_file))
    script_params = params["plots"]["plot_individual_histograms"][
        args.script_parameter_group
    ]
    agent_summaries = pd.read_csv(script_params["logistic_file"])
    logging.info("Loaded input data")
    all_runs = pd.read_json(args.all_runs_file, lines=True)

    # Q-Q plots
    qq_output = args.output_file.parent / f"qq_plots.{args.plot_format}"
    plot_qq_residuals(
        params["plots"],
        agent_summaries,
        all_runs,
        script_params["include_agents"],
        qq_output,
    )


if __name__ == "__main__":
    main()
