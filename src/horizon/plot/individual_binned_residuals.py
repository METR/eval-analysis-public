import argparse
import logging
import pathlib
from typing import Sequence

import dvc.api
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.special import expit

from horizon.utils import plots


def plot_binned_residuals(
    plot_params: plots.PlotParams,
    agent_summaries: pd.DataFrame,
    all_runs: pd.DataFrame,
    focus_agents: Sequence[str],
    output_file: pathlib.Path,
) -> None:
    """
    Create binned residual plots for each agent: ±SE bands centered at 0,
    and larger bins (fewer bins in total).
    """

    agent_summaries = agent_summaries[agent_summaries["agent"].isin(focus_agents)]
    n_agents = len(focus_agents)
    n_cols = 3
    n_rows = (n_agents + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()  # Flatten to make indexing easier

    for idx, (_, agent_info) in enumerate(agent_summaries.groupby("agent")):
        agent = agent_info.iloc[0]["agent"]

        # Filter runs for this agent
        agent_runs = all_runs[all_runs["alias"] == agent]
        n_bins = int(np.sqrt(len(agent_runs)))
        # Calculate predicted probabilities
        times = agent_runs["human_minutes"]
        predicted_probs = expit(
            agent_info.iloc[0]["coefficient"] * np.log2(times)
            + agent_info.iloc[0]["intercept"]
        )

        # Calculate deviance residuals
        residuals = np.array(agent_runs["score_binarized"] - predicted_probs)

        # Create bins based on predicted probabilities
        bins = np.linspace(predicted_probs.min(), predicted_probs.max(), n_bins + 1)
        bin_indices = np.digitize(predicted_probs, bins)

        bin_means_x = []
        bin_means_y = []
        bin_ses = []
        bin_sizes = []

        for bin_idx in range(1, len(bins)):
            mask = bin_indices == bin_idx
            n_points = np.sum(mask)
            if n_points >= 1:
                bin_pred_prob_mean = np.mean(predicted_probs[mask])
                bin_mean_residual = np.mean(residuals[mask])
                bin_residual_std = np.std(residuals[mask], ddof=1)

                bin_means_x.append(bin_pred_prob_mean)
                bin_means_y.append(bin_mean_residual)

                # Standard error of the mean residual
                bin_se = bin_residual_std / np.sqrt(n_points)
                bin_ses.append(bin_se)
                bin_sizes.append(n_points)

        bin_means_x = np.array(bin_means_x)
        bin_means_y = np.array(bin_means_y)
        bin_ses = np.array(bin_ses)
        bin_sizes = np.array(bin_sizes)

        upper_band_2se = 2 * np.abs(bin_ses)
        lower_band_2se = -2 * np.abs(bin_ses)

        axes[idx].fill_between(
            bin_means_x,
            lower_band_2se,
            upper_band_2se,
            color="gray",
            alpha=0.15,
            label="±2 SE",
        )

        upper_band_1se = np.abs(bin_ses)
        lower_band_1se = -np.abs(bin_ses)
        axes[idx].fill_between(
            bin_means_x,
            lower_band_1se,
            upper_band_1se,
            color="gray",
            alpha=0.25,
            label="±1 SE",
        )

        # Plot binned means with no error bars around the bin's own average
        sizes = 50 * (bin_sizes / bin_sizes.max()) if len(bin_sizes) else 50
        axes[idx].scatter(
            bin_means_x,
            bin_means_y,
            color="black",
            s=sizes,
            zorder=3,
            label="Binned averages",
        )

        # Reference lines
        axes[idx].axhline(y=0, color="gray", linestyle="--", alpha=0.8)

        # Title, labels, etc.
        axes[idx].set_title(f"Binned Residuals: {agent}")
        axes[idx].set_xlabel("Predicted Probability")
        axes[idx].set_ylabel("Residuals")
        # axes[idx].set_ylim(-1, 1)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend(loc="upper right")

    # Remove empty subplots if any
    for idx in range(len(focus_agents), len(axes)):
        axes[idx].remove()

    fig.tight_layout()
    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True)
    fig.savefig(output_file)
    logging.info(f"Saved zero-centered binned residual plots to {output_file}")


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

    params = dvc.api.params_show(stages="plot_individual_binned_residuals", deps=True)
    # Shares params with plot_individual_histograms stage
    fig_params = params["figs"]["plot_individual_histograms"][
        args.script_parameter_group
    ]
    agent_summaries = pd.read_csv(fig_params["logistic_file"])
    logging.info("Loaded input data")
    all_runs = pd.read_json(args.all_runs_file, lines=True)

    # Binned residual plots
    binned_output = args.output_file
    plot_binned_residuals(
        params["plots"],
        agent_summaries,
        all_runs,
        fig_params["include_agents"],
        binned_output,
    )


if __name__ == "__main__":
    main()
