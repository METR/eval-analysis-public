#!/usr/bin/env python3

import argparse
import logging
import pathlib
from typing import Any

import dvc.api
import numpy as np
import pandas as pd
import yaml
from matplotlib.dates import date2num

from horizon.plot.logistic import fit_trendline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def perform_bootstrap_test(
    bootstrap_results: pd.DataFrame,
    test_agent: str,
    comparison_agent: str,
    confidence_level: float,
) -> dict[str, Any]:
    """
    Performs a bootstrap test to compare a test agent against a comparison agent.

    Returns:
        Dictionary containing the test metrics for the comparison
    """
    n_bootstraps = len(bootstrap_results)

    # Check that both agents have columns in the data
    test_p50_col = f"{test_agent}_p50"
    comparison_p50_col = f"{comparison_agent}_p50"

    if test_p50_col not in bootstrap_results.columns:
        logger.error(
            f"50%-time horizon column for test agent '{test_agent}' not found in bootstrap results."
        )
        return {}

    if comparison_p50_col not in bootstrap_results.columns:
        logger.error(
            f"50%-time horizon column for comparison agent '{comparison_agent}' not found in bootstrap results."
        )
        return {}

    # Initialize tracking variables
    stats = {
        "win_count": 0,
        "tie_count": 0,
        "loss_count": 0,
        "valid_comparisons": 0,
        "actual_p50s": [],
        "comparison_p50s": [],
    }

    # Single pass through bootstrap samples
    for sample_idx in range(n_bootstraps):
        sample_row = bootstrap_results.iloc[sample_idx]

        # Get p50 values for both agents
        actual_p50_raw = sample_row.get(test_p50_col)
        comparison_p50_raw = sample_row.get(comparison_p50_col)

        if actual_p50_raw is None or comparison_p50_raw is None:
            continue

        actual_p50 = pd.to_numeric(actual_p50_raw, errors="coerce")
        comparison_p50 = pd.to_numeric(comparison_p50_raw, errors="coerce")

        if (
            pd.isna(actual_p50)
            or np.isinf(actual_p50)
            or actual_p50 < 1e-3
            or pd.isna(comparison_p50)
            or np.isinf(comparison_p50)
            or comparison_p50 < 1e-3
        ):
            continue

        # Valid comparison
        stats["valid_comparisons"] += 1
        stats["actual_p50s"].append(actual_p50)
        stats["comparison_p50s"].append(comparison_p50)

        if actual_p50 > comparison_p50:
            stats["win_count"] += 1
        elif actual_p50 == comparison_p50:
            stats["tie_count"] += 1
        else:
            stats["loss_count"] += 1

    if stats["valid_comparisons"] == 0:
        logger.error(
            f"No valid comparisons could be made for agent {test_agent} vs {comparison_agent}"
        )
        return {}

    # Calculate rates
    win_rate = stats["win_count"] / stats["valid_comparisons"]
    loss_rate = stats["loss_count"] / stats["valid_comparisons"]
    tie_rate = stats["tie_count"] / stats["valid_comparisons"]

    median_actual_p50 = np.nanmedian(stats["actual_p50s"])
    median_comparison_p50 = np.nanmedian(stats["comparison_p50s"])
    median_ratio = median_actual_p50 / median_comparison_p50

    # Determine significance
    significance_threshold = confidence_level
    if win_rate >= significance_threshold:
        significance_result = "significantly_stronger"
        is_significant = True
    elif loss_rate >= significance_threshold:
        significance_result = "significantly_weaker"
        is_significant = True
    else:
        significance_result = "not_significant"
        is_significant = False

    # Prepare metrics output
    metrics = {
        "test_agent": test_agent,
        "comparison_agent": comparison_agent,
        "num_valid_comparisons": stats["valid_comparisons"],
        "total_bootstrap_samples": n_bootstraps,
        "win_count": stats["win_count"],
        "loss_count": stats["loss_count"],
        "tie_count": stats["tie_count"],
        "win_rate": float(win_rate),
        "loss_rate": float(loss_rate),
        "tie_rate": float(tie_rate),
        "median_actual_horizon_minutes": float(median_actual_p50),
        "median_comparison_horizon_minutes": float(median_comparison_p50),
        "median_ratio_actual_vs_comparison": float(median_ratio),
        "confidence_level": confidence_level,
        "significance_result": significance_result,
        "is_significant": is_significant,
        "interpretation": f"{test_agent} beats {comparison_agent} in {win_rate * 100:.1f}% of bootstrap samples. "
        f"At {confidence_level * 100:.0f}% confidence level, {test_agent} is "
        f"{'significantly stronger than' if significance_result == 'significantly_stronger' else 'significantly weaker than' if significance_result == 'significantly_weaker' else 'not significantly different from'} {comparison_agent}.",
    }

    # Print results
    logger.info("-" * 30)
    logger.info(f"Bootstrap Test Results for Agent: {test_agent}")
    logger.info(f"Comparing against: {comparison_agent}")
    logger.info(
        f"Based on {stats['valid_comparisons']} / {n_bootstraps} valid bootstrap samples."
    )
    logger.info(f"Win rate: {win_rate * 100:.1f}% ({stats['win_count']} wins)")
    logger.info(f"Loss rate: {loss_rate * 100:.1f}% ({stats['loss_count']} losses)")
    logger.info(f"Tie rate: {tie_rate * 100:.1f}% ({stats['tie_count']} ties)")
    logger.info(f"Median Actual Horizon (minutes): {median_actual_p50:.3f}")
    logger.info(
        f"Median {comparison_agent} Horizon (minutes): {median_comparison_p50:.3f}"
    )
    logger.info(f"Median Ratio: {median_ratio:.3f}")
    logger.info(
        f"Significance ({confidence_level * 100:.0f}% threshold): {significance_result.replace('_', ' ').title()}"
    )
    logger.info(f"Interpretation: {metrics['interpretation']}")
    logger.info("-" * 30)

    return metrics


def perform_bootstrap_test_vs_max(
    bootstrap_results: pd.DataFrame,
    release_dates: dict[str, str],
    most_recent_agents: list[str],
    most_recent_date: pd.Timestamp,
    agents_to_exclude: list[str],
    confidence_level: float,
) -> dict[str, dict[str, Any]]:
    """
    Performs a bootstrap test to compare the latest agents against the strongest previous agent.

    Returns:
        Dictionary containing the test metrics for each agent
    """
    all_metrics = {}
    agents_to_exclude_with_recent = agents_to_exclude + most_recent_agents

    # Check that all most recent agents have columns in the data
    valid_recent_agents = []
    for agent in most_recent_agents:
        if f"{agent}_p50" in bootstrap_results.columns:
            valid_recent_agents.append(agent)
        else:
            logger.error(
                f"50%-time horizon column for most recent agent '{agent}' not found in bootstrap results."
            )

    if not valid_recent_agents:
        return {}

    agents_for_comparison = sorted(
        [
            agent
            for agent in release_dates
            if agent not in agents_to_exclude_with_recent
        ],
        key=lambda x: release_dates[x],
    )

    n_bootstraps = len(bootstrap_results)

    # First, identify the strongest previous agent
    # Count how many times each agent is the best in bootstrap samples
    best_agent_counts = {agent: 0 for agent in agents_for_comparison}

    for sample_idx in range(n_bootstraps):
        sample_row = bootstrap_results.iloc[sample_idx]

        # Find which agent has the highest p50 in this sample
        best_p50 = -1
        best_agent = None

        for agent in agents_for_comparison:
            p50_col_name = f"{agent}_p50"
            if p50_col_name not in sample_row:
                continue
            p50 = pd.to_numeric(sample_row[p50_col_name], errors="coerce")
            if pd.isna(p50) or np.isinf(p50) or p50 < 1e-3:
                continue
            if p50 > best_p50:
                best_p50 = p50
                best_agent = agent

        if best_agent:
            best_agent_counts[best_agent] += 1

    # Identify the agent that is best in the most samples
    strongest_agent = max(best_agent_counts, key=lambda k: best_agent_counts[k])
    strongest_agent_wins = best_agent_counts[strongest_agent]

    logger.info(
        f"Strongest previous agent: {strongest_agent} (best in {strongest_agent_wins}/{n_bootstraps} samples)"
    )
    logger.info(f"Agent win counts: {best_agent_counts}")

    for agent in valid_recent_agents:
        agent_metrics = perform_bootstrap_test(
            bootstrap_results=bootstrap_results,
            test_agent=agent,
            comparison_agent=strongest_agent,
            confidence_level=confidence_level,
        )

        if agent_metrics:
            agent_metrics["most_recent_agent"] = agent_metrics.pop("test_agent")
            agent_metrics["strongest_previous_agent"] = agent_metrics.pop(
                "comparison_agent"
            )
            agent_metrics["strongest_agent_win_count"] = strongest_agent_wins
            agent_metrics["median_strongest_previous_horizon_minutes"] = (
                agent_metrics.pop("median_comparison_horizon_minutes")
            )
            agent_metrics["median_ratio_actual_vs_strongest"] = agent_metrics.pop(
                "median_ratio_actual_vs_comparison"
            )

            all_metrics[agent] = agent_metrics

    return all_metrics


def perform_bootstrap_test_from_config(
    bootstrap_results: pd.DataFrame,
    agent_pairs: list[dict[str, str]],
    confidence_level: float,
) -> dict[str, dict[str, Any]]:
    """
    Performs bootstrap tests for configured agent pairs.

    Args:
        bootstrap_results: DataFrame containing bootstrap results
        agent_pairs: List of dicts with 'test_agent' and 'comparison_agent' keys
        confidence_level: Confidence level for significance testing

    Returns:
        Dictionary containing the test metrics for each pair
    """
    all_metrics = {}

    for pair in agent_pairs:
        test_agent = pair.get("test_agent")
        comparison_agent = pair.get("comparison_agent")

        if not test_agent or not comparison_agent:
            logger.error(f"Invalid agent pair configuration: {pair}")
            continue

        pair_key = f"{test_agent} vs {comparison_agent}"

        agent_metrics = perform_bootstrap_test(
            bootstrap_results=bootstrap_results,
            test_agent=test_agent,
            comparison_agent=comparison_agent,
            confidence_level=confidence_level,
        )

        if agent_metrics:
            all_metrics[pair_key] = agent_metrics

    return all_metrics


def perform_bootstrap_test_vs_trend(
    bootstrap_results: pd.DataFrame,
    release_dates: dict[str, str],
    most_recent_agents: list[str],
    most_recent_date: pd.Timestamp,
    agents_to_exclude: list[str],
    confidence_level: float,
) -> dict[str, dict[str, Any]]:
    """
    Performs bootstrap tests to compare the latest agents' performance against the trend.

    Returns:
        Dictionary containing the test metrics for each agent
    """

    all_metrics = {}
    agents_to_exclude_with_recent = agents_to_exclude + most_recent_agents

    # Check that all most recent agents have columns in the data
    valid_recent_agents = []
    for agent in most_recent_agents:
        if f"{agent}_p50" in bootstrap_results.columns:
            valid_recent_agents.append(agent)
        else:
            logger.error(
                f"P50 column for most recent agent '{agent}' not found in bootstrap results."
            )

    if not valid_recent_agents:
        return {}

    agents_for_trend = sorted(
        [
            agent
            for agent in release_dates
            if agent not in agents_to_exclude_with_recent
        ],
        key=lambda x: release_dates[x],
    )

    n_bootstraps = len(bootstrap_results)

    # Initialize tracking variables for each recent agent
    agent_stats = {}
    for agent in valid_recent_agents:
        agent_stats[agent] = {
            "ratios": [],
            "actual_p50s": [],
            "predicted_p50s": [],
        }

    # Single pass through bootstrap samples
    for sample_idx in range(n_bootstraps):
        sample_row = bootstrap_results.iloc[sample_idx]

        # Collect valid p50 values and dates for trend fitting agents in this sample
        valid_p50s_trend = []
        valid_dates_trend = []
        for agent in agents_for_trend:
            p50_col_name = f"{agent}_p50"
            if p50_col_name not in sample_row:
                continue
            p50 = pd.to_numeric(sample_row[p50_col_name], errors="coerce")
            if pd.isna(p50) or np.isinf(p50) or p50 < 1e-3:
                continue
            valid_p50s_trend.append(p50)
            valid_dates_trend.append(release_dates[agent])

        if len(valid_p50s_trend) < 2:
            continue  # Not enough data points for trend fitting

        # Fit exponential trendline
        try:
            reg, _ = fit_trendline(
                pd.Series(valid_p50s_trend),
                pd.Series(pd.to_datetime(valid_dates_trend)),
                log_scale=True,
            )

            # Predict p50 for the most recent date
            most_recent_date_num = date2num(most_recent_date)
            predicted_log_p50 = reg.predict(np.array([[most_recent_date_num]]))
            predicted_p50_recent = np.exp(predicted_log_p50[0])

            if predicted_p50_recent <= 0 or np.isinf(predicted_p50_recent):
                continue

            # Calculate ratio for each recent agent
            for agent in valid_recent_agents:
                agent_p50_col = f"{agent}_p50"
                actual_p50_raw = sample_row.get(agent_p50_col)
                if actual_p50_raw is None:
                    continue

                actual_p50 = pd.to_numeric(actual_p50_raw, errors="coerce")
                if pd.isna(actual_p50) or np.isinf(actual_p50) or actual_p50 < 1e-3:
                    continue

                ratio = actual_p50 / predicted_p50_recent
                agent_stats[agent]["ratios"].append(ratio)
                agent_stats[agent]["actual_p50s"].append(actual_p50)
                agent_stats[agent]["predicted_p50s"].append(predicted_p50_recent)

        except Exception as e:
            logger.error(
                f"Sample {sample_idx}: Error during trend fitting or prediction: {e}. Skipping."
            )
            continue

    # Calculate metrics for each agent
    for agent in valid_recent_agents:
        stats = agent_stats[agent]

        if not stats["ratios"]:
            logger.error(f"No valid ratios could be calculated for agent {agent}")
            continue

        # Calculate metrics
        ratios_array = np.array(stats["ratios"])
        median_ratio = np.nanmedian(ratios_array)
        median_actual_p50 = np.nanmedian(stats["actual_p50s"])
        median_predicted_p50 = np.nanmedian(stats["predicted_p50s"])

        # Calculate two-sided p-value
        p_less_than_1 = np.nanmean(ratios_array < 1)
        p_value = 2 * min(float(p_less_than_1), 1.0 - float(p_less_than_1))

        # Calculate confidence interval for the ratio
        alpha = 1 - confidence_level
        lower_bound = np.nanpercentile(ratios_array, alpha / 2 * 100)
        upper_bound = np.nanpercentile(ratios_array, (1 - alpha / 2) * 100)

        is_significant = p_value < 0.05

        # Prepare metrics output
        all_metrics[agent] = {
            "most_recent_agent": agent,
            "num_bootstrap_samples_used": len(stats["ratios"]),
            "total_bootstrap_samples": n_bootstraps,
            "median_actual_horizon_minutes": float(median_actual_p50),
            "median_predicted_horizon_minutes": float(median_predicted_p50),
            "median_ratio_actual_vs_predicted": float(median_ratio),
            "confidence_interval_ratio": {
                "level": confidence_level,
                "lower": float(lower_bound),
                "upper": float(upper_bound),
            },
            "p_value_two_sided": float(p_value),
            "significant_at_0.05": bool(is_significant),
            "interpretation": f"The trendline predicts {agent}'s median 50%-time horizon to be {median_predicted_p50} minutes, whereas it is actually {median_actual_p50} minutes. The ratio of actual 50%-time horizon ({agent}) to predicted 50%-time horizon is {median_ratio:.2f}. "
            f"The {confidence_level * 100:.0f}% CI is [{lower_bound:.2f}, {upper_bound:.2f}]. "
            f"P-value = {p_value:.4f}. "
            f"The result is {'significant' if is_significant else 'not significant'} at alpha=0.05.",
        }

        # Print results
        logger.info("-" * 30)
        logger.info(f"Bootstrap Test Results vs Trend for Agent: {agent}")
        logger.info(
            f"Based on {len(stats['ratios'])} / {n_bootstraps} valid bootstrap samples."
        )
        logger.info(f"Median Ratio (Actual/Predicted): {median_ratio:.3f}")
        logger.info(f"Median Actual Horizon (minutes): {median_actual_p50:.3f}")
        logger.info(f"Median Predicted Horizon (minutes): {median_predicted_p50:.3f}")
        logger.info(
            f"{confidence_level * 100:.0f}% CI for Ratio: [{lower_bound:.3f}, {upper_bound:.3f}]"
        )
        logger.info(f"Two-sided p-value: {p_value:.4f}")
        logger.info(
            f"Significant difference from trend (alpha=0.05)? {'Yes' if is_significant else 'No'}"
        )
        logger.info(f"Interpretation: {all_metrics[agent]['interpretation']}")
        logger.info("-" * 30)

    return all_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Perform bootstrap tests for latest agent vs. trend and vs. max."
    )
    parser.add_argument(
        "--bootstrap-results-file",
        type=pathlib.Path,
        default=pathlib.Path("data/wrangled/bootstrap/headline.csv"),
        help="Path to the bootstrap results CSV file.",
    )
    parser.add_argument(
        "--release-dates-file",
        type=pathlib.Path,
        default=pathlib.Path("data/external/release_dates.yaml"),
        help="Path to the release dates YAML file.",
    )
    parser.add_argument(
        "--output-metrics-file",
        type=pathlib.Path,
        default=pathlib.Path("metrics/compare_trend/headline_latest_agent_test.yaml"),
        help="Path to save the trend comparison metrics YAML file.",
    )
    parser.add_argument(
        "--output-max-metrics-file",
        type=pathlib.Path,
        default=pathlib.Path("metrics/compare_max/headline_latest_agent_test.yaml"),
        help="Path to save the max comparison metrics YAML file.",
    )
    parser.add_argument(
        "--output-config-metrics-file",
        type=pathlib.Path,
        default=pathlib.Path(
            "metrics/compare_config_headline_agent_comparisons_test.yaml"
        ),
        help="Path to save the configured agent comparison metrics YAML file.",
    )

    args = parser.parse_args()

    params = dvc.api.params_show(stages="compare_trend", deps=True)["figs"][
        "compare_trend"
    ]

    agents_to_exclude = params.get("exclude_agents", []) + params.get(
        "exclude_agents_from_all_fits", []
    )
    confidence_level = params.get("confidence_level", 0.95)

    # Load data
    try:
        bootstrap_results = pd.read_csv(args.bootstrap_results_file)
        agents_under_consideration = [
            col.rsplit("_", 1)[0]
            for col in bootstrap_results.columns
            if col.endswith("_p50")
        ]
        release_dates_data = yaml.safe_load(args.release_dates_file.read_text())
        release_dates = {
            alias: date
            for alias, date in release_dates_data["date"].items()
            if alias in agents_under_consideration
        }
    except FileNotFoundError as e:
        logger.error(f"Error loading data file: {e}")
        return
    except Exception as e:
        logger.error(f"Error processing data files: {e}")
        return

    if not isinstance(release_dates, dict):
        logger.error(
            "Release dates file format is incorrect. Expected a dictionary under the 'date' key."
        )
        return

    # Identify the most recent agents
    try:
        most_recent_date_string = max(release_dates.values())
        most_recent_agents = [
            agent
            for agent, date_str in release_dates.items()
            if date_str == most_recent_date_string
        ]
        most_recent_date = pd.to_datetime(most_recent_date_string)
        logger.info(
            f"Most recent agents: {most_recent_agents} released on {most_recent_date.date()}"
        )
    except ValueError:
        logger.error("Could not determine the most recent agent from release dates.")
        return

    # Perform trend comparison
    logger.info("=" * 50)
    logger.info("Performing trend comparison...")
    logger.info("=" * 50)
    trend_metrics = perform_bootstrap_test_vs_trend(
        bootstrap_results=bootstrap_results,
        release_dates=release_dates,
        most_recent_agents=most_recent_agents,
        most_recent_date=most_recent_date,
        agents_to_exclude=agents_to_exclude,
        confidence_level=confidence_level,
    )

    # Save trend metrics
    args.output_metrics_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(args.output_metrics_file, "w") as f:
            yaml.dump(
                trend_metrics, f, indent=2, default_flow_style=False, sort_keys=False
            )
        logger.info(f"Trend metrics saved to {args.output_metrics_file}")
    except Exception as e:
        logger.error(f"Failed to save trend metrics file: {e}")

    # Perform max comparison
    logger.info("=" * 50)
    logger.info("Performing max comparison...")
    logger.info("=" * 50)
    max_metrics = perform_bootstrap_test_vs_max(
        bootstrap_results=bootstrap_results,
        release_dates=release_dates,
        most_recent_agents=most_recent_agents,
        most_recent_date=most_recent_date,
        agents_to_exclude=agents_to_exclude,
        confidence_level=confidence_level,
    )

    # Save max metrics
    args.output_max_metrics_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(args.output_max_metrics_file, "w") as f:
            yaml.dump(
                max_metrics, f, indent=2, default_flow_style=False, sort_keys=False
            )
        logger.info(f"Max metrics saved to {args.output_max_metrics_file}")
    except Exception as e:
        logger.error(f"Failed to save max metrics file: {e}")

    # Perform configured agent comparisons
    agent_comparisons = params.get("agent_comparisons", [])
    if agent_comparisons:
        logger.info("=" * 50)
        logger.info("Performing configured agent comparisons...")
        logger.info("=" * 50)
        config_metrics = perform_bootstrap_test_from_config(
            bootstrap_results=bootstrap_results,
            agent_pairs=agent_comparisons,
            confidence_level=confidence_level,
        )
    else:
        logger.info(
            "No agent_comparisons configured, creating empty configured agent comparisons metrics file."
        )
        config_metrics = {}

    # Save configured agent comparisons metrics
    args.output_config_metrics_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(args.output_config_metrics_file, "w") as f:
            yaml.dump(
                config_metrics,
                f,
                indent=2,
                default_flow_style=False,
                sort_keys=False,
            )
        logger.info(
            f"Configured agent comparisons metrics saved to {args.output_config_metrics_file}"
        )
    except Exception as e:
        logger.error(f"Failed to save configured agent comparisons metrics file: {e}")


if __name__ == "__main__":
    main()
