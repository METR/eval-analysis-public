#!/usr/bin/env python3

import argparse
import logging
import pathlib

import pandas as pd
import yaml

from horizon.plot.bootstrap_ci import (
    compute_bootstrap_confidence_region,
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute trendline confidence interval metrics from bootstrap samples."
    )
    parser.add_argument("--input-file", type=pathlib.Path, required=True)
    parser.add_argument("--agent-summaries-file", type=pathlib.Path, required=True)
    parser.add_argument("--release-dates", type=pathlib.Path, required=True)
    parser.add_argument("--output-metrics-file", type=pathlib.Path, required=True)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--after-date", type=str, default="2019-01-01")
    parser.add_argument("--before-date", type=str, default="2030-01-01")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )

    confidence_level = 0.95

    bootstrap_results = pd.read_csv(args.input_file)
    agent_summaries = pd.read_csv(args.agent_summaries_file)
    release_dates = yaml.safe_load(args.release_dates.read_text())

    stats, _, _, _ = compute_bootstrap_confidence_region(
        agent_summaries=agent_summaries,
        bootstrap_results=bootstrap_results,
        release_dates=release_dates,
        after_date=args.after_date,
        sota_before_date=args.before_date,
        trendline_end_date=args.before_date,
        confidence_level=confidence_level,
    )

    date_range = f"after {args.after_date}"
    if args.before_date:
        date_range += f", before {args.before_date}"
    logger.info(f"SOTA agents ({date_range}): {stats.sota_agents}")
    logger.info(f"Using {len(stats.sota_agents)} SOTA agents for trendline")

    metrics = {
        "after_date": args.after_date,
        "before_date": args.before_date,
        "confidence_level": confidence_level,
        "n_bootstrap_samples": stats.n_samples,
        "sota_agents": stats.sota_agents,
        "doubling_time_days": {
            "point_estimate": round(stats.point_estimate, 2),
            "median": round(stats.median, 2),
            "ci_lower": round(stats.ci_lower, 2),
            "ci_upper": round(stats.ci_upper, 2),
            "pct_above": round(stats.pct_above, 2),
            "pct_below": round(stats.pct_below, 2),
        },
    }

    logger.info(
        f"95% CI for doubling times: [{stats.ci_lower:.0f}, {stats.ci_upper:.0f}] days "
        f"(+{stats.pct_above:.0%}/-{stats.pct_below:.0%})"
    )

    args.output_metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_metrics_file, "w") as f:
        yaml.dump(metrics, f, indent=2, default_flow_style=False, sort_keys=False)
    logger.info(f"Metrics saved to {args.output_metrics_file}")


if __name__ == "__main__":
    main()
