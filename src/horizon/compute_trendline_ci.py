#!/usr/bin/env python3

import argparse
import logging
import pathlib

import numpy as np
import pandas as pd
import yaml

from horizon.plot.bootstrap_ci import (
    compute_bootstrap_confidence_region,
)

logger = logging.getLogger(__name__)


def get_sota_agents(
    agent_summaries: pd.DataFrame,
    release_dates: dict[str, str],
    after_date: str | None = None,
    before_date: str | None = None,
) -> list[str]:
    """Determine which agents are SOTA based on p50 horizon at release time.

    An agent is SOTA if its p50 horizon is >= the highest p50 seen among
    all agents released on or before the same date.

    If after_date is provided, only returns SOTA agents released on or after that date.
    If before_date is provided, only returns SOTA agents released before that date.
    """
    agents_with_dates = []
    for _, row in agent_summaries.iterrows():
        agent = row["agent"]
        if agent == "human":
            continue

        assert agent in release_dates, f"Agent {agent} not found in release dates"
        p50 = row["p50"]
        assert not pd.isna(p50) and not np.isinf(
            p50
        ), f"Agent {agent} has invalid p50: {p50}"
        agents_with_dates.append(
            {
                "agent": agent,
                "release_date": pd.to_datetime(release_dates[agent]).date(),
                "p50": p50,
            }
        )

    df = pd.DataFrame(agents_with_dates)
    assert not df.empty, "No agents with valid p50s found"

    df = df.sort_values("release_date")

    # Then, we filter to after_date and before_date if provided
    if after_date:
        df = df[df["release_date"] >= pd.to_datetime(after_date).date()]
    if before_date:
        df = df[df["release_date"] < pd.to_datetime(before_date).date()]

    sota_agents = []
    highest_horizon_so_far = float("-inf")

    for release_date in df["release_date"].unique():
        agents_on_date = df[df["release_date"] == release_date]
        max_horizon_on_date = agents_on_date["p50"].max()
        highest_horizon_so_far = max(highest_horizon_so_far, max_horizon_on_date)

        for _, row in agents_on_date.iterrows():
            if row["p50"] >= highest_horizon_so_far:
                sota_agents.append(row["agent"])

    assert len(sota_agents) > 0, "No SOTA agents found after filtering"
    return sota_agents


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
    parser.add_argument("--before-date", type=str, default=None)
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

    sota_agents = get_sota_agents(
        agent_summaries, release_dates["date"], args.after_date, args.before_date
    )
    date_range = f"after {args.after_date}"
    if args.before_date:
        date_range += f", before {args.before_date}"
    logger.info(f"SOTA agents ({date_range}): {sota_agents}")

    # Assert all the SOTA agents have bootstrap results
    for agent in sota_agents:
        if f"{agent}_p50" not in bootstrap_results.columns:
            raise ValueError(f"Agent {agent} not found in bootstrap results")

    logger.info(f"Using {len(sota_agents)} SOTA agents for trendline")

    agent_summaries_for_fitting = agent_summaries[
        agent_summaries["agent"].isin(sota_agents)
    ]
    assert len(agent_summaries_for_fitting) == len(sota_agents)
    bootstrap_results_for_fitting = bootstrap_results[
        [f"{agent}_p50" for agent in sota_agents]
    ]

    stats, _, _, _ = compute_bootstrap_confidence_region(
        agent_summaries=agent_summaries_for_fitting,
        bootstrap_results=bootstrap_results_for_fitting,
        release_dates=release_dates,
        after_date=args.after_date,
        max_date=pd.to_datetime(args.before_date or "2027-01-01"),
        confidence_level=confidence_level,
    )

    metrics = {
        "after_date": args.after_date,
        "before_date": args.before_date,
        "confidence_level": confidence_level,
        "n_bootstrap_samples": stats.n_samples,
        "sota_agents": sota_agents,
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
