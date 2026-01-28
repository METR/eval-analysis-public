from __future__ import annotations

import argparse
import logging
import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import ruamel.yaml
import yaml
from matplotlib.dates import date2num

from horizon.plot.logistic import fit_trendline

logger = logging.getLogger(__name__)


@dataclass
class TrendSample:
    slope: float
    intercept: float


def get_frontier_agents(
    logistic_fits_df: pd.DataFrame,
    release_dates: dict[str, Any],
    success_percent: int,
) -> set[str]:
    """Get set of agents that were frontier (best) at their release date."""
    date_lookup = release_dates["date"]
    p_col = f"p{success_percent}"

    df = logistic_fits_df[logistic_fits_df["agent"].isin(date_lookup)].copy()
    agents_with_dates = [
        {
            "agent": row["agent"],
            "release_date": pd.to_datetime(date_lookup[row["agent"]]),
            "p_val": row[p_col],
        }
        for _, row in df.iterrows()
        if pd.notna(row[p_col]) and np.isfinite(float(row[p_col]))
    ]

    if not agents_with_dates:
        return set()

    df = pd.DataFrame(agents_with_dates).sort_values("release_date")
    running_max = df["p_val"].expanding().max()
    return set(df.loc[df["p_val"] >= running_max, "agent"])


def _fit_trend_sample(
    row: pd.Series[Any],
    cols: list[str],
    agent_dates: pd.Series[Any],
) -> TrendSample | None:
    """Fit a single trend sample from bootstrap row. Returns None if insufficient data."""
    vals = np.array([row[col] for col in cols])
    valid = ~(np.isnan(vals) | np.isinf(vals) | (vals < 1e-6))
    if valid.sum() < 2:
        return None
    reg, _ = fit_trendline(
        pd.Series(vals[valid]),
        agent_dates[valid],
        log_scale=True,
    )
    assert (
        reg.coef_[0] > 0
    ), f"Negative slope {reg.coef_[0]} for frontier models is impossible"
    return TrendSample(slope=reg.coef_[0], intercept=reg.intercept_)


def _sample_trends(
    bootstrap_df: pd.DataFrame,
    release_dates: dict[str, Any],
    frontier_agents: set[str],
    success_percent: int,
    n_minimum_models_for_trend: int | None,
    rng: np.random.Generator | None,
) -> list[TrendSample]:
    """Sample trends from bootstrap, optionally with window uncertainty.

    If n_minimum_models_for_trend is None, uses all frontier agents for each sample.
    Otherwise, uniformly samples window start indices to capture bias-variance tradeoff.

    Note: frontier_agents is computed once over all time, then windows select subsets.
    We do NOT recompute frontier per window. This is intentional: the frontier property
    is considered a historical fact (model X was SOTA when released), not dependent on
    which window we use for trend estimation. (Possibly this should change and we should
    bootstrap over models before computing frontier, but this is not currently implemented.)
    We are attempting merely to capture uncertainty about which historical data to use for
    trend estimation.
    """
    date_lookup = release_dates["date"]
    agents = sorted(frontier_agents, key=lambda a: pd.to_datetime(date_lookup[a]))
    n_agents = len(agents)

    cols = [f"{agent}_p{success_percent}" for agent in agents]
    agent_dates = pd.to_datetime(pd.Series([date_lookup[agent] for agent in agents]))

    use_window_uncertainty = n_minimum_models_for_trend is not None
    if use_window_uncertainty:
        assert n_minimum_models_for_trend >= 5
        assert n_agents >= n_minimum_models_for_trend
        assert rng is not None
        max_start_idx = n_agents - n_minimum_models_for_trend
        window_starts = rng.integers(
            0, max_start_idx, size=len(bootstrap_df), endpoint=True
        )
    else:
        window_starts = [0] * len(bootstrap_df)

    trend_samples = []
    for (_, row), start_idx in zip(bootstrap_df.iterrows(), window_starts):
        sample = _fit_trend_sample(row, cols[start_idx:], agent_dates.iloc[start_idx:])
        if sample is not None:
            trend_samples.append(sample)

    return trend_samples


DAYS_PER_MONTH = 365.25 / 12


def sample_gap_days_from_survey(
    gap_months: list[float],
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
    """Sample gaps (in days) from a normal distribution fit to survey data (in months).

    We sample from a fitted normal rather than using raw values for anonymization:
    we don't want to reveal exact lag values reported by individual companies.
    If only one value is provided, return that value deterministically (no sampling).
    """
    assert all(g >= 0 for g in gap_months)
    if len(gap_months) == 1:
        return np.full(n_samples, gap_months[0] * DAYS_PER_MONTH)
    mean = np.mean(gap_months)
    std = np.std(gap_months, ddof=1)
    samples_months = rng.normal(mean, std, n_samples)
    samples_months = np.clip(samples_months, 0, None)
    return samples_months * DAYS_PER_MONTH


def compute_extrapolation_distribution(
    trend_samples: list[TrendSample],
    gap_samples_days: np.ndarray[Any, np.dtype[np.floating[Any]]],
    target_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    """Compute distribution of extrapolated horizons at each target date.

    Pairs trend samples with gap samples 1:1 (deterministically).
    """
    n_samples = min(len(trend_samples), len(gap_samples_days))
    n_dates = len(target_dates)

    slopes = np.array([t.slope for t in trend_samples[:n_samples]])
    intercepts = np.array([t.intercept for t in trend_samples[:n_samples]])
    gap_days = gap_samples_days[:n_samples]

    # Shift target dates forward by gap: internal capabilities lead external deployment,
    # so we extrapolate to where internal capabilities will be at target_date + gap.
    x_effective = np.zeros((n_samples, n_dates))
    for i, target in enumerate(target_dates):
        x_effective[:, i] = date2num(target) + gap_days

    log_horizons = intercepts[:, np.newaxis] + slopes[:, np.newaxis] * x_effective
    horizons = np.exp(log_horizons)

    sample_idx = np.repeat(np.arange(n_samples), n_dates)
    target_date_iso = np.tile([d.isoformat() for d in target_dates], n_samples)
    assert np.all(slopes > 0), "All bootstrap samples should have positive slopes"
    doubling_days = np.log(2) / slopes

    return pd.DataFrame(
        {
            "sample_idx": sample_idx,
            "target_date": target_date_iso,
            "horizon_minutes": horizons.ravel(),
            "gap_days": np.repeat(gap_days, n_dates),
            "slope": np.repeat(slopes, n_dates),
            "intercept": np.repeat(intercepts, n_dates),
            "doubling_days": np.repeat(doubling_days, n_dates),
        }
    )


def summarize_extrapolations(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize extrapolation samples into quantiles per target date.

    Computes quantiles on log scale then exponentiates, so bands are
    straight lines in log space. Uses geometric mean as central estimate.
    """
    quantiles = [0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975]

    def _summarize_group(group: pd.DataFrame) -> dict[str, float]:
        log_h = np.log(group["horizon_minutes"].to_numpy(dtype=float))  # type: ignore[assignment]
        return {
            "geomean": float(np.exp(log_h.mean())),
            "log_std": float(np.std(log_h)),
            **{f"q{q:.3f}": float(np.exp(np.quantile(log_h, q))) for q in quantiles},
        }

    return pd.DataFrame(
        [
            {"target_date": date, **_summarize_group(group)}
            for date, group in df.groupby("target_date")
        ]
    )


def _get_baseline_and_targets(
    logistic_fits_df: pd.DataFrame,
    extrapolation_months: int,
) -> tuple[pd.Timestamp, list[pd.Timestamp]]:
    """Get baseline date (most recent model) and target dates for extrapolation."""
    df = logistic_fits_df.copy()
    df["release_date"] = pd.to_datetime(df["release_date"])
    baseline_date = df["release_date"].max()
    target_dates = [
        baseline_date + pd.DateOffset(months=m) for m in range(extrapolation_months + 1)
    ]
    return baseline_date, target_dates


def extrapolate_frontier_trend(
    bootstrap_df: pd.DataFrame,
    logistic_fits_df: pd.DataFrame,
    release_dates: dict[str, Any],
    success_percent: int,
    seed: int,
    gap_months: list[float] | None = None,
    n_minimum_models_for_trend: int | None = None,
    extrapolation_months: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame, set[str], pd.Timestamp]:
    """Extrapolate frontier model trend with optional gap and window uncertainty.

    1. **Frontier model selection**: From all evaluated models, select only those that
       were state-of-the-art (SOTA) at their release date.

    2. **Trend fitting**: Fit a log-linear trend (horizon = exp(slope * date + intercept))
       to the frontier models using bootstrap samples from logistic fits.
       If n_minimum_models_for_trend is provided, uniformly samples window start indices
       to capture bias-variance tradeoff (shorter windows may capture recent dynamics
       better but have higher variance).

    3. **Capability gap adjustment**: If gap_months is provided, shift extrapolation by
       an "internal-external capability gap" (in months) to predict internal capabilities
       ahead of external deployment. Multiple values sample from a normal distribution
       fit to those values.

    4. **Extrapolation**: For each bootstrap sample, extrapolate the trend to target
       dates (0 to extrapolation_months from baseline). Baseline is the most recent model
       release date in the dataset (not necessarily SOTA), since we extrapolate from when
       we have data.

    Args:
        gap_months: If None, no gap adjustment (gap=0). Otherwise, list of gap values
            in months to sample from.
        n_minimum_models_for_trend: If None, uses all frontier agents. Otherwise,
            uniformly samples windows of at least this many models.
        extrapolation_months: How many months into the future to extrapolate. Default 12.

    Returns (samples_df, summary_df, frontier_agents, baseline_date).
    """
    rng = np.random.default_rng(seed)

    frontier_agents = get_frontier_agents(
        logistic_fits_df, release_dates, success_percent
    )

    trend_samples = _sample_trends(
        bootstrap_df,
        release_dates,
        frontier_agents,
        success_percent,
        n_minimum_models_for_trend,
        rng if n_minimum_models_for_trend is not None else None,
    )

    if gap_months is not None:
        gap_samples_days = sample_gap_days_from_survey(
            gap_months, len(trend_samples), rng
        )
    else:
        gap_samples_days = np.zeros(len(trend_samples))

    baseline_date, target_dates = _get_baseline_and_targets(
        logistic_fits_df, extrapolation_months
    )

    samples_df = compute_extrapolation_distribution(
        trend_samples, gap_samples_days, target_dates
    )
    summary_df = summarize_extrapolations(samples_df)

    return samples_df, summary_df, frontier_agents, baseline_date


def _build_metrics(
    samples_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    frontier_agents: set[str],
    baseline_date: pd.Timestamp,
    release_dates: dict[str, Any],
) -> dict[str, Any]:
    """Build metrics dict for output."""
    gap_days = samples_df.groupby("sample_idx")["gap_days"].first()
    slopes = samples_df.groupby("sample_idx")["slope"].first()
    date_lookup = release_dates["date"]
    sorted_agents = sorted(frontier_agents, key=lambda a: date_lookup[a])
    return {
        "n_bootstrap_samples": len(slopes),
        "n_frontier_agents": len(frontier_agents),
        "frontier_agents": sorted_agents,
        "gap_months_mean": round(float(gap_days.mean() / DAYS_PER_MONTH), 1),
        "gap_months_std": round(float(gap_days.std() / DAYS_PER_MONTH), 1),
        "mean_doubling_days": round(float(np.log(2) / slopes.mean()), 1),
        "baseline_date": str(baseline_date.date()),
        "extrapolation_summary": {
            row["target_date"]: {
                "ci_high": round(float(row["q0.975"]), 1),
                "ci_low": round(float(row["q0.025"]), 1),
                "geomean_minutes": round(float(row["geomean"]), 1),
                "median_minutes": round(float(row["q0.500"]), 1),
            }
            for _, row in summary_df.iterrows()
        },
    }


def _run_extrapolation(
    extrapolation_params: dict[str, Any],
    bootstrap_df: pd.DataFrame,
    logistic_fits_df: pd.DataFrame,
    release_dates: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, set[str], pd.Timestamp]:
    """Run extrapolation based on params."""
    return extrapolate_frontier_trend(
        bootstrap_df=bootstrap_df,
        logistic_fits_df=logistic_fits_df,
        release_dates=release_dates,
        success_percent=extrapolation_params["success_percent"],
        seed=extrapolation_params["seed"],
        gap_months=extrapolation_params.get("gap_months"),
        n_minimum_models_for_trend=extrapolation_params.get(
            "n_minimum_models_for_trend"
        ),
        extrapolation_months=extrapolation_params.get("extrapolation_months", 12),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script-parameter-group", type=str, required=True)
    parser.add_argument("--fig-params-file", type=pathlib.Path, required=True)
    parser.add_argument("--bootstrap-file", type=pathlib.Path, required=True)
    parser.add_argument("--logistic-fits-file", type=pathlib.Path, required=True)
    parser.add_argument("--release-dates-file", type=pathlib.Path, required=True)
    parser.add_argument("--output-samples-file", type=pathlib.Path, required=True)
    parser.add_argument("--output-summary-file", type=pathlib.Path, required=True)
    parser.add_argument("--output-metrics-file", type=pathlib.Path, required=True)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s - %(levelname)s - %(message)s"
    )

    yaml_loader = ruamel.yaml.YAML(typ="safe")
    with open(args.fig_params_file) as f:
        params = yaml_loader.load(f)
    extrapolation_params = params["wrangle_out_of_sample_extrapolation"][
        args.script_parameter_group
    ]

    bootstrap_df = pd.read_csv(args.bootstrap_file)
    logistic_fits_df = pd.read_csv(args.logistic_fits_file)
    release_dates = yaml.safe_load(args.release_dates_file.read_text())

    logger.info(f"Loaded {len(bootstrap_df)} bootstrap samples")

    samples_df, summary_df, frontier_agents, baseline_date = _run_extrapolation(
        extrapolation_params, bootstrap_df, logistic_fits_df, release_dates
    )

    logger.info(f"Frontier agents ({len(frontier_agents)}): {sorted(frontier_agents)}")
    logger.info(f"Computed {len(samples_df)} extrapolation samples")
    logger.info(f"Baseline (most recent model): {baseline_date.date()}")

    metrics = _build_metrics(
        samples_df, summary_df, frontier_agents, baseline_date, release_dates
    )

    args.output_samples_file.parent.mkdir(parents=True, exist_ok=True)
    samples_df.to_csv(args.output_samples_file, index=False)
    summary_df.to_csv(args.output_summary_file, index=False)

    args.output_metrics_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_metrics_file.write_text(yaml.dump(metrics, sort_keys=False))


if __name__ == "__main__":
    main()
