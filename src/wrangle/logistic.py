# %%

from __future__ import annotations

import argparse
import logging
import pathlib
from typing import TYPE_CHECKING, Any, TypedDict

import dvc.api
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression

from src.utils.logistic import get_x_for_quantile, logistic_regression

if TYPE_CHECKING:
    from numpy.typing import NDArray


class WrangleParams(TypedDict):
    runs_file: pathlib.Path
    weighting: str
    categories: list[str]
    regularization: float
    exclude: list[str]
    success_percents: list[int]
    confidence_level: float


def empirical_success_rates(
    x: NDArray[Any],
    y: NDArray[Any],
    time_buckets: list[int],
    weights: NDArray[Any],
) -> tuple[pd.Series[Any], float]:
    use_weighted_mean = True
    # Calculate empirical success rates for different time buckets
    empirical_rates = []
    for i in range(len(time_buckets) - 1):
        mask = (np.exp2(x).reshape(-1) >= time_buckets[i]) & (
            np.exp2(x).reshape(-1) < time_buckets[i + 1]
        )
        success_rate = (
            np.sum(y[mask] * weights[mask]) / np.sum(weights[mask])
            if use_weighted_mean
            else np.mean(y[mask])
        )
        empirical_rates.append(success_rate)

    average = np.sum(y * weights) / np.sum(weights)
    indices = [
        f"{start}-{end} min" for start, end in zip(time_buckets[:-1], time_buckets[1:])
    ]
    return pd.Series(empirical_rates, index=indices), average


def get_bce_loss(
    x: NDArray[Any],
    y: NDArray[Any],
    model: LogisticRegression,
    weights: NDArray[Any],
) -> float:
    y_pred = model.predict_proba(x)[:, 1]

    # Calculate weighted BCE loss
    # can't use sklearn.metrics.log_loss because it doesn't support continuous y
    epsilon = 1e-15  # small constant to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    weights = weights / weights.mean()
    bce = -weights * (y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return np.mean(bce).item()


def agent_regression(
    x: NDArray[Any],
    y: NDArray[Any],
    weights: NDArray[Any],
    agent_name: str,
    regularization: float,
    success_percents: list[int],
    confidence_level: float,
    bootstrap_results: pd.DataFrame | None = None,
    include_empirical_rates: bool = True,
) -> pd.Series[Any]:
    time_buckets = [1, 4, 16, 64, 256, 960]
    x = np.log2(x).reshape(-1, 1)

    empirical_rates, average = None, None
    if include_empirical_rates:
        empirical_rates, average = empirical_success_rates(x, y, time_buckets, weights)

    # Build indices based on success_percents
    indices = ["coefficient", "intercept", "bce_loss", "average"]
    low_q = (1 - confidence_level) / 2
    high_q = 1 - low_q
    for p in success_percents:
        indices.extend([f"p{p}", f"p{p}q{low_q:.3f}", f"p{p}q{high_q:.3f}"])

    if np.all(y == 0):
        # Return zeros for all metrics
        values = [
            -np.inf,  # coefficient
            0,  # intercept
            0,  # bce_loss
            0,  # average
        ]
        for _ in success_percents:
            values.extend([0, 0, 0])  # p{n}, p{n}q10, p{n}q90
        result = pd.Series(values, index=indices)  # type: ignore[reportCallIssue]
        if include_empirical_rates:
            result = result._append(empirical_rates)
        return result

    model = logistic_regression(
        x, y, sample_weight=weights, regularization=regularization
    )
    if model.coef_[0][0] > 0:
        logging.warning(f"Warning: {agent_name} has positive slope {model.coef_[0][0]}")

    # Calculate metrics
    values = [
        model.coef_[0][0],
        model.intercept_[0],  # type: ignore
        get_bce_loss(x, y, model, weights),
        float("nan") if not include_empirical_rates else average,
    ]

    # Calculate percentiles and confidence intervals
    for p in success_percents:
        horizon = np.exp2(get_x_for_quantile(model, p / 100))

        if (
            bootstrap_results is not None
            and f"{agent_name}_p{p}" in bootstrap_results.columns
        ):
            horizon_low_q = np.nanquantile(
                bootstrap_results[f"{agent_name}_p{p}"], low_q
            )
            horizon_high_q = np.nanquantile(
                bootstrap_results[f"{agent_name}_p{p}"], high_q
            )
        else:
            horizon_low_q = float("nan")
            horizon_high_q = float("nan")

        values.extend([horizon, horizon_low_q, horizon_high_q])

    result = pd.Series(values, index=indices)  # type: ignore[reportCallIssue]
    if include_empirical_rates:
        result = result._append(empirical_rates)
    return result


def run_logistic_regressions(
    runs: pd.DataFrame,
    release_dates_file: pathlib.Path,
    wrangle_params: WrangleParams,
    bootstrap_file: pathlib.Path | None = None,
    include_empirical_rates: bool = True,
) -> pd.DataFrame:
    release_dates = yaml.safe_load(release_dates_file.read_text())

    weights_fn = lambda x: x[wrangle_params["weighting"]].values  # noqa: E731
    # rename alias to agent
    runs.rename(columns={"alias": "agent"}, inplace=True)
    if wrangle_params["exclude"] is not None:
        unique_task_sources = runs["task_source"].unique()
        excluding_task_sources = set(wrangle_params["exclude"])
        assert set(wrangle_params["exclude"]) <= set(
            unique_task_sources
        ), "All excluded task sources must be present in the data"
        logging.info(f"Excluding task sources: {excluding_task_sources}")
        runs = runs[~runs["task_source"].isin(excluding_task_sources)]

    # Load bootstrap results if available
    bootstrap_results = None
    if bootstrap_file is not None and bootstrap_file.exists():
        bootstrap_results = pd.read_csv(bootstrap_file)
        logging.info(f"Loaded bootstrap results from {bootstrap_file}")

    logging.info(f"Running logistic regressions for {len(runs)} runs")
    score_col = wrangle_params.get("score_col", "score_binarized")
    regressions = runs.groupby("agent", as_index=False).apply(
        lambda x: agent_regression(
            x["human_minutes"].values,  # type: ignore
            x[score_col].values,  # type: ignore
            weights=weights_fn(x),  # type: ignore
            agent_name=x.name,  # type: ignore
            regularization=wrangle_params["regularization"],
            success_percents=wrangle_params["success_percents"],
            confidence_level=wrangle_params["confidence_level"],
            bootstrap_results=bootstrap_results,
            include_empirical_rates=include_empirical_rates,
        )  # type: ignore
    )  # type: ignore

    regressions["release_date"] = regressions["agent"].map(release_dates["date"])
    # Round numeric columns to 6 decimal places
    numeric_columns = regressions.select_dtypes(include=["float64", "float32"]).columns
    regressions[numeric_columns] = regressions[numeric_columns].round(6)
    return regressions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig-name", type=str, required=True)
    parser.add_argument("--runs-file", type=pathlib.Path, required=True)
    parser.add_argument("--output-logistic-fits-file", type=pathlib.Path, required=True)
    parser.add_argument("--release-dates", type=pathlib.Path, required=True)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--bootstrap-file", type=pathlib.Path)
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    params = dvc.api.params_show(stages="wrangle_logistic_regression", deps=True)
    wrangle_params = params["figs"]["wrangle_logistic"][args.fig_name]

    runs = pd.read_json(
        args.runs_file, lines=True, orient="records", convert_dates=False
    )
    logging.info(f"Loaded {len(runs)} runs")

    regressions = run_logistic_regressions(
        runs,
        args.release_dates,
        wrangle_params,
        args.bootstrap_file,
    )
    logging.info("\n" + str(regressions))
    logging.info(f"Mean BCE loss: {regressions.bce_loss.mean():.3f}")
    regressions.to_csv(args.output_logistic_fits_file)

    logging.info(f"Saved logistic fits to {args.output_logistic_fits_file}")


if __name__ == "__main__":
    main()

# %%
