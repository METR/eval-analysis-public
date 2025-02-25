# %%

from __future__ import annotations

import argparse
import logging
import pathlib
from typing import TYPE_CHECKING, Any

import dvc.api
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import yaml

from src.utils.logistic import get_x_for_quantile, logistic_regression

if TYPE_CHECKING:
    from numpy.typing import NDArray


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
    weights: NDArray[Any] | None,
    agent_name: str,
    regularization: float,
    bootstrap_results: pd.DataFrame | None = None,
) -> pd.Series[Any]:
    logging.info(f"Analyzing {agent_name}")
    time_buckets = [1, 4, 16, 64, 256, 960]
    assert np.all((y == 0) | (y == 1)), "y values must be 0 or 1"
    x = np.log2(x).reshape(-1, 1)
    if weights is None:
        weights = np.ones_like(y, dtype=np.float64)

    empirical_rates, average = empirical_success_rates(x, y, time_buckets, weights)

    indices = [
        "coefficient",
        "intercept",
        "bce_loss",
        "50%",
        "50_low",
        "50_high",
        "average",
    ]
    if np.all(y == 0):
        return pd.Series(
            [
                -np.inf,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            index=indices,
        )._append(empirical_rates)  # type: ignore[reportCallIssue]

    model = logistic_regression(
        x, y, sample_weight=weights, regularization=regularization
    )
    if model.coef_[0][0] > 0:
        logging.warning(f"Warning: {agent_name} has positive slope {model.coef_[0][0]}")

    p50_full = np.exp2(get_x_for_quantile(model, 0.5))

    # Get confidence intervals from bootstrap results if available
    if bootstrap_results is not None and agent_name in bootstrap_results.columns:
        p50_low = np.nanquantile(bootstrap_results[agent_name], 0.1)
        p50_high = np.nanquantile(bootstrap_results[agent_name], 0.9)
    else:
        p50_low = float("nan")
        p50_high = float("nan")
        logging.warning(f"No bootstrap results for {agent_name}, using point estimate")

    bce_loss = get_bce_loss(x, y, model, weights)

    return pd.Series(
        [
            model.coef_[0][0],
            model.intercept_[0],  # type: ignore
            bce_loss,
            p50_full,
            p50_low,
            p50_high,
            average,
        ],
        index=indices,
    )._append(empirical_rates)


def run_logistic_regressions(
    runs: pd.DataFrame,
    release_dates_file: pathlib.Path,
    weighting: str,
    regularization: float,
    exclude_task_sources: list[str] | None,
    bootstrap_file: pathlib.Path | None = None,
) -> pd.DataFrame:
    release_dates = yaml.safe_load(release_dates_file.read_text())

    weights_fn = lambda x: x[weighting].values  # noqa: E731
    # rename alias to agent
    runs.rename(columns={"alias": "agent"}, inplace=True)
    if exclude_task_sources is not None:
        unique_task_sources = runs["task_source"].unique()
        excluding_task_sources = set(unique_task_sources) & set(exclude_task_sources)
        logging.info(f"Excluding task sources: {excluding_task_sources}")
        runs = runs[~runs["task_source"].isin(excluding_task_sources)]

    # Load bootstrap results if available
    bootstrap_results = None
    if bootstrap_file is not None and bootstrap_file.exists():
        bootstrap_results = pd.read_csv(bootstrap_file)
        logging.info(f"Loaded bootstrap results from {bootstrap_file}")

    logging.info(f"Running logistic regressions for {len(runs)} runs")
    regressions = runs.groupby("agent", as_index=False).apply(
        lambda x: agent_regression(
            x["human_minutes"].values,  # type: ignore
            x["score_binarized"].values,  # type: ignore
            weights=weights_fn(x),  # type: ignore
            agent_name=x.name,  # type: ignore
            regularization=regularization,
            bootstrap_results=bootstrap_results,
        )  # type: ignore
    )  # type: ignore

    regressions["release_date"] = regressions["agent"].map(release_dates["date"])
    logging.info(regressions)
    logging.info(f"Mean BCE loss: {regressions.bce_loss.mean():.3f}")
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

    params = dvc.api.params_show("public/params.yaml", deps=True)
    fig_params = params["figs"]["plot_logistic_regression"][args.fig_name]

    runs = pd.read_json(
        args.runs_file, lines=True, orient="records", convert_dates=False
    )
    logging.info(f"Loaded {len(runs)} runs")

    regressions = run_logistic_regressions(
        runs,
        args.release_dates,
        fig_params["weighting"],
        fig_params["regularization"],
        fig_params["exclude"] if "exclude" in fig_params else None,
        args.bootstrap_file,
    )
    regressions.to_csv(args.output_logistic_fits_file)


if __name__ == "__main__":
    main()

# %%
