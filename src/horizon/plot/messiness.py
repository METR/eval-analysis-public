from __future__ import annotations

import argparse
import logging
import pathlib
from typing import Any, Dict, List, Literal, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import (
    BinaryResultsWrapper,
    L1BinaryResultsWrapper,
)
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa

# Binary factors to sum (excluding implicit generalizability which needs special handling)
BINARY_FACTORS = [
    "real life source",
    # "# sourced from metr internal work",
    "resource limited",
    "not easily resettable",
    "irreversible mistake availability",
    "dynamic environment",
    "difficult counterfactuals",
    "not purely automatic scoring",
    "non explicit scoring description",
    "is suboptimal behavior exploited",
    "no provided verification mechanisms",
    "realtime coordination",
    "self modification required",
    "self improvement required",
    "information seeking required",
    "novel situation",
]

EXCLUDE_LABELLERS = ["Machine User"]

IMPLICIT_GENERALIZABILITY_FACTORS = [
    "implicit generalizability required",
    "implicit generalizability requirements are scored",
]
IMPLICIT_GEN_COL_NAME = "implicit_generalizability_required_and_scored"
TIME_BUCKETS = [(1, 15), (16, 60), (60, 240), (240, 2_000_000)]


def _get_tasks(
    df: pd.DataFrame,
    messiness_tasks_path: pathlib.Path,
) -> List[str]:
    tasks_in_df = set(df["task id"].unique())
    tasks_in_messiness = set(pd.read_csv(messiness_tasks_path)["task_id"].unique())
    tasks_in_both = tasks_in_df & tasks_in_messiness
    logging.info(f"Number of tasks in both df and messiness: {len(tasks_in_both)}")
    return list(tasks_in_both)


def _create_factor_sum_column(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a factor sum column to the dataframe."""
    # First check the data types and values
    logging.debug("Factor values before conversion:")
    for factor in BINARY_FACTORS:
        unique_vals = df[factor].unique()
        logging.debug(
            f"{factor}: unique values = {unique_vals}, dtype = {df[factor].dtype}"
        )
    binary_factor_sum = df[BINARY_FACTORS].sum(axis=1)

    implicit_gen = df[IMPLICIT_GENERALIZABILITY_FACTORS].all(axis=1)
    logging.debug(
        f"Implicit generalizability: unique values = {implicit_gen.unique()}, dtype = {implicit_gen.dtype}"
    )

    factor_sum = binary_factor_sum + implicit_gen
    logging.debug(
        f"Factor sum: unique values = {factor_sum.unique()}, dtype = {factor_sum.dtype}"
    )
    return df.assign(factor_sum=factor_sum)


def _exclude_tasks_not_in_column(
    df: pd.DataFrame,
    column: str,
    messiness_tasks_path: pathlib.Path,
) -> pd.DataFrame:
    df_messiness_tasks = pd.read_csv(messiness_tasks_path)
    in_test_set = df_messiness_tasks[df_messiness_tasks[column]]["task_id"].unique()
    return df[df["task id"].isin(in_test_set)]


def _exclude_tasks_marked_for_exclusion_by_person(
    df: pd.DataFrame,
    person: str = "Megan Kinniment",
    column: str = "exclude task from analysis",
) -> pd.DataFrame:
    person_labels = df[df["labeller"] == person]
    tasks_marked_for_exclusion_by_person = person_labels[
        person_labels[column].fillna(False)
    ]["task id"].unique()
    return df[~df["task id"].isin(tasks_marked_for_exclusion_by_person)]


def _only_model_alias(runs_df: pd.DataFrame, alias: str) -> pd.DataFrame:
    return runs_df[runs_df["alias"] == alias]


def _prepare_data_for_analysis(
    df_messiness: pd.DataFrame,
    df_runs: pd.DataFrame,
    alias: str | None = None,
    remove_ai_rd: bool = True,
) -> pd.DataFrame:
    """Prepare data for analysis by merging messiness and runs data, and creating derived columns.
    If there are multiple labellers for a task, their labels will be averaged."""
    # First average the labels across labellers for each task
    label_columns = BINARY_FACTORS + [
        "implicit generalizability required",
        "implicit generalizability requirements are scored",
    ]

    logging.info(
        f"Initial number of unique tasks in messiness data: {df_messiness['task id'].nunique()}"
    )
    logging.info(
        f"Initial number of unique tasks in runs data: {df_runs['task_id'].nunique()}"
    )
    if remove_ai_rd:
        df_runs = df_runs[~df_runs["task_id"].str.contains("ai_rd_")]
        logging.info(
            f"Number of unique tasks in runs data after removing AI-RD: {df_runs['task_id'].nunique()}"
        )
        df_messiness = df_messiness[~df_messiness["task id"].str.contains("ai_rd_")]

    if alias:
        df_runs = _only_model_alias(df_runs, alias)
    df_messiness_avg = (
        df_messiness.groupby("task id")[label_columns].mean().reset_index()
    )

    logging.info(
        f"Number of unique tasks after averaging labels: {len(df_messiness_avg)}"
    )

    # Create factor sum from averaged labels
    binary_factor_sum = df_messiness_avg[BINARY_FACTORS].sum(axis=1)
    implicit_gen = (
        df_messiness_avg["implicit generalizability required"].astype(bool)
        & df_messiness_avg["implicit generalizability requirements are scored"]
        .fillna(False)
        .astype(bool)
    ).astype(float)
    df_messiness_avg["factor_sum"] = binary_factor_sum + implicit_gen

    # Keep individual runs instead of averaging
    logging.info(f"Number of runs before merging: {len(df_runs)}")

    # Merge with averaged messiness data
    analysis_df = df_messiness_avg.merge(
        df_runs, left_on="task id", right_on="task_id", how="inner"
    )
    logging.info(
        f"Number of runs after merging with messiness data: {len(analysis_df)}"
    )
    logging.info(
        f"Number of unique tasks after merging: {analysis_df['task_id'].nunique()}"
    )

    # Create Log10 Human Time-To-Complete (Minutes), handling zeros by adding a small constant
    analysis_df["log_human_time"] = np.log10(
        analysis_df["human_minutes"],
    )
    print(
        f"Human Time-To-Complete (Minutes) stats: {analysis_df['human_minutes'].describe()}"
    )
    print(
        f"Log10 Human Time-To-Complete (Minutes) stats: {analysis_df['log_human_time'].describe()}"
    )

    # Create implicit generalizability column (AND of required and scored)
    analysis_df["implicit_gen"] = implicit_gen

    # Log tasks that were in messiness data but not in runs data
    missing_tasks = set(df_messiness_avg["task id"]) - set(df_runs["task_id"])
    if missing_tasks:
        logging.info(
            f"Tasks in messiness data but not in runs data: {sorted(missing_tasks)}"
        )

    # Log tasks that were in runs data but not in messiness data
    extra_tasks = set(df_runs["task_id"]) - set(df_messiness_avg["task id"])
    if extra_tasks:
        logging.info(
            f"Tasks in runs data but not in messiness data: {sorted(extra_tasks)}"
        )

    return analysis_df


def _add_implicit_generalizability_required_and_scored(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df["implicit_generalizability_required_and_scored"] = (
        df["implicit generalizability required"].astype(bool)
        & df["implicit generalizability requirements are scored"]
        .fillna(False)
        .astype(bool)
    ).astype(float)
    return df


def _analyze_factor_relationships_linear(
    df_messiness: pd.DataFrame, df_runs: pd.DataFrame
) -> None:
    """Analyze relationships between factor sum, human time, and model performance using OLS."""
    # Calculate mean human time and success rate per task
    task_stats = (
        df_runs.groupby("task_id")
        .agg(
            {
                "human_minutes": "mean",
                "score_binarized": "mean",  # Average performance across all models
            }
        )
        .reset_index()
    )

    # Merge with messiness data
    analysis_df = df_messiness.merge(
        task_stats, left_on="task id", right_on="task_id", how="inner"
    )

    # Create Log10 Human Time-To-Complete (Minutes), handling zeros by adding a small constant
    analysis_df["log_human_time"] = np.log(analysis_df["human_minutes"].clip(lower=0.1))
    analysis_df = _add_implicit_generalizability_required_and_scored(analysis_df)
    # Drop any rows with NaN values
    analysis_df = analysis_df.dropna(
        subset=["score_binarized", "factor_sum", "log_human_time"]
        + BINARY_FACTORS
        + ["implicit_generalizability_required_and_scored"]
    )

    # Prepare variables for regression
    y_perf = analysis_df["score_binarized"]

    # Single-factor regressions (without controls)
    X_factor = sm.add_constant(analysis_df["factor_sum"].astype(float))

    # Add debug logging for data types
    logging.debug("\nData Types:")
    logging.debug(f"y_perf dtype: {y_perf.dtype}")
    if isinstance(X_factor, pd.DataFrame):
        logging.debug("\nX_factor dtypes:")
        logging.debug(X_factor.dtypes)
    else:
        logging.debug(f"\nX_factor type: {type(X_factor)}")
        if hasattr(X_factor, "dtype"):
            logging.debug(f"X_factor dtype: {X_factor.dtype}")

    X_time = sm.add_constant(analysis_df["log_human_time"].astype(float))
    model_perf_factor = sm.OLS(y_perf, X_factor).fit()
    model_perf_time = sm.OLS(y_perf, X_time).fit()

    # Multiple regression (controlling for both factors)
    X_both = sm.add_constant(
        analysis_df[["factor_sum", "log_human_time"]].astype(float)
    )
    model_perf_both = sm.OLS(y_perf, X_both).fit()

    # Multiple regression with all individual factors
    # Convert all binary factors to float
    binary_factors_df = analysis_df[BINARY_FACTORS].astype(float)

    # Handle implicit generalizability
    implicit_gen = (
        analysis_df["implicit generalizability required"].astype(bool)
        & analysis_df[IMPLICIT_GEN_COL_NAME].fillna(False).astype(bool)
    ).astype(float)

    # Combine all features
    X_all_factors = pd.concat(
        [
            binary_factors_df,
            pd.Series(
                implicit_gen,
                name=IMPLICIT_GEN_COL_NAME,
            ),
            analysis_df["log_human_time"].astype(float),
        ],
        axis=1,
    )

    # Add constant and ensure no NaN values
    X_all_factors = sm.add_constant(X_all_factors)

    # Log the shape of data and check for any remaining NaN values
    logging.debug("\nData shape before regression:")
    logging.debug(f"y_perf shape: {y_perf.shape}")
    logging.debug(f"X_all_factors shape: {X_all_factors.shape}")
    logging.debug(f"Number of NaN values in y_perf: {np.isnan(y_perf).sum()}")
    logging.debug(
        f"Number of NaN values in X_all_factors: {np.isnan(X_all_factors).sum()}"
    )

    model_perf_all = sm.OLS(y_perf, X_all_factors).fit()

    logging.info("\nRegression Results:")

    logging.info("\nModel 1: Model Performance ~ Factor Sum (without controls)")
    logging.info(f"R-squared: {model_perf_factor.rsquared:.3f}")
    logging.info(model_perf_factor.summary().tables[1])

    logging.info(
        "\nModel 2: Model Performance ~ log10(human minutes-to-complete) (without controls)"
    )
    logging.info(f"R-squared: {model_perf_time.rsquared:.3f}")
    logging.info(model_perf_time.summary().tables[1])

    logging.info(
        "\nModel 3: Model Performance ~ Factor Sum + Log10 Human Time-To-Complete (Minutes) (with controls)"
    )
    logging.info(f"R-squared: {model_perf_both.rsquared:.3f}")
    logging.info(model_perf_both.summary().tables[1])

    logging.info(
        "\nModel 4: Model Performance ~ All Individual Factors + Log10 Human Time-To-Complete (Minutes)"
    )
    logging.info(f"R-squared: {model_perf_all.rsquared:.3f}")
    logging.info(model_perf_all.summary().tables[1])


def _create_logistic_heatmap_plot(
    X: pd.DataFrame,
    y: Union[pd.Series[int], pd.Series[float]],
    model: Union[BinaryResultsWrapper, L1BinaryResultsWrapper],
    alpha: float,
    output_plots_dir: pathlib.Path,
    test_set: str,
    label_type: str,
) -> None:
    """Create a heatmap plot comparing predicted probabilities with actual observations."""
    # Create a grid of points for the heatmap
    factor_sum_range = np.linspace(X["factor_sum"].min(), X["factor_sum"].max(), 100)
    log_time_range = np.linspace(
        X["log_human_time"].min(), X["log_human_time"].max(), 100
    )
    xx, yy = np.meshgrid(factor_sum_range, log_time_range)

    # Create input data for predictions
    grid_points = np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()]

    # Get predictions
    z = model.predict(grid_points)
    z = z.reshape(xx.shape)

    # Create the figure with three subplots
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(24, 12))

    # Adjust spacing between subplot rows
    plt.subplots_adjust(hspace=0.4)

    # Plot 1: Model predictions
    im1 = ax1.pcolormesh(xx, yy, z, cmap="RdYlBu", shading="auto")
    plt.colorbar(im1, ax=ax1, label="Predicted Probability")
    ax1.set_xlabel("Task Messiness Score")
    ax1.set_ylabel("Log10 Human Time-To-Complete (Minutes)")
    ax1.set_title(
        f"Model Predictions\n(alpha={alpha:.3f})\n(factor_sum={float(model.params[1]):.2f}\nconst={float(model.params[0]):.2f}, log_time={float(model.params[2]):.2f}",
        fontsize=10,
    )

    # Plot 2: Task-level averaged data
    task_data = (
        pd.DataFrame(
            {
                "factor_sum": X["factor_sum"],
                "log_human_time": X["log_human_time"],
                "outcome": y,
            }
        )
        .groupby(["factor_sum", "log_human_time"])["outcome"]
        .mean()
        .reset_index()
    )

    scatter2 = ax2.scatter(
        task_data["factor_sum"],
        task_data["log_human_time"],
        c=task_data["outcome"],
        cmap="RdYlBu",
        alpha=0.6,
        s=50,
    )
    plt.colorbar(scatter2, ax=ax2, label="Observed Success Rate")
    ax2.set_xlabel("Task Messiness Score")
    ax2.set_ylabel("Task Length for Human\n(log10 minutes)")
    ax2.set_title(
        "Observed Model Success Rates\nby Human Time-to-Complete and Messiness"
    )

    # Plot 3: Individual runs with jitter
    jitter_amount = 0.1
    jittered_x = X["factor_sum"] + np.random.normal(0, jitter_amount, len(X))
    jittered_y = X["log_human_time"] + np.random.normal(0, jitter_amount, len(X))

    scatter3 = ax3.scatter(
        jittered_x,
        jittered_y,
        c=y,
        cmap="RdYlBu",
        alpha=0.15,
        s=20,
        marker="x",
    )
    plt.colorbar(scatter3, ax=ax3, label="Run Outcome (0/1)")
    ax3.set_xlabel("Task Messiness Score")
    ax3.set_ylabel("Log10 Human Time-To-Complete (Minutes)")
    ax3.set_title(f"Individual Runs (n={len(y)}, jitter={jitter_amount})")

    # Plot 4: Residuals (observed - predicted)
    # Get model predictions for actual data points
    X_with_const = sm.add_constant(X)
    predicted_probs = model.predict(X_with_const)

    # Calculate task-level residuals
    task_residuals = (
        pd.DataFrame(
            {
                "factor_sum": X["factor_sum"],
                "log_human_time": X["log_human_time"],
                "observed": y,
                "predicted": predicted_probs,
            }
        )
        .groupby(["factor_sum", "log_human_time"])
        .agg(
            {
                "observed": "mean",
                "predicted": "mean",
            }
        )
        .reset_index()
    )
    task_residuals["residual"] = (
        task_residuals["observed"] - task_residuals["predicted"]
    )

    scatter4 = ax4.scatter(
        task_residuals["factor_sum"],
        task_residuals["log_human_time"],
        c=task_residuals["residual"],
        cmap="RdBu",  # Red-Blue diverging colormap centered at 0
        alpha=0.6,
        s=50,
    )
    plt.colorbar(scatter4, ax=ax4, label="Residual (Observed - Predicted)")
    ax4.set_xlabel("Task Messiness Score")
    ax4.set_ylabel("Log10 Human Time-To-Complete (Minutes)")
    ax4.set_title("Model Residuals by Messiness and Time")

    # Plot 5: Residuals from time-only model
    # Create DataFrame from X array
    X_df = pd.DataFrame(X, columns=["factor_sum", "log_human_time"])

    # Fit logistic regression with just Log10 Human Time-To-Complete (Minutes)
    X_time_only = sm.add_constant(X_df[["log_human_time"]])
    time_only_model = sm.Logit(y, X_time_only).fit(method="newton", maxiter=1000)
    print(f"Time-only model summary: {time_only_model.summary()}")
    predicted_probs_time_only = time_only_model.predict(X_time_only)

    # Calculate task-level residuals for time-only model
    task_residuals_time = (
        pd.DataFrame(
            {
                "factor_sum": X_df["factor_sum"],
                "log_human_time": X_df["log_human_time"],
                "observed": y,
                "predicted": predicted_probs_time_only,
            }
        )
        .groupby(["factor_sum", "log_human_time"])
        .agg(
            {
                "observed": "mean",
                "predicted": "mean",
            }
        )
        .reset_index()
    )
    task_residuals_time["residual"] = (
        task_residuals_time["observed"] - task_residuals_time["predicted"]
    )

    scatter5 = ax5.scatter(
        task_residuals_time["factor_sum"],
        task_residuals_time["log_human_time"],
        c=task_residuals_time["residual"],
        cmap="RdBu",  # Red-Blue diverging colormap centered at 0
        alpha=0.6,
        s=50,
    )
    plt.colorbar(
        scatter5,
        ax=ax5,
        label="Residual (Observed - Predicted from Human Time-To-Complete)",
    )
    ax5.set_xlabel("Task Messiness Score")
    ax5.set_ylabel("Residual (Observed - Predicted from Time)")
    ax5.set_title(
        "Excess Success Rate compared to \nprediction from Log10 Human Time-To-Complete (Minutes)"
    )

    # Plot trend line using factor sum
    factor_sum_range = np.linspace(
        task_residuals_time["factor_sum"].min(),
        task_residuals_time["factor_sum"].max(),
        100,
    )
    # Fit a simple linear regression between factor sum and residuals for the trend line
    z = np.polyfit(
        task_residuals_time["factor_sum"], task_residuals_time["residual"], 1
    )
    p = np.poly1d(z)
    ax6.plot(
        factor_sum_range,
        p(factor_sum_range),
        "k--",
        alpha=0.5,
    )

    # Plot 6: Time-only residuals vs Task Messiness Score
    _ = ax6.scatter(
        task_residuals_time["factor_sum"],
        task_residuals_time["residual"],
        # c=task_residuals_time["observed"],  # Color by observed success rate
        cmap="RdYlBu",
        marker="x",
        alpha=0.6,
        s=50,
    )
    # plt.colorbar(scatter6, ax=ax6, label="Observed Success Rate")

    # Add a horizontal line at y=0 to show where residuals cross zero
    ax6.axhline(y=0, color="gray", linestyle="dotted", alpha=0.3)

    # Calculate R-squared of the trend line
    residuals = task_residuals_time["residual"].to_numpy() - p(
        task_residuals_time["factor_sum"].to_numpy()
    )
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum(
        (
            task_residuals_time["residual"].to_numpy()
            - np.mean(task_residuals_time["residual"].to_numpy())
        )
        ** 2
    )
    r_squared = 1 - (ss_res / ss_tot)

    ax6.set_xlabel("Task Messiness Score")
    ax6.set_ylabel("Residual (Observed - Predicted Success Rate)")
    # ax6.set_title(
    #     f"Observed vs Predicted Success\nRate over Task Messiness\n(slope={z[0]:.3f}, R²={r_squared:.3f})"
    # )
    ax6.set_title("Observed vs Predicted Success\nRate over Task Messiness")

    # Save just plot 6
    six_plot_path = (
        output_plots_dir
        / f"messiness_effect_{test_set}_{label_type}_alpha_{alpha:.3f}.png"
    )
    six_fig = plt.figure(figsize=(6, 6))
    six_ax = six_fig.add_subplot(111)
    six_ax.plot(
        factor_sum_range,
        p(factor_sum_range),
        "k--",
        alpha=0.5,
    )
    six_ax.scatter(
        task_residuals_time["factor_sum"],
        task_residuals_time["residual"],
        # c=task_residuals_time["observed"],  # Color by observed success rate
        color="#eb4f34",
        marker="x",
        alpha=0.6,
        s=50,
    )
    six_ax.axhline(y=0, color="gray", linestyle="dotted", alpha=0.3)

    # Calculate R-squared of the trend line
    residuals = task_residuals_time["residual"].to_numpy() - p(
        task_residuals_time["factor_sum"].to_numpy()
    )
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum(
        (
            task_residuals_time["residual"].to_numpy()
            - np.mean(task_residuals_time["residual"].to_numpy())
        )
        ** 2
    )
    r_squared = 1 - (ss_res / ss_tot)

    six_ax.set_xlabel("Task Messiness Score", fontsize=15)
    six_ax.set_ylabel("Residual (Observed - Predicted Success Rate)", fontsize=15)
    # six_ax.set_title(
    #     f"Observed vs Predicted Success\nRate over Task Messiness\n(slope={z[0]:.3f}, R²={r_squared:.3f})"
    # )
    six_ax.grid(True, alpha=0.15)
    six_ax.set_title("Excess Success Rate vs Task Messiness Score", y=1.05, fontsize=18)
    # Annotate in upper right with R²
    six_ax.text(
        0.95,
        0.95,
        f"R²={r_squared:.3f}",
        transform=six_ax.transAxes,
        ha="right",
        fontsize=15,
    )
    six_fig.savefig(six_plot_path, dpi=500, bbox_inches="tight")
    plt.close()

    # Add overall title
    plt.suptitle(
        f"Logistic Regression Results - {test_set.title()} Test Set, {label_type.title()} Labels",
        y=1,
    )

    # Save the plot
    plot_path = (
        output_plots_dir
        / f"logistic_heatmap_{test_set}_{label_type}_alpha_{alpha:.3f}.png"
    )
    plt.savefig(plot_path, dpi=500, bbox_inches="tight")
    plt.close()


def _analyze_logistic_regression_sensitivity(
    df_messiness: pd.DataFrame,
    df_runs: pd.DataFrame,
    analysis_file: pathlib.Path,
    output_plots_dir: pathlib.Path,
    n_splits: int = 5,
    alphas: List[float] = [0.01],
    test_set: str = "",
    label_type: str = "",
    alias: str | None = None,
) -> Dict[str, Dict[Any, Any]]:
    """Analyze logistic regression sensitivity to data splits and regularization."""
    analysis_df = _prepare_data_for_analysis(df_messiness, df_runs, alias=alias)
    analysis_df = _add_implicit_generalizability_required_and_scored(analysis_df)
    results: Dict[str, Dict[Any, Any]] = {
        "alpha_sweep": {},
        "split_sensitivity": {},
    }
    model = None

    with open(analysis_file, "a") as f:
        f.write("\nLogistic Regression Sensitivity Analysis\n")
        f.write(f"Test Set: {test_set}, Label Type: {label_type}\n")
        f.write(f"Number of splits: {n_splits}, Alpha values: {alphas}\n")
        f.write("=" * 80 + "\n\n")

    # Use individual run outcomes directly
    y = analysis_df["score_binarized"].astype(int).to_numpy()
    X = analysis_df[["factor_sum", "log_human_time"]].astype(float).to_numpy()
    X_with_const = sm.add_constant(X)
    # Create heatmap plot for this alpha value
    X_df = pd.DataFrame(X, columns=["factor_sum", "log_human_time"])
    y_series = pd.Series(y, dtype=int, name="score_binarized")

    # Alpha sweep - try both unregularized and regularized fits
    for alpha in alphas:
        try:
            # Try unregularized fit first
            model_unreg = sm.Logit(y, X_with_const).fit(method="newton", maxiter=1000)

            # Then try regularized fit
            model_l1 = sm.Logit(y, X_with_const).fit_regularized(
                method="l1",
                alpha=alpha,  # type: ignore
                maxiter=1000,  # type: ignore
            )

            # Use the model with better pseudo R-squared
            model = (
                model_unreg if model_unreg.prsquared > model_l1.prsquared else model_l1
            )

            results["alpha_sweep"][alpha] = {
                "pseudo_r2": model.prsquared,
                "coefficients": {
                    "const": float(model.params[0]),
                    "factor_sum": float(model.params[1]),
                    "log_human_time": float(model.params[2]),
                },
            }

        except Exception as e:
            logging.warning(f"Failed to fit model with alpha={alpha}: {str(e)}")
        if results["alpha_sweep"].get(alpha) is None:
            continue
        else:
            assert model is not None
            _create_logistic_heatmap_plot(
                X=X_df,
                y=y_series,
                model=model,
                alpha=alpha,
                output_plots_dir=output_plots_dir,
                test_set=test_set,
                label_type=label_type,
            )

            with open(analysis_file, "a") as f:
                f.write(f"\nAlpha = {alpha:.3f} \n")
                f.write("-" * 40 + "\n")
                f.write(f"Pseudo R²: {model.prsquared:12.3f}\n")
                f.write("Coefficients:\n")
                f.write(f"{'const':>20}: {float(model.params[0]):10.3f}\n")
                f.write(f"{'factor_sum':>20}: {float(model.params[1]):10.3f}\n")
                f.write(f"{'log_human_time':>20}: {float(model.params[2]):10.3f}\n")
                f.write("-" * 40 + "\n")

    # Split sensitivity
    with open(analysis_file, "a") as f:
        f.write("\nSplit Sensitivity Results:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Split':>6} {'Train R²':>10} {'Test Acc':>10} {'Coefficients'}\n")
        f.write("-" * 80 + "\n")

    n_samples = len(analysis_df)
    for split in range(n_splits):
        try:
            # Generate train/test indices
            rng = np.random.default_rng()
            all_indices = np.arange(n_samples)
            rng.shuffle(all_indices)
            split_size = n_samples // 2
            train_indices = all_indices[:split_size]
            test_indices = all_indices[split_size:]

            # Select data using numpy indexing
            train_X = X_with_const[train_indices]
            train_y = y[train_indices]
            test_X = X_with_const[test_indices]
            test_y = y[test_indices]

            # Try both unregularized and regularized fits
            model_unreg = sm.Logit(train_y, train_X).fit(method="newton", maxiter=1000)
            model_l1 = sm.Logit(train_y, train_X).fit_regularized(
                method="l1",
                alpha=0.01,  # type: ignore
                maxiter=1000,  # type: ignore
            )
            model = (
                model_unreg if model_unreg.prsquared > model_l1.prsquared else model_l1
            )

            train_pseudo_r2 = model.prsquared
            test_pred = model.predict(test_X)
            test_acc = ((test_pred >= 0.5) == test_y).mean()

            results["split_sensitivity"][split] = {
                "train_pseudo_r2": train_pseudo_r2,
                "test_accuracy": test_acc,
                "coefficients": {
                    "const": float(model.params[0]),
                    "factor_sum": float(model.params[1]),
                    "log_human_time": float(model.params[2]),
                },
            }
            with open(analysis_file, "a") as f:
                f.write(
                    f"{split:6d} {train_pseudo_r2:10.3f} {test_acc:10.3f} {dict(zip(['const', 'factor_sum', 'log_human_time'], model.params))} \n"
                )
        except Exception as e:
            logging.warning(f"Failed to fit model for split {split}: {str(e)}")

    with open(analysis_file, "a") as f:
        f.write("\n" + "=" * 80 + "\n\n")

    return results


def _get_high_low_messiness_by_time_bucket(
    df_messiness: pd.DataFrame,
    df_runs: pd.DataFrame,
    n_tasks: int = 5,
) -> Tuple[pd.DataFrame, Dict[Tuple[int, int], pd.DataFrame]]:
    """Get the highest and lowest messiness tasks overall and within each time bucket."""
    analysis_df = _prepare_data_for_analysis(df_messiness, df_runs)

    # First get overall top and bottom tasks by factor sum
    sorted_df = analysis_df.sort_values("factor_sum", ascending=False)
    high_mess = sorted_df.head(20)[["task id", "factor_sum", "human_minutes"]]
    low_mess = sorted_df.tail(20)[["task id", "factor_sum", "human_minutes"]]
    overall_results = pd.concat(
        [high_mess.assign(category="high"), low_mess.assign(category="low")]
    )

    # Then get top/bottom tasks within each time bucket
    bucket_results = {}
    for start, end in TIME_BUCKETS:
        bucket_df = analysis_df[
            (analysis_df["human_minutes"] >= start)
            & (analysis_df["human_minutes"] < end)
        ]

        # Sort by factor sum and get top/bottom n tasks
        sorted_bucket_df = bucket_df.sort_values("factor_sum", ascending=False)
        high_mess = sorted_bucket_df.head(n_tasks)[
            ["task id", "factor_sum", "human_minutes"]
        ]
        low_mess = sorted_bucket_df.tail(n_tasks)[
            ["task id", "factor_sum", "human_minutes"]
        ]

        bucket_results[(start, end)] = pd.concat(
            [high_mess.assign(category="high"), low_mess.assign(category="low")]
        )

    return overall_results, bucket_results


def _calculate_fleiss_kappa(
    df_messiness: pd.DataFrame, test_set: str, output_file: pathlib.Path
) -> Dict[str, float]:
    """Calculate Fleiss' kappa for inter-rater agreement on binary factors."""
    kappa_scores = {}
    results_data = []

    # First calculate kappa for binary factors
    for factor in BINARY_FACTORS + IMPLICIT_GENERALIZABILITY_FACTORS:
        try:
            # Get all ratings for this factor
            # First ensure we have unique task IDs
            task_ratings = df_messiness.pivot_table(
                index="task id",
                columns="labeller",
                values=factor,
                aggfunc="first",  # Take first value if there are duplicates
            )

            # Drop any tasks that don't have ratings from all raters
            task_ratings = task_ratings.dropna()

            if (
                len(task_ratings) > 0 and len(task_ratings.columns) >= 2
            ):  # Need at least 2 raters and 1 task
                # Convert to numpy array and ensure integer type
                ratings_array = task_ratings.astype(int).to_numpy()

                # Convert to format expected by fleiss_kappa using aggregate_raters
                # Since we have binary data (0/1), we know n_cat=2
                table, _ = aggregate_raters(ratings_array, n_cat=2)

                # Calculate kappa
                kappa = float(fleiss_kappa(table, method="fleiss"))
                kappa_scores[factor] = kappa
                results_data.append(
                    {
                        "Factor": factor,
                        "Kappa": kappa,
                        "Tasks": len(task_ratings),
                        "Raters": len(task_ratings.columns),
                    }
                )
        except Exception as e:
            logging.warning(f"Could not calculate kappa for {factor}: {str(e)}")
            results_data.append(
                {"Factor": factor, "Kappa": float("nan"), "Tasks": 0, "Raters": 0}
            )

    # Calculate agreement on total factor sum
    try:
        # Calculate factor sum per labeller
        df_with_sum = _create_factor_sum_column(df_messiness)
        task_sums = df_with_sum.pivot_table(
            index="task id", columns="labeller", values="factor_sum", aggfunc="first"
        )
        task_sums = task_sums.dropna()

        if len(task_sums) > 0 and len(task_sums.columns) >= 2:
            # Calculate correlation between raters
            correlations = task_sums.corr()
            # Average correlation (excluding self-correlations)
            mask = ~np.eye(correlations.shape[0], dtype=bool)
            mean_correlation = correlations.where(mask).mean().mean()

            results_data.append(
                {
                    "Factor": "Total Messiness Score",
                    "Kappa": mean_correlation,  # Using correlation instead of kappa for continuous measure
                    "Tasks": len(task_sums),
                    "Raters": len(task_sums.columns),
                }
            )
            kappa_scores["Total Messiness Score"] = mean_correlation
    except Exception as e:
        logging.warning(f"Could not calculate agreement for total factor sum: {str(e)}")
        results_data.append(
            {
                "Factor": "Total Messiness Score",
                "Kappa": float("nan"),
                "Tasks": 0,
                "Raters": 0,
            }
        )

    # Create and display results table
    results_df = pd.DataFrame(results_data)
    results_df = results_df.sort_values("Kappa", ascending=False)

    # Format the table
    table_str = f"\nInter-rater Agreement Analysis ({test_set.upper()} Test Set)\n"
    table_str += "=" * 80 + "\n"
    table_str += "Analysis of agreement between human raters on binary factors\n"
    table_str += f"Total tasks analyzed: {results_df['Tasks'].max()}, Number of raters: {results_df['Raters'].max()}\n"
    table_str += "=" * 80 + "\n"
    table_str += f"{'Factor':<40} {'Kappa':>10} {'Tasks':>10}\n"
    table_str += "-" * 80 + "\n"

    # First output total messiness score if present
    total_score_row = results_df[results_df["Factor"] == "Total Messiness Score"]
    if not total_score_row.empty:
        row = total_score_row.iloc[0]
        corr_str = f"{row['Kappa']:.3f}" if not pd.isna(row["Kappa"]) else "nan"
        table_str += f"{row['Factor']:<40} {corr_str:>10} {row['Tasks']:>10}\n"
        table_str += "-" * 80 + "\n"

    # Then output individual factors
    for _, row in results_df[
        results_df["Factor"] != "Total Messiness Score"
    ].iterrows():
        kappa_str = f"{row['Kappa']:.3f}" if not pd.isna(row["Kappa"]) else "nan"
        table_str += f"{row['Factor']:<40} {kappa_str:>10} {row['Tasks']:>10}\n"

    table_str += "=" * 80 + "\n\n"

    # Write to file
    with open(output_file, "a") as f:
        f.write(table_str)

    logging.info(table_str)
    return kappa_scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze messiness factors across different test sets and label types"
    )
    parser.add_argument(
        "--runs-file",
        type=pathlib.Path,
        required=True,
        help="Path to the JSONL file containing run data",
    )
    parser.add_argument(
        "--messiness-file",
        type=pathlib.Path,
        required=True,
        help="Path to CSV file containing messiness labels",
    )
    parser.add_argument(
        "--messiness-tasks-file",
        type=pathlib.Path,
        required=True,
        help="Path to CSV file containing messiness tasks",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    parser.add_argument(
        "--output-plots-dir",
        type=pathlib.Path,
        required=True,
        help="Directory where analysis outputs will be saved (will be created if it doesn't exist)",
    )
    parser.add_argument(
        "--output-data-dir",
        type=pathlib.Path,
        required=True,
        help="Directory where analysis outputs will be saved (will be created if it doesn't exist)",
    )
    parser.add_argument(
        "--alias",
        type=str,
        default=None,
        help="Alias to use for model labels",
    )
    parser.add_argument("--exclude-agent", action="append", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    # Create output directory if it doesn't exist
    args.output_plots_dir.mkdir(parents=True, exist_ok=True)
    args.output_data_dir.mkdir(parents=True, exist_ok=True)

    # Create a single output file for all tables
    analysis_results_file = args.output_data_dir / "analysis_results.txt"

    # Write header to the output file
    with open(analysis_results_file, "w") as f:
        f.write("Messiness Analysis Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated on: {pd.Timestamp.now()}\n")
        f.write("Input files:\n")
        f.write(f"  Runs file: {args.runs_file}\n")
        f.write(f"  Messiness file: {args.messiness_file}\n")

    try:
        # Read data and exclude machine user labeller
        df_messiness = pd.read_csv(args.messiness_file)
        tasks_in_both = _get_tasks(df_messiness, args.messiness_tasks_file)
        df_messiness = df_messiness[df_messiness["task id"].isin(tasks_in_both)]
        df_messiness = df_messiness[~df_messiness["labeller"].isin(EXCLUDE_LABELLERS)]
        # Exclude tasks marked for exclusion by person
        # df_messiness = _exclude_tasks_marked_for_exclusion_by_person(df_messiness)
        df_runs = pd.read_json(args.runs_file, lines=True, orient="records")
        df_runs = df_runs[df_runs["task_id"].isin(tasks_in_both)]
        df_runs = df_runs[~df_runs["alias"].isin(args.exclude_agent)]
        # Create separate dataframes for different label sources
        df_megan = df_messiness[df_messiness["labeller"] == "Megan Kinniment"]
        df_contractor = df_messiness[df_messiness["labeller"] != "Megan Kinniment"]

        test_sets: tuple[Literal["original", "expanded"], ...] = (
            #   "original",
            "expanded",
        )
        label_types: tuple[Literal["megan", "contractor", "combined"], ...] = (
            # "megan",
            # "contractor",
            "combined",
        )

        for test_set in test_sets:
            logging.info(f"\nAnalyzing {test_set} test set...")
            logging.info(
                f"Number of unique tasks: {len(df_messiness['task id'].unique())}"
            )

            for label_type in label_types:
                logging.info(f"\nAnalyzing {label_type} labels...")

                # Select appropriate dataset
                if label_type == "megan":
                    curr_df = df_megan
                elif label_type == "contractor":
                    curr_df = df_contractor
                else:  # combined
                    curr_df = df_messiness

                # Filter to test set
                curr_df = _exclude_tasks_not_in_column(
                    curr_df,
                    f"in_{test_set}_test_set",
                    messiness_tasks_path=args.messiness_tasks_file,
                )
                curr_df = _create_factor_sum_column(curr_df)

                # 1. High vs Low Messiness Analysis
                overall_results, bucket_results = (
                    _get_high_low_messiness_by_time_bucket(curr_df, df_runs)
                )

                # Write overall high/low messiness results
                with open(analysis_results_file, "a") as f:
                    f.write("\nHigh/Low Messiness Analysis\n")
                    f.write(f"Test Set: {test_set}, Label Type: {label_type}\n")
                    f.write("=" * 80 + "\n")
                    f.write("Overall Top and Bottom 10 Tasks by Factor Sum\n")
                    f.write("-" * 80 + "\n")
                    f.write(overall_results.to_string())
                    f.write("\n\n")

                # Write time bucket results
                for (start, end), df in bucket_results.items():
                    with open(analysis_results_file, "a") as f:
                        f.write(f"\nTime Bucket: {start}-{end} minutes\n")
                        f.write("-" * 80 + "\n")
                        f.write(df.to_string())
                        f.write("\n\n")

                # 2. Logistic Regression Analysis
                _ = _analyze_logistic_regression_sensitivity(
                    curr_df,
                    df_runs,
                    alias=args.alias,
                    analysis_file=analysis_results_file,
                    output_plots_dir=args.output_plots_dir,
                    test_set=test_set,
                    label_type=label_type,
                )

                # 3. Inter-rater Agreement (only for combined)
                if label_type == "combined":
                    _ = _calculate_fleiss_kappa(
                        curr_df, test_set, analysis_results_file
                    )

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
