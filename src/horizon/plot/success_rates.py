import argparse
import pathlib
from typing import Dict, List

import matplotlib
import matplotlib.axes
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from pandas.core.series import Series
from sklearn import metrics
from sklearn.linear_model import LinearRegression


def _save_df_as_image(df: pd.DataFrame, path: pathlib.Path) -> None:
    # Calculate figure size based on dataframe dimensions
    # Allow roughly 1 inch per column and 0.3 inches per row
    fig_width = max(8, len(df.columns) * 1.4)
    fig_height = max(6, len(df.index) * 0.3)

    plt.figure(figsize=(fig_width, fig_height))

    # Extract numeric values for coloring
    numeric_df = df.map(
        lambda x: float(x.split()[0]) if isinstance(x, str) and x != "0/0" else 0
    )

    # Round all numbers to 2dp
    numeric_df = numeric_df.round(2)
    df = df.round(2)

    # Create heatmap with white-to-blue gradient
    cmap = sns.light_palette("blue", as_cmap=True)
    plot = sns.heatmap(
        numeric_df, annot=df, fmt="", cmap=cmap, cbar=True, vmin=0, vmax=1
    )

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")

    fig = plot.get_figure()
    fig.tight_layout()  # type: ignore
    fig.savefig(path, bbox_inches="tight", dpi=300)  # type: ignore
    plt.close(fig)  # type: ignore


def _format_time_label(seconds: float) -> str:
    seconds = round(seconds)
    hours = seconds / 3600
    if hours >= 24 * 8:
        return f"{int(hours / 24)}d"
    if hours >= 1:
        return f"{int(hours)} hr" + ("s" if int(hours) > 1 else "")
    if hours >= 1 / 60:
        return f"{int(hours * 60)} min"
    return f"{int(seconds)} sec"


POSSIBLE_TICKS = np.array(
    [
        1 / 60,
        2 / 60,
        4 / 60,
        8 / 60,
        15 / 60,
        30 / 60,
        1,
        2,
        4,
        8,
        15,
        30,
        60,
        120,
        240,
        480,
        960,
        40 * 60,
    ]
)


def _log_x_axis(
    ax: matplotlib.axes.Axes, low_limit: int | None = None, unit: str = "minutes"
) -> None:
    ax.set_xscale("log")
    x_min, x_max = ax.get_xlim()

    multiplier = 60 if unit == "minutes" else 3600
    if low_limit is not None:
        x_min = max(x_min, low_limit / multiplier)
        ax.set_xlim(left=x_min)

    xticks = POSSIBLE_TICKS[(POSSIBLE_TICKS >= x_min) & (POSSIBLE_TICKS <= x_max)]
    labels = [_format_time_label(tick * multiplier) for tick in xticks]

    ax.set_xticks(xticks)
    ax.set_xticklabels(labels)
    ax.xaxis.set_major_locator(
        matplotlib.ticker.FixedLocator([float(x) for x in xticks])
    )
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


# df = pd.read_json(
#     "/Users/megan/Documents/Code/eval-pipeline-2/data/external/all_runs.jsonl",
#     lines=True,
# )
def _load_runs(runs_file: pathlib.Path, exclude_agents: List[str]) -> pd.DataFrame:
    df = pd.read_json(runs_file, lines=True)
    # Drop the human alias
    df = df[df["alias"] != "human"]
    # Drop o3-mini
    df = df[df["alias"] != "o3-mini"]
    df = df[~df["alias"].isin(exclude_agents)]
    # df = _fill_missing_gpt2_runs(df)
    return df


def _load_release_dates(release_dates_file: pathlib.Path) -> dict[str, str]:
    with open(release_dates_file, "r") as f:
        release_dates_raw = yaml.safe_load(f)
    return release_dates_raw["date"]


# with open(
#     "/Users/megan/Documents/Code/eval-pipeline-2/public/data/external/release_dates.yaml",
#     "r",
# ) as f:
#     release_dates_raw = yaml.safe_load(f)


def _calculate_mean(x: Series) -> float:  # type: ignore
    return float(np.round(x.astype(float).mean(), 2))


def _calculate_value(x: Series) -> str:  # type: ignore
    # Makes a string with the number of successes / total
    if len(x) == 0:
        return "0/0"
    return f"{_calculate_mean(x)} (N={len(x)})"


def _make_family_pivot(df: pd.DataFrame, release_dates: dict[str, str]) -> pd.DataFrame:
    family_pivot = df.pivot_table(
        index="task_family",
        columns="alias",
        values="score_binarized",
        aggfunc={"score_binarized": _calculate_value},
    )
    # order columns by release date
    family_pivot = family_pivot[
        sorted(family_pivot.columns, key=lambda x: release_dates[x])
    ]
    return family_pivot


def _save_output(
    output: pd.DataFrame,
    output_plots_dir: pathlib.Path,
    output_data_dir: pathlib.Path,
    file_name: str,
) -> None:
    # Image file extensions
    IMAGE_EXTENSIONS = [".png", ".svg", ".pdf", ".jpg"]
    DATA_EXTENSIONS = [".csv", ".json", ".yaml", ".jsonl"]

    assert output_plots_dir.exists()
    assert output_data_dir.exists()

    if any(file_name.endswith(ext) for ext in IMAGE_EXTENSIONS):
        _save_df_as_image(output, output_plots_dir / file_name)
    elif any(file_name.endswith(ext) for ext in DATA_EXTENSIONS):
        output.to_csv(output_data_dir / file_name)
    else:
        raise ValueError(
            f"File name must end with an image or data extension: {IMAGE_EXTENSIONS + DATA_EXTENSIONS}"
        )

    return None


def _make_task_pivot(df: pd.DataFrame, release_dates: dict[str, str]) -> pd.DataFrame:
    task_pivot = df.pivot_table(
        index="task_id",
        columns="alias",
        values="score_binarized",
        aggfunc={"score_binarized": _calculate_value},
    )

    # order columns by release date
    task_pivot = task_pivot[sorted(task_pivot.columns, key=lambda x: release_dates[x])]

    return task_pivot


def _make_and_save_family_pivot(
    df: pd.DataFrame,
    release_dates: dict[str, str],
    output_plots_dir: pathlib.Path,
    output_data_dir: pathlib.Path,
) -> pd.DataFrame:
    family_pivot = _make_family_pivot(df, release_dates)
    _save_output(family_pivot, output_plots_dir, output_data_dir, "family_pivot.csv")
    _save_output(family_pivot, output_plots_dir, output_data_dir, "family_pivot.png")
    return family_pivot


def _make_and_save_task_pivot(
    df: pd.DataFrame,
    release_dates: dict[str, str],
    output_plots_dir: pathlib.Path,
    output_data_dir: pathlib.Path,
) -> pd.DataFrame:
    task_pivot = _make_task_pivot(df, release_dates)
    _save_output(task_pivot, output_plots_dir, output_data_dir, "task_pivot.csv")
    _save_output(task_pivot, output_plots_dir, output_data_dir, "task_pivot.png")
    return task_pivot


def _make_family_success_ordered_pivot(df: pd.DataFrame) -> pd.DataFrame:
    family_success_ordered_pivot = df.pivot_table(
        index="task_family",
        columns="alias",
        values="score_binarized",
        aggfunc={"score_binarized": _calculate_mean},
    )
    family_success_ordered_pivot["average"] = family_success_ordered_pivot.mean(axis=1)
    family_success_ordered_pivot = family_success_ordered_pivot.sort_values(
        by="average", ascending=False
    )
    family_success_ordered_pivot["human_minutes"] = df.groupby("task_family")[
        "human_minutes"
    ].mean()
    # Move the average success rate and human minutes to the start columns of the pivot
    family_success_ordered_pivot = family_success_ordered_pivot[
        ["average", "human_minutes"]
        + [
            col
            for col in family_success_ordered_pivot.columns
            if col not in ["average", "human_minutes"]
        ]
    ]
    # Rename average and human_minutes to "Average Success Rate" and "Human Minutes"
    family_success_ordered_pivot.rename(
        columns={
            "average": "Average Model Success Rate",
            "human_minutes": "Human Time-to-Complete (mins)",
        },
        inplace=True,
    )
    return family_success_ordered_pivot


def _make_and_save_family_success_ordered_pivot(
    df: pd.DataFrame,
    output_plots_dir: pathlib.Path,
    output_data_dir: pathlib.Path,
) -> pd.DataFrame:
    family_success_ordered_pivot = _make_family_success_ordered_pivot(df)
    _save_output(
        output=family_success_ordered_pivot,
        output_plots_dir=output_plots_dir,
        output_data_dir=output_data_dir,
        file_name="family_success_ordered_pivot.csv",
    )
    _save_output(
        output=family_success_ordered_pivot,
        output_plots_dir=output_plots_dir,
        output_data_dir=output_data_dir,
        file_name="family_success_ordered_pivot.png",
    )
    return family_success_ordered_pivot


def _make_task_success_ordered_pivot(
    df: pd.DataFrame,
) -> pd.DataFrame:
    task_success_ordered_pivot = df.pivot_table(
        index="task_id",
        columns="alias",
        values="score_binarized",
        aggfunc={"score_binarized": _calculate_mean},
    )

    task_success_ordered_pivot["average"] = task_success_ordered_pivot.mean(axis=1)
    task_success_ordered_pivot = task_success_ordered_pivot.sort_values(
        by="average", ascending=False
    )
    task_success_ordered_pivot["human_minutes"] = df.groupby("task_id")[
        "human_minutes"
    ].mean()
    # Move the average success rate and human minutes to the start columns of the pivot
    task_success_ordered_pivot = task_success_ordered_pivot[
        ["average", "human_minutes"]
        + [
            col
            for col in task_success_ordered_pivot.columns
            if col not in ["average", "human_minutes"]
        ]
    ]
    # Rename average and human_minutes to "Average Model Success Rate" and "Human Time-to-Complete (mins)"
    task_success_ordered_pivot.rename(
        columns={
            "average": "Average Model Success Rate",
            "human_minutes": "Human Time-to-Complete (mins)",
        },
        inplace=True,
    )
    return task_success_ordered_pivot


def _make_and_save_task_success_ordered_pivot(
    df: pd.DataFrame,
    output_plots_dir: pathlib.Path,
    output_data_dir: pathlib.Path,
) -> pd.DataFrame:
    task_success_ordered_pivot = _make_task_success_ordered_pivot(df)
    _save_output(
        output=task_success_ordered_pivot,
        output_plots_dir=output_plots_dir,
        output_data_dir=output_data_dir,
        file_name="task_success_ordered_pivot.csv",
    )
    _save_output(
        output=task_success_ordered_pivot,
        output_plots_dir=output_plots_dir,
        output_data_dir=output_data_dir,
        file_name="task_success_ordered_pivot.png",
    )
    return task_success_ordered_pivot


def _make_and_save_task_success_rate_vs_human_completion_time_plot(
    df: pd.DataFrame,
    output_plots_dir: pathlib.Path,
    task_source_styling: Dict[str, Dict[str, str]],
) -> None:
    task_success_ordered_pivot = _make_task_success_ordered_pivot(df)

    # Create scatter plot with regression line
    fig, ax = plt.subplots(figsize=(5, 5))

    y = task_success_ordered_pivot["Average Model Success Rate"]
    x = task_success_ordered_pivot["Human Time-to-Complete (mins)"]

    task_sources = df.groupby("task_id")["task_source"].first()
    for task_source in task_sources.unique():
        ax.scatter(
            x[task_sources == task_source],
            y[task_sources == task_source],
            alpha=0.5,
            marker="x",
            color=task_source_styling[task_source]["color"],
            label=task_source,
        )

    # Fit regression line on log-transformed x data
    model = LinearRegression()
    X = np.log2(x.values).reshape(-1, 1)  # type: ignore
    model.fit(X, y)

    y_pred = model.predict(X)
    r_squared = model.score(X, y)
    adjusted_r_squared = 1 - (1 - r_squared) * (len(y) - 1) / (len(y) - X.shape[1] - 1)
    mse = metrics.mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y, y_pred)
    slope = model.coef_[0]
    intercept = model.intercept_
    num_doublings = -1 / 2 * 1 / slope
    success_rate_drop_after_one_doubling = -slope

    print("\nModel Statistics:")
    print(f"R² Score: {r_squared:.3f}")
    print(f"Adjusted R²: {adjusted_r_squared:.3f}")
    print(f"Root Mean Square Error: {rmse:.3f}")
    print(f"Mean Absolute Error: {mae:.3f}")
    print(f"Slope: {slope:.3f}")
    print(f"Intercept: {intercept:.3f}")
    print(f"Number of Doublings to Halve Success Rate: {num_doublings:.3f}")
    print(
        f"Success Rate Drop After One Doubling: {success_rate_drop_after_one_doubling:.3f}"
    )
    # Generate points for the regression line
    x_range = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
    X_range = np.log2(x_range).reshape(-1, 1)
    y_pred = model.predict(X_range)

    # Set up log x-axis and plot regression line
    _log_x_axis(ax)
    ax.plot(x_range, y_pred, color="grey", linestyle="--")
    # Rotate x ticks 45 degrees
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    # Add legend
    ax.legend(loc="lower left")

    # Skip
    # Calculate statistics
    # Add grid
    ax.grid(True, alpha=0.2)

    r_squared = model.score(X, y)
    slope = model.coef_[0]

    # Add labels and title
    ax.set_ylabel("Mean Model Success Rate")
    ax.set_xlabel("Human Time-to-Complete")
    ax.set_title("Model Success Rate vs\nHuman Completion Time")
    # Add annotations
    ax.annotate(
        f"R² = {r_squared:.2f}",
        xy=(1 - 0.05, 0.95),
        xycoords="axes fraction",
        fontsize=12,
        ha="right",
    )

    plt.tight_layout()
    # Save the plot
    plt.savefig(
        output_plots_dir / "model_success_rate_vs_human_completion_time.png",
        dpi=300,
    )


def _make_dirs_if_not_exists(
    output_plots_dir: pathlib.Path, output_data_dir: pathlib.Path
) -> None:
    # make output dir if it doesn't exist
    output_plots_dir.mkdir(parents=True, exist_ok=True)

    output_data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created output directories: {output_plots_dir} and {output_data_dir}")
    return None


def main(
    runs_file: str,
    output_plots_dir: str,
    output_data_dir: str,
    exclude_agent: List[str],
    task_source_styling: Dict[str, Dict[str, str]],
) -> None:
    df = _load_runs(pathlib.Path(runs_file), exclude_agent)
    _make_dirs_if_not_exists(
        output_plots_dir=pathlib.Path(output_plots_dir),
        output_data_dir=pathlib.Path(output_data_dir),
    )
    _make_and_save_family_success_ordered_pivot(
        df,
        output_plots_dir=pathlib.Path(output_plots_dir),
        output_data_dir=pathlib.Path(output_data_dir),
    )
    _make_and_save_task_success_ordered_pivot(
        df,
        output_plots_dir=pathlib.Path(output_plots_dir),
        output_data_dir=pathlib.Path(output_data_dir),
    )
    _make_and_save_task_success_rate_vs_human_completion_time_plot(
        df,
        output_plots_dir=pathlib.Path(output_plots_dir),
        task_source_styling=task_source_styling,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-file", type=str, required=True)
    parser.add_argument("--output-plots-dir", type=str, required=True)
    parser.add_argument("--output-data-dir", type=str, required=True)
    parser.add_argument("--exclude-agent", action="append", default=None)
    parser.add_argument("--params-file", type=str, required=True)
    args = parser.parse_args()

    with open(args.params_file, "r") as f:
        params = yaml.safe_load(f)
    task_source_styling: Dict[str, Dict[str, str]] = params["plots"][
        "task_source_styling"
    ]

    main(
        runs_file=args.runs_file,
        output_plots_dir=args.output_plots_dir,
        output_data_dir=args.output_data_dir,
        exclude_agent=args.exclude_agent,
        task_source_styling=task_source_styling,
    )
