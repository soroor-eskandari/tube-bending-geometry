from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric_by_tree_count(
    metrics_df: pd.DataFrame,
    metric: str,
    axis: str,
    minimum_coverage: float | None = None,
) -> None:
    plot_df = metrics_df[
        metrics_df["axis"].eq(axis)
    ].copy()

    if plot_df.empty:
        raise ValueError(
            f"No rows are available for axis='{axis}'."
        )

    required = {"split_index", "n_estimators", metric}
    missing = required.difference(plot_df.columns)

    if missing:
        raise KeyError(
            f"Metrics DataFrame is missing columns: "
            f"{sorted(missing)}"
        )

    fig, ax = plt.subplots(figsize=(12, 6))

    for split_index, split_df in plot_df.groupby(
        "split_index",
        sort=True,
    ):
        split_df = split_df.sort_values("n_estimators")

        ax.plot(
            split_df["n_estimators"],
            split_df[metric],
            marker="o",
            label=f"split {int(split_index)}",
        )

    if (
        metric == "coverage_percent"
        and minimum_coverage is not None
    ):
        ax.axhline(
            minimum_coverage,
            linestyle="--",
            label=(
                "Minimum acceptable coverage "
                f"= {minimum_coverage:g}%"
            ),
        )

    ax.set_title(
        f"{axis.capitalize()} axis: {metric} by tree count"
    )
    ax.set_xlabel("Number of trees")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_prediction_interval(
    predictions_df: pd.DataFrame,
    run_id: str,
) -> None:
    plot_df = predictions_df[
        predictions_df["run_id"].eq(run_id)
    ].copy()

    if plot_df.empty:
        raise ValueError(
            f"No predictions found for run_id='{run_id}'."
        )

    plot_df = plot_df.sort_values(
        "Angle[degree]ORDistance[mm]"
    )

    x = plot_df["Angle[degree]ORDistance[mm]"]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.fill_between(
        x,
        plot_df["y_lower"],
        plot_df["y_upper"],
        alpha=0.25,
        label="Prediction interval",
    )
    ax.plot(
        x,
        plot_df["y_true"],
        label="Observed",
    )
    ax.plot(
        x,
        plot_df["y_median"],
        label="Median prediction",
    )

    ax.set_title(run_id)
    ax.set_xlabel("Angle or distance")
    ax.set_ylabel("Axis value [mm]")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()
