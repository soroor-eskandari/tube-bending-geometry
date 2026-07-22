from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _validate_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
) -> None:
    missing = set(required_columns).difference(df.columns)
    if missing:
        raise KeyError(
            "DataFrame is missing required columns: "
            f"{sorted(missing)}"
        )


def _min_max_scale(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="raise").astype(float)
    minimum = values.min()
    maximum = values.max()

    if np.isclose(minimum, maximum):
        return pd.Series(
            np.zeros(len(values), dtype=float),
            index=values.index,
        )

    return (values - minimum) / (maximum - minimum)


def build_qrf_split_ranking(
    metrics_df: pd.DataFrame,
    *,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    n_estimators: int | None = 500,
    axes: Iterable[str] | None = ("main", "secondary"),
    coverage_weight: float = 0.60,
    rmse_weight: float = 0.20,
    mae_weight: float = 0.10,
    interval_width_weight: float = 0.10,
) -> pd.DataFrame:
    """
    Rank splits using QRF calibration and prediction quality.

    Lower qrf_score is better.

    The default score is:
      60% coverage error from nominal coverage
      20% RMSE
      10% MAE
      10% mean interval width

    Every component is min-max scaled across splits before weighting.
    """
    required = {
        "split_index",
        "axis",
        "n_estimators",
        "coverage_percent",
        "rmse",
        "mae",
        "mean_interval_width",
    }
    _validate_columns(metrics_df, required)

    ranking_input = metrics_df.copy()

    if "split_name" not in ranking_input.columns:
        ranking_input["split_name"] = (
            "split " + ranking_input["split_index"].astype(str)
        )

    if n_estimators is not None:
        ranking_input = ranking_input[
            pd.to_numeric(
                ranking_input["n_estimators"],
                errors="coerce",
            ).eq(int(n_estimators))
        ].copy()

    if axes is not None:
        ranking_input = ranking_input[
            ranking_input["axis"].isin(set(axes))
        ].copy()

    if ranking_input.empty:
        raise ValueError(
            "No metrics remain after filtering by tree count and axes."
        )

    target_coverage = (
        float(upper_quantile) - float(lower_quantile)
    ) * 100.0

    ranking_input["coverage_error"] = (
        ranking_input["coverage_percent"].astype(float)
        - target_coverage
    ).abs()

    ranking_df = (
        ranking_input
        .groupby(
            ["split_index", "split_name"],
            as_index=False,
            dropna=False,
        )
        .agg(
            mean_coverage_percent=("coverage_percent", "mean"),
            mean_coverage_error=("coverage_error", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_mae=("mae", "mean"),
            mean_interval_width=("mean_interval_width", "mean"),
            evaluated_axes=("axis", "nunique"),
            evaluated_runs=("axis", "size"),
        )
    )

    raw_weights = {
        "coverage_error_scaled": float(coverage_weight),
        "rmse_scaled": float(rmse_weight),
        "mae_scaled": float(mae_weight),
        "interval_width_scaled": float(interval_width_weight),
    }

    if any(weight < 0 for weight in raw_weights.values()):
        raise ValueError("Ranking weights must be non-negative.")

    total_weight = sum(raw_weights.values())
    if np.isclose(total_weight, 0.0):
        raise ValueError(
            "At least one ranking weight must be greater than zero."
        )

    weights = {
        name: value / total_weight
        for name, value in raw_weights.items()
    }

    ranking_df["coverage_error_scaled"] = _min_max_scale(
        ranking_df["mean_coverage_error"]
    )
    ranking_df["rmse_scaled"] = _min_max_scale(
        ranking_df["mean_rmse"]
    )
    ranking_df["mae_scaled"] = _min_max_scale(
        ranking_df["mean_mae"]
    )
    ranking_df["interval_width_scaled"] = _min_max_scale(
        ranking_df["mean_interval_width"]
    )

    ranking_df["qrf_score"] = (
        ranking_df["coverage_error_scaled"]
        * weights["coverage_error_scaled"]
        + ranking_df["rmse_scaled"]
        * weights["rmse_scaled"]
        + ranking_df["mae_scaled"]
        * weights["mae_scaled"]
        + ranking_df["interval_width_scaled"]
        * weights["interval_width_scaled"]
    )

    ranking_df["target_coverage_percent"] = target_coverage

    ranking_df = (
        ranking_df
        .sort_values(
            [
                "qrf_score",
                "mean_coverage_error",
                "mean_interval_width",
                "mean_rmse",
                "mean_mae",
            ],
            ascending=True,
        )
        .reset_index(drop=True)
    )

    ranking_df.insert(
        0,
        "qrf_rank",
        range(1, len(ranking_df) + 1),
    )

    return ranking_df


def select_top_and_worst_splits(
    ranking_df: pd.DataFrame,
    *,
    top_n: int = 3,
    worst_n: int = 3,
) -> pd.DataFrame:
    _validate_columns(
        ranking_df,
        {"qrf_rank", "split_index", "split_name", "qrf_score"},
    )

    if top_n < 0 or worst_n < 0:
        raise ValueError("top_n and worst_n must be non-negative.")

    top_df = ranking_df.head(top_n).copy()
    top_df["rank_group"] = "Top"

    worst_df = ranking_df.tail(worst_n).copy()
    worst_df["rank_group"] = "Worst"

    return (
        pd.concat([top_df, worst_df], ignore_index=True)
        .drop_duplicates(subset=["split_index"])
        .sort_values("qrf_rank")
        .reset_index(drop=True)
    )


def selected_split_indices(
    selected_splits_df: pd.DataFrame,
) -> list[int]:
    _validate_columns(selected_splits_df, {"split_index"})
    return (
        selected_splits_df["split_index"]
        .astype(int)
        .tolist()
    )


def save_qrf_ranking(
    ranking_df: pd.DataFrame,
    output_directory: str | Path,
    *,
    selected_splits_df: pd.DataFrame | None = None,
) -> Path:
    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    ranking_df.to_csv(
        output_directory / "qrf_split_ranking.csv",
        index=False,
    )
    ranking_df.to_parquet(
        output_directory / "qrf_split_ranking.parquet",
        index=False,
    )

    if selected_splits_df is not None:
        selected_splits_df.to_csv(
            output_directory / "qrf_selected_top_worst_splits.csv",
            index=False,
        )
        selected_splits_df.to_parquet(
            output_directory / "qrf_selected_top_worst_splits.parquet",
            index=False,
        )

    return output_directory


def rank_and_select_qrf_splits(
    metrics_df: pd.DataFrame,
    *,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    n_estimators: int | None = 500,
    axes: Iterable[str] | None = ("main", "secondary"),
    top_n: int = 3,
    worst_n: int = 3,
    coverage_weight: float = 0.60,
    rmse_weight: float = 0.20,
    mae_weight: float = 0.10,
    interval_width_weight: float = 0.10,
    output_directory: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranking_df = build_qrf_split_ranking(
        metrics_df=metrics_df,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        n_estimators=n_estimators,
        axes=axes,
        coverage_weight=coverage_weight,
        rmse_weight=rmse_weight,
        mae_weight=mae_weight,
        interval_width_weight=interval_width_weight,
    )

    selected_df = select_top_and_worst_splits(
        ranking_df=ranking_df,
        top_n=top_n,
        worst_n=worst_n,
    )

    if output_directory is not None:
        save_qrf_ranking(
            ranking_df=ranking_df,
            selected_splits_df=selected_df,
            output_directory=output_directory,
        )

    return ranking_df, selected_df
