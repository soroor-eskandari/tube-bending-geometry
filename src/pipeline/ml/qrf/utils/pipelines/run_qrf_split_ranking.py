from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ============================================================
# Constants
# ============================================================

TARGET_TO_AXIS = {
    "Main-axis [mm]": "main",
    "Secondary-axis [mm]": "secondary",
}


# Coverage constraint used before MAE and interval width.
MIN_COVERAGE_PERCENT = 85.0
MAX_COVERAGE_PERCENT = 95.0
MIN_TRAIN_GROUP_PERCENT = 0.0
MIN_TEST_GROUP_PERCENT = 0.0
MAX_TEST_GROUP_PERCENT = 100.0
REQUIRE_GROUP_BALANCE = False


# Columns written into various_splits.parquet.
MAIN_RANKING_COLUMNS = {
    "qrf_rank": "qrf_rank_main",
    "qrf_score": "qrf_score_main",
    "coverage_percent": "qrf_coverage_main",
    "coverage_error": "qrf_coverage_error_main",
    "coverage_eligible": "qrf_coverage_eligible_main",
    "coverage_constraint_distance": (
        "qrf_coverage_constraint_distance_main"
    ),
    "ranking_status": "qrf_ranking_status_main",
    "rmse": "qrf_rmse_main",
    "mae": "qrf_mae_main",
    "mean_interval_width": (
        "qrf_mean_interval_width_main"
    ),
    "train_group_percent": (
        "qrf_train_group_percent_main"
    ),
    "test_group_percent": (
        "qrf_test_group_percent_main"
    ),
    "split_design_eligible": (
        "qrf_split_design_eligible_main"
    ),
    "min_test_group_requirement_met": (
        "qrf_min_test_group_requirement_met_main"
    ),
    "max_test_group_requirement_met": (
        "qrf_max_test_group_requirement_met_main"
    ),
    "group_balance_requirement_enabled": (
        "qrf_group_balance_requirement_enabled_main"
    ),
    "min_test_group_percent": (
        "qrf_min_test_group_percent_main"
    ),
    "max_test_group_percent": (
        "qrf_max_test_group_percent_main"
    ),
    "train_group_requirement_met": (
        "qrf_train_group_requirement_met_main"
    ),
    "train_group_requirement_enabled": (
        "qrf_train_group_requirement_enabled_main"
    ),
    "min_train_group_percent": (
        "qrf_min_train_group_percent_main"
    ),
    "split_is_eligible": "qrf_split_is_eligible_main",
    "split_quality_flag": "qrf_split_quality_flag_main",
    "split_quality_reason": "qrf_split_quality_reason_main",
    "ranking_metric_mode": "qrf_ranking_metric_mode_main",
    "median_curve_distance": "qrf_median_curve_distance_main",
    "p90_curve_distance": "qrf_p90_curve_distance_main",
    "median_trend_error": "qrf_median_trend_error_main",
    "p90_trend_error": "qrf_p90_trend_error_main",
    "median_pinaw": "qrf_median_pinaw_main",
    "p90_pinaw": "qrf_p90_pinaw_main",
}


SECONDARY_RANKING_COLUMNS = {
    "qrf_rank": "qrf_rank_secondary",
    "qrf_score": "qrf_score_secondary",
    "coverage_percent": "qrf_coverage_secondary",
    "coverage_error": "qrf_coverage_error_secondary",
    "coverage_eligible": (
        "qrf_coverage_eligible_secondary"
    ),
    "coverage_constraint_distance": (
        "qrf_coverage_constraint_distance_secondary"
    ),
    "ranking_status": (
        "qrf_ranking_status_secondary"
    ),
    "rmse": "qrf_rmse_secondary",
    "mae": "qrf_mae_secondary",
    "mean_interval_width": (
        "qrf_mean_interval_width_secondary"
    ),
    "train_group_percent": (
        "qrf_train_group_percent_secondary"
    ),
    "test_group_percent": (
        "qrf_test_group_percent_secondary"
    ),
    "split_design_eligible": (
        "qrf_split_design_eligible_secondary"
    ),
    "min_test_group_requirement_met": (
        "qrf_min_test_group_requirement_met_secondary"
    ),
    "max_test_group_requirement_met": (
        "qrf_max_test_group_requirement_met_secondary"
    ),
    "group_balance_requirement_enabled": (
        "qrf_group_balance_requirement_enabled_secondary"
    ),
    "min_test_group_percent": (
        "qrf_min_test_group_percent_secondary"
    ),
    "max_test_group_percent": (
        "qrf_max_test_group_percent_secondary"
    ),
    "train_group_requirement_met": (
        "qrf_train_group_requirement_met_secondary"
    ),
    "train_group_requirement_enabled": (
        "qrf_train_group_requirement_enabled_secondary"
    ),
    "min_train_group_percent": (
        "qrf_min_train_group_percent_secondary"
    ),
    "split_is_eligible": "qrf_split_is_eligible_secondary",
    "split_quality_flag": "qrf_split_quality_flag_secondary",
    "split_quality_reason": "qrf_split_quality_reason_secondary",
    "ranking_metric_mode": "qrf_ranking_metric_mode_secondary",
    "median_curve_distance": "qrf_median_curve_distance_secondary",
    "p90_curve_distance": "qrf_p90_curve_distance_secondary",
    "median_trend_error": "qrf_median_trend_error_secondary",
    "p90_trend_error": "qrf_p90_trend_error_secondary",
    "median_pinaw": "qrf_median_pinaw_secondary",
    "p90_pinaw": "qrf_p90_pinaw_secondary",
}


# ============================================================
# Dynamic project paths
# ============================================================

def find_repo_root(
    start: Path,
) -> Path:
    """
    Find the tube-bending-geometry repository root dynamically.
    """
    current = start.resolve()

    while True:
        expected_qrf_path = (
            current
            / "src"
            / "pipeline"
            / "ml"
            / "qrf"
        )

        if (
            current.name == "tube-bending-geometry"
            or expected_qrf_path.exists()
        ):
            return current

        if current == current.parent:
            raise RuntimeError(
                "Could not locate the repository root. "
                "Expected a parent directory containing "
                "'src/pipeline/ml/qrf'."
            )

        current = current.parent


SCRIPT_PATH = Path(__file__).resolve()

QRF_PIPELINES_ROOT = SCRIPT_PATH.parent

QRF_ROOT = (
    QRF_PIPELINES_ROOT
    .parent
    .parent
)

REPO_ROOT = find_repo_root(
    SCRIPT_PATH.parent
)


for import_root in (
    REPO_ROOT,
    QRF_ROOT,
):
    import_root_value = str(
        import_root
    )

    if import_root_value not in sys.path:
        sys.path.insert(
            0,
            import_root_value,
        )


from src.pipeline.ml.qrf.mode.experiments.qrf_pipeline import (
    load_config,
    run_experiments,
)


# ============================================================
# Generic validation
# ============================================================

def validate_columns(
    frame: pd.DataFrame,
    required_columns: Iterable[str],
    frame_name: str,
) -> None:
    """
    Validate that a DataFrame contains all required columns.
    """
    missing_columns = set(
        required_columns
    ).difference(
        frame.columns
    )

    if missing_columns:
        raise KeyError(
            f"{frame_name} is missing required columns: "
            f"{sorted(missing_columns)}. "
            f"Available columns: "
            f"{frame.columns.tolist()}"
        )


def validate_unique_split_metadata(
    split_metadata_df: pd.DataFrame,
) -> None:
    """
    Validate that split metadata contains exactly one row per split.
    """
    required_columns = {
        "split_index",
        "split_name",
        "train_group_ids",
        "test_group_ids",
        "train_experiment_ids",
        "test_experiment_ids",
    }

    validate_columns(
        frame=split_metadata_df,
        required_columns=required_columns,
        frame_name="Split metadata",
    )

    if split_metadata_df.empty:
        raise ValueError(
            "Split metadata is empty."
        )

    split_indices = pd.to_numeric(
        split_metadata_df["split_index"],
        errors="raise",
    ).astype(int)

    if split_indices.duplicated().any():
        duplicated_indices = sorted(
            split_indices[
                split_indices.duplicated(
                    keep=False
                )
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "various_splits.parquet must contain exactly "
            "one row per split_index. Duplicated indices: "
            f"{duplicated_indices}"
        )

    if split_metadata_df[
        "split_name"
    ].duplicated().any():
        duplicated_names = (
            split_metadata_df.loc[
                split_metadata_df[
                    "split_name"
                ].duplicated(
                    keep=False
                ),
                "split_name",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "various_splits.parquet contains duplicate "
            "split_name values: "
            f"{duplicated_names[:20]}"
        )


# ============================================================
# Split catalog
# ============================================================

def unique_split_catalog(
    split_metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return one validated row per split.

    The current various_splits.parquet is expected to already contain
    one row per split. This function does not silently deduplicate rows.
    """
    validate_unique_split_metadata(
        split_metadata_df
    )

    catalog_columns = [
        "split_index",
        "split_name",
        "train_group_ids",
        "test_group_ids",
        "train_experiment_ids",
        "test_experiment_ids",
    ]

    available_optional_columns = [
        column
        for column in (
            "split_type",
            "n_conditions",
            "test_experiment_percentage",
            "test_group_percentage",
        )
        if column in split_metadata_df.columns
    ]

    catalog_df = (
        split_metadata_df[
            [
                *catalog_columns,
                *available_optional_columns,
            ]
        ]
        .copy()
        .sort_values(
            "split_index"
        )
        .reset_index(drop=True)
    )

    catalog_df["split_index"] = (
        pd.to_numeric(
            catalog_df["split_index"],
            errors="raise",
        )
        .astype(int)
    )

    return catalog_df


# ============================================================
# Axis-specific ranking
# ============================================================

def min_max_normalize(
    series: pd.Series,
) -> pd.Series:
    """
    Normalize a numeric Series to the [0, 1] interval.

    Lower normalized values are always considered better.

    If every value is identical, all normalized values are set to
    zero because that metric cannot distinguish between splits.
    """
    numeric_series = pd.to_numeric(
        series,
        errors="raise",
    ).astype(float)

    if not np.isfinite(
        numeric_series.to_numpy()
    ).all():
        invalid_indices = (
            numeric_series.index[
                ~np.isfinite(
                    numeric_series.to_numpy()
                )
            ]
            .tolist()
        )

        raise ValueError(
            "Cannot normalize a metric containing NaN or "
            "infinite values. Invalid row indices: "
            f"{invalid_indices[:20]}"
        )

    minimum_value = float(
        numeric_series.min()
    )

    maximum_value = float(
        numeric_series.max()
    )

    metric_range = (
        maximum_value
        - minimum_value
    )

    if np.isclose(
        metric_range,
        0.0,
    ):
        return pd.Series(
            0.0,
            index=numeric_series.index,
            dtype=float,
        )

    return (
        numeric_series
        - minimum_value
    ) / metric_range


def normalize_ranking_weights(
    *,
    coverage_weight: float,
    rmse_weight: float,
    mae_weight: float,
    interval_width_weight: float,
) -> dict[str, float]:
    """
    Validate and normalize ranking weights so that they sum to 1.

    The caller may provide weights such as 60, 20, 10, 10 or
    0.60, 0.20, 0.10, 0.10. Both produce the same result.
    """
    raw_weights = {
        "coverage": float(
            coverage_weight
        ),
        "rmse": float(
            rmse_weight
        ),
        "mae": float(
            mae_weight
        ),
        "interval_width": float(
            interval_width_weight
        ),
    }

    for weight_name, weight_value in (
        raw_weights.items()
    ):
        if not np.isfinite(
            weight_value
        ):
            raise ValueError(
                f"{weight_name}_weight must be finite. "
                f"Received: {weight_value}"
            )

        if weight_value < 0.0:
            raise ValueError(
                f"{weight_name}_weight cannot be negative. "
                f"Received: {weight_value}"
            )

    total_weight = sum(
        raw_weights.values()
    )

    if np.isclose(
        total_weight,
        0.0,
    ):
        raise ValueError(
            "At least one ranking weight must be greater "
            "than zero."
        )

    return {
        weight_name: (
            weight_value
            / total_weight
        )
        for weight_name, weight_value
        in raw_weights.items()
    }

def _first_existing_column(
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
) -> str | None:
    """Return the first available column from a list of aliases."""
    return next(
        (column for column in candidates if column in frame.columns),
        None,
    )


def build_axis_ranking(
    metrics_df: pd.DataFrame,
    *,
    axis: str,
    lower_quantile: float,
    upper_quantile: float,
    n_estimators: int,
    coverage_weight: float = 0.0,
    rmse_weight: float = 0.0,
    mae_weight: float = 0.0,
    interval_width_weight: float = 0.0,
    require_group_balance: bool = REQUIRE_GROUP_BALANCE,
    min_train_group_percent: float = MIN_TRAIN_GROUP_PERCENT,
    min_test_group_percent: float = MIN_TEST_GROUP_PERCENT,
    max_test_group_percent: float = MAX_TEST_GROUP_PERCENT,
) -> pd.DataFrame:
    """
    Rank every split without deleting any split.

    Ranking is lexicographic. Structural train/test balance is checked
    before model-performance metrics, so a split with too few training
    groups cannot become rank 1 merely because its test metrics look good.

    Order
    -----
    1. Structurally valid splits first.
    2. Fully eligible splits first (valid structure and coverage >= 85%).
    3. Lowest median curve distance.
    4. Lowest P90 curve distance.
    5. Lowest median trend error.
    6. Lowest P90 trend error.
    7. Lowest median PINAW.
    8. Lowest P90 PINAW.
    9. Highest coverage.
    10. Lowest split_index as deterministic tie-breaker.

    No split is removed. Invalid splits receive explicit flags and remain
    available for top/worst plotting.

    The legacy weight arguments are retained only for backward
    compatibility and are intentionally ignored.
    """
    del coverage_weight, rmse_weight, mae_weight, interval_width_weight

    if axis not in {"main", "secondary"}:
        raise ValueError(
            "axis must be either 'main' or 'secondary'. "
            f"Received: {axis!r}"
        )

    for name, value in {
        "min_train_group_percent": min_train_group_percent,
        "min_test_group_percent": min_test_group_percent,
        "max_test_group_percent": max_test_group_percent,
    }.items():
        if not np.isfinite(float(value)) or not 0.0 <= float(value) <= 100.0:
            raise ValueError(f"{name} must be finite and between 0 and 100.")

    if float(min_test_group_percent) > float(max_test_group_percent):
        raise ValueError(
            "min_test_group_percent cannot exceed max_test_group_percent."
        )

    required_columns = {
        "split_index", "split_name", "axis", "n_estimators",
        "train_groups", "test_groups", "coverage_percent",
        "rmse", "mae", "mean_interval_width",
    }
    validate_columns(metrics_df, required_columns, "QRF metrics")

    axis_metrics_df = metrics_df[
        metrics_df["axis"].eq(axis)
        & pd.to_numeric(metrics_df["n_estimators"], errors="coerce").eq(
            int(n_estimators)
        )
    ].copy()

    if axis_metrics_df.empty:
        raise ValueError(
            f"No QRF metrics were found for axis={axis!r} and "
            f"n_estimators={n_estimators}."
        )

    axis_metrics_df["split_index"] = pd.to_numeric(
        axis_metrics_df["split_index"], errors="raise"
    ).astype(int)

    for column in (
        "coverage_percent", "rmse", "mae", "mean_interval_width",
        "train_groups", "test_groups",
    ):
        axis_metrics_df[column] = pd.to_numeric(
            axis_metrics_df[column], errors="raise"
        ).astype(float)
        if not np.isfinite(axis_metrics_df[column].to_numpy()).all():
            raise ValueError(
                f"Metric {column!r} contains NaN or infinite values."
            )

    advanced_aliases = {
        "median_curve_distance": (
            "median_curve_distance_norm",
            "median_group_curve_distance_norm",
            "cv_curve_distance",
        ),
        "p90_curve_distance": (
            "p90_curve_distance_norm",
            "p90_group_curve_distance_norm",
        ),
        "median_trend_error": (
            "median_trend_shape_loss",
            "median_group_trend_shape_loss",
            "cv_trend_error",
        ),
        "p90_trend_error": (
            "p90_trend_shape_loss",
            "p90_group_trend_shape_loss",
        ),
        "median_pinaw": (
            "median_group_pinaw", "pinaw", "pinaw_test",
        ),
        "p90_pinaw": ("p90_group_pinaw",),
        "ranking_coverage": (
            "median_group_coverage", "coverage_percent",
        ),
    }

    resolved = {
        name: _first_existing_column(axis_metrics_df, aliases)
        for name, aliases in advanced_aliases.items()
    }
    has_complete_advanced_metrics = all(
        resolved[name] is not None
        for name in (
            "median_curve_distance", "p90_curve_distance",
            "median_trend_error", "p90_trend_error",
            "median_pinaw", "p90_pinaw",
        )
    )

    if has_complete_advanced_metrics:
        metric_mode = "group_curve_trend_pinaw"
        for output_name, source_column in resolved.items():
            axis_metrics_df[output_name] = pd.to_numeric(
                axis_metrics_df[source_column], errors="raise"
            ).astype(float)
    else:
        metric_mode = "global_metric_fallback"
        axis_metrics_df["median_curve_distance"] = axis_metrics_df["mae"]
        axis_metrics_df["p90_curve_distance"] = axis_metrics_df["rmse"]
        axis_metrics_df["median_trend_error"] = 0.0
        axis_metrics_df["p90_trend_error"] = 0.0
        axis_metrics_df["median_pinaw"] = axis_metrics_df[
            "mean_interval_width"
        ]
        axis_metrics_df["p90_pinaw"] = axis_metrics_df[
            "mean_interval_width"
        ]
        axis_metrics_df["ranking_coverage"] = axis_metrics_df[
            "coverage_percent"
        ]

    ranking_df = (
        axis_metrics_df
        .groupby(["split_index", "split_name"], as_index=False, dropna=False)
        .agg(
            coverage_percent=("coverage_percent", "mean"),
            ranking_coverage=("ranking_coverage", "mean"),
            rmse=("rmse", "mean"),
            mae=("mae", "mean"),
            mean_interval_width=("mean_interval_width", "mean"),
            median_curve_distance=("median_curve_distance", "mean"),
            p90_curve_distance=("p90_curve_distance", "mean"),
            median_trend_error=("median_trend_error", "mean"),
            p90_trend_error=("p90_trend_error", "mean"),
            median_pinaw=("median_pinaw", "mean"),
            p90_pinaw=("p90_pinaw", "mean"),
            train_groups=("train_groups", "mean"),
            test_groups=("test_groups", "mean"),
            evaluated_runs=("axis", "size"),
        )
    )

    total_groups = ranking_df["train_groups"] + ranking_df["test_groups"]
    if total_groups.le(0).any():
        raise ValueError("A split has zero total train/test groups.")

    ranking_df["train_group_percent"] = (
        100.0 * ranking_df["train_groups"] / total_groups
    )
    ranking_df["test_group_percent"] = (
        100.0 * ranking_df["test_groups"] / total_groups
    )

    ranking_df["min_train_group_percent"] = float(min_train_group_percent)
    ranking_df["min_test_group_percent"] = float(min_test_group_percent)
    ranking_df["max_test_group_percent"] = float(max_test_group_percent)
    ranking_df["group_balance_requirement_enabled"] = bool(
        require_group_balance
    )
    # Kept for compatibility with the previous metadata schema.
    ranking_df["train_group_requirement_enabled"] = bool(
        require_group_balance
    )

    ranking_df["train_group_requirement_met"] = ranking_df[
        "train_group_percent"
    ].ge(float(min_train_group_percent))
    ranking_df["min_test_group_requirement_met"] = ranking_df[
        "test_group_percent"
    ].ge(float(min_test_group_percent))
    ranking_df["max_test_group_requirement_met"] = ranking_df[
        "test_group_percent"
    ].le(float(max_test_group_percent))

    ranking_df["split_design_eligible"] = (
        ranking_df["train_group_requirement_met"]
        & ranking_df["min_test_group_requirement_met"]
        & ranking_df["max_test_group_requirement_met"]
    )

    target_coverage_percent = (
        float(upper_quantile) - float(lower_quantile)
    ) * 100.0
    if not 0.0 < target_coverage_percent <= 100.0:
        raise ValueError("Configured quantiles produce invalid coverage.")

    ranking_df["target_coverage_percent"] = target_coverage_percent
    ranking_df["min_allowed_coverage_percent"] = MIN_COVERAGE_PERCENT
    ranking_df["max_allowed_coverage_percent"] = MAX_COVERAGE_PERCENT
    ranking_df["coverage_error"] = (
        ranking_df["ranking_coverage"] - target_coverage_percent
    ).abs()
    ranking_df["coverage_eligible"] = ranking_df[
        "ranking_coverage"
    ].ge(MIN_COVERAGE_PERCENT)
    ranking_df["coverage_constraint_distance"] = np.maximum(
        MIN_COVERAGE_PERCENT - ranking_df["ranking_coverage"], 0.0
    )

    design_ok = (
        ranking_df["split_design_eligible"]
        if require_group_balance
        else pd.Series(True, index=ranking_df.index, dtype=bool)
    )
    ranking_df["split_is_eligible"] = (
        design_ok & ranking_df["coverage_eligible"]
    )

    low_train = ~ranking_df["train_group_requirement_met"]
    too_few_test = ~ranking_df["min_test_group_requirement_met"]
    too_many_test = ~ranking_df["max_test_group_requirement_met"]
    low_coverage = ~ranking_df["coverage_eligible"]
    high_coverage = ranking_df["ranking_coverage"].gt(
        MAX_COVERAGE_PERCENT
    )

    ranking_df["split_quality_flag"] = np.select(
        [
            low_train & low_coverage,
            too_many_test,
            low_train,
            too_few_test,
            low_coverage,
            high_coverage,
        ],
        [
            "bad_low_train_and_coverage",
            "bad_too_many_test_groups",
            "bad_too_few_train_groups",
            "bad_too_few_test_groups",
            "bad_low_coverage",
            "warning_high_coverage_check_width",
        ],
        default="good",
    )
    ranking_df["split_quality_reason"] = np.select(
        [
            low_train & low_coverage,
            too_many_test,
            low_train,
            too_few_test,
            low_coverage,
            high_coverage,
        ],
        [
            "train_group_percent_below_minimum;coverage_below_minimum",
            "test_group_percent_above_maximum",
            "train_group_percent_below_minimum",
            "test_group_percent_below_minimum",
            "coverage_below_minimum",
            "coverage_above_reference_range;retain_and_check_pinaw",
        ],
        default="eligible",
    )

    ranking_df["ranking_status"] = ranking_df["split_quality_flag"]
    ranking_df["ranking_metric_mode"] = metric_mode
    ranking_df["axis"] = axis
    ranking_df["qrf_score"] = ranking_df["median_curve_distance"]

    # Never silently select a structurally invalid rank-1 split when
    # structural constraints are enabled.
    if require_group_balance and not ranking_df[
        "split_design_eligible"
    ].any():
        raise RuntimeError(
            f"No structurally valid splits were found for axis={axis!r}. "
            f"Required train >= {min_train_group_percent:.1f}%, "
            f"test >= {min_test_group_percent:.1f}% and "
            f"test <= {max_test_group_percent:.1f}%. "
            "The existing parquet file has not been overwritten."
        )

    ranking_df = (
        ranking_df
        .sort_values(
            by=[
                "split_design_eligible",
                "split_is_eligible",
                "median_curve_distance",
                "p90_curve_distance",
                "median_trend_error",
                "p90_trend_error",
                "median_pinaw",
                "p90_pinaw",
                "ranking_coverage",
                "split_index",
            ],
            ascending=[
                False, False, True, True, True,
                True, True, True, False, True,
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    ranking_df.insert(0, "qrf_rank", range(1, len(ranking_df) + 1))

    validate_axis_ranking(ranking_df=ranking_df, axis=axis)
    return ranking_df


def validate_axis_ranking(
    ranking_df: pd.DataFrame,
    axis: str,
) -> None:
    """
    Validate one complete axis-specific ranking table.
    """
    required_columns = {
        "qrf_rank",
        "split_index",
        "split_name",
        "axis",
        "qrf_score",
        "coverage_percent",
        "coverage_error",
        "coverage_eligible",
        "coverage_constraint_distance",
        "ranking_status",
        "rmse",
        "mae",
        "mean_interval_width",
        "split_design_eligible",
        "split_design_eligible",
        "split_is_eligible",
        "split_quality_flag",
        "split_quality_reason",
        "ranking_metric_mode",
        "median_curve_distance",
        "p90_curve_distance",
        "median_trend_error",
        "p90_trend_error",
        "median_pinaw",
        "p90_pinaw",
    }

    validate_columns(
        frame=ranking_df,
        required_columns=required_columns,
        frame_name=f"{axis} ranking",
    )

    if ranking_df.empty:
        raise ValueError(
            f"{axis} ranking is empty."
        )

    if not ranking_df[
        "axis"
    ].eq(axis).all():
        raise ValueError(
            f"{axis} ranking contains rows from another axis."
        )

    if ranking_df[
        "split_index"
    ].duplicated().any():
        duplicated_indices = (
            ranking_df.loc[
                ranking_df[
                    "split_index"
                ].duplicated(
                    keep=False
                ),
                "split_index",
            ]
            .astype(int)
            .tolist()
        )

        raise ValueError(
            f"{axis} ranking contains duplicated split indices: "
            f"{duplicated_indices}"
        )

    actual_ranks = (
        pd.to_numeric(
            ranking_df["qrf_rank"],
            errors="raise",
        )
        .astype(int)
        .tolist()
    )

    expected_ranks = list(
        range(
            1,
            len(ranking_df) + 1,
        )
    )

    if actual_ranks != expected_ranks:
        raise ValueError(
            f"{axis} ranking is not sequential from 1 to "
            f"{len(ranking_df)}."
        )

    rank_one_rows = ranking_df[
        ranking_df[
            "qrf_rank"
        ].eq(1)
    ]

    if len(rank_one_rows) != 1:
        raise ValueError(
            f"{axis} ranking must contain exactly one rank-1 "
            f"split, but found {len(rank_one_rows)}."
        )


# ============================================================
# Attach rankings to the same metadata rows
# ============================================================

def remove_previous_ranking_columns(
    split_metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove previous QRF ranking columns before attaching new rankings.
    """
    all_ranking_columns = set(
        MAIN_RANKING_COLUMNS.values()
    ).union(
        SECONDARY_RANKING_COLUMNS.values()
    )

    # Remove older generic columns as well if they exist.
    all_ranking_columns.update(
        {
            "qrf_axis",
            "qrf_rank",
            "qrf_score",
        }
    )

    existing_ranking_columns = [
        column
        for column in all_ranking_columns
        if column in split_metadata_df.columns
    ]

    if not existing_ranking_columns:
        return split_metadata_df.copy()

    return split_metadata_df.drop(
        columns=existing_ranking_columns
    ).copy()


def attach_rankings_to_metadata(
    split_metadata_df: pd.DataFrame,
    main_ranking_df: pd.DataFrame,
    secondary_ranking_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach main and secondary rankings to the existing split rows.

    No split rows are appended. The output must have exactly the same
    number of rows and split indices as the input metadata.
    """
    validate_unique_split_metadata(
        split_metadata_df
    )

    validate_axis_ranking(
        ranking_df=main_ranking_df,
        axis="main",
    )

    validate_axis_ranking(
        ranking_df=secondary_ranking_df,
        axis="secondary",
    )

    original_row_count = len(
        split_metadata_df
    )

    original_split_indices = set(
        pd.to_numeric(
            split_metadata_df[
                "split_index"
            ],
            errors="raise",
        )
        .astype(int)
        .tolist()
    )

    main_split_indices = set(
        pd.to_numeric(
            main_ranking_df[
                "split_index"
            ],
            errors="raise",
        )
        .astype(int)
        .tolist()
    )

    secondary_split_indices = set(
        pd.to_numeric(
            secondary_ranking_df[
                "split_index"
            ],
            errors="raise",
        )
        .astype(int)
        .tolist()
    )

    if main_split_indices != original_split_indices:
        missing_main = sorted(
            original_split_indices
            - main_split_indices
        )

        unexpected_main = sorted(
            main_split_indices
            - original_split_indices
        )

        raise ValueError(
            "Main ranking split indices do not match the "
            "metadata catalog. "
            f"Missing: {missing_main}; "
            f"unexpected: {unexpected_main}"
        )

    if (
        secondary_split_indices
        != original_split_indices
    ):
        missing_secondary = sorted(
            original_split_indices
            - secondary_split_indices
        )

        unexpected_secondary = sorted(
            secondary_split_indices
            - original_split_indices
        )

        raise ValueError(
            "Secondary ranking split indices do not match the "
            "metadata catalog. "
            f"Missing: {missing_secondary}; "
            f"unexpected: {unexpected_secondary}"
        )

    output_df = remove_previous_ranking_columns(
        split_metadata_df
    )

    main_lookup_df = (
        main_ranking_df[
            [
                "split_index",
                *MAIN_RANKING_COLUMNS.keys(),
            ]
        ]
        .rename(
            columns=MAIN_RANKING_COLUMNS
        )
        .copy()
    )

    secondary_lookup_df = (
        secondary_ranking_df[
            [
                "split_index",
                *SECONDARY_RANKING_COLUMNS.keys(),
            ]
        ]
        .rename(
            columns=SECONDARY_RANKING_COLUMNS
        )
        .copy()
    )

    output_df = output_df.merge(
        main_lookup_df,
        on="split_index",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    output_df = output_df.merge(
        secondary_lookup_df,
        on="split_index",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    validate_updated_metadata(
        original_metadata_df=(
            split_metadata_df
        ),
        updated_metadata_df=output_df,
        expected_row_count=original_row_count,
        expected_split_indices=(
            original_split_indices
        ),
    )

    return output_df


def validate_updated_metadata(
    *,
    original_metadata_df: pd.DataFrame,
    updated_metadata_df: pd.DataFrame,
    expected_row_count: int,
    expected_split_indices: set[int],
) -> None:
    """
    Ensure ranking only added/replaced columns and never changed rows.
    """
    if len(updated_metadata_df) != expected_row_count:
        raise RuntimeError(
            "Attaching rankings changed the number of rows. "
            f"Before={expected_row_count}, "
            f"after={len(updated_metadata_df)}."
        )

    updated_split_indices = set(
        pd.to_numeric(
            updated_metadata_df[
                "split_index"
            ],
            errors="raise",
        )
        .astype(int)
        .tolist()
    )

    if updated_split_indices != expected_split_indices:
        raise RuntimeError(
            "Attaching rankings changed the split catalog. "
            f"Expected indices: "
            f"{sorted(expected_split_indices)}; "
            f"actual indices: "
            f"{sorted(updated_split_indices)}"
        )

    if updated_metadata_df[
        "split_index"
    ].duplicated().any():
        duplicated_indices = (
            updated_metadata_df.loc[
                updated_metadata_df[
                    "split_index"
                ].duplicated(
                    keep=False
                ),
                "split_index",
            ]
            .astype(int)
            .tolist()
        )

        raise RuntimeError(
            "Ranking attachment created duplicate rows. "
            f"Duplicated split indices: "
            f"{duplicated_indices}"
        )

    required_ranking_columns = set(
        MAIN_RANKING_COLUMNS.values()
    ).union(
        SECONDARY_RANKING_COLUMNS.values()
    )

    validate_columns(
        frame=updated_metadata_df,
        required_columns=required_ranking_columns,
        frame_name="Updated split metadata",
    )

    if updated_metadata_df[
        "qrf_rank_main"
    ].isna().any():
        missing_main_indices = (
            updated_metadata_df.loc[
                updated_metadata_df[
                    "qrf_rank_main"
                ].isna(),
                "split_index",
            ]
            .astype(int)
            .tolist()
        )

        raise RuntimeError(
            "Some splits do not have a main-axis rank: "
            f"{missing_main_indices}"
        )

    if updated_metadata_df[
        "qrf_rank_secondary"
    ].isna().any():
        missing_secondary_indices = (
            updated_metadata_df.loc[
                updated_metadata_df[
                    "qrf_rank_secondary"
                ].isna(),
                "split_index",
            ]
            .astype(int)
            .tolist()
        )

        raise RuntimeError(
            "Some splits do not have a secondary-axis rank: "
            f"{missing_secondary_indices}"
        )

    if (
        updated_metadata_df[
            "qrf_rank_main"
        ].duplicated().any()
    ):
        raise RuntimeError(
            "Main-axis ranks are not unique."
        )

    if (
        updated_metadata_df[
            "qrf_rank_secondary"
        ].duplicated().any()
    ):
        raise RuntimeError(
            "Secondary-axis ranks are not unique."
        )

    original_non_ranking_columns = [
        column
        for column in original_metadata_df.columns
        if not (
            column.startswith("qrf_")
        )
    ]

    for column in original_non_ranking_columns:
        if column not in updated_metadata_df.columns:
            raise RuntimeError(
                "A non-ranking metadata column was removed: "
                f"{column!r}"
            )


# ============================================================
# Atomic overwrite
# ============================================================

def overwrite_parquet_atomically(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> Path:
    """
    Replace the existing Parquet file atomically.

    The file is first written to a temporary path and then moved over
    the previous file. This avoids leaving a partially written output.
    """
    output_path = output_path.resolve()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_name(
        f".{output_path.stem}"
        f"__ranking_temporary"
        f"{output_path.suffix}"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    dataframe.to_parquet(
        temporary_path,
        index=False,
    )

    # Validate the written file before replacing the original.
    reloaded_df = pd.read_parquet(
        temporary_path
    )

    if len(reloaded_df) != len(dataframe):
        temporary_path.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            "Temporary ranking file validation failed. "
            f"Expected {len(dataframe)} rows, "
            f"read back {len(reloaded_df)} rows."
        )

    temporary_path.replace(
        output_path
    )

    return output_path


# ============================================================
# Main ranking pipeline
# ============================================================

def run_qrf_ranking_pipeline(
    config_path: str | Path,
    *,
    n_estimators: int = 200,
    coverage_weight: float = 1.0,
    rmse_weight: float = 0.0,
    mae_weight: float = 0.0,
    interval_width_weight: float = 0.0,
    require_group_balance: bool = REQUIRE_GROUP_BALANCE,
    min_train_group_percent: float = MIN_TRAIN_GROUP_PERCENT,
    min_test_group_percent: float = MIN_TEST_GROUP_PERCENT,
    max_test_group_percent: float = MAX_TEST_GROUP_PERCENT,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    Path,
]:
    """
    Evaluate every stored split, rank both axes independently and
    overwrite various_splits.parquet with the new ranking columns.
    """
    config_path = Path(
        config_path
    )

    if not config_path.is_absolute():
        config_path = (
            REPO_ROOT
            / config_path
        )

    config_path = config_path.resolve()

    config = load_config(
        config_path=config_path,
        project_root=REPO_ROOT,
    )

    if "paths" not in config:
        raise KeyError(
            "Config is missing the 'paths' section."
        )

    if (
        "split_metadata"
        not in config["paths"]
    ):
        raise KeyError(
            "Config is missing "
            "paths.split_metadata."
        )

    split_metadata_path = Path(
        config["paths"][
            "split_metadata"
        ]
    )

    if not split_metadata_path.is_absolute():
        split_metadata_path = (
            REPO_ROOT
            / split_metadata_path
        )

    split_metadata_path = (
        split_metadata_path.resolve()
    )

    if not split_metadata_path.exists():
        raise FileNotFoundError(
            "Split metadata does not exist: "
            f"{split_metadata_path}"
        )

    original_metadata_df = pd.read_parquet(
        split_metadata_path
    )

    validate_unique_split_metadata(
        original_metadata_df
    )

    split_catalog_df = unique_split_catalog(
        original_metadata_df
    )

    split_indices = (
        split_catalog_df[
            "split_index"
        ]
        .astype(int)
        .tolist()
    )

    ranking_config = dict(
        config
    )

    ranking_config["search"] = dict(
        config.get(
            "search",
            {}
        )
    )

    ranking_config["output"] = dict(
        config.get(
            "output",
            {}
        )
    )

    ranking_config[
        "search"
    ][
        "split_indices"
    ] = split_indices

    ranking_config[
        "search"
    ][
        "n_estimators"
    ] = [
        int(n_estimators)
    ]

    ranking_config[
        "search"
    ][
        "axes"
    ] = [
        "main",
        "secondary",
    ]

    ranking_config[
        "output"
    ][
        "save_predictions"
    ] = False
    ranking_config[
        "output"
    ][
        "save_run_artifacts"
    ] = False

    print()
    print("=" * 78)
    print("QRF split ranking")
    print("=" * 78)
    print(
        f"Config:              {config_path}"
    )
    print(
        f"Split metadata:      {split_metadata_path}"
    )
    print(
        f"Number of splits:    {len(split_indices)}"
    )
    print(
        f"Number of trees:     {n_estimators}"
    )
    print(
        "Ranking axes:        main, secondary"
    )
    print(
        "Group-balance rule:  "
        + (
            f"enabled; train >= {float(min_train_group_percent):.2f}%, "
            f"test between {float(min_test_group_percent):.2f}% and "
            f"{float(max_test_group_percent):.2f}%"
            if require_group_balance
            else "disabled"
        )
    )

    geometry_source = (
        ranking_config
        .get(
            "data",
            {}
        )
        .get(
            "geometry_source"
        )
    )

    if geometry_source:
        print(
            f"Ranking data source: {geometry_source}"
        )

    print()

    metrics_df, _, _ = (
        run_experiments(
            ranking_config
        )
    )

    lower_quantile = float(
        ranking_config[
            "model"
        ][
            "lower_quantile"
        ]
    )

    upper_quantile = float(
        ranking_config[
            "model"
        ][
            "upper_quantile"
        ]
    )

    # Main ranking receives only rows where axis == "main".
    main_ranking_df = build_axis_ranking(
        metrics_df=metrics_df,
        axis="main",
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        n_estimators=n_estimators,
        coverage_weight=coverage_weight,
        rmse_weight=rmse_weight,
        mae_weight=mae_weight,
        interval_width_weight=(
            interval_width_weight
        ),
        require_group_balance=require_group_balance,
        min_train_group_percent=min_train_group_percent,
        min_test_group_percent=min_test_group_percent,
        max_test_group_percent=max_test_group_percent,
    )

    # Secondary ranking receives only rows where axis == "secondary".
    secondary_ranking_df = build_axis_ranking(
        metrics_df=metrics_df,
        axis="secondary",
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        n_estimators=n_estimators,
        coverage_weight=coverage_weight,
        rmse_weight=rmse_weight,
        mae_weight=mae_weight,
        interval_width_weight=(
            interval_width_weight
        ),
        require_group_balance=require_group_balance,
        min_train_group_percent=min_train_group_percent,
        min_test_group_percent=min_test_group_percent,
        max_test_group_percent=max_test_group_percent,
    )

    if (
        len(main_ranking_df)
        != len(split_catalog_df)
    ):
        raise RuntimeError(
            "Main-axis ranking does not contain every split. "
            f"Expected {len(split_catalog_df)}, "
            f"received {len(main_ranking_df)}."
        )

    if (
        len(secondary_ranking_df)
        != len(split_catalog_df)
    ):
        raise RuntimeError(
            "Secondary-axis ranking does not contain every split. "
            f"Expected {len(split_catalog_df)}, "
            f"received {len(secondary_ranking_df)}."
        )

    updated_metadata_df = (
        attach_rankings_to_metadata(
            split_metadata_df=(
                original_metadata_df
            ),
            main_ranking_df=(
                main_ranking_df
            ),
            secondary_ranking_df=(
                secondary_ranking_df
            ),
        )
    )

    stored_metadata_path = (
        overwrite_parquet_atomically(
            dataframe=updated_metadata_df,
            output_path=split_metadata_path,
        )
    )

    top_main_row = (
        main_ranking_df
        .loc[
            main_ranking_df[
                "qrf_rank"
            ].eq(1)
        ]
        .iloc[0]
    )

    top_secondary_row = (
        secondary_ranking_df
        .loc[
            secondary_ranking_df[
                "qrf_rank"
            ].eq(1)
        ]
        .iloc[0]
    )

    print()
    print("=" * 78)
    print("Ranking completed successfully")
    print("=" * 78)
    print(
        "Metadata rows before ranking: "
        f"{len(original_metadata_df)}"
    )
    print(
        "Metadata rows after ranking:  "
        f"{len(updated_metadata_df)}"
    )
    print(
        f"Updated metadata file:        "
        f"{stored_metadata_path}"
    )

    print()
    print("Top-1 main-axis split")
    print(
        f"  split_index: "
        f"{int(top_main_row['split_index'])}"
    )
    print(
        f"  split_name:  "
        f"{top_main_row['split_name']}"
    )
    print(
        f"  coverage:    "
        f"{top_main_row['coverage_percent']:.2f}%"
    )
    print(
        f"  MAE:         "
        f"{top_main_row['mae']:.6f}"
    )
    print(
        f"  train groups:"
        f" {top_main_row['train_group_percent']:.2f}%"
    )
    print(
        f"  curve median:"
        f" {top_main_row['median_curve_distance']:.6f}"
    )
    print(
        f"  PINAW/width: "
        f"{top_main_row['median_pinaw']:.6f}"
    )
    print(
        f"  quality:     "
        f"{top_main_row['split_quality_flag']}"
    )

    print()
    print("Top-1 secondary-axis split")
    print(
        f"  split_index: "
        f"{int(top_secondary_row['split_index'])}"
    )
    print(
        f"  split_name:  "
        f"{top_secondary_row['split_name']}"
    )
    print(
        f"  coverage:    "
        f"{top_secondary_row['coverage_percent']:.2f}%"
    )
    print(
        f"  MAE:         "
        f"{top_secondary_row['mae']:.6f}"
    )
    print(
        f"  train groups:"
        f" {top_secondary_row['train_group_percent']:.2f}%"
    )
    print(
        f"  curve median:"
        f" {top_secondary_row['median_curve_distance']:.6f}"
    )
    print(
        f"  PINAW/width: "
        f"{top_secondary_row['median_pinaw']:.6f}"
    )
    print(
        f"  quality:     "
        f"{top_secondary_row['split_quality_flag']}"
    )

    return (
        main_ranking_df,
        secondary_ranking_df,
        stored_metadata_path,
    )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every split using one configured geometry "
            "source, rank main and secondary axes independently, "
            "and overwrite various_splits.parquet without adding "
            "or removing split rows."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=(
            QRF_ROOT
            / "configs"
            / "qrf_split_ranking.yaml"
        ),
        help=(
            "Path to the QRF split-ranking YAML config."
        ),
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help=(
            "Number of QRF trees used to evaluate every split."
        ),
    )

    # Retained for compatibility with previous command-line usage.
    parser.add_argument(
        "--coverage-weight",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--rmse-weight",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--mae-weight",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--interval-width-weight",
        type=float,
        default=0.6,
    )

    parser.add_argument(
        "--require-group-balance",
        action=argparse.BooleanOptionalAction,
        default=REQUIRE_GROUP_BALANCE,
        help=(
            "Require structurally valid train/test group percentages "
            "before model metrics are considered."
        ),
    )

    parser.add_argument(
        "--min-train-group-percent",
        type=float,
        default=MIN_TRAIN_GROUP_PERCENT,
        help="Minimum percentage of groups required in training.",
    )

    parser.add_argument(
        "--min-test-group-percent",
        type=float,
        default=MIN_TEST_GROUP_PERCENT,
        help="Minimum percentage of groups required in testing.",
    )

    parser.add_argument(
        "--max-test-group-percent",
        type=float,
        default=MAX_TEST_GROUP_PERCENT,
        help="Maximum percentage of groups allowed in testing.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    (
        main_ranking_df,
        secondary_ranking_df,
        _,
    ) = run_qrf_ranking_pipeline(
        config_path=args.config,
        n_estimators=args.n_estimators,
        coverage_weight=args.coverage_weight,
        rmse_weight=args.rmse_weight,
        mae_weight=args.mae_weight,
        interval_width_weight=(
            args.interval_width_weight
        ),
        require_group_balance=args.require_group_balance,
        min_train_group_percent=args.min_train_group_percent,
        min_test_group_percent=args.min_test_group_percent,
        max_test_group_percent=args.max_test_group_percent,
    )

    display_columns = [
        "qrf_rank",
        "split_index",
        "split_name",
        "split_design_eligible",
        "split_is_eligible",
        "split_quality_flag",
        "ranking_metric_mode",
        "ranking_coverage",
        "median_curve_distance",
        "p90_curve_distance",
        "median_trend_error",
        "p90_trend_error",
        "median_pinaw",
        "p90_pinaw",
        "train_group_percent",
        "test_group_percent",
        "mae",
        "rmse",
    ]

    print()
    print("Top 3 — main axis")
    print(
        main_ranking_df[
            display_columns
        ]
        .head(3)
        .to_string(
            index=False
        )
    )

    print()
    print("Worst 3 — main axis")
    print(
        main_ranking_df[
            display_columns
        ]
        .tail(3)
        .to_string(
            index=False
        )
    )

    print()
    print("Top 3 — secondary axis")
    print(
        secondary_ranking_df[
            display_columns
        ]
        .head(3)
        .to_string(
            index=False
        )
    )

    print()
    print("Worst 3 — secondary axis")
    print(
        secondary_ranking_df[
            display_columns
        ]
        .tail(3)
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
