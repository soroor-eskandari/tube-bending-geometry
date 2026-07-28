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

def build_axis_ranking(
    metrics_df: pd.DataFrame,
    *,
    axis: str,
    lower_quantile: float,
    upper_quantile: float,
    n_estimators: int,
    coverage_weight: float,
    rmse_weight: float,
    mae_weight: float,
    interval_width_weight: float,
) -> pd.DataFrame:
    """
    Rank splits independently for one target axis using a normalized
    weighted composite score.

    Ranking policy
    --------------
    1. Calculate the absolute error between empirical coverage and
       the target quantile coverage.
    2. Normalize coverage error, MAE, RMSE and mean interval width
       independently to the [0, 1] interval.
    3. Calculate a weighted composite score:

       qrf_score =
           coverage_weight * normalized coverage error
           + rmse_weight * normalized RMSE
           + mae_weight * normalized MAE
           + interval_width_weight * normalized interval width

    4. Lower qrf_score is better.

    Main and secondary axes are evaluated and ranked independently.
    """
    if axis not in {
        "main",
        "secondary",
    }:
        raise ValueError(
            "axis must be either 'main' or 'secondary'. "
            f"Received: {axis!r}"
        )

    required_columns = {
        "split_index",
        "split_name",
        "axis",
        "n_estimators",
        "coverage_percent",
        "rmse",
        "mae",
        "mean_interval_width",
    }

    validate_columns(
        frame=metrics_df,
        required_columns=required_columns,
        frame_name="QRF metrics",
    )

    normalized_weights = (
        normalize_ranking_weights(
            coverage_weight=coverage_weight,
            rmse_weight=rmse_weight,
            mae_weight=mae_weight,
            interval_width_weight=(
                interval_width_weight
            ),
        )
    )

    axis_metrics_df = metrics_df[
        metrics_df["axis"].eq(axis)
        & pd.to_numeric(
            metrics_df["n_estimators"],
            errors="coerce",
        ).eq(
            int(n_estimators)
        )
    ].copy()

    if axis_metrics_df.empty:
        raise ValueError(
            "No QRF metrics were found for "
            f"axis={axis!r} and "
            f"n_estimators={n_estimators}."
        )

    axis_metrics_df["split_index"] = (
        pd.to_numeric(
            axis_metrics_df["split_index"],
            errors="raise",
        )
        .astype(int)
    )

    numeric_metric_columns = [
        "coverage_percent",
        "rmse",
        "mae",
        "mean_interval_width",
    ]

    for column in numeric_metric_columns:
        axis_metrics_df[column] = (
            pd.to_numeric(
                axis_metrics_df[column],
                errors="raise",
            )
            .astype(float)
        )

        invalid_mask = ~np.isfinite(
            axis_metrics_df[
                column
            ].to_numpy()
        )

        if invalid_mask.any():
            invalid_splits = (
                axis_metrics_df.loc[
                    invalid_mask,
                    "split_index",
                ]
                .astype(int)
                .tolist()
            )

            raise ValueError(
                f"Metric {column!r} contains NaN or "
                "infinite values for split indices: "
                f"{invalid_splits[:20]}"
            )

    target_coverage_percent = (
        float(upper_quantile)
        - float(lower_quantile)
    ) * 100.0

    if not (
        0.0
        < target_coverage_percent
        <= 100.0
    ):
        raise ValueError(
            "The quantile interval produces an invalid "
            "target coverage. "
            f"lower_quantile={lower_quantile}, "
            f"upper_quantile={upper_quantile}, "
            f"target={target_coverage_percent}%."
        )

    axis_metrics_df[
        "coverage_error"
    ] = (
        axis_metrics_df[
            "coverage_percent"
        ]
        - target_coverage_percent
    ).abs()

    # Normally one row exists per split, axis and tree count.
    # Grouping allows repeated deterministic evaluations to be
    # aggregated safely.
    ranking_df = (
        axis_metrics_df
        .groupby(
            [
                "split_index",
                "split_name",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            coverage_percent=(
                "coverage_percent",
                "mean",
            ),
            coverage_error=(
                "coverage_error",
                "mean",
            ),
            rmse=(
                "rmse",
                "mean",
            ),
            mae=(
                "mae",
                "mean",
            ),
            mean_interval_width=(
                "mean_interval_width",
                "mean",
            ),
            evaluated_runs=(
                "axis",
                "size",
            ),
        )
    )

    # Retain these columns for compatibility with the existing
    # metadata schema and UI, although coverage is no longer used
    # as a hard constraint.
    ranking_df[
        "coverage_eligible"
    ] = (
        ranking_df[
            "coverage_percent"
        ].between(
            MIN_COVERAGE_PERCENT,
            MAX_COVERAGE_PERCENT,
            inclusive="both",
        )
    )

    ranking_df[
        "coverage_constraint_distance"
    ] = np.where(
        ranking_df[
            "coverage_percent"
        ].lt(
            MIN_COVERAGE_PERCENT
        ),
        (
            MIN_COVERAGE_PERCENT
            - ranking_df[
                "coverage_percent"
            ]
        ),
        np.where(
            ranking_df[
                "coverage_percent"
            ].gt(
                MAX_COVERAGE_PERCENT
            ),
            (
                ranking_df[
                    "coverage_percent"
                ]
                - MAX_COVERAGE_PERCENT
            ),
            0.0,
        ),
    )

    ranking_df[
        "ranking_status"
    ] = np.where(
        ranking_df[
            "coverage_eligible"
        ],
        "within_reference_coverage_range",
        "outside_reference_coverage_range",
    )

    # All metrics are loss-type metrics: lower is better.
    ranking_df[
        "normalized_coverage_error"
    ] = min_max_normalize(
        ranking_df[
            "coverage_error"
        ]
    )

    ranking_df[
        "normalized_rmse"
    ] = min_max_normalize(
        ranking_df[
            "rmse"
        ]
    )

    ranking_df[
        "normalized_mae"
    ] = min_max_normalize(
        ranking_df[
            "mae"
        ]
    )

    ranking_df[
        "normalized_mean_interval_width"
    ] = min_max_normalize(
        ranking_df[
            "mean_interval_width"
        ]
    )

    ranking_df[
        "coverage_score_component"
    ] = (
        normalized_weights[
            "coverage"
        ]
        * ranking_df[
            "normalized_coverage_error"
        ]
    )

    ranking_df[
        "rmse_score_component"
    ] = (
        normalized_weights[
            "rmse"
        ]
        * ranking_df[
            "normalized_rmse"
        ]
    )

    ranking_df[
        "mae_score_component"
    ] = (
        normalized_weights[
            "mae"
        ]
        * ranking_df[
            "normalized_mae"
        ]
    )

    ranking_df[
        "interval_width_score_component"
    ] = (
        normalized_weights[
            "interval_width"
        ]
        * ranking_df[
            "normalized_mean_interval_width"
        ]
    )

    ranking_df[
        "qrf_score"
    ] = (
        ranking_df[
            "coverage_score_component"
        ]
        + ranking_df[
            "rmse_score_component"
        ]
        + ranking_df[
            "mae_score_component"
        ]
        + ranking_df[
            "interval_width_score_component"
        ]
    )

    ranking_df["axis"] = axis

    ranking_df[
        "target_coverage_percent"
    ] = target_coverage_percent

    ranking_df[
        "min_allowed_coverage_percent"
    ] = MIN_COVERAGE_PERCENT

    ranking_df[
        "max_allowed_coverage_percent"
    ] = MAX_COVERAGE_PERCENT

    # Store the effective normalized weights in every ranking row
    # so the output is self-describing and reproducible.
    ranking_df[
        "coverage_weight"
    ] = normalized_weights[
        "coverage"
    ]

    ranking_df[
        "rmse_weight"
    ] = normalized_weights[
        "rmse"
    ]

    ranking_df[
        "mae_weight"
    ] = normalized_weights[
        "mae"
    ]

    ranking_df[
        "interval_width_weight"
    ] = normalized_weights[
        "interval_width"
    ]

    # Primary ranking is based on the weighted score.
    # Metrics with zero weight must not affect the ranking,
    # even as tie-breakers.
    sort_columns = [
        "qrf_score",
    ]

    sort_ascending = [
        True,
    ]

    if normalized_weights["coverage"] > 0.0:
        sort_columns.append(
            "coverage_error"
        )
        sort_ascending.append(
            True
        )

    if normalized_weights["rmse"] > 0.0:
        sort_columns.append(
            "rmse"
        )
        sort_ascending.append(
            True
        )

    if normalized_weights["mae"] > 0.0:
        sort_columns.append(
            "mae"
        )
        sort_ascending.append(
            True
        )

    if normalized_weights["interval_width"] > 0.0:
        sort_columns.append(
            "mean_interval_width"
        )
        sort_ascending.append(
            True
        )

    # Final deterministic tie-breaker.
    # This does not represent model quality; it only guarantees
    # reproducible ordering when all active ranking metrics are equal.
    sort_columns.append(
        "split_index"
    )

    sort_ascending.append(
        True
    )

    ranking_df = (
        ranking_df
        .sort_values(
            by=sort_columns,
            ascending=sort_ascending,
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    ranking_df.insert(
        0,
        "qrf_rank",
        range(
            1,
            len(ranking_df) + 1,
        ),
    )

    validate_axis_ranking(
        ranking_df=ranking_df,
        axis=axis,
    )

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

    metrics_df, _, run_dir = (
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

    ranking_output_dir = (
        run_dir
        / "axis_rankings"
    )

    ranking_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    main_ranking_df.to_csv(
        ranking_output_dir
        / "qrf_ranking_main.csv",
        index=False,
    )

    main_ranking_df.to_parquet(
        ranking_output_dir
        / "qrf_ranking_main.parquet",
        index=False,
    )

    secondary_ranking_df.to_csv(
        ranking_output_dir
        / "qrf_ranking_secondary.csv",
        index=False,
    )

    secondary_ranking_df.to_parquet(
        ranking_output_dir
        / "qrf_ranking_secondary.parquet",
        index=False,
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
    print(
        f"Detailed ranking directory:   "
        f"{ranking_output_dir}"
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
        f"  width:       "
        f"{top_main_row['mean_interval_width']:.6f}"
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
        f"  width:       "
        f"{top_secondary_row['mean_interval_width']:.6f}"
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
        default=0.4,
    )

    parser.add_argument(
        "--rmse-weight",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--mae-weight",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--interval-width-weight",
        type=float,
        default=0.2,
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
    )

    display_columns = [
        "qrf_rank",
        "split_index",
        "split_name",
        "coverage_percent",
        "coverage_error",
        "normalized_coverage_error",
        "rmse",
        "normalized_rmse",
        "mae",
        "normalized_mae",
        "mean_interval_width",
        "normalized_mean_interval_width",
        "qrf_score",
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