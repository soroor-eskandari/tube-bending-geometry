"""
Rank selected bending-setup features based on their relationship with
within-group geometry spread.

Data flow
---------
1. Load augmented experiment geometry.
2. Load unique bending setups.
3. Merge bending setup values into geometry data using group_id.
4. Convert every experiment curve into one numeric vector.
5. Calculate the mean pairwise Euclidean distance between experiments
   inside each group.
6. Train two completely independent Random Forest models:
   - Main geometry
   - Secondary geometry
7. Calculate held-out permutation importance.
8. Print one final DataFrame in the terminal.
9. Save one compressed Parquet file containing Main and Secondary rankings.

The output file is overwritten on every run.

No plots or trained models are stored.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Types
# =============================================================================

GeometryType = Literal["main", "secondary"]


# =============================================================================
# Dynamic project paths
# =============================================================================

CURRENT_FILE = Path(__file__).resolve()


def find_project_root(start_path: Path) -> Path:
    """
    Find the repository root dynamically.

    The project root is identified as the nearest parent containing both:
    - src/
    - data/
    """
    search_locations = [
        start_path,
        *start_path.parents,
    ]

    for candidate in search_locations:
        if (
            (candidate / "src").is_dir()
            and (candidate / "data").is_dir()
        ):
            return candidate

    raise FileNotFoundError(
        "Could not determine the project root.\n"
        "Expected a parent directory containing both 'src' and 'data'.\n"
        f"Started searching from:\n{start_path}"
    )


PROJECT_ROOT = find_project_root(
    CURRENT_FILE.parent
)

SRC_DIR = PROJECT_ROOT / "src"

QRF_DIR = (
    SRC_DIR
    / "pipeline"
    / "ml"
    / "qrf"
)

AUGMENTED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "rf_augmented"
    / "ui_data"
    / "final_geometry_sensor_augmented_noise__time_wrapping__scaling__jittering.csv"
)

REAL_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "geometry.csv"
)

BENDING_SETUP_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "rf_augmented"
    / "ui_data"
    / "unique_bending_setups.csv"
)

OUTPUT_PATH = (
    QRF_DIR
    / "data"
    / "bending_setup_group_spread_feature_ranking.parquet"
)


# =============================================================================
# Dataset columns
# =============================================================================

GEOMETRY_GROUP_COLUMN = "group_id"
SETUP_GROUP_COLUMN = "Group_ID"

EXPERIMENT_COLUMN = "Experiment_ID"
COORDINATE_COLUMN = "Angle[degree]ORDistance[mm]"

CURVE_INSTANCE_COLUMN = "_curve_instance_id"
AUGMENTATION_INSTANCE_COLUMN = "_augmentation_instance"

MAIN_AXIS_COLUMN = "Main-axis [mm]"
SECONDARY_AXIS_COLUMN = "Secondary-axis [mm]"

# Only these columns are used as Random Forest input features.
BENDING_SETUP_COLUMNS = [
    "Pressure-die distance",
    "Pressure-die boost",
    "Mandrel position",
    "Mandrel retraction timing",
    "Collet boost",
    "Clamp-die lateral position",
]


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class RankingConfig:
    """
    Random Forest and validation configuration.
    """

    n_estimators: int = 500
    max_depth: int | None = None
    min_samples_leaf: int = 2
    max_features: str | float | int | None = 1.0

    cv_splits: int = 5
    permutation_repeats: int = 30
    random_state: int = 42

    # False is recommended because Main and Secondary values are already
    # expressed in millimetres and are analysed separately.
    scale_curve_points: bool = False


# =============================================================================
# Input loading and validation
# =============================================================================

def validate_file_exists(
    path: Path,
    description: str,
) -> None:
    """
    Raise a clear error when an input file is missing.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} was not found:\n{path}"
        )


def validate_required_columns(
    data: pd.DataFrame,
    required_columns: Sequence[str],
    dataset_name: str,
) -> None:
    """
    Ensure all required columns exist in a DataFrame.
    """
    missing_columns = sorted(
        set(required_columns).difference(data.columns)
    )

    if missing_columns:
        raise KeyError(
            f"Missing columns in {dataset_name}:\n"
            f"{missing_columns}"
        )


def load_and_merge_input_data() -> pd.DataFrame:
    """
    Load augmented geometry and unique bending setups, then merge them using
    the group identifier.

    Returns
    -------
    pd.DataFrame
        Experiment-level geometry rows containing the six selected bending
        setup features.
    """
    validate_file_exists(
        REAL_DATA_PATH,
        "Augmented geometry dataset",
    )

    validate_file_exists(
        BENDING_SETUP_DATA_PATH,
        "Unique bending-setup dataset",
    )

    geometry_df = pd.read_csv(
        REAL_DATA_PATH,
        low_memory=False,
    )

    setup_df = pd.read_csv(
        BENDING_SETUP_DATA_PATH,
        low_memory=False,
    )

    validate_required_columns(
        data=geometry_df,
        required_columns=[
            EXPERIMENT_COLUMN,
            COORDINATE_COLUMN,
            MAIN_AXIS_COLUMN,
            SECONDARY_AXIS_COLUMN,
            GEOMETRY_GROUP_COLUMN,
        ],
        dataset_name=REAL_DATA_PATH.name,
    )

    validate_required_columns(
        data=setup_df,
        required_columns=[
            SETUP_GROUP_COLUMN,
            *BENDING_SETUP_COLUMNS,
        ],
        dataset_name=BENDING_SETUP_DATA_PATH.name,
    )

    if geometry_df.empty:
        raise ValueError(
            f"Augmented geometry dataset is empty:\n"
            f"{REAL_DATA_PATH}"
        )

    if setup_df.empty:
        raise ValueError(
            f"Bending-setup dataset is empty:\n"
            f"{BENDING_SETUP_DATA_PATH}"
        )

    geometry_df[GEOMETRY_GROUP_COLUMN] = pd.to_numeric(
        geometry_df[GEOMETRY_GROUP_COLUMN],
        errors="raise",
    )

    setup_df[SETUP_GROUP_COLUMN] = pd.to_numeric(
        setup_df[SETUP_GROUP_COLUMN],
        errors="raise",
    )

    duplicated_setup_groups = setup_df.loc[
        setup_df[SETUP_GROUP_COLUMN].duplicated(
            keep=False
        ),
        SETUP_GROUP_COLUMN,
    ].drop_duplicates()

    if not duplicated_setup_groups.empty:
        raise ValueError(
            "unique_bending_setups.csv contains duplicate Group_ID values:\n"
            f"{duplicated_setup_groups.tolist()}"
        )

    selected_setup_df = setup_df[
        [
            SETUP_GROUP_COLUMN,
            *BENDING_SETUP_COLUMNS,
        ]
    ].copy()

    merged_df = geometry_df.merge(
        selected_setup_df,
        how="left",
        left_on=GEOMETRY_GROUP_COLUMN,
        right_on=SETUP_GROUP_COLUMN,
        validate="many_to_one",
    )

    unmatched_group_mask = merged_df[
        BENDING_SETUP_COLUMNS
    ].isna().all(axis=1)

    unmatched_groups = (
        merged_df.loc[
            unmatched_group_mask,
            GEOMETRY_GROUP_COLUMN,
        ]
        .drop_duplicates()
        .tolist()
    )

    if unmatched_groups:
        raise ValueError(
            "Some geometry group IDs do not exist in "
            "unique_bending_setups.csv:\n"
            f"{unmatched_groups}"
        )

    missing_setup_mask = merged_df[
        BENDING_SETUP_COLUMNS
    ].isna().any(axis=1)

    if missing_setup_mask.any():
        missing_groups = (
            merged_df.loc[
                missing_setup_mask,
                GEOMETRY_GROUP_COLUMN,
            ]
            .drop_duplicates()
            .tolist()
        )

        missing_columns = [
            column
            for column in BENDING_SETUP_COLUMNS
            if merged_df.loc[
                missing_setup_mask,
                column,
            ].isna().any()
        ]

        raise ValueError(
            "Missing values exist in selected bending-setup features.\n"
            f"Groups: {missing_groups}\n"
            f"Columns: {missing_columns}"
        )

    merged_df = merged_df.drop(
        columns=[SETUP_GROUP_COLUMN]
    )

    for feature in BENDING_SETUP_COLUMNS:
        merged_df[feature] = pd.to_numeric(
            merged_df[feature],
            errors="raise",
        )

    return merged_df


# =============================================================================
# Bending-setup validation
# =============================================================================

def validate_setup_consistency_inside_groups(
    data: pd.DataFrame,
) -> None:
    """
    Ensure every selected setup feature has one value per group.
    """
    violations: list[str] = []

    grouped = data.groupby(
        GEOMETRY_GROUP_COLUMN,
        sort=False,
        dropna=False,
    )

    for feature in BENDING_SETUP_COLUMNS:
        unique_counts = grouped[feature].nunique(
            dropna=False
        )

        invalid_groups = unique_counts[
            unique_counts > 1
        ]

        for group_id, unique_count in invalid_groups.items():
            violations.append(
                f"group_id={group_id!r}, "
                f"feature={feature!r}, "
                f"unique_values={int(unique_count)}"
            )

    if violations:
        preview = "\n".join(
            f"  - {item}"
            for item in violations[:30]
        )

        raise ValueError(
            "Bending-setup values are not constant inside some groups.\n"
            f"{preview}"
        )


# =============================================================================
# Experiment curve construction
# =============================================================================

def prepare_geometry_columns(
    data: pd.DataFrame,
    axis_column: str,
    geometry_type: GeometryType,
) -> pd.DataFrame:
    """
    Prepare geometry rows and create a unique curve identifier for every
    augmented curve instance.

    In the augmented dataset, one Experiment_ID can occur several times.
    Therefore Experiment_ID alone cannot identify one complete curve.
    """
    required_columns = [
        EXPERIMENT_COLUMN,
        GEOMETRY_GROUP_COLUMN,
        COORDINATE_COLUMN,
        axis_column,
        *BENDING_SETUP_COLUMNS,
    ]

    validate_required_columns(
        data=data,
        required_columns=required_columns,
        dataset_name=f"{geometry_type} merged geometry data",
    )

    working_df = data[
        required_columns
    ].copy()

    working_df[COORDINATE_COLUMN] = pd.to_numeric(
        working_df[COORDINATE_COLUMN],
        errors="coerce",
    )

    working_df[axis_column] = pd.to_numeric(
        working_df[axis_column],
        errors="coerce",
    )

    invalid_geometry_mask = working_df[
        [
            COORDINATE_COLUMN,
            axis_column,
        ]
    ].isna().any(axis=1)

    if invalid_geometry_mask.any():
        invalid_rows = working_df.loc[
            invalid_geometry_mask,
            [
                EXPERIMENT_COLUMN,
                GEOMETRY_GROUP_COLUMN,
                COORDINATE_COLUMN,
                axis_column,
            ],
        ].head(20)

        raise ValueError(
            f"Invalid numeric geometry values were found for "
            f"{geometry_type}:\n"
            f"{invalid_rows.to_string(index=False)}"
        )

    # Keep the original CSV row order.
    working_df = working_df.reset_index(
        names="_original_row_order"
    )

    # The same coordinate may appear several times for one Experiment_ID,
    # because the experiment has several augmented curve realizations.
    #
    # Example:
    # coordinate 0, first occurrence  -> augmentation instance 0
    # coordinate 0, second occurrence -> augmentation instance 1
    #
    # The same logic is applied independently to every coordinate.
    working_df[AUGMENTATION_INSTANCE_COLUMN] = (
        working_df.groupby(
            [
                GEOMETRY_GROUP_COLUMN,
                EXPERIMENT_COLUMN,
                COORDINATE_COLUMN,
            ],
            sort=False,
            dropna=False,
        )
        .cumcount()
    )

    working_df[CURVE_INSTANCE_COLUMN] = (
        working_df[EXPERIMENT_COLUMN].astype(str)
        + "__aug_"
        + working_df[AUGMENTATION_INSTANCE_COLUMN].astype(str)
    )

    duplicate_coordinate_mask = working_df.duplicated(
        subset=[
            GEOMETRY_GROUP_COLUMN,
            CURVE_INSTANCE_COLUMN,
            COORDINATE_COLUMN,
        ],
        keep=False,
    )

    if duplicate_coordinate_mask.any():
        duplicated_rows = working_df.loc[
            duplicate_coordinate_mask,
            [
                GEOMETRY_GROUP_COLUMN,
                EXPERIMENT_COLUMN,
                CURVE_INSTANCE_COLUMN,
                COORDINATE_COLUMN,
            ],
        ].head(30)

        raise ValueError(
            "Could not construct unique augmented curve instances:\n"
            f"{duplicated_rows.to_string(index=False)}"
        )

    # Validate that all generated curve instances have the same number of
    # coordinate points inside each group.
    point_counts = (
        working_df.groupby(
            [
                GEOMETRY_GROUP_COLUMN,
                CURVE_INSTANCE_COLUMN,
            ],
            sort=False,
            dropna=False,
        )[COORDINATE_COLUMN]
        .nunique()
    )

    group_length_variation = point_counts.groupby(
        level=GEOMETRY_GROUP_COLUMN
    ).nunique()

    inconsistent_groups = group_length_variation[
        group_length_variation > 1
    ]

    if not inconsistent_groups.empty:
        raise ValueError(
            "Some generated curve instances have different numbers of "
            "coordinates inside the same group:\n"
            f"{inconsistent_groups.to_string()}"
        )

    return working_df


def build_group_experiment_matrix(
    group_df: pd.DataFrame,
    axis_column: str,
    geometry_type: GeometryType,
    group_id: object,
) -> tuple[np.ndarray, list[object]]:
    """
    Convert all real and augmented curve instances inside one group into a
    matrix.

    Output shape:
        n_curve_instances x n_curve_points
    """
    curve_vectors: list[np.ndarray] = []
    curve_instance_ids: list[object] = []

    reference_coordinates: np.ndarray | None = None

    grouped_curves = group_df.groupby(
        CURVE_INSTANCE_COLUMN,
        sort=False,
        dropna=False,
    )

    for curve_instance_id, curve_df in grouped_curves:
        experiment_id = curve_df[
            EXPERIMENT_COLUMN
        ].iloc[0]

        curve_df = curve_df.sort_values(
            COORDINATE_COLUMN
        )

        coordinates = curve_df[
            COORDINATE_COLUMN
        ].to_numpy(dtype=float)

        axis_values = curve_df[
            axis_column
        ].to_numpy(dtype=float)

        if reference_coordinates is None:
            reference_coordinates = coordinates
        else:
            if len(coordinates) != len(reference_coordinates):
                raise ValueError(
                    f"{geometry_type} group {group_id!r} contains curve "
                    "instances with different lengths.\n"
                    f"Curve {curve_instance_id!r}, original experiment "
                    f"{experiment_id!r}, has {len(coordinates)} points; "
                    f"expected {len(reference_coordinates)}."
                )

            if not np.allclose(
                coordinates,
                reference_coordinates,
                rtol=0.0,
                atol=1e-8,
            ):
                raise ValueError(
                    f"{geometry_type} group {group_id!r} contains curve "
                    "instances with different coordinate grids.\n"
                    f"Problematic curve: {curve_instance_id!r}, "
                    f"original experiment: {experiment_id!r}"
                )

        if not np.isfinite(axis_values).all():
            raise ValueError(
                f"Non-finite {geometry_type} values exist in group "
                f"{group_id!r}, curve {curve_instance_id!r}."
            )

        curve_vectors.append(
            axis_values
        )

        curve_instance_ids.append(
            curve_instance_id
        )

    if not curve_vectors:
        return np.empty((0, 0)), []

    geometry_matrix = np.vstack(
        curve_vectors
    )

    return geometry_matrix, curve_instance_ids


# =============================================================================
# Group-spread target
# =============================================================================

def build_group_spread_dataframe(
    data: pd.DataFrame,
    axis_column: str,
    geometry_type: GeometryType,
    scale_curve_points: bool,
) -> pd.DataFrame:
    """
    Create one row per group.

    Target definition
    -----------------
    group_spread = mean pairwise Euclidean distance between all experiment
    curve vectors inside the group.
    """
    working_df = prepare_geometry_columns(
        data=data,
        axis_column=axis_column,
        geometry_type=geometry_type,
    )

    records: list[dict[str, object]] = []
    skipped_groups: list[object] = []

    grouped = working_df.groupby(
        GEOMETRY_GROUP_COLUMN,
        sort=False,
        dropna=False,
    )

    for group_id, group_df in grouped:
        geometry_matrix, experiment_ids = (
            build_group_experiment_matrix(
                group_df=group_df,
                axis_column=axis_column,
                geometry_type=geometry_type,
                group_id=group_id,
            )
        )

        n_experiments = geometry_matrix.shape[0]

        if n_experiments < 2:
            skipped_groups.append(group_id)
            continue

        if scale_curve_points:
            scaler = StandardScaler()

            # Each curve position is treated as one geometry dimension.
            geometry_matrix = scaler.fit_transform(
                geometry_matrix
            )

        pairwise_distances = pdist(
            geometry_matrix,
            metric="euclidean",
        )

        if pairwise_distances.size == 0:
            skipped_groups.append(group_id)
            continue

        setup_values = group_df[
            BENDING_SETUP_COLUMNS
        ].iloc[0]

        records.append(
            {
                GEOMETRY_GROUP_COLUMN: group_id,
                "geometry_type": geometry_type,
                **setup_values.to_dict(),
                "group_spread": float(
                    pairwise_distances.mean()
                ),
                "group_spread_std": float(
                    pairwise_distances.std(ddof=0)
                ),
                "group_spread_min": float(
                    pairwise_distances.min()
                ),
                "group_spread_max": float(
                    pairwise_distances.max()
                ),
                "n_experiments": int(
                    n_experiments
                ),
                "n_curve_points": int(
                    geometry_matrix.shape[1]
                ),
                "n_pairwise_distances": int(
                    pairwise_distances.size
                ),
            }
        )

    result_df = pd.DataFrame.from_records(
        records
    )

    if result_df.empty:
        raise ValueError(
            f"No valid group-level rows were created for "
            f"{geometry_type}."
        )

    if skipped_groups:
        print(
            f"\n[{geometry_type}] Groups skipped because they contain "
            "fewer than two experiments:"
        )
        print(skipped_groups)

    return result_df.reset_index(
        drop=True
    )


# =============================================================================
# Random Forest validation
# =============================================================================

def validate_model_features(
    group_spread_df: pd.DataFrame,
    geometry_type: GeometryType,
) -> list[str]:
    """
    Validate RF input columns and report constant features.

    Constant selected features remain in the result with zero importance.
    """
    feature_columns = list(
        BENDING_SETUP_COLUMNS
    )

    non_numeric_features = [
        feature
        for feature in feature_columns
        if not pd.api.types.is_numeric_dtype(
            group_spread_df[feature]
        )
    ]

    if non_numeric_features:
        raise TypeError(
            f"{geometry_type}: non-numeric bending-setup features:\n"
            f"{non_numeric_features}"
        )

    features_with_missing_values = [
        feature
        for feature in feature_columns
        if group_spread_df[feature].isna().any()
    ]

    if features_with_missing_values:
        raise ValueError(
            f"{geometry_type}: missing values exist in RF features:\n"
            f"{features_with_missing_values}"
        )

    constant_features = [
        feature
        for feature in feature_columns
        if group_spread_df[feature].nunique(
            dropna=False
        ) <= 1
    ]

    if constant_features:
        print(
            f"\n[{geometry_type}] Constant setup features detected. "
            "Their importance should be zero:"
        )
        print(constant_features)

    if len(constant_features) == len(feature_columns):
        raise ValueError(
            f"{geometry_type}: all selected bending-setup features are "
            "constant across groups."
        )

    return feature_columns


def create_random_forest(
    config: RankingConfig,
) -> RandomForestRegressor:
    """
    Construct the Random Forest model.
    """
    return RandomForestRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        max_features=config.max_features,
        random_state=config.random_state,
        n_jobs=-1,
    )


def create_cross_validator(
    n_groups: int,
    config: RankingConfig,
) -> KFold:
    """
    Build a reproducible K-fold validator.
    """
    if n_groups < 3:
        raise ValueError(
            "At least three group-level samples are required for "
            "cross-validation."
        )

    n_splits = min(
        config.cv_splits,
        n_groups,
    )

    if n_splits < 2:
        raise ValueError(
            "At least two cross-validation folds are required."
        )

    return KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=config.random_state,
    )


# =============================================================================
# Feature ranking
# =============================================================================

def train_and_rank_geometry(
    group_spread_df: pd.DataFrame,
    geometry_type: GeometryType,
    config: RankingConfig,
) -> pd.DataFrame:
    """
    Train one RF model and calculate held-out permutation importance.

    Permutation importance is calculated separately on every validation fold
    and then combined.
    """
    feature_columns = validate_model_features(
        group_spread_df=group_spread_df,
        geometry_type=geometry_type,
    )

    X = group_spread_df[
        feature_columns
    ].copy()

    y = group_spread_df[
        "group_spread"
    ].astype(float)

    cv = create_cross_validator(
        n_groups=len(group_spread_df),
        config=config,
    )

    base_model = create_random_forest(
        config=config,
    )

    oof_predictions = np.full(
        shape=len(group_spread_df),
        fill_value=np.nan,
        dtype=float,
    )

    baseline_oof_predictions = np.full(
        shape=len(group_spread_df),
        fill_value=np.nan,
        dtype=float,
    )

    fold_importance_arrays: list[np.ndarray] = []

    for fold_number, (
        train_indices,
        validation_indices,
    ) in enumerate(
        cv.split(X),
        start=1,
    ):
        model = clone(
            base_model
        )

        X_train = X.iloc[
            train_indices
        ]

        y_train = y.iloc[
            train_indices
        ]

        X_validation = X.iloc[
            validation_indices
        ]

        y_validation = y.iloc[
            validation_indices
        ]

        model.fit(
            X_train,
            y_train,
        )

        oof_predictions[
            validation_indices
        ] = model.predict(
            X_validation
        )

        baseline_oof_predictions[
            validation_indices
        ] = float(
            y_train.mean()
        )

        fold_permutation = permutation_importance(
            estimator=model,
            X=X_validation,
            y=y_validation,
            scoring="neg_mean_absolute_error",
            n_repeats=config.permutation_repeats,
            random_state=(
                config.random_state + fold_number
            ),
            n_jobs=-1,
        )

        fold_importance_arrays.append(
            fold_permutation.importances
        )

    if np.isnan(oof_predictions).any():
        raise RuntimeError(
            f"{geometry_type}: some OOF predictions were not generated."
        )

    combined_importances = np.concatenate(
        fold_importance_arrays,
        axis=1,
    )

    oof_mae = float(
        mean_absolute_error(
            y,
            oof_predictions,
        )
    )

    oof_rmse = float(
        np.sqrt(
            mean_squared_error(
                y,
                oof_predictions,
            )
        )
    )

    oof_r2 = float(
        r2_score(
            y,
            oof_predictions,
        )
    )

    baseline_mae = float(
        mean_absolute_error(
            y,
            baseline_oof_predictions,
        )
    )

    final_model = clone(
        base_model
    )

    final_model.fit(
        X,
        y,
    )

    ranking_df = pd.DataFrame(
        {
            "geometry_type": geometry_type,
            "feature": feature_columns,
            "permutation_importance_mean": (
                combined_importances.mean(axis=1)
            ),
            "permutation_importance_std": (
                combined_importances.std(
                    axis=1,
                    ddof=1,
                )
            ),
            "permutation_importance_min": (
                combined_importances.min(axis=1)
            ),
            "permutation_importance_max": (
                combined_importances.max(axis=1)
            ),
            "rf_impurity_importance": (
                final_model.feature_importances_
            ),
            "feature_unique_values": [
                int(X[feature].nunique(dropna=False))
                for feature in feature_columns
            ],
        }
    )

    ranking_df["rank"] = (
        ranking_df[
            "permutation_importance_mean"
        ]
        .rank(
            method="dense",
            ascending=False,
        )
        .astype(int)
    )

    ranking_df["n_groups"] = int(
        len(group_spread_df)
    )

    ranking_df["mean_group_spread"] = float(
        y.mean()
    )

    ranking_df["std_group_spread"] = float(
        y.std(ddof=1)
    )

    ranking_df["oof_mae"] = oof_mae
    ranking_df["oof_rmse"] = oof_rmse
    ranking_df["oof_r2"] = oof_r2
    ranking_df["baseline_oof_mae"] = baseline_mae

    ranking_df["beats_mean_baseline"] = bool(
        oof_mae < baseline_mae
    )

    ranking_df["ranking_reliability_warning"] = bool(
        oof_mae >= baseline_mae
        or oof_r2 <= 0
    )

    ranking_df = ranking_df[
        [
            "geometry_type",
            "feature",
            "rank",
            "permutation_importance_mean",
            "permutation_importance_std",
            "permutation_importance_min",
            "permutation_importance_max",
            "rf_impurity_importance",
            "feature_unique_values",
            "n_groups",
            "mean_group_spread",
            "std_group_spread",
            "oof_mae",
            "oof_rmse",
            "oof_r2",
            "baseline_oof_mae",
            "beats_mean_baseline",
            "ranking_reliability_warning",
        ]
    ]

    return ranking_df.sort_values(
        by=[
            "rank",
            "feature",
        ],
        ascending=[
            True,
            True,
        ],
    ).reset_index(
        drop=True
    )


# =============================================================================
# Output persistence
# =============================================================================

def save_final_ranking(
    ranking_df: pd.DataFrame,
) -> None:
    """
    Atomically save one compressed Parquet file.

    The previous output is overwritten on every successful execution.
    """
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_output_path = OUTPUT_PATH.with_name(
        f"{OUTPUT_PATH.stem}.tmp{OUTPUT_PATH.suffix}"
    )

    try:
        ranking_df.to_parquet(
            temporary_output_path,
            index=False,
            engine="pyarrow",
            compression="zstd",
        )

        temporary_output_path.replace(
            OUTPUT_PATH
        )

    except Exception:
        if temporary_output_path.exists():
            temporary_output_path.unlink()

        raise


# =============================================================================
# Terminal output
# =============================================================================

def print_group_spread_summary(
    main_group_df: pd.DataFrame,
    secondary_group_df: pd.DataFrame,
) -> None:
    """
    Print a concise summary of the constructed RF targets.
    """
    summary_df = pd.DataFrame(
        [
            {
                "geometry_type": "main",
                "n_groups": len(main_group_df),
                "mean_group_spread": main_group_df[
                    "group_spread"
                ].mean(),
                "std_group_spread": main_group_df[
                    "group_spread"
                ].std(ddof=1),
                "min_group_spread": main_group_df[
                    "group_spread"
                ].min(),
                "max_group_spread": main_group_df[
                    "group_spread"
                ].max(),
            },
            {
                "geometry_type": "secondary",
                "n_groups": len(secondary_group_df),
                "mean_group_spread": secondary_group_df[
                    "group_spread"
                ].mean(),
                "std_group_spread": secondary_group_df[
                    "group_spread"
                ].std(ddof=1),
                "min_group_spread": secondary_group_df[
                    "group_spread"
                ].min(),
                "max_group_spread": secondary_group_df[
                    "group_spread"
                ].max(),
            },
        ]
    )

    print("\n")
    print("=" * 120)
    print("GROUP SPREAD TARGET SUMMARY")
    print("=" * 120)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )


def print_final_ranking(
    ranking_df: pd.DataFrame,
) -> None:
    """
    Print the final Main and Secondary rankings in the terminal.
    """
    print("\n")
    print("=" * 160)
    print("BENDING SETUP GROUP-SPREAD FEATURE RANKING")
    print("=" * 160)

    terminal_columns = [
        "geometry_type",
        "feature",
        "rank",
        "permutation_importance_mean",
        "permutation_importance_std",
        "rf_impurity_importance",
        "feature_unique_values",
        "n_groups",
        "oof_mae",
        "oof_rmse",
        "oof_r2",
        "baseline_oof_mae",
        "beats_mean_baseline",
        "ranking_reliability_warning",
    ]

    print(
        ranking_df[
            terminal_columns
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\n")
    print("-" * 160)
    print(f"Project root:          {PROJECT_ROOT}")
    print(f"Augmented data:        {REAL_DATA_PATH}")
    print(f"Bending setup data:    {BENDING_SETUP_DATA_PATH}")
    print(f"Output overwritten at: {OUTPUT_PATH}")
    print("-" * 160)


# =============================================================================
# Complete pipeline
# =============================================================================

def run_bending_setup_ranking(
    config: RankingConfig | None = None,
) -> pd.DataFrame:
    """
    Run the complete Main and Secondary bending-setup ranking pipeline.

    Returns
    -------
    pd.DataFrame
        Combined Main and Secondary feature rankings.
    """
    config = config or RankingConfig()

    merged_data = load_and_merge_input_data()

    validate_setup_consistency_inside_groups(
        data=merged_data
    )

    main_group_spread_df = build_group_spread_dataframe(
        data=merged_data,
        axis_column=MAIN_AXIS_COLUMN,
        geometry_type="main",
        scale_curve_points=config.scale_curve_points,
    )

    secondary_group_spread_df = build_group_spread_dataframe(
        data=merged_data,
        axis_column=SECONDARY_AXIS_COLUMN,
        geometry_type="secondary",
        scale_curve_points=config.scale_curve_points,
    )

    print_group_spread_summary(
        main_group_df=main_group_spread_df,
        secondary_group_df=secondary_group_spread_df,
    )

    main_ranking_df = train_and_rank_geometry(
        group_spread_df=main_group_spread_df,
        geometry_type="main",
        config=config,
    )

    secondary_ranking_df = train_and_rank_geometry(
        group_spread_df=secondary_group_spread_df,
        geometry_type="secondary",
        config=config,
    )

    final_ranking_df = pd.concat(
        [
            main_ranking_df,
            secondary_ranking_df,
        ],
        ignore_index=True,
    )

    geometry_sort_order = pd.Categorical(
        final_ranking_df["geometry_type"],
        categories=[
            "main",
            "secondary",
        ],
        ordered=True,
    )

    final_ranking_df = (
        final_ranking_df
        .assign(
            _geometry_sort_order=geometry_sort_order
        )
        .sort_values(
            by=[
                "_geometry_sort_order",
                "rank",
                "feature",
            ],
            ascending=[
                True,
                True,
                True,
            ],
        )
        .drop(
            columns="_geometry_sort_order"
        )
        .reset_index(
            drop=True
        )
    )

    save_final_ranking(
        ranking_df=final_ranking_df
    )

    print_final_ranking(
        ranking_df=final_ranking_df
    )

    return final_ranking_df


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    run_bending_setup_ranking(
        config=RankingConfig(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            max_features=1.0,
            cv_splits=5,
            permutation_repeats=30,
            random_state=42,

            # Main and Secondary are already in millimetres and are trained
            # independently. Therefore no point-wise scaling is applied.
            scale_curve_points=False,
        )
    )