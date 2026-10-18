from __future__ import annotations

import copy
import importlib.util
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.pipeline.rf_augmentation.rf_data_generator_pipeline import (
    RFDataGeneratorPipeline,
)


# ============================================================
# Configuration
# ============================================================

MIN_EXPERIMENTS_PER_GROUP = 1
MAX_EXPERIMENTS_PER_GROUP = 25
RANDOM_SEED = 42

# Choose the augmented source you want to study.
# This default matches one of the generated sources already configured
# in 3_1_run_qrf_interval_predictor.py.
AUGMENTED_SOURCE_NAME = "within_group_interpolation_raw"
AUGMENTED_SOURCE_RELATIVE_PATH = Path(
    "data/rf_augmented/final_geometry_within_group_interpolation_raw.parquet"
)

# Generate/refresh the augmented source automatically before the HGP sweep.
AUTO_GENERATE_AUGMENTED_DATA = True

# This must match AUGMENTED_SOURCE_NAME / AUGMENTED_SOURCE_RELATIVE_PATH above.
AUGMENTATION_FEATURE_SAMPLING_MODE = "within-group-interpolation"
AUGMENTATION_SENSOR_MODE = "raw"
AUGMENTATION_INCLUDE_ALL_FEATURES = False
AUGMENTATION_SENSOR_NOISE_SNR_DB = 40.0

# Final result requested for the experiment.
OUTPUT_RELATIVE_PATH = Path(
    "ml_result/experiments_per_group/hgp_experiments_per_group_hgp.parquet"
)

# Existing HGP runner/configuration file.
HGP_DRIVER_FILENAME = "3_2_run_hgp_interval_predictor.py"

# One source key is enough because this experiment evaluates only the
# selected augmented source.
CURATED_SOURCE_KEY = "experiments_per_group_curated"

# Requested output columns.
REQUESTED_METRIC_ALIASES = {
    "r2": (
        "r2",
        "r2_score",
        "r_squared",
    ),
    "mse": (
        "mse",
        "mean_squared_error",
    ),
    "rmse": (
        "rmse",
        "root_mean_squared_error",
    ),
    "mae": (
        "mae",
        "mean_absolute_error",
    ),
    "std": (
        "std",
        "std_dev",
        "standard_deviation",
        "residual_std",
        "prediction_std",
    ),
    "coverage": (
        "coverage",
        "interval_coverage",
        "prediction_interval_coverage",
        "picp",
    ),
    "interval_width": (
        "interval_width",
        "mean_interval_width",
        "avg_interval_width",
        "average_interval_width",
        "mpiw",
    ),
    "bias": (
        "bias",
        "mean_bias",
        "prediction_bias",
    ),
}


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# Existing HGP driver loader
# ============================================================

def load_hgp_driver(project_root: Path):
    """
    Load 3_1_run_qrf_interval_predictor.py as a module.

    The filename starts with a digit, so a normal Python import cannot be used.
    """
    driver_path = project_root / HGP_DRIVER_FILENAME

    if not driver_path.exists():
        raise FileNotFoundError(
            f"HGP driver was not found: {driver_path}"
        )

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    spec = importlib.util.spec_from_file_location(
        "qrf_interval_predictor_driver",
        driver_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load HGP driver from {driver_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ============================================================
# IO helpers
# ============================================================

def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    raise ValueError(
        f"Unsupported table format: {path}"
    )


def resolve_group_column(df: pd.DataFrame) -> str:
    if "Group_ID" in df.columns:
        return "Group_ID"

    if "group_id" in df.columns:
        return "group_id"

    raise KeyError(
        "No Group_ID/group_id column was found. "
        f"Available columns: {df.columns.tolist()}"
    )


def normalize_experiment_ids(values: Any) -> list[int]:
    if isinstance(values, np.ndarray):
        values = values.tolist()

    if isinstance(values, pd.Series):
        values = values.tolist()

    if not isinstance(values, (list, tuple, set)):
        values = [values]

    return sorted(
        {
            int(value)
            for value in values
            if pd.notna(value)
        }
    )


# ============================================================
# Metric helpers
# ============================================================

def normalize_metric_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("%", "pct")
    )


def flatten_metrics(
    value: Any,
    prefix: str = "",
) -> dict[str, Any]:
    """
    Flatten nested metric dictionaries.

    Example:
        {"point": {"mae": 1.2}}
    becomes:
        {"point.mae": 1.2}
    """
    if not isinstance(value, dict):
        return {prefix: value}

    flattened: dict[str, Any] = {}

    for key, child in value.items():
        child_key = (
            f"{prefix}.{key}"
            if prefix
            else str(key)
        )
        flattened.update(
            flatten_metrics(
                child,
                prefix=child_key,
            )
        )

    return flattened


def extract_requested_metrics(
    result: dict[str, Any],
) -> dict[str, float]:
    """
    Extract the requested HGP metrics.

    Current aggregate metrics:
        coverage_percent
        mae
        mean_interval_width
        rmse

    Derived:
        mse = rmse ** 2

    Point-prediction metrics are computed from predictions_df:
        r2
        bias = mean(y_pred - y_true)
        std  = std(y_pred - y_true)
    """
    metrics = result.get("metrics", {})
    flat_metrics = flatten_metrics(metrics)

    normalized_lookup = {
        normalize_metric_name(key): value
        for key, value in flat_metrics.items()
    }

    def find_metric(*aliases: str) -> float | None:
        normalized_aliases = [
            normalize_metric_name(alias)
            for alias in aliases
        ]

        for alias in normalized_aliases:
            if alias in normalized_lookup:
                return float(normalized_lookup[alias])

        for metric_key, metric_value in normalized_lookup.items():
            for alias in normalized_aliases:
                if metric_key.endswith(f".{alias}"):
                    return float(metric_value)

        return None

    rmse = find_metric(
        "rmse",
        "root_mean_squared_error",
    )
    mae = find_metric(
        "mae",
        "mean_absolute_error",
    )
    interval_width = find_metric(
        "mean_interval_width",
        "interval_width",
        "avg_interval_width",
        "average_interval_width",
        "mpiw",
    )
    coverage = find_metric(
        "coverage_percent",
        "coverage",
        "interval_coverage",
        "prediction_interval_coverage",
        "picp",
    )

    required_values = {
        "rmse": rmse,
        "mae": mae,
        "coverage": coverage,
        "interval_width": interval_width,
    }

    missing = [
        name
        for name, value in required_values.items()
        if value is None
    ]

    if missing:
        raise KeyError(
            "HGP result is missing required aggregate metrics: "
            f"{missing}. Available metric keys: {sorted(flat_metrics)}"
        )

    mse = float(rmse ** 2)

    # --------------------------------------------------------
    # Compute R2 / bias / residual std from predictions_df
    # --------------------------------------------------------
    predictions_df = result.get("predictions_df")

    if not isinstance(predictions_df, pd.DataFrame):
        raise TypeError(
            "HGP result['predictions_df'] is required to compute "
            "r2, bias, and std, but it is missing or is not a DataFrame."
        )

    if predictions_df.empty:
        raise ValueError(
            "HGP result['predictions_df'] is empty."
        )

    normalized_columns = {
        normalize_metric_name(column): column
        for column in predictions_df.columns
    }

    def find_column(
        exact_candidates: tuple[str, ...],
        contains_candidates: tuple[str, ...],
        *,
        exclude_contains: tuple[str, ...] = (),
    ) -> str | None:
        # Exact match first.
        for candidate in exact_candidates:
            candidate_norm = normalize_metric_name(candidate)
            if candidate_norm in normalized_columns:
                return normalized_columns[candidate_norm]

        # Then conservative substring matching.
        for normalized_name, original_name in normalized_columns.items():
            if any(
                excluded in normalized_name
                for excluded in exclude_contains
            ):
                continue

            if any(
                candidate in normalized_name
                for candidate in contains_candidates
            ):
                return original_name

        return None

    y_true_column = find_column(
        exact_candidates=(
            "y_true",
            "y_test",
            "actual",
            "actual_value",
            "true_value",
            "target",
            "target_value",
            "observed",
        ),
        contains_candidates=(
            "y_true",
            "actual",
            "true",
            "observed",
        ),
        exclude_contains=(
            "lower",
            "upper",
            "pred",
            "error",
        ),
    )

    y_pred_column = find_column(
        exact_candidates=(
            "y_pred",
            "prediction",
            "predicted",
            "predicted_value",
            "median_prediction",
            "point_prediction",
            "q50",
            "median",
        ),
        contains_candidates=(
            "y_pred",
            "prediction",
            "predicted",
            "median",
            "q50",
        ),
        exclude_contains=(
            "lower",
            "upper",
            "interval",
            "width",
        ),
    )

    if y_true_column is None or y_pred_column is None:
        raise KeyError(
            "Could not identify true/predicted value columns in predictions_df. "
            f"Available columns: {predictions_df.columns.tolist()}"
        )

    y_true = pd.to_numeric(
        predictions_df[y_true_column],
        errors="coerce",
    ).to_numpy(dtype=float)

    y_pred = pd.to_numeric(
        predictions_df[y_pred_column],
        errors="coerce",
    ).to_numpy(dtype=float)

    valid_mask = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )

    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    if y_true.size == 0:
        raise ValueError(
            "No valid y_true/y_pred pairs were found in predictions_df."
        )

    residual = y_pred - y_true

    bias = float(
        np.mean(residual)
    )

    std = float(
        np.std(
            residual,
            ddof=0,
        )
    )

    ss_res = float(
        np.sum(
            (y_true - y_pred) ** 2
        )
    )

    ss_tot = float(
        np.sum(
            (y_true - np.mean(y_true)) ** 2
        )
    )

    r2 = (
        float(1.0 - ss_res / ss_tot)
        if ss_tot > 0.0
        else np.nan
    )

    logger.info(
        "Prediction columns | y_true=%s | y_pred=%s | n=%s",
        y_true_column,
        y_pred_column,
        y_true.size,
    )

    return {
        "r2": float(r2),
        "mse": mse,
        "rmse": float(rmse),
        "mae": float(mae),
        "std": std,
        "coverage": float(coverage),
        "interval_width": float(interval_width),
        "bias": bias,
    }


def add_train_group_ids_from_augmented_geometry(
    top_splits_by_axis: dict[str, dict],
    augmented_df: pd.DataFrame,
) -> dict[str, dict]:
    """
    Add train_group_ids to the HGP split metadata.

    The original real geometry.csv does not contain Group_ID. The generated
    augmented geometry does contain Group_ID and also contains the original
    real Experiment_ID values, so it is the correct source for deriving the
    training groups used by the stored split.
    """
    group_column = resolve_group_column(augmented_df)

    required_columns = {
        "Experiment_ID",
        group_column,
    }
    missing = required_columns.difference(augmented_df.columns)

    if missing:
        raise KeyError(
            "Augmented geometry is missing columns required to derive HGP "
            f"training groups: {sorted(missing)}"
        )

    lookup_df = augmented_df[
        ["Experiment_ID", group_column]
    ].copy()

    lookup_df["Experiment_ID"] = pd.to_numeric(
        lookup_df["Experiment_ID"],
        errors="raise",
    ).astype(int)

    lookup_df[group_column] = pd.to_numeric(
        lookup_df[group_column],
        errors="raise",
    ).astype(int)

    output = copy.deepcopy(top_splits_by_axis)

    for axis in ("main", "secondary"):
        train_exp = normalize_experiment_ids(
            output[axis]["train_exp"]
        )

        matched = lookup_df[
            lookup_df["Experiment_ID"].isin(train_exp)
        ].copy()

        present = set(
            matched["Experiment_ID"].unique().tolist()
        )
        missing_exp = sorted(set(train_exp).difference(present))

        if missing_exp:
            raise ValueError(
                f"Could not map all HGP training experiments to Group_ID "
                f"for axis={axis!r}. Missing Experiment_IDs: {missing_exp}"
            )

        output[axis]["train_group_ids"] = sorted(
            matched[group_column].unique().astype(int).tolist()
        )

    return output


# ============================================================
# Build nested experiment ordering
# ============================================================

def build_experiment_order_by_group(
    augmented_df: pd.DataFrame,
    real_df: pd.DataFrame,
    candidate_group_ids: set[int],
    excluded_experiments: set[int],
    random_seed: int,
) -> dict[int, list[int]]:
    """
    Build one deterministic random experiment ordering per group.

    Policy:
      1. Real experiments always come first.
      2. Augmented experiments come after the real experiments.
      3. Within the real and augmented partitions, order is randomized.
      4. N experiments/group means taking the first N IDs.

    Therefore the learning curve is nested:
        selection(N=1) subset selection(N=2) subset ... subset selection(N=15)

    This is preferable to independently re-sampling every N because the effect
    of adding experiments is easier to interpret.
    """
    group_column = resolve_group_column(augmented_df)

    required_columns = {
        group_column,
        "Experiment_ID",
    }
    missing_columns = required_columns.difference(
        augmented_df.columns
    )

    if missing_columns:
        raise KeyError(
            "Augmented geometry is missing columns: "
            f"{sorted(missing_columns)}"
        )

    real_experiment_ids = set(
        pd.to_numeric(
            real_df["Experiment_ID"],
            errors="raise",
        )
        .astype(int)
        .unique()
        .tolist()
    )

    work_df = augmented_df.copy()
    work_df[group_column] = pd.to_numeric(
        work_df[group_column],
        errors="raise",
    ).astype(int)
    work_df["Experiment_ID"] = pd.to_numeric(
        work_df["Experiment_ID"],
        errors="raise",
    ).astype(int)

    work_df = work_df[
        work_df[group_column].isin(candidate_group_ids)
    ].copy()

    work_df = work_df[
        ~work_df["Experiment_ID"].isin(
            excluded_experiments
        )
    ].copy()

    rng = np.random.default_rng(random_seed)
    experiment_order_by_group: dict[int, list[int]] = {}

    for group_id in sorted(candidate_group_ids):
        group_experiment_ids = (
            work_df.loc[
                work_df[group_column].eq(group_id),
                "Experiment_ID",
            ]
            .drop_duplicates()
            .astype(int)
            .tolist()
        )

        if not group_experiment_ids:
            raise ValueError(
                f"No usable experiments found for Group_ID={group_id}."
            )

        real_ids = sorted(
            exp_id
            for exp_id in group_experiment_ids
            if exp_id in real_experiment_ids
        )

        augmented_ids = sorted(
            exp_id
            for exp_id in group_experiment_ids
            if exp_id not in real_experiment_ids
        )

        if not real_ids:
            raise ValueError(
                f"Group_ID={group_id} has no real experiment. "
                "The requested policy requires N=1 to contain a real sample."
            )

        real_ids = list(
            rng.permutation(real_ids)
        )
        augmented_ids = list(
            rng.permutation(augmented_ids)
        )

        ordered_ids = [
            int(exp_id)
            for exp_id in (
                real_ids
                + augmented_ids
            )
        ]

        if len(ordered_ids) < MAX_EXPERIMENTS_PER_GROUP:
            raise ValueError(
                f"Group_ID={group_id} has only {len(ordered_ids)} usable "
                f"experiments, but MAX_EXPERIMENTS_PER_GROUP="
                f"{MAX_EXPERIMENTS_PER_GROUP}."
            )

        experiment_order_by_group[group_id] = ordered_ids

        logger.info(
            "Prepared group | group=%s | real=%s | augmented=%s | total=%s",
            group_id,
            len(real_ids),
            len(augmented_ids),
            len(ordered_ids),
        )

    return experiment_order_by_group


def selected_train_experiments(
    train_group_ids: list[int],
    experiment_order_by_group: dict[int, list[int]],
    experiments_per_group: int,
) -> list[int]:
    selected: list[int] = []

    for group_id in train_group_ids:
        group_id = int(group_id)

        if group_id not in experiment_order_by_group:
            raise KeyError(
                f"No experiment ordering exists for Group_ID={group_id}."
            )

        selected.extend(
            experiment_order_by_group[group_id][
                :experiments_per_group
            ]
        )

    return normalize_experiment_ids(selected)


# ============================================================
# Curated geometry + axis-specific split
# ============================================================

def build_curated_geometry(
    augmented_df: pd.DataFrame,
    selected_train_ids_by_axis: dict[str, list[int]],
    fixed_test_ids_by_axis: dict[str, list[int]],
) -> pd.DataFrame:
    """
    Keep only rows needed by either axis.

    Training IDs vary with N.
    Test IDs are fixed original real IDs from the stored Top-1 split.
    """
    required_ids: set[int] = set()

    for ids in selected_train_ids_by_axis.values():
        required_ids.update(ids)

    for ids in fixed_test_ids_by_axis.values():
        required_ids.update(ids)

    experiment_ids = pd.to_numeric(
        augmented_df["Experiment_ID"],
        errors="raise",
    ).astype(int)

    curated_df = augmented_df[
        experiment_ids.isin(required_ids)
    ].copy()

    present_ids = set(
        pd.to_numeric(
            curated_df["Experiment_ID"],
            errors="raise",
        )
        .astype(int)
        .unique()
        .tolist()
    )

    missing_ids = sorted(
        required_ids.difference(present_ids)
    )

    if missing_ids:
        raise ValueError(
            "The augmented source does not contain all required train/test "
            f"Experiment_ID values. Missing IDs: {missing_ids}"
        )

    return curated_df


def build_curated_split_by_axis(
    top_splits_by_axis: dict[str, dict],
    selected_train_ids_by_axis: dict[str, list[int]],
) -> dict[str, dict]:
    """
    Preserve the independent best main/secondary split metadata, but replace
    train_exp with the controlled N-experiments-per-group subset.

    test_exp remains unchanged and real.
    """
    split_by_axis: dict[str, dict] = {}

    for axis in ("main", "secondary"):
        split_config = copy.deepcopy(
            top_splits_by_axis[axis]
        )

        train_exp = normalize_experiment_ids(
            selected_train_ids_by_axis[axis]
        )
        test_exp = normalize_experiment_ids(
            split_config["test_exp"]
        )

        overlap = set(train_exp).intersection(
            test_exp
        )

        if overlap:
            raise ValueError(
                f"Train/test leakage for axis={axis!r}. "
                f"Overlapping Experiment_IDs: {sorted(overlap)}"
            )

        split_config["train_exp"] = train_exp
        split_config["test_exp"] = test_exp
        split_config["split_selection_mode"] = (
            "hgp_best_rank_experiments_per_group"
        )

        split_by_axis[axis] = split_config

    return split_by_axis


# ============================================================
# Augmented-data generation
# ============================================================

def generate_augmented_source(
    project_root: Path,
) -> Path:
    """
    Generate/refresh the augmented geometry before the HGP sweep.

    MAX_EXPERIMENTS_PER_GROUP controls both:
      1. how many samples per group are generated/available;
      2. the upper end of the QRF experiments-per-group sweep.

    Example:
        MAX_EXPERIMENTS_PER_GROUP = 25
    first regenerates the source with n_new_samples=25, then runs
    QRF for N = 1, 2, ..., 25.
    """
    output_dir = (
        project_root
        / "data"
        / "rf_augmented"
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "Generating augmented data before HGP sweep | "
        "mode=%s | n_new_samples=%s | sensor_mode=%s",
        AUGMENTATION_FEATURE_SAMPLING_MODE,
        MAX_EXPERIMENTS_PER_GROUP,
        AUGMENTATION_SENSOR_MODE,
    )

    RFDataGeneratorPipeline.run(
        project_root=project_root,
        n_new_samples=MAX_EXPERIMENTS_PER_GROUP,
        output_dir=output_dir,
        feature_sampling_mode=(
            AUGMENTATION_FEATURE_SAMPLING_MODE
        ),
        include_all_features=(
            AUGMENTATION_INCLUDE_ALL_FEATURES
        ),
        sensor_augmentation_mode=(
            AUGMENTATION_SENSOR_MODE
        ),
        regenerate_sensor_data=True,
        sensor_noise_snr_db=(
            AUGMENTATION_SENSOR_NOISE_SNR_DB
        ),
    )

    augmented_path = (
        project_root
        / AUGMENTED_SOURCE_RELATIVE_PATH
    ).resolve()

    if not augmented_path.exists():
        raise FileNotFoundError(
            "Augmentation finished, but the expected augmented geometry "
            f"was not found: {augmented_path}"
        )

    logger.info(
        "Augmented source generated successfully: %s",
        augmented_path,
    )

    return augmented_path


# ============================================================
# Main experiment
# ============================================================

def main() -> None:
    project_root = Path(
        __file__
    ).resolve().parent

    hgp_driver = load_hgp_driver(
        project_root
    )

    if AUTO_GENERATE_AUGMENTED_DATA:
        augmented_path = generate_augmented_source(
            project_root=project_root,
        )
    else:
        augmented_path = (
            project_root
            / AUGMENTED_SOURCE_RELATIVE_PATH
        ).resolve()

    real_geometry_path = (
        project_root
        / "data"
        / "processed"
        / "geometry.csv"
    ).resolve()

    output_path = (
        project_root
        / OUTPUT_RELATIVE_PATH
    ).resolve()

    bending_setups_path = (
        project_root
        / hgp_driver.BENDING_SETUPS_PATH
    ).resolve()

    for path, label in (
        (augmented_path, "augmented geometry"),
        (real_geometry_path, "real geometry"),
        (bending_setups_path, "bending setups"),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {label}: {path}"
            )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "Augmented source: %s",
        augmented_path,
    )
    logger.info(
        "Output: %s",
        output_path,
    )

    augmented_df = read_table(
        augmented_path
    )
    real_df = read_table(
        real_geometry_path
    )

    # --------------------------------------------------------
    # Independent best split for main and secondary
    # --------------------------------------------------------
    top_splits_by_axis = (
        hgp_driver.load_top_splits_by_axis(
            project_root=project_root,
        )
    )

    top_splits_by_axis = (
        add_train_group_ids_from_augmented_geometry(
            top_splits_by_axis=top_splits_by_axis,
            augmented_df=augmented_df,
        )
    )

    # --------------------------------------------------------
    # Best QRF config
    # --------------------------------------------------------
    # Load tuned params BEFORE modifying split_selection_mode.
    # This preserves compatibility with the existing tuning metadata.
    tuned_params_by_axis = (
        hgp_driver.load_tuned_hgp_params_by_axis(
            project_root=project_root,
            selected_splits_by_axis=top_splits_by_axis,
        )
    )

    if tuned_params_by_axis is None:
        # Match the behavior of the existing HGP driver:
        # if tuned parameters do not match the selected Top-1 split metadata,
        # use the hard-coded BEST_QRF_PARAMS_BY_DATASET configuration for the
        # selected augmented source.
        if AUGMENTED_SOURCE_NAME not in hgp_driver.BEST_QRF_PARAMS_BY_DATASET:
            raise KeyError(
                "No hard-coded best HGP configuration exists for "
                f"AUGMENTED_SOURCE_NAME={AUGMENTED_SOURCE_NAME!r}. "
                "Available sources: "
                f"{sorted(hgp_driver.BEST_QRF_PARAMS_BY_DATASET)}"
            )

        best_dataset_params = dict(
            hgp_driver.BEST_QRF_PARAMS_BY_DATASET[
                AUGMENTED_SOURCE_NAME
            ]
        )

        tuned_params_by_axis = {
            "main": dict(best_dataset_params),
            "secondary": dict(best_dataset_params),
        }

        logger.warning(
            "Tuned QRF parameters do not match the selected Top-1 split. "
            "Using hard-coded BEST_QRF_PARAMS_BY_DATASET for source=%r.",
            AUGMENTED_SOURCE_NAME,
        )

    logger.info(
        "Best HGP params | main=%s",
        tuned_params_by_axis["main"],
    )
    logger.info(
        "Best HGP params | secondary=%s",
        tuned_params_by_axis["secondary"],
    )

    # --------------------------------------------------------
    # Build one nested real-first random ordering per train group
    # --------------------------------------------------------
    all_train_group_ids = {
        int(group_id)
        for axis in ("main", "secondary")
        for group_id in top_splits_by_axis[
            axis
        ]["train_group_ids"]
    }

    excluded_experiments = {
        int(exp_id)
        for exp_id
        in hgp_driver.HGP_EXCLUDED_EXPERIMENTS
    }

    experiment_order_by_group = (
        build_experiment_order_by_group(
            augmented_df=augmented_df,
            real_df=real_df,
            candidate_group_ids=all_train_group_ids,
            excluded_experiments=excluded_experiments,
            random_seed=RANDOM_SEED,
        )
    )

    fixed_test_ids_by_axis = {
        axis: normalize_experiment_ids(
            top_splits_by_axis[axis][
                "test_exp"
            ]
        )
        for axis in ("main", "secondary")
    }

    result_rows: list[dict[str, Any]] = []

    # ========================================================
    # Sweep N = MIN_EXPERIMENTS_PER_GROUP ... MAX_EXPERIMENTS_PER_GROUP
    # ========================================================
    for experiments_per_group in range(
        MIN_EXPERIMENTS_PER_GROUP,
        MAX_EXPERIMENTS_PER_GROUP + 1,
    ):
        logger.info(
            "Running HGP experiment | experiments_per_group=%s",
            experiments_per_group,
        )

        selected_train_ids_by_axis = {
            axis: selected_train_experiments(
                train_group_ids=top_splits_by_axis[
                    axis
                ]["train_group_ids"],
                experiment_order_by_group=(
                    experiment_order_by_group
                ),
                experiments_per_group=(
                    experiments_per_group
                ),
            )
            for axis in ("main", "secondary")
        }

        curated_split_by_axis = (
            build_curated_split_by_axis(
                top_splits_by_axis=(
                    top_splits_by_axis
                ),
                selected_train_ids_by_axis=(
                    selected_train_ids_by_axis
                ),
            )
        )

        curated_df = build_curated_geometry(
            augmented_df=augmented_df,
            selected_train_ids_by_axis=(
                selected_train_ids_by_axis
            ),
            fixed_test_ids_by_axis=(
                fixed_test_ids_by_axis
            ),
        )

        # Temporary source/model paths keep the main project outputs clean.
        with tempfile.TemporaryDirectory(
            prefix=(
                f"hgp_exp_per_group_"
                f"{experiments_per_group:02d}_"
            )
        ) as temp_dir_name:
            temp_dir = Path(
                temp_dir_name
            )

            curated_path = (
                temp_dir
                / "curated_geometry.parquet"
            )
            curated_df.to_parquet(
                curated_path,
                index=False,
            )

            geometry_sources = {
                CURATED_SOURCE_KEY: curated_path
            }

            splits_by_source = {
                CURATED_SOURCE_KEY: (
                    curated_split_by_axis
                )
            }

            model_params_by_source = (
                hgp_driver.build_model_params_by_source_from_axis_params(
                    geometry_sources=(
                        geometry_sources
                    ),
                    tuned_params_by_axis=(
                        tuned_params_by_axis
                    ),
                )
            )

            model_root = (
                temp_dir
                / "models"
            )

            results = hgp_driver.run(
                project_root=project_root,
                geometry_sources=geometry_sources,
                splits_by_source=splits_by_source,
                feature_columns=list(
                    hgp_driver.HGP_FEATURE_COLUMNS
                ),
                excluded_experiments=list(
                    hgp_driver.HGP_EXCLUDED_EXPERIMENTS
                ),
                model_params_by_source=(
                    model_params_by_source
                ),
                model_root=model_root,
                bending_setups_path=(
                    bending_setups_path
                ),
            )

            source_results = results[
                CURATED_SOURCE_KEY
            ]

            # ------------------------------------------------
            # One output row for main + one for secondary
            # ------------------------------------------------
            for axis in (
                "main",
                "secondary",
            ):
                result = source_results[
                    axis
                ]

                extracted_metrics = (
                    extract_requested_metrics(
                        result
                    )
                )

                train_groups = (
                    top_splits_by_axis[
                        axis
                    ]["train_group_ids"]
                )

                row = {
                    "experiments_per_group": (
                        experiments_per_group
                    ),
                    "target_axis": axis,
                    "geometry_source": (
                        AUGMENTED_SOURCE_NAME
                    ),
                    "random_seed": RANDOM_SEED,
                    "split_index": int(
                        top_splits_by_axis[
                            axis
                        ]["split_index"]
                    ),
                    "split_name": str(
                        top_splits_by_axis[
                            axis
                        ]["split_name"]
                    ),
                    "hgp_rank": int(
                        top_splits_by_axis[
                            axis
                        ]["hgp_rank"]
                    ),
                    "hgp_score": float(
                        top_splits_by_axis[
                            axis
                        ]["source_score"]
                    ),
                    "n_train_groups": len(
                        train_groups
                    ),
                    "n_train_experiments": len(
                        selected_train_ids_by_axis[
                            axis
                        ]
                    ),
                    "n_test_experiments": len(
                        fixed_test_ids_by_axis[
                            axis
                        ]
                    ),
                    **extracted_metrics,
                }

                result_rows.append(
                    row
                )

                logger.info(
                    "Result | N=%s | axis=%s | "
                    "R2=%.6f | MSE=%.6f | RMSE=%.6f | "
                    "MAE=%.6f | STD=%.6f | coverage=%.6f | "
                    "interval_width=%.6f | bias=%.6f",
                    experiments_per_group,
                    axis,
                    row["r2"],
                    row["mse"],
                    row["rmse"],
                    row["mae"],
                    row["std"],
                    row["coverage"],
                    row["interval_width"],
                    row["bias"],
                )

    # ========================================================
    # Final parquet
    # ========================================================
    result_df = pd.DataFrame(
        result_rows
    ).sort_values(
        [
            "experiments_per_group",
            "target_axis",
        ]
    ).reset_index(
        drop=True
    )

    expected_rows = (
        (
            MAX_EXPERIMENTS_PER_GROUP
            - MIN_EXPERIMENTS_PER_GROUP
            + 1
        )
        * 2
    )

    if len(result_df) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} result rows, "
            f"but received {len(result_df)}."
        )

    ordered_columns = [
        "experiments_per_group",
        "target_axis",
        "geometry_source",
        "random_seed",
        "split_index",
        "split_name",
        "hgp_rank",
        "hgp_score",
        "n_train_groups",
        "n_train_experiments",
        "n_test_experiments",
        "r2",
        "mse",
        "rmse",
        "mae",
        "std",
        "coverage",
        "interval_width",
        "bias",
    ]

    result_df = result_df[
        ordered_columns
    ]

    result_df.to_parquet(
        output_path,
        index=False,
    )

    logger.info(
        "Completed experiments-per-group sweep."
    )
    logger.info(
        "Stored %s rows in %s",
        len(result_df),
        output_path,
    )


if __name__ == "__main__":
    main()
