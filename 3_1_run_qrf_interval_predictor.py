from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.pipeline.ml.qrf.mode.experiments.qrf_pipeline import (
    run,
)


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger(__name__)


# ============================================================
# QRF configuration
# ============================================================

QRF_FEATURE_COLUMNS = [
    "Group_ID",
    "Angle[degree]ORDistance[mm]",
]


QRF_EXCLUDED_EXPERIMENTS = [
    1,
    48,
    166,
]


IS_BEST_CONFIG = True


BEST_QRF_PARAMS_BY_DATASET = {
    "real": {
        "n_estimators": 200,
        "max_depth": 5,
        "min_samples_leaf": 10,
        "min_samples_split": 10,
        "max_features": 0.7,
        "bootstrap": True,
        "lower_quantile": 0.05,
        "upper_quantile": 0.95,
    },
    (
        "sensor_augmented_noise__"
        "time_wrapping__scaling__jittering"
    ): {
        "n_estimators": 200,
        "max_depth": 5,
        "min_samples_leaf": 10,
        "min_samples_split": 10,
        "max_features": 0.7,
        "bootstrap": True,
        "lower_quantile": 0.05,
        "upper_quantile": 0.95,
    },
    "within_group_interpolation_raw": {
        "n_estimators": 200,
        "max_depth": 5,
        "min_samples_leaf": 10,
        "min_samples_split": 10,
        "max_features": 0.7,
        "bootstrap": True,
        "lower_quantile": 0.05,
        "upper_quantile": 0.95,
    },
}


STORED_QRF_SPLIT_PATH = Path(
    "src"
    "/pipeline"
    "/ml"
    "/qrf"
    "/data"
    "/various_splits.parquet"
)


STORED_QRF_TUNING_PATH = Path(
    "src"
    "/pipeline"
    "/ml"
    "/qrf"
    "/data"
    "/qrf_hyperparameter_tuning_results.parquet"
)


STORED_QRF_MODEL_DIR = Path(
    "src"
    "/pipeline"
    "/ml"
    "/qrf"
    "/result"
    "/models"
)


BENDING_SETUPS_PATH = Path(
    "data"
    "/rf_augmented"
    "/ui_data"
    "/unique_bending_setups.csv"
)


# ============================================================
# Generic helpers
# ============================================================

def _resolve_project_path(
    project_root: Path,
    path: str | Path,
) -> Path:
    """
    Resolve a project-relative or absolute path.
    """
    resolved_path = Path(path)

    if not resolved_path.is_absolute():
        resolved_path = (
            project_root
            / resolved_path
        )

    return resolved_path.resolve()


def _normalize_stored_experiment_ids(
    value: Any,
) -> list[int]:
    """
    Convert stored experiment IDs to a sorted unique integer list.

    Supported input formats include:
      - Python list;
      - tuple;
      - set;
      - NumPy array;
      - Pandas Series;
      - string representation of a list.
    """
    if isinstance(value, str):
        stripped_value = value.strip()

        if not stripped_value:
            return []

        value = ast.literal_eval(
            stripped_value
        )

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, pd.Series):
        value = value.tolist()

    if not isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        if pd.isna(value):
            return []

        value = [value]

    normalized_ids = sorted(
        {
            int(experiment_id)
            for experiment_id in value
            if pd.notna(experiment_id)
        }
    )

    return normalized_ids


def _validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    dataframe_name: str,
) -> None:
    """
    Validate required DataFrame columns.
    """
    missing_columns = (
        required_columns.difference(
            dataframe.columns
        )
    )

    if missing_columns:
        raise KeyError(
            f"{dataframe_name} is missing columns: "
            f"{sorted(missing_columns)}. "
            f"Available columns: "
            f"{dataframe.columns.tolist()}"
        )


# ============================================================
# Geometry sources
# ============================================================

def qrf_training_geometry_sources(
    project_root: Path,
) -> dict[str, Path]:
    """
    Return all geometry sources used for final QRF training.
    """
    source_paths = {
        "real": (
            project_root
            / "data"
            / "processed"
            / "geometry.csv"
        ),
        (
            "sensor_augmented_noise__"
            "time_wrapping__scaling__jittering"
        ): (
            project_root
            / "data"
            / "rf_augmented"
            / "ui_data"
            / (
                "final_geometry_sensor_augmented_noise__"
                "time_wrapping__scaling__jittering.csv"
            )
        ),
        "within_group_interpolation_raw": (
            project_root
            / "data"
            / "rf_augmented"
            / "ui_data"
            / (
                "final_geometry_within_group_"
                "interpolation_raw.csv"
            )
        ),
    }

    return {
        source_name: source_path.resolve()
        for source_name, source_path
        in source_paths.items()
    }


def validate_geometry_sources(
    geometry_sources: dict[str, Path],
) -> None:
    """
    Validate that all configured geometry source files exist.
    """
    if not geometry_sources:
        raise ValueError(
            "No QRF geometry sources were configured."
        )

    missing_sources = {
        source_name: source_path
        for source_name, source_path
        in geometry_sources.items()
        if not source_path.exists()
    }

    if missing_sources:
        formatted_missing_sources = "\n".join(
            (
                f"  - {source_name}: "
                f"{source_path}"
            )
            for source_name, source_path
            in missing_sources.items()
        )

        raise FileNotFoundError(
            "Some QRF geometry sources were not found:\n"
            f"{formatted_missing_sources}"
        )


# ============================================================
# Load independent Top-1 splits
# ============================================================

def load_top_qrf_splits_by_axis(
    project_root: Path,
) -> dict[str, dict]:
    """
    Load the independent Top-1 split for each target axis.

    Selection policy
    ----------------
    Main model:
        Select the unique row where qrf_rank_main == 1.

    Secondary model:
        Select the unique row where qrf_rank_secondary == 1.

    The stored rankings are shared across every geometry source.
    Therefore, various_splits.parquet does not need a
    geometry_source column.
    """
    split_path = _resolve_project_path(
        project_root=project_root,
        path=STORED_QRF_SPLIT_PATH,
    )

    if not split_path.exists():
        raise FileNotFoundError(
            "Stored QRF split metadata was not found: "
            f"{split_path}"
        )

    required_columns = {
        "split_index",
        "split_name",
        "train_experiment_ids",
        "test_experiment_ids",
        "qrf_rank_main",
        "qrf_score_main",
        "qrf_rank_secondary",
        "qrf_score_secondary",
    }

    split_df = pd.read_parquet(
        split_path,
        columns=sorted(
            required_columns
        ),
    )

    if split_df.empty:
        raise ValueError(
            "Stored QRF split metadata is empty: "
            f"{split_path}"
        )

    _validate_required_columns(
        dataframe=split_df,
        required_columns=required_columns,
        dataframe_name=(
            "Stored QRF split metadata"
        ),
    )

    split_df["split_index"] = (
        pd.to_numeric(
            split_df["split_index"],
            errors="raise",
        )
        .astype(int)
    )

    if split_df[
        "split_index"
    ].duplicated().any():
        duplicated_indices = sorted(
            split_df.loc[
                split_df[
                    "split_index"
                ].duplicated(
                    keep=False
                ),
                "split_index",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "various_splits.parquet must contain "
            "exactly one row per split_index. "
            f"Duplicated split indices: "
            f"{duplicated_indices}"
        )

    axis_configuration = {
        "main": {
            "rank_column": (
                "qrf_rank_main"
            ),
            "score_column": (
                "qrf_score_main"
            ),
            "target_column": (
                "Main-axis [mm]"
            ),
        },
        "secondary": {
            "rank_column": (
                "qrf_rank_secondary"
            ),
            "score_column": (
                "qrf_score_secondary"
            ),
            "target_column": (
                "Secondary-axis [mm]"
            ),
        },
    }

    selected_splits: dict[
        str,
        dict,
    ] = {}

    for (
        axis,
        axis_config,
    ) in axis_configuration.items():
        rank_column = (
            axis_config[
                "rank_column"
            ]
        )

        score_column = (
            axis_config[
                "score_column"
            ]
        )

        axis_df = split_df.dropna(
            subset=[
                rank_column,
                score_column,
            ]
        ).copy()

        if axis_df.empty:
            raise ValueError(
                "No ranked QRF splits are available "
                f"for axis={axis!r}. "
                "Run run_qrf_split_ranking.py first."
            )

        axis_df[rank_column] = (
            pd.to_numeric(
                axis_df[
                    rank_column
                ],
                errors="raise",
            )
            .astype(int)
        )

        axis_df[score_column] = (
            pd.to_numeric(
                axis_df[
                    score_column
                ],
                errors="raise",
            )
            .astype(float)
        )

        rank_one_df = (
            axis_df.loc[
                axis_df[
                    rank_column
                ].eq(1)
            ]
            .copy()
            .reset_index(drop=True)
        )

        if len(rank_one_df) != 1:
            rank_one_indices = (
                rank_one_df[
                    "split_index"
                ]
                .astype(int)
                .tolist()
            )

            raise ValueError(
                "Expected exactly one rank-1 QRF "
                f"split for axis={axis!r}, but "
                f"found {len(rank_one_df)}. "
                f"Split indices: "
                f"{rank_one_indices}"
            )

        row = rank_one_df.iloc[0]

        train_experiment_ids = (
            _normalize_stored_experiment_ids(
                row[
                    "train_experiment_ids"
                ]
            )
        )

        test_experiment_ids = (
            _normalize_stored_experiment_ids(
                row[
                    "test_experiment_ids"
                ]
            )
        )

        if not train_experiment_ids:
            raise ValueError(
                "Rank-1 split has no training "
                f"experiments for axis={axis!r}."
            )

        if not test_experiment_ids:
            raise ValueError(
                "Rank-1 split has no test "
                f"experiments for axis={axis!r}."
            )

        experiment_overlap = set(
            train_experiment_ids
        ).intersection(
            test_experiment_ids
        )

        if experiment_overlap:
            raise ValueError(
                "Experiment leakage was found in "
                f"the rank-1 {axis} split. "
                f"Overlapping experiment IDs: "
                f"{sorted(experiment_overlap)}"
            )

        selected_splits[axis] = {
            "axis": axis,
            "target_column": (
                axis_config[
                    "target_column"
                ]
            ),
            "split_index": int(
                row["split_index"]
            ),
            "split_name": str(
                row["split_name"]
            ),
            "qrf_rank": int(
                row[rank_column]
            ),
            "qrf_score": float(
                row[score_column]
            ),
            "train_exp": (
                train_experiment_ids
            ),
            "test_exp": (
                test_experiment_ids
            ),
        }

    if set(
        selected_splits
    ) != {
        "main",
        "secondary",
    }:
        raise RuntimeError(
            "Top-1 split selection did not return "
            "both main and secondary axes."
        )

    return selected_splits


# ============================================================
# Prepare source-specific split configuration
# ============================================================

def build_splits_by_source(
    geometry_sources: dict[str, Path],
    top_splits_by_axis: dict[str, dict],
) -> dict[str, dict[str, dict]]:
    """
    Reuse the independent Top-1 main and secondary splits for every
    geometry source.

    A new dictionary is created for each source and axis so later
    modifications cannot unintentionally affect another source.
    """
    required_axes = {
        "main",
        "secondary",
    }

    missing_axes = (
        required_axes.difference(
            top_splits_by_axis
        )
    )

    if missing_axes:
        raise KeyError(
            "Top split configuration is missing axes: "
            f"{sorted(missing_axes)}"
        )

    splits_by_source: dict[
        str,
        dict[str, dict],
    ] = {}

    for geometry_source in (
        geometry_sources
    ):
        splits_by_source[
            geometry_source
        ] = {
            "main": {
                key: (
                    list(value)
                    if isinstance(
                        value,
                        list,
                    )
                    else value
                )
                for key, value
                in top_splits_by_axis[
                    "main"
                ].items()
            },
            "secondary": {
                key: (
                    list(value)
                    if isinstance(
                        value,
                        list,
                    )
                    else value
                )
                for key, value
                in top_splits_by_axis[
                    "secondary"
                ].items()
            },
        }

    return splits_by_source


# ============================================================
# Load axis-specific tuned hyperparameters
# ============================================================

def load_tuned_qrf_params_by_axis(
    project_root: Path,
) -> dict[str, dict]:
    """
    Load the best tuned QRF parameters for main and secondary axes.
    """
    tuning_path = _resolve_project_path(
        project_root=project_root,
        path=STORED_QRF_TUNING_PATH,
    )

    if not tuning_path.exists():
        raise FileNotFoundError(
            "QRF hyperparameter tuning result was not found: "
            f"{tuning_path}. Run HyperParameterTuning first."
        )

    tuning_df = pd.read_parquet(
        tuning_path
    )

    required_columns = {
        "target_axis",
        "best_n_estimators",
        "best_max_depth",
        "best_min_samples_leaf",
        "best_min_samples_split",
        "best_max_features",
        "best_bootstrap",
        "best_lower_quantile",
        "best_upper_quantile",
    }

    _validate_required_columns(
        dataframe=tuning_df,
        required_columns=required_columns,
        dataframe_name=(
            "QRF hyperparameter tuning result"
        ),
    )

    tuned_params_by_axis: dict[
        str,
        dict,
    ] = {}

    for axis in (
        "main",
        "secondary",
    ):
        axis_df = tuning_df[
            tuning_df[
                "target_axis"
            ].astype(str).eq(axis)
        ].copy()

        if axis_df.empty:
            raise ValueError(
                "No tuned QRF parameters were found "
                f"for axis={axis!r} in {tuning_path}."
            )

        axis_df = axis_df.sort_values(
            "selection_score"
            if "selection_score" in axis_df.columns
            else "target_axis"
        )

        row = axis_df.iloc[0]

        tuned_params_by_axis[axis] = {
            "n_estimators": int(
                row["best_n_estimators"]
            ),
            "max_depth": (
                None
                if pd.isna(
                    row["best_max_depth"]
                )
                else int(row["best_max_depth"])
            ),
            "min_samples_leaf": int(
                row["best_min_samples_leaf"]
            ),
            "min_samples_split": int(
                row["best_min_samples_split"]
            ),
            "max_features": (
                float(row["best_max_features"])
                if isinstance(
                    row["best_max_features"],
                    (int, float, np.integer, np.floating),
                )
                and not isinstance(
                    row["best_max_features"],
                    bool,
                )
                else row["best_max_features"]
            ),
            "bootstrap": bool(
                row["best_bootstrap"]
            ),
            "lower_quantile": float(
                row["best_lower_quantile"]
            ),
            "upper_quantile": float(
                row["best_upper_quantile"]
            ),
        }

    return tuned_params_by_axis


def build_model_params_by_source(
    geometry_sources: dict[str, Path],
    tuned_params_by_axis: dict[str, dict],
) -> dict[str, dict[str, dict]]:
    """
    Reuse the tuned main/secondary parameters for every geometry source.
    """
    return {
        source_name: {
            axis: dict(
                tuned_params_by_axis[axis]
            )
            for axis in (
                "main",
                "secondary",
            )
        }
        for source_name in geometry_sources
    }


def build_default_model_params_by_source(
    geometry_sources: dict[str, Path],
) -> dict[str, dict[str, dict]]:
    """
    Use the original hard-coded dataset parameters for both axes.
    """
    missing_sources = set(
        geometry_sources
    ).difference(
        BEST_QRF_PARAMS_BY_DATASET
    )

    if missing_sources:
        raise KeyError(
            "Default QRF model parameters are missing for "
            f"geometry sources: {sorted(missing_sources)}"
        )

    return {
        source_name: {
            "main": dict(
                BEST_QRF_PARAMS_BY_DATASET[
                    source_name
                ]
            ),
            "secondary": dict(
                BEST_QRF_PARAMS_BY_DATASET[
                    source_name
                ]
            ),
        }
        for source_name in geometry_sources
    }


# ============================================================
# Model parameter validation
# ============================================================

def validate_model_parameters(
    geometry_sources: dict[str, Path],
    model_params_by_source: dict[
        str,
        dict[str, dict],
    ],
) -> None:
    """
    Validate that every geometry source has QRF model parameters.
    """
    missing_sources = set(
        geometry_sources
    ).difference(
        model_params_by_source
    )

    if missing_sources:
        raise KeyError(
            "QRF model parameters are missing for "
            f"geometry sources: "
            f"{sorted(missing_sources)}"
        )

    required_model_parameters = {
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "min_samples_split",
        "max_features",
        "bootstrap",
        "lower_quantile",
        "upper_quantile",
    }

    for (
        geometry_source,
        source_model_parameters,
    ) in model_params_by_source.items():
        missing_axes = {
            "main",
            "secondary",
        }.difference(
            source_model_parameters
        )

        if missing_axes:
            raise KeyError(
                "Axis-specific model parameters are "
                "missing for "
                f"geometry_source={geometry_source!r}: "
                f"{sorted(missing_axes)}"
            )

        for (
            target_axis,
            model_parameters,
        ) in source_model_parameters.items():
            missing_parameters = (
                required_model_parameters.difference(
                    model_parameters
                )
            )

            if missing_parameters:
                raise KeyError(
                    "Model parameters are missing for "
                    f"geometry_source="
                    f"{geometry_source!r}, "
                    f"target_axis={target_axis!r}: "
                    f"{sorted(missing_parameters)}"
                )


# ============================================================
# Main execution
# ============================================================

def main() -> None:
    project_root = Path(
        __file__
    ).resolve().parent

    logger.info(
        "Project root: %s",
        project_root,
    )

    geometry_sources = (
        qrf_training_geometry_sources(
            project_root=project_root,
        )
    )

    validate_geometry_sources(
        geometry_sources
    )

    logger.info(
        "Available geometry sources: %s",
        sorted(
            geometry_sources
        ),
    )

    top_splits_by_axis = (
        load_top_qrf_splits_by_axis(
            project_root=project_root,
        )
    )

    for (
        axis,
        split_config,
    ) in top_splits_by_axis.items():
        logger.info(
            "Selected independent Top-1 split | "
            "axis=%s | "
            "split_index=%s | "
            "rank=%s | "
            "score=%.6f | "
            "train_experiments=%s | "
            "test_experiments=%s | "
            "split=%s",
            axis,
            split_config[
                "split_index"
            ],
            split_config[
                "qrf_rank"
            ],
            split_config[
                "qrf_score"
            ],
            len(
                split_config[
                    "train_exp"
                ]
            ),
            len(
                split_config[
                    "test_exp"
                ]
            ),
            split_config[
                "split_name"
            ],
        )

    if (
        top_splits_by_axis[
            "main"
        ][
            "split_index"
        ]
        == top_splits_by_axis[
            "secondary"
        ][
            "split_index"
        ]
    ):
        logger.info(
            "Main and secondary independently selected "
            "the same rank-1 split_index=%s.",
            top_splits_by_axis[
                "main"
            ][
                "split_index"
            ],
        )
    else:
        logger.info(
            "Independent rank-1 splits selected | "
            "main split_index=%s | "
            "secondary split_index=%s",
            top_splits_by_axis[
                "main"
            ][
                "split_index"
            ],
            top_splits_by_axis[
                "secondary"
            ][
                "split_index"
            ],
        )

    selected_splits_by_source = (
        build_splits_by_source(
            geometry_sources=(
                geometry_sources
            ),
            top_splits_by_axis=(
                top_splits_by_axis
            ),
        )
    )

    if IS_BEST_CONFIG:
        tuned_params_by_axis = (
            load_tuned_qrf_params_by_axis(
                project_root=project_root,
            )
        )

        for (
            axis,
            tuned_params,
        ) in tuned_params_by_axis.items():
            logger.info(
                "Loaded tuned QRF params | "
                "axis=%s | params=%s",
                axis,
                tuned_params,
            )

        model_params_by_source = (
            build_model_params_by_source(
                geometry_sources=geometry_sources,
                tuned_params_by_axis=(
                    tuned_params_by_axis
                ),
            )
        )
    else:
        logger.info(
            "IS_BEST_CONFIG=False; using default "
            "hard-coded QRF params."
        )
        model_params_by_source = (
            build_default_model_params_by_source(
                geometry_sources=geometry_sources,
            )
        )

    validate_model_parameters(
        geometry_sources=(
            geometry_sources
        ),
        model_params_by_source=(
            model_params_by_source
        ),
    )

    model_root = (
        _resolve_project_path(
            project_root=project_root,
            path=STORED_QRF_MODEL_DIR,
        )
    )

    bending_setups_path = (
        _resolve_project_path(
            project_root=project_root,
            path=BENDING_SETUPS_PATH,
        )
    )

    if not bending_setups_path.exists():
        raise FileNotFoundError(
            "Bending setups file was not found: "
            f"{bending_setups_path}"
        )

    logger.info(
        "Starting final QRF model training."
    )

    logger.info(
        "Main axis uses split_index=%s "
        "for every geometry source.",
        top_splits_by_axis[
            "main"
        ][
            "split_index"
        ],
    )

    logger.info(
        "Secondary axis uses split_index=%s "
        "for every geometry source.",
        top_splits_by_axis[
            "secondary"
        ][
            "split_index"
        ],
    )

    results = run(
        project_root=project_root,
        geometry_sources=(
            geometry_sources
        ),
        splits_by_source=(
            selected_splits_by_source
        ),
        feature_columns=list(
            QRF_FEATURE_COLUMNS
        ),
        excluded_experiments=list(
            QRF_EXCLUDED_EXPERIMENTS
        ),
        model_params_by_source=(
            model_params_by_source
        ),
        model_root=model_root,
        bending_setups_path=(
            bending_setups_path
        ),
    )

    expected_sources = set(
        geometry_sources
    )

    actual_sources = set(
        results
    )

    if actual_sources != expected_sources:
        raise RuntimeError(
            "Final QRF results do not contain all "
            "configured geometry sources. "
            f"Expected: {sorted(expected_sources)}; "
            f"received: {sorted(actual_sources)}"
        )

    for (
        geometry_source,
        source_results,
    ) in results.items():
        expected_axes = {
            "main",
            "secondary",
        }

        actual_axes = set(
            source_results
        )

        if actual_axes != expected_axes:
            raise RuntimeError(
                "Final QRF results do not contain "
                "both axes for "
                f"geometry_source="
                f"{geometry_source!r}. "
                f"Received axes: "
                f"{sorted(actual_axes)}"
            )

        for (
            target_axis,
            result,
        ) in source_results.items():
            selected_split = (
                top_splits_by_axis[
                    target_axis
                ]
            )

            logger.info(
                "Stored final model | "
                "source=%s | "
                "axis=%s | "
                "split_index=%s | "
                "rank=%s | "
                "model=%s | "
                "metrics=%s",
                geometry_source,
                target_axis,
                selected_split[
                    "split_index"
                ],
                selected_split[
                    "qrf_rank"
                ],
                result[
                    "model_path"
                ],
                result[
                    "metrics"
                ],
            )

    logger.info(
        "Final QRF training completed successfully."
    )


if __name__ == "__main__":
    main()
