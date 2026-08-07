from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.pipeline.ml.hgp.utils.experiments.hgp_pipeline import (
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
# HGP configuration
# ============================================================

HGP_FEATURE_COLUMNS = [
    "Angle[degree]ORDistance[mm]",
    "Collet boost",
    "Pressure-die distance",
    "Mandrel retraction timing",
    "Pressure-die boost",
    "Clamp-die lateral position",
    "Mandrel position",
]


HGP_EXCLUDED_EXPERIMENTS = [
    1,
    48,
    166,
]


IS_BEST_CONFIG = True


# This runner uses the already-ranked Top-1 main and secondary splits
# stored in common/data/various_splits.parquet.
#
# Using the same splits as QRF gives a direct model comparison.
STORED_SPLIT_PATH = Path(
    "src"
    "/pipeline"
    "/ml"
    "/common"
    "/data"
    "/various_splits.parquet"
)


STORED_HGP_MODEL_DIR = Path(
    "src"
    "/pipeline"
    "/ml"
    "/hgp"
    "/results"
    "/models"
)


STORED_HGP_TUNING_PATH = Path(
    "src"
    "/pipeline"
    "/ml"
    "/hgp"
    "/data"
    "/hgp_hyperparameter_tuning_results.parquet"
)


BENDING_SETUPS_PATH = Path(
    "data"
    "/rf_augmented"
    "/ui_data"
    "/unique_bending_setups.csv"
)


# One configuration is supplied for each geometry source.
# Each source may have separate main and secondary configurations.
#
# Start with n_restarts_optimizer=0 for a fast pipeline check.
# After the pipeline runs correctly, increase it to 3 or 5.
DEFAULT_HGP_PARAMS = {
    "mean_kernel": "matern_2.5",
    "noise_kernel": "rbf",
    "mean_kernel_params": {
        "initial_length_scale": 1.0,
        "length_scale_bounds": (
            1e-2,
            1e2,
        ),
        "constant_value": 1.0,
        "constant_value_bounds": (
            1e-3,
            1e3,
        ),
    },
    "noise_kernel_params": {
        "initial_length_scale": 1.0,
        "length_scale_bounds": (
            1e-2,
            1e2,
        ),
        "constant_value": 1.0,
        "constant_value_bounds": (
            1e-3,
            1e3,
        ),
    },
    "noise_gp_alpha": 1e-4,
    "residual_variance_epsilon": 1e-8,
    "noise_variance_floor": 1e-6,
    "noise_variance_ceiling": 10.0,
    "confidence_level": 0.90,
    "n_restarts_optimizer": 0,
    "random_state": 1100,
    "group_column": "Group_ID",
}


BEST_HGP_PARAMS_BY_DATASET = {
    "real": {
        "main": {
            **DEFAULT_HGP_PARAMS,
        },
        "secondary": {
            **DEFAULT_HGP_PARAMS,
        },
    },
    (
        "sensor_augmented_noise__"
        "time_wrapping__scaling__jittering"
    ): {
        "main": {
            **DEFAULT_HGP_PARAMS,
        },
        "secondary": {
            **DEFAULT_HGP_PARAMS,
        },
    },
    "within_group_interpolation_raw": {
        "main": {
            **DEFAULT_HGP_PARAMS,
        },
        "secondary": {
            **DEFAULT_HGP_PARAMS,
        },
    },
}


# ============================================================
# Generic helpers
# ============================================================

def _resolve_project_path(
    project_root: Path,
    path: str | Path,
) -> Path:
    """Resolve a project-relative or absolute path."""
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

    Supported values:
      - list, tuple or set;
      - NumPy array;
      - Pandas Series;
      - string representation of a list;
      - scalar experiment ID.
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

    return sorted(
        {
            int(experiment_id)
            for experiment_id in value
            if pd.notna(experiment_id)
        }
    )


def _validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    dataframe_name: str,
) -> None:
    """Validate required DataFrame columns."""
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


def _deep_copy_configuration(
    value: Any,
) -> Any:
    """
    Copy nested model configuration without sharing dictionaries or lists.
    """
    if isinstance(value, dict):
        return {
            key: _deep_copy_configuration(
                item
            )
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _deep_copy_configuration(
                item
            )
            for item in value
        ]

    if isinstance(value, tuple):
        return tuple(
            _deep_copy_configuration(
                item
            )
            for item in value
        )

    return value


# ============================================================
# Geometry sources
# ============================================================

def hgp_training_geometry_sources(
    project_root: Path,
) -> dict[str, Path]:
    """Return all geometry sources used for final HGP training."""
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
            / (
                "final_geometry_sensor_augmented_noise__"
                "time_wrapping__scaling__jittering.parquet"
            )
        ),
        "within_group_interpolation_raw": (
            project_root
            / "data"
            / "rf_augmented"
            / (
                "final_geometry_within_group_"
                "interpolation_raw.parquet"
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
    """Validate that all configured geometry source files exist."""
    if not geometry_sources:
        raise ValueError(
            "No HGP geometry sources were configured."
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
            "Some HGP geometry sources were not found:\n"
            f"{formatted_missing_sources}"
        )


# ============================================================
# Load independent Top-1 splits
# ============================================================

def load_top_splits_by_axis(
    project_root: Path,
) -> dict[str, dict]:
    """
    Load the independent Top-1 split for main and secondary axes.

    This runner intentionally uses qrf_rank_main and
    qrf_rank_secondary from the shared split table. This keeps
    train/test data identical between QRF and HGP.

    If HGP-specific ranking columns are added later, replace:
        qrf_rank_main       -> hgp_rank_main
        qrf_score_main      -> hgp_score_main
        qrf_rank_secondary  -> hgp_rank_secondary
        qrf_score_secondary -> hgp_score_secondary
    """
    split_path = _resolve_project_path(
        project_root=project_root,
        path=STORED_SPLIT_PATH,
    )

    if not split_path.exists():
        raise FileNotFoundError(
            "Stored split metadata was not found: "
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
            "Stored split metadata is empty: "
            f"{split_path}"
        )

    _validate_required_columns(
        dataframe=split_df,
        required_columns=required_columns,
        dataframe_name=(
            "Stored split metadata"
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
            "one row per split_index. "
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
                "No ranked splits are available "
                f"for axis={axis!r}."
            )

        axis_df[rank_column] = (
            pd.to_numeric(
                axis_df[rank_column],
                errors="raise",
            )
            .astype(int)
        )
        axis_df[score_column] = (
            pd.to_numeric(
                axis_df[score_column],
                errors="raise",
            )
            .astype(float)
        )

        selected_df = (
            axis_df.loc[
                axis_df[
                    rank_column
                ].eq(1)
            ]
            .copy()
            .reset_index(drop=True)
        )

        if len(selected_df) != 1:
            rank_one_indices = (
                selected_df[
                    "split_index"
                ]
                .astype(int)
                .tolist()
            )

            raise ValueError(
                "Expected exactly one rank-1 split "
                f"for axis={axis!r}, but found "
                f"{len(selected_df)}. "
                f"Split indices: {rank_one_indices}"
            )

        row = selected_df.iloc[0]

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
                f"Overlapping IDs: "
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
            # hgp_pipeline accepts these fields but does not
            # require rank or score for model storage.
            "hgp_rank": int(
                row[rank_column]
            ),
            "source_rank_column": (
                rank_column
            ),
            "source_score": float(
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
            "both axes."
        )

    return selected_splits


# ============================================================
# Source-specific split configuration
# ============================================================

def build_splits_by_source(
    geometry_sources: dict[str, Path],
    top_splits_by_axis: dict[str, dict],
) -> dict[str, dict[str, dict]]:
    """
    Reuse the independent Top-1 main and secondary splits for
    every geometry source.
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

    for geometry_source in geometry_sources:
        splits_by_source[
            geometry_source
        ] = {
            "main": (
                _deep_copy_configuration(
                    top_splits_by_axis[
                        "main"
                    ]
                )
            ),
            "secondary": (
                _deep_copy_configuration(
                    top_splits_by_axis[
                        "secondary"
                    ]
                )
            ),
        }

    return splits_by_source


# ============================================================
# Model configuration
# ============================================================

def build_model_params_by_source(
    geometry_sources: dict[str, Path],
) -> dict[str, dict[str, dict]]:
    """
    Build source-specific and axis-specific HGP parameters.
    """
    missing_sources = set(
        geometry_sources
    ).difference(
        BEST_HGP_PARAMS_BY_DATASET
    )

    if missing_sources:
        raise KeyError(
            "HGP model parameters are missing for "
            f"geometry sources: "
            f"{sorted(missing_sources)}"
        )

    return {
        source_name: {
            "main": (
                _deep_copy_configuration(
                    BEST_HGP_PARAMS_BY_DATASET[
                        source_name
                    ][
                        "main"
                    ]
                )
            ),
            "secondary": (
                _deep_copy_configuration(
                    BEST_HGP_PARAMS_BY_DATASET[
                        source_name
                    ][
                        "secondary"
                    ]
                )
            ),
        }
        for source_name in geometry_sources
    }


def _normalize_config_value(value: Any) -> Any:
    """
    Convert pandas/numpy values from parquet to JSON-like Python values.
    """
    if isinstance(value, np.ndarray):
        return [
            _normalize_config_value(item)
            for item in value.tolist()
        ]

    if isinstance(value, dict):
        return {
            key: _normalize_config_value(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _normalize_config_value(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return tuple(
            _normalize_config_value(item)
            for item in value
        )

    if isinstance(value, np.generic):
        return value.item()

    return value


def _merge_hgp_config(
    base_config: dict,
    tuned_config: dict,
) -> dict:
    merged = _deep_copy_configuration(
        base_config
    )

    for key, value in tuned_config.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            nested = dict(merged[key])
            nested.update(
                _normalize_config_value(value)
            )
            merged[key] = nested
        else:
            merged[key] = _normalize_config_value(
                value
            )

    # The current trainer uses Group_ID for aggregation, not as a GP input.
    merged["group_column"] = "Group_ID"
    merged["optimizer"] = None

    return merged


def load_tuned_hgp_params_by_axis(
    project_root: Path,
    selected_splits_by_axis: dict[str, dict],
) -> dict[str, dict] | None:
    """
    Load tuned HGP parameters for the selected best-ranked splits.
    """
    tuning_path = _resolve_project_path(
        project_root=project_root,
        path=STORED_HGP_TUNING_PATH,
    )

    if not tuning_path.exists():
        logger.warning(
            "HGP hyperparameter tuning result was not found: %s. "
            "Falling back to hard-coded HGP parameters.",
            tuning_path,
        )
        return None

    tuning_df = pd.read_parquet(
        tuning_path
    )

    required_columns = {
        "target_axis",
        "split_index",
        "split_name",
        "best_model_config",
    }

    _validate_required_columns(
        dataframe=tuning_df,
        required_columns=required_columns,
        dataframe_name=(
            "HGP hyperparameter tuning result"
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
        selected_split = (
            selected_splits_by_axis[axis]
        )

        axis_df = tuning_df[
            tuning_df["target_axis"]
            .astype(str)
            .eq(axis)
        ].copy()

        axis_df = axis_df[
            pd.to_numeric(
                axis_df["split_index"],
                errors="raise",
            )
            .astype(int)
            .eq(
                int(
                    selected_split[
                        "split_index"
                    ]
                )
            )
        ].copy()

        axis_df = axis_df[
            axis_df["split_name"].astype(str).eq(
                str(
                    selected_split[
                        "split_name"
                    ]
                )
            )
        ].copy()

        if axis_df.empty:
            logger.warning(
                "No tuned HGP parameters match the selected "
                "best-rank split for axis=%r "
                "(split_index=%s, split_name=%r). "
                "Falling back to hard-coded HGP parameters.",
                axis,
                selected_split["split_index"],
                selected_split["split_name"],
            )
            return None

        if "selection_score" in axis_df.columns:
            axis_df = axis_df.sort_values(
                "selection_score"
            )
        elif "test_curve_distance_norm" in axis_df.columns:
            axis_df = axis_df.sort_values(
                "test_curve_distance_norm"
            )

        row = axis_df.iloc[0]
        best_model_config = row[
            "best_model_config"
        ]

        if not isinstance(best_model_config, dict):
            if isinstance(best_model_config, str):
                best_model_config = ast.literal_eval(
                    best_model_config
                )
            else:
                raise TypeError(
                    "best_model_config must be stored as "
                    f"a dict or string, got "
                    f"{type(best_model_config).__name__}."
                )

        tuned_params_by_axis[axis] = _merge_hgp_config(
            base_config=DEFAULT_HGP_PARAMS,
            tuned_config=best_model_config,
        )

    return tuned_params_by_axis


def build_model_params_by_source_from_axis_params(
    geometry_sources: dict[str, Path],
    tuned_params_by_axis: dict[str, dict],
) -> dict[str, dict[str, dict]]:
    return {
        source_name: {
            "main": _deep_copy_configuration(
                tuned_params_by_axis["main"]
            ),
            "secondary": _deep_copy_configuration(
                tuned_params_by_axis["secondary"]
            ),
        }
        for source_name in geometry_sources
    }


def validate_model_parameters(
    geometry_sources: dict[str, Path],
    model_params_by_source: dict[
        str,
        dict[str, dict],
    ],
) -> None:
    """
    Validate every source and axis HGP configuration.
    """
    missing_sources = set(
        geometry_sources
    ).difference(
        model_params_by_source
    )

    if missing_sources:
        raise KeyError(
            "HGP model parameters are missing for "
            f"geometry sources: "
            f"{sorted(missing_sources)}"
        )

    required_model_parameters = {
        "mean_kernel",
        "noise_kernel",
        "noise_gp_alpha",
        "noise_variance_floor",
        "noise_variance_ceiling",
        "confidence_level",
        "n_restarts_optimizer",
        "random_state",
        "group_column",
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
                "Axis-specific parameters are missing "
                f"for geometry_source="
                f"{geometry_source!r}: "
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
                    "HGP parameters are missing for "
                    f"geometry_source="
                    f"{geometry_source!r}, "
                    f"target_axis={target_axis!r}: "
                    f"{sorted(missing_parameters)}"
                )

            confidence_level = float(
                model_parameters[
                    "confidence_level"
                ]
            )

            if not 0.0 < confidence_level < 1.0:
                raise ValueError(
                    "confidence_level must be between "
                    "0 and 1 for "
                    f"source={geometry_source!r}, "
                    f"axis={target_axis!r}."
                )

            noise_variance_floor = float(
                model_parameters[
                    "noise_variance_floor"
                ]
            )
            noise_variance_ceiling = float(
                model_parameters[
                    "noise_variance_ceiling"
                ]
            )

            if noise_variance_floor <= 0.0:
                raise ValueError(
                    "noise_variance_floor must be positive."
                )

            if (
                noise_variance_ceiling
                <= noise_variance_floor
            ):
                raise ValueError(
                    "noise_variance_ceiling must be "
                    "larger than noise_variance_floor."
                )


# ============================================================
# Main execution
# ============================================================

def main() -> None:
    # The requested file location is:
    # /Users/soroureskandari/Master Thesis /
    # tube-bending-geometry/3_2_run_hgp_interval_predictor.py
    #
    # Therefore, the file's parent is the project root.
    project_root = Path(
        __file__
    ).resolve().parent

    logger.info(
        "Project root: %s",
        project_root,
    )

    geometry_sources = (
        hgp_training_geometry_sources(
            project_root=project_root,
        )
    )

    validate_geometry_sources(
        geometry_sources
    )

    logger.info(
        "Available HGP geometry sources: %s",
        sorted(
            geometry_sources
        ),
    )

    top_splits_by_axis = (
        load_top_splits_by_axis(
            project_root=project_root,
        )
    )

    for (
        axis,
        split_config,
    ) in top_splits_by_axis.items():
        logger.info(
            "Selected Top-1 split | "
            "axis=%s | "
            "split_index=%s | "
            "source_rank_column=%s | "
            "source_score=%.6f | "
            "train_experiments=%s | "
            "test_experiments=%s | "
            "split=%s",
            axis,
            split_config[
                "split_index"
            ],
            split_config[
                "source_rank_column"
            ],
            split_config[
                "source_score"
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
            "Main and secondary use the same "
            "split_index=%s.",
            top_splits_by_axis[
                "main"
            ][
                "split_index"
            ],
        )
    else:
        logger.info(
            "Independent splits selected | "
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
            load_tuned_hgp_params_by_axis(
                project_root=project_root,
                selected_splits_by_axis=(
                    top_splits_by_axis
                ),
            )
        )

        if tuned_params_by_axis is None:
            model_params_by_source = (
                build_model_params_by_source(
                    geometry_sources=(
                        geometry_sources
                    ),
                )
            )
        else:
            for (
                axis,
                tuned_params,
            ) in tuned_params_by_axis.items():
                logger.info(
                    "Loaded tuned HGP params | "
                    "axis=%s | params=%s",
                    axis,
                    tuned_params,
                )

            model_params_by_source = (
                build_model_params_by_source_from_axis_params(
                    geometry_sources=(
                        geometry_sources
                    ),
                    tuned_params_by_axis=(
                        tuned_params_by_axis
                    ),
                )
            )
    else:
        logger.info(
            "IS_BEST_CONFIG=False; using hard-coded "
            "HGP params."
        )
        model_params_by_source = (
            build_model_params_by_source(
                geometry_sources=(
                    geometry_sources
                ),
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

    for (
        geometry_source,
        source_params,
    ) in model_params_by_source.items():
        for (
            axis,
            axis_params,
        ) in source_params.items():
            logger.info(
                "HGP configuration | "
                "source=%s | "
                "axis=%s | "
                "mean_kernel=%s | "
                "noise_kernel=%s | "
                "confidence=%.3f | "
                "optimizer=None | "
                "training_mode=aggregated_group_variance",
                geometry_source,
                axis,
                axis_params[
                    "mean_kernel"
                ],
                axis_params[
                    "noise_kernel"
                ],
                axis_params[
                    "confidence_level"
                ],
            )

    model_root = _resolve_project_path(
        project_root=project_root,
        path=STORED_HGP_MODEL_DIR,
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
        "Starting final HGP model training."
    )

    logger.info(
        "Models are stored with stable names and "
        "existing files are overwritten."
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
            HGP_FEATURE_COLUMNS
        ),
        excluded_experiments=list(
            HGP_EXCLUDED_EXPERIMENTS
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
            "Final HGP results do not contain all "
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
                "Final HGP results do not contain "
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
                "Stored final HGP model | "
                "source=%s | "
                "axis=%s | "
                "split_index=%s | "
                "model=%s | "
                "metrics=%s",
                geometry_source,
                target_axis,
                selected_split[
                    "split_index"
                ],
                result[
                    "model_path"
                ],
                result[
                    "metrics"
                ],
            )

    logger.info(
        "Final HGP training completed successfully."
    )


if __name__ == "__main__":
    main()
