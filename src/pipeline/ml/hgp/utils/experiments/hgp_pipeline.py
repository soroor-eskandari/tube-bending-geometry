from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.pipeline.ml.common.geometry_data_preprocessor import (
    load_bending_setups,
    load_selected_geometry_source,
)
from src.pipeline.ml.hgp.utils.experiments.hgp_evaluator import (
    evaluate_predictions,
)
from src.pipeline.ml.hgp.utils.experiments.hgp_model_trainer import (
    HGPModelBundle,
    HGPPredictions,
    train_and_predict,
)


HGP_TARGET_COLUMNS_BY_AXIS = {
    "main": "Main-axis [mm]",
    "secondary": "Secondary-axis [mm]",
}


def _resolve_path(
    project_root: Path,
    value: str | Path,
) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = project_root / path

    return path.resolve()


def _split_geometry_by_experiment_ids(
    geometry_df: pd.DataFrame,
    train_experiment_ids: list[int],
    test_experiment_ids: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create train/test dataframes using explicit Experiment_ID values.
    """
    if "Experiment_ID" not in geometry_df.columns:
        raise KeyError(
            "Geometry data must contain 'Experiment_ID'."
        )

    train_ids = {
        int(value)
        for value in train_experiment_ids
    }
    test_ids = {
        int(value)
        for value in test_experiment_ids
    }

    if not train_ids:
        raise ValueError(
            "train_experiment_ids cannot be empty."
        )

    if not test_ids:
        raise ValueError(
            "test_experiment_ids cannot be empty."
        )

    overlapping_ids = train_ids.intersection(
        test_ids
    )

    if overlapping_ids:
        raise ValueError(
            "Train/test experiment leakage detected. "
            f"Overlapping Experiment_ID values: "
            f"{sorted(overlapping_ids)}"
        )

    geometry_df = geometry_df.copy()

    geometry_df["Experiment_ID"] = pd.to_numeric(
        geometry_df["Experiment_ID"],
        errors="raise",
    ).astype(int)

    train_df = geometry_df[
        geometry_df["Experiment_ID"].isin(
            train_ids
        )
    ].copy()

    test_df = geometry_df[
        geometry_df["Experiment_ID"].isin(
            test_ids
        )
    ].copy()

    if train_df.empty:
        raise ValueError(
            "Training geometry is empty."
        )

    if test_df.empty:
        raise ValueError(
            "Test geometry is empty."
        )

    available_train_ids = set(
        train_df["Experiment_ID"].unique()
    )
    available_test_ids = set(
        test_df["Experiment_ID"].unique()
    )

    missing_train_ids = (
        train_ids - available_train_ids
    )
    missing_test_ids = (
        test_ids - available_test_ids
    )

    if missing_train_ids:
        raise ValueError(
            "Training experiments absent from the selected "
            f"geometry source: {sorted(missing_train_ids)}"
        )

    if missing_test_ids:
        raise ValueError(
            "Test experiments absent from the selected "
            f"geometry source: {sorted(missing_test_ids)}"
        )

    return (
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def _resolve_axis_model_config(
    model_params_by_source: dict[str, dict],
    geometry_source: str,
    target_axis: str,
) -> dict[str, Any]:
    if geometry_source not in model_params_by_source:
        raise KeyError(
            "No model parameters were supplied for "
            f"geometry_source={geometry_source!r}."
        )

    source_config = model_params_by_source[
        geometry_source
    ]

    return dict(
        source_config.get(
            target_axis,
            source_config,
        )
    )


def _model_directory(
    model_root: Path,
    geometry_source: str,
    target_axis: str,
) -> Path:
    """
    Return the stable model directory.

    The directory and file name do not contain timestamps, split indices,
    ranks, or run IDs. Every new run overwrites the existing model file.
    """
    return (
        model_root
        / geometry_source
        / target_axis
    )


def _model_path(
    model_root: Path,
    geometry_source: str,
    target_axis: str,
) -> Path:
    return (
        _model_directory(
            model_root=model_root,
            geometry_source=geometry_source,
            target_axis=target_axis,
        )
        / "hgp_model.joblib"
    )


def train_single_model(
    *,
    geometry_df: pd.DataFrame,
    geometry_source: str,
    target_axis: str,
    split_config: dict,
    feature_columns: list[str],
    model_config: dict[str, Any],
    model_root: Path,
) -> dict[str, Any]:
    """
    Train and overwrite one HGP model.

    Exactly one model file is kept for every:
        geometry_source × target_axis

    Existing model files are overwritten by joblib.dump.
    No timestamped run directory, metrics file, metadata file,
    or prediction file is stored.
    """
    if target_axis not in HGP_TARGET_COLUMNS_BY_AXIS:
        raise ValueError(
            "target_axis must be one of "
            f"{sorted(HGP_TARGET_COLUMNS_BY_AXIS)}. "
            f"Received {target_axis!r}."
        )

    required_split_keys = {
        "train_exp",
        "test_exp",
    }

    missing_split_keys = (
        required_split_keys.difference(
            split_config
        )
    )

    if missing_split_keys:
        raise KeyError(
            "Split configuration is missing keys: "
            f"{sorted(missing_split_keys)}"
        )

    target_column = (
        HGP_TARGET_COLUMNS_BY_AXIS[
            target_axis
        ]
    )

    required_columns = {
        "Experiment_ID",
        target_column,
        *feature_columns,
    }

    missing_columns = required_columns.difference(
        geometry_df.columns
    )

    if missing_columns:
        raise KeyError(
            "Geometry data is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    train_df, test_df = (
        _split_geometry_by_experiment_ids(
            geometry_df=geometry_df,
            train_experiment_ids=list(
                split_config["train_exp"]
            ),
            test_experiment_ids=list(
                split_config["test_exp"]
            ),
        )
    )

    model_bundle, predictions = train_and_predict(
        train_df=train_df,
        test_df=test_df,
        feature_columns=feature_columns,
        target_column=target_column,
        model_config=model_config,
        group_column=str(
            model_config.get(
                "group_column",
                "Experiment_ID",
            )
        ),
    )

    y_true = pd.to_numeric(
        test_df[target_column],
        errors="raise",
    ).to_numpy(dtype=float)

    confidence_level = float(
        model_config.get(
            "confidence_level",
            0.90,
        )
    )

    metrics = evaluate_predictions(
        y_true=y_true,
        y_lower=predictions.lower,
        y_mean=predictions.mean,
        y_upper=predictions.upper,
        y_total_std=predictions.total_std,
        confidence_level=confidence_level,
    )

    model_directory = _model_directory(
        model_root=model_root,
        geometry_source=geometry_source,
        target_axis=target_axis,
    )
    model_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = _model_path(
        model_root=model_root,
        geometry_source=geometry_source,
        target_axis=target_axis,
    )

    # joblib.dump overwrites the existing file at the same path.
    joblib.dump(
        model_bundle,
        model_path,
    )

    print(
        "Stored HGP model | "
        f"source={geometry_source} | "
        f"axis={target_axis} | "
        f"coverage={metrics['coverage_percent']:.2f}% | "
        f"rmse={metrics['rmse']:.4f} | "
        f"overwritten_path={model_path}"
    )

    return {
        "geometry_source": geometry_source,
        "target_axis": target_axis,
        "target_column": target_column,
        "model_path": model_path,
        "model": model_bundle,
        "predictions": predictions,
        "metrics": metrics,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
    }


def run(
    *,
    project_root: str | Path,
    geometry_sources: dict[
        str,
        str | Path,
    ],
    splits_by_source: dict[
        str,
        dict[str, dict],
    ],
    feature_columns: list[str],
    excluded_experiments: list[int],
    model_params_by_source: dict[
        str,
        dict,
    ],
    model_root: str | Path,
    bending_setups_path: str | Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Train final HGP models for all sources and both axes.

    Stored files:
        <model_root>/<geometry_source>/main/hgp_model.joblib
        <model_root>/<geometry_source>/secondary/hgp_model.joblib

    Running this function again overwrites those same files.
    """
    project_root = Path(
        project_root
    ).resolve()

    model_root = _resolve_path(
        project_root,
        model_root,
    )

    bending_setups_path = _resolve_path(
        project_root,
        bending_setups_path,
    )

    if not bending_setups_path.exists():
        raise FileNotFoundError(
            "Bending setups file was not found: "
            f"{bending_setups_path}"
        )

    bending_setups_df = load_bending_setups(
        path=bending_setups_path,
        excluded_experiments=(
            excluded_experiments
        ),
    )

    results: dict[
        str,
        dict[str, dict[str, Any]],
    ] = {}

    for (
        geometry_source,
        geometry_path_value,
    ) in geometry_sources.items():
        geometry_path = _resolve_path(
            project_root,
            geometry_path_value,
        )

        if not geometry_path.exists():
            raise FileNotFoundError(
                f"Geometry source {geometry_source!r} "
                f"was not found: {geometry_path}"
            )

        if geometry_source not in splits_by_source:
            raise KeyError(
                "No axis-specific splits were supplied for "
                f"geometry_source={geometry_source!r}."
            )

        source_splits = splits_by_source[
            geometry_source
        ]

        missing_axes = {
            "main",
            "secondary",
        }.difference(
            source_splits
        )

        if missing_axes:
            raise KeyError(
                "Source split configuration is missing axes: "
                f"{sorted(missing_axes)} for "
                f"geometry_source={geometry_source!r}."
            )

        geometry_df, loaded_source, _ = (
            load_selected_geometry_source(
                project_root=project_root,
                paths_config={
                    "geometry_sources": {
                        geometry_source: str(
                            geometry_path
                        )
                    }
                },
                data_config={
                    "geometry_source": geometry_source,
                    "excluded_experiments": list(
                        excluded_experiments
                    ),
                },
                bending_setups_df=(
                    bending_setups_df
                ),
            )
        )

        source_results: dict[
            str,
            dict[str, Any],
        ] = {}

        for target_axis in (
            "main",
            "secondary",
        ):
            model_config = (
                _resolve_axis_model_config(
                    model_params_by_source=(
                        model_params_by_source
                    ),
                    geometry_source=(
                        geometry_source
                    ),
                    target_axis=target_axis,
                )
            )

            source_results[target_axis] = (
                train_single_model(
                    geometry_df=geometry_df,
                    geometry_source=loaded_source,
                    target_axis=target_axis,
                    split_config=source_splits[
                        target_axis
                    ],
                    feature_columns=list(
                        feature_columns
                    ),
                    model_config=model_config,
                    model_root=model_root,
                )
            )

        results[geometry_source] = (
            source_results
        )

    return results


def load_stored_model(
    *,
    project_root: str | Path,
    model_root: str | Path,
    geometry_source: str,
    target_axis: str,
) -> HGPModelBundle:
    """Load the single stored model for one source and axis."""
    if target_axis not in HGP_TARGET_COLUMNS_BY_AXIS:
        raise ValueError(
            "target_axis must be one of "
            f"{sorted(HGP_TARGET_COLUMNS_BY_AXIS)}."
        )

    project_root = Path(
        project_root
    ).resolve()
    model_root = _resolve_path(
        project_root,
        model_root,
    )

    model_path = _model_path(
        model_root=model_root,
        geometry_source=geometry_source,
        target_axis=target_axis,
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Stored HGP model was not found: {model_path}"
        )

    return joblib.load(
        model_path
    )
