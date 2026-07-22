from __future__ import annotations

import shutil
from datetime import datetime
from itertools import product
from pathlib import Path

import pandas as pd
import yaml
import json
import joblib

from src.pipeline.ml.qrf.mode.experiments.data_splittor import (
    load_split_metadata,
    make_train_test_split,
)
from src.pipeline.ml.qrf.mode.experiments.geometry_data_preprocessor import (
    load_bending_setups,
    load_selected_geometry_source,
)
from src.pipeline.ml.qrf.mode.experiments.qrf_evaluator import (
    evaluate_predictions,
)
from src.pipeline.ml.qrf.mode.experiments.qrf_model_trainer import (
    train_and_predict,
)


QRF_TARGET_COLUMNS_BY_AXIS = {
    "main": "Main-axis [mm]",
    "secondary": "Secondary-axis [mm]",
}


def _resolve_target_axes(
    target_axes: str | list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    """
    Resolve target axes while preserving the existing behavior.

    target_axes=None:
        Train all available axes.

    target_axes="main":
        Train only the main axis.

    target_axes=("main", "secondary"):
        Train both axes.
    """
    if target_axes is None:
        return tuple(
            QRF_TARGET_COLUMNS_BY_AXIS.keys()
        )

    if isinstance(target_axes, str):
        target_axes = (target_axes,)
    else:
        target_axes = tuple(target_axes)

    unknown_axes = set(target_axes).difference(
        QRF_TARGET_COLUMNS_BY_AXIS
    )

    if unknown_axes:
        raise ValueError(
            "Unknown QRF target axes: "
            f"{sorted(unknown_axes)}. "
            "Available axes are: "
            f"{sorted(QRF_TARGET_COLUMNS_BY_AXIS)}"
        )

    # Remove duplicates while preserving order.
    return tuple(
        dict.fromkeys(target_axes)
    )

def load_config(
    config_path: str | Path,
    project_root: str | Path,
) -> dict:
    config_path = Path(config_path)
    project_root = Path(project_root).resolve()

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config["_project_root"] = project_root
    config["_config_path"] = config_path.resolve()

    return config


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _split_name(split_row: pd.Series, split_index: int) -> str:
    for column in ("split_name", "Split name"):
        value = split_row.get(column)
        if pd.notna(value):
            return str(value)

    return f"split_{split_index}"


def run_experiments(
    config: dict,
    target_axes: (
        str
        | list[str]
        | tuple[str, ...]
        | None
    ) = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Run split × tree-count × axis experiments."""
    project_root = Path(config["_project_root"])
    paths = config["paths"]

    # Loaded for validation and reproducibility.
    bending_setups_df = load_bending_setups(
        path=_resolve_path(
            project_root,
            paths["bending_setups"],
        ),
        excluded_experiments=config["data"].get(
            "excluded_experiments",
            [],
        ),
    )

    geometry_df, geometry_source, geometry_path = (
        load_selected_geometry_source(
            project_root=project_root,
            paths_config=paths,
            data_config=config["data"],
            bending_setups_df=bending_setups_df,
        )
    )

    split_metadata_df = load_split_metadata(
        _resolve_path(
            project_root,
            paths["split_metadata"],
        )
    )

    feature_columns = list(
        config["features"]["input_columns"]
    )
    target_columns = dict(
        config["features"]["target_columns"]
    )

    configured_axes = tuple(
        config["search"]["axes"]
    )

    if target_axes is None:
        selected_axes = configured_axes
    else:
        selected_axes = _resolve_target_axes(
            target_axes
        )

        unavailable_axes = set(
            selected_axes
        ).difference(configured_axes)

        if unavailable_axes:
            raise ValueError(
                "Requested target axes are not enabled "
                "in config['search']['axes']: "
                f"{sorted(unavailable_axes)}. "
                f"Configured axes: {list(configured_axes)}"
            )

    missing_target_columns = set(
        selected_axes
    ).difference(target_columns)

    if missing_target_columns:
        raise KeyError(
            "No target columns are configured for axes: "
            f"{sorted(missing_target_columns)}"
        )

    metrics_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    combinations = product(
        config["search"]["split_indices"],
        config["search"]["n_estimators"],
        selected_axes,
    )

    for split_index, n_estimators, axis in combinations:
        split_index = int(split_index)
        n_estimators = int(n_estimators)

        if axis not in target_columns:
            raise KeyError(
                f"No target column configured for axis='{axis}'."
            )

        train_df, test_df, split_row = make_train_test_split(
            geometry_df=geometry_df,
            split_metadata_df=split_metadata_df,
            split_index=split_index,
        )

        target_column = target_columns[axis]

        _, predictions = train_and_predict(
            train_df=train_df,
            test_df=test_df,
            feature_columns=feature_columns,
            target_column=target_column,
            n_estimators=n_estimators,
            model_config=config["model"],
        )

        y_true = test_df[target_column].to_numpy()

        metrics = evaluate_predictions(
            y_true=y_true,
            y_lower=predictions.lower,
            y_median=predictions.median,
            y_upper=predictions.upper,
        )

        run_id = (
            f"split-{split_index:03d}"
            f"__trees-{n_estimators:04d}"
            f"__axis-{axis}"
        )

        split_name = str(
            split_row.get(
                "split_name",
                f"split_{split_index}",
            )
        )

        summary_df_index = split_row.get(
            "summary_df_index"
        )

        split_rank = (
            int(summary_df_index) + 1
            if pd.notna(summary_df_index)
            else None
        )

        metrics_rows.append(
            {
                "experiment_name": config["experiment_name"],
                "run_id": run_id,
                "split_index": split_index,
                "split_name": split_name,
                "split_rank": split_rank,
                "axis": axis,
                "n_estimators": n_estimators,
                "target_column": target_column,
                "lower_quantile": float(
                    config["model"]["lower_quantile"]
                ),
                "upper_quantile": float(
                    config["model"]["upper_quantile"]
                ),
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "train_groups": train_df["Group_ID"].nunique(),
                "test_groups": test_df["Group_ID"].nunique(),
                "geometry_source": geometry_source,
                "geometry_path": str(geometry_path),
                **metrics,
            }
        )

        if config.get("output", {}).get(
            "save_predictions",
            True,
        ):
            prediction_df = test_df[
                feature_columns
            ].copy()

            prediction_df["experiment_name"] = (
                config["experiment_name"]
            )
            prediction_df["run_id"] = run_id
            prediction_df["split_index"] = split_index
            prediction_df["axis"] = axis
            prediction_df["n_estimators"] = n_estimators
            prediction_df["y_true"] = y_true
            prediction_df["y_lower"] = predictions.lower
            prediction_df["y_median"] = predictions.median
            prediction_df["y_upper"] = predictions.upper
            prediction_df["geometry_source"] = geometry_source

            prediction_frames.append(prediction_df)

        print(
            f"Finished {run_id}: "
            f"coverage={metrics['coverage_percent']:.2f}% | "
            f"rmse={metrics['rmse']:.4f}"
        )

    metrics_df = pd.DataFrame(metrics_rows)

    predictions_df = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = _resolve_path(
        project_root,
        paths["output_directory"],
    )
    run_dir = (
        output_root
        / f"{config['experiment_name']}__{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    metrics_df.to_parquet(
        run_dir / "metrics.parquet",
        index=False,
    )
    metrics_df.to_csv(
        run_dir / "metrics.csv",
        index=False,
    )

    if not predictions_df.empty:
        predictions_df.to_parquet(
            run_dir / "predictions.parquet",
            index=False,
        )

    shutil.copy2(
        config["_config_path"],
        run_dir / "config.yaml",
    )

    return metrics_df, predictions_df, run_dir


def _resolve_external_path(
    project_root: Path,
    path: str | Path,
) -> Path:
    resolved_path = Path(path)

    if not resolved_path.is_absolute():
        resolved_path = project_root / resolved_path

    return resolved_path.resolve()


def _split_geometry_by_experiment_ids(
    geometry_df: pd.DataFrame,
    train_experiment_ids: list[int],
    test_experiment_ids: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create train/test frames using explicit Experiment_ID values.
    """
    if "Experiment_ID" not in geometry_df.columns:
        raise KeyError(
            "Geometry data must contain 'Experiment_ID' "
            "for experiment-based splitting."
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
            "Training geometry is empty after filtering "
            "by train_experiment_ids."
        )

    if test_df.empty:
        raise ValueError(
            "Test geometry is empty after filtering "
            "by test_experiment_ids."
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
            "Some training experiments are absent from "
            "the selected geometry source: "
            f"{sorted(missing_train_ids)}"
        )

    if missing_test_ids:
        raise ValueError(
            "Some test experiments are absent from "
            "the selected geometry source: "
            f"{sorted(missing_test_ids)}"
        )

    return (
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def _train_and_store_single_qrf_model(
    *,
    project_root: Path,
    geometry_df: pd.DataFrame,
    geometry_source: str,
    geometry_path: Path,
    target_axis: str,
    split_config: dict,
    feature_columns: list[str],
    model_config: dict,
    model_root: Path,
) -> dict:
    """
    Train and store one model for one geometry source and one axis.
    """
    if target_axis not in QRF_TARGET_COLUMNS_BY_AXIS:
        raise ValueError(
            "target_axis must be one of "
            f"{sorted(QRF_TARGET_COLUMNS_BY_AXIS)}, "
            f"received {target_axis!r}."
        )

    required_split_keys = {
        "split_index",
        "split_name",
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
        QRF_TARGET_COLUMNS_BY_AXIS[
            target_axis
        ]
    )

    required_geometry_columns = set(
        feature_columns
    ).union(
        {
            "Experiment_ID",
            target_column,
        }
    )

    missing_geometry_columns = (
        required_geometry_columns.difference(
            geometry_df.columns
        )
    )

    if missing_geometry_columns:
        raise KeyError(
            "Geometry data is missing required columns: "
            f"{sorted(missing_geometry_columns)}"
        )

    train_df, test_df = (
        _split_geometry_by_experiment_ids(
            geometry_df=geometry_df,
            train_experiment_ids=(
                split_config["train_exp"]
            ),
            test_experiment_ids=(
                split_config["test_exp"]
            ),
        )
    )

    if "n_estimators" not in model_config:
        raise KeyError(
            "model_config must contain 'n_estimators'."
        )

    n_estimators = int(
        model_config["n_estimators"]
    )

    model, predictions = train_and_predict(
        train_df=train_df,
        test_df=test_df,
        feature_columns=feature_columns,
        target_column=target_column,
        n_estimators=n_estimators,
        model_config=model_config,
    )

    y_true = test_df[
        target_column
    ].to_numpy()

    metrics = evaluate_predictions(
        y_true=y_true,
        y_lower=predictions.lower,
        y_median=predictions.median,
        y_upper=predictions.upper,
    )

    split_index = int(
        split_config["split_index"]
    )

    qrf_rank = split_config.get(
        "qrf_rank"
    )

    artifact_name = (
        f"qrf_geometry_{geometry_source}_"
        f"{target_axis}_"
        f"split_{split_index:03d}"
    )

    if qrf_rank is not None:
        artifact_name += (
            f"_rank_{int(qrf_rank):02d}"
        )

    artifact_dir = (
        model_root
        / geometry_source
        / target_axis
        / artifact_name
    )

    if artifact_dir.exists():
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        artifact_dir = artifact_dir.with_name(
            f"{artifact_dir.name}__{timestamp}"
        )

    artifact_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    model_path = (
        artifact_dir
        / "qrf_model.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    prediction_columns = list(
        dict.fromkeys(
            [
                "Experiment_ID",
                "Group_ID",
                *feature_columns,
                target_column,
            ]
        )
    )

    prediction_columns = [
        column
        for column in prediction_columns
        if column in test_df.columns
    ]

    predictions_df = test_df[
        prediction_columns
    ].copy()

    predictions_df["geometry_source"] = (
        geometry_source
    )

    predictions_df["target_axis"] = (
        target_axis
    )

    predictions_df["target_column"] = (
        target_column
    )

    predictions_df["split_index"] = (
        split_index
    )

    predictions_df["split_name"] = str(
        split_config["split_name"]
    )

    predictions_df["y_true"] = y_true
    predictions_df["y_lower"] = (
        predictions.lower
    )
    predictions_df["y_median"] = (
        predictions.median
    )
    predictions_df["y_upper"] = (
        predictions.upper
    )

    predictions_df.to_parquet(
        artifact_dir
        / "test_predictions.parquet",
        index=False,
    )

    predictions_df.to_csv(
        artifact_dir
        / "test_predictions.csv",
        index=False,
    )

    metrics_row = {
        "geometry_source": geometry_source,
        "geometry_path": str(
            geometry_path
        ),
        "target_axis": target_axis,
        "target_column": target_column,
        "split_index": split_index,
        "split_name": str(
            split_config["split_name"]
        ),
        "qrf_rank": qrf_rank,
        "qrf_score": split_config.get(
            "qrf_score"
        ),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_experiments": (
            train_df["Experiment_ID"].nunique()
        ),
        "test_experiments": (
            test_df["Experiment_ID"].nunique()
        ),
        **metrics,
    }

    pd.DataFrame(
        [metrics_row]
    ).to_csv(
        artifact_dir / "metrics.csv",
        index=False,
    )

    pd.DataFrame(
        [metrics_row]
    ).to_parquet(
        artifact_dir / "metrics.parquet",
        index=False,
    )

    metadata = {
        **metrics_row,
        "feature_columns": list(
            feature_columns
        ),
        "model_config": dict(
            model_config
        ),
        "train_experiment_ids": sorted(
            {
                int(value)
                for value in split_config[
                    "train_exp"
                ]
            }
        ),
        "test_experiment_ids": sorted(
            {
                int(value)
                for value in split_config[
                    "test_exp"
                ]
            }
        ),
        "model_path": str(model_path),
        "artifact_dir": str(artifact_dir),
    }

    with (
        artifact_dir
        / "metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            default=str,
        )

    print(
        "Stored QRF model | "
        f"source={geometry_source} | "
        f"axis={target_axis} | "
        f"split={split_index} | "
        f"coverage={metrics['coverage_percent']:.2f}% | "
        f"rmse={metrics['rmse']:.4f} | "
        f"path={model_path}"
    )

    return {
        "geometry_source": geometry_source,
        "target_axis": target_axis,
        "split_index": split_index,
        "artifact_dir": artifact_dir,
        "model_path": model_path,
        "metrics": metrics,
        "model": model,
        "predictions_df": predictions_df,
    }


def run(
    *,
    project_root: str | Path,
    geometry_sources: dict[str, str | Path],
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
) -> dict[str, dict[str, dict]]:
    """
    Train and store final QRF models for all geometry sources.

    For every source:
        Top-1 main split      -> main model
        Top-1 secondary split -> secondary model

    All model parameters and preprocessing settings are supplied
    by the caller, typically 3_1_run_qrf_interval_predictor.py.
    """
    project_root = Path(
        project_root
    ).resolve()

    model_root = _resolve_external_path(
        project_root,
        model_root,
    )

    bending_setups_path = (
        _resolve_external_path(
            project_root,
            bending_setups_path,
        )
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
        dict[str, dict],
    ] = {}

    for (
        geometry_source,
        geometry_path_value,
    ) in geometry_sources.items():
        geometry_path = _resolve_external_path(
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

        if geometry_source not in (
            model_params_by_source
        ):
            raise KeyError(
                "No model parameters were supplied for "
                f"geometry_source={geometry_source!r}."
            )

        source_splits = (
            splits_by_source[
                geometry_source
            ]
        )

        missing_axes = {
            "main",
            "secondary",
        }.difference(source_splits)

        if missing_axes:
            raise KeyError(
                "Source split configuration is missing axes: "
                f"{sorted(missing_axes)} for "
                f"geometry_source={geometry_source!r}."
            )

        paths_config = {
            "geometry_sources": {
                geometry_source: str(
                    geometry_path
                )
            }
        }

        data_config = {
            "geometry_source": geometry_source,
            "excluded_experiments": list(
                excluded_experiments
            ),
        }

        geometry_df, loaded_source, loaded_path = (
            load_selected_geometry_source(
                project_root=project_root,
                paths_config=paths_config,
                data_config=data_config,
                bending_setups_df=(
                    bending_setups_df
                ),
            )
        )

        source_results: dict[
            str,
            dict,
        ] = {}

        for target_axis in (
            "main",
            "secondary",
        ):
            source_results[target_axis] = (
                _train_and_store_single_qrf_model(
                    project_root=project_root,
                    geometry_df=geometry_df,
                    geometry_source=loaded_source,
                    geometry_path=loaded_path,
                    target_axis=target_axis,
                    split_config=source_splits[
                        target_axis
                    ],
                    feature_columns=list(
                        feature_columns
                    ),
                    model_config=dict(
                        model_params_by_source[
                            geometry_source
                        ]
                    ),
                    model_root=model_root,
                )
            )

        results[geometry_source] = (
            source_results
        )

    return results