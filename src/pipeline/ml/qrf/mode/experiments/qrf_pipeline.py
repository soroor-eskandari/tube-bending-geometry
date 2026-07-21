from __future__ import annotations

import shutil
from datetime import datetime
from itertools import product
from pathlib import Path

import pandas as pd
import yaml

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

    metrics_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    combinations = product(
        config["search"]["split_indices"],
        config["search"]["n_estimators"],
        config["search"]["axes"],
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

        metrics_rows.append(
            {
                "experiment_name": config["experiment_name"],
                "run_id": run_id,
                "split_index": split_index,
                "split_name": _split_name(
                    split_row,
                    split_index,
                ),
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
