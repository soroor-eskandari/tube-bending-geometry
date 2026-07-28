import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "src"))

import json
import ast

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.lines import Line2D

from pipeline.ml.qrf.mode.experiments.geometry_data_preprocessor import (
    load_bending_setups,
    load_geometry_data,
    read_table,
)

st.set_page_config(
    page_title="QRF Visualization",
    layout="wide",
)

st.title("QRF Prediction Interval Visualization")

project_root = Path(__file__).resolve().parent
result_dir = (
    project_root
    / "src"
    / "pipeline"
    / "ml"
    / "qrf"
    / "result"
)

model_dir = result_dir / "models"

split_metadata_path = (
    project_root
    / "src"
    / "pipeline"
    / "ml"
    / "qrf"
    / "data"
    / "various_splits.parquet"
)


bending_setup_catalog_path = (
    project_root
    / "data"
    / "rf_augmented"
    / "ui_data"
    / "unique_bending_setups.csv"
)


QRF_EXCLUDED_EXPERIMENTS = [
    1,
    48,
    166,
]

TARGETS_BY_AXIS = {
    "main": "Main-axis [mm]",
    "secondary": "Secondary-axis [mm]",
}

DATASET_LABELS = {
    "real": "Real geometry",
    "within_group_interpolation_raw": "Within group interpolation",
    "sensor_augmented_noise__time_wrapping__scaling__jittering": (
        "Sensor augmented"
    ),
}


def dataset_label(geometry_source: str) -> str:
    return DATASET_LABELS.get(
        geometry_source,
        geometry_source.replace("__", " + ").replace("_", " ").title(),
    )


def format_group(group_id) -> str:
    return f"Group {int(group_id)}"


def qrf_training_geometry_sources(project_root: Path) -> dict[str, Path]:
    return {
        "real": (
            project_root
            / "data"
            / "processed"
            / "geometry.csv"
        ),
        "sensor_augmented_noise__time_wrapping__scaling__jittering": (
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


@st.cache_data
def load_csv(path, file_mtime):
    return pd.read_csv(path)



@st.cache_data
def load_bending_setup_catalog(
    catalog_path: str,
    catalog_mtime: float,
) -> pd.DataFrame:
    """
    Load one bending-setup row per Group_ID for UI display.
    """
    del catalog_mtime

    catalog_df = read_table(
        Path(catalog_path)
    ).copy()

    if "Group_ID" not in catalog_df.columns:
        raise KeyError(
            "Bending setup catalog is missing 'Group_ID'."
        )

    catalog_df["Group_ID"] = pd.to_numeric(
        catalog_df["Group_ID"],
        errors="raise",
    ).astype(int)

    if catalog_df["Group_ID"].duplicated().any():
        duplicated_groups = sorted(
            catalog_df.loc[
                catalog_df["Group_ID"].duplicated(keep=False),
                "Group_ID",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Bending setup catalog must contain one row per "
            f"Group_ID. Duplicated groups: {duplicated_groups}"
        )

    return catalog_df.sort_values(
        "Group_ID"
    ).reset_index(drop=True)


def decode_experiment_ids(value: object) -> list[int]:
    if isinstance(value, str):
        value = ast.literal_eval(value)

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if not isinstance(value, (list, tuple, set)):
        raise TypeError(
            "Experiment IDs must be list-like, "
            f"received {type(value).__name__}."
        )

    return sorted({int(item) for item in value})


@st.cache_data
def load_preprocessed_geometry(
    geometry_path: str,
    geometry_mtime: float,
) -> pd.DataFrame:
    del geometry_mtime

    bending_df = load_bending_setups(
        path=(
            project_root
            / "data"
            / "rf_augmented"
            / "ui_data"
            / "unique_bending_setups.csv"
        ),
        excluded_experiments=QRF_EXCLUDED_EXPERIMENTS,
    )

    geometry_clean = load_geometry_data(
        path=Path(geometry_path),
        bending_setups_df=bending_df,
        excluded_experiments=QRF_EXCLUDED_EXPERIMENTS,
    )

    return geometry_clean.reset_index(drop=True)


@st.cache_data
def load_best_splits_by_target(
    metadata_path: str,
    metadata_mtime: float,
) -> dict[str, dict]:
    """
    Load one independently ranked Top-1 split per target axis.

    The split catalog contains one row per split_index. Main and
    secondary ranking values are stored in separate columns, so no
    geometry_source column is required.
    """
    del metadata_mtime

    required_columns = {
        "split_index",
        "split_name",
        "train_group_ids",
        "test_group_ids",
        "train_experiment_ids",
        "test_experiment_ids",
        "qrf_rank_main",
        "qrf_score_main",
        "qrf_rank_secondary",
        "qrf_score_secondary",
    }

    split_df = pd.read_parquet(
        metadata_path,
        columns=sorted(required_columns),
    )

    if split_df.empty:
        raise ValueError(
            f"Stored split metadata is empty: {metadata_path}"
        )

    missing_columns = required_columns.difference(
        split_df.columns
    )

    if missing_columns:
        raise KeyError(
            "Split metadata is missing columns: "
            f"{sorted(missing_columns)}"
        )

    split_df["split_index"] = pd.to_numeric(
        split_df["split_index"],
        errors="raise",
    ).astype(int)

    if split_df["split_index"].duplicated().any():
        duplicated_indices = sorted(
            split_df.loc[
                split_df["split_index"].duplicated(keep=False),
                "split_index",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "various_splits.parquet must contain exactly one row "
            "per split_index. Duplicated indices: "
            f"{duplicated_indices}"
        )

    target_rank_columns = {
        "Main-axis [mm]": {
            "axis": "main",
            "rank": "qrf_rank_main",
            "score": "qrf_score_main",
        },
        "Secondary-axis [mm]": {
            "axis": "secondary",
            "rank": "qrf_rank_secondary",
            "score": "qrf_score_secondary",
        },
    }

    selected: dict[str, dict] = {}

    for target_name, columns in target_rank_columns.items():
        rank_column = columns["rank"]
        score_column = columns["score"]

        ranked_df = split_df.dropna(
            subset=[rank_column, score_column]
        ).copy()

        if ranked_df.empty:
            raise ValueError(
                f"No QRF ranking exists for {target_name!r}. "
                "Run run_qrf_split_ranking.py first."
            )

        ranked_df[rank_column] = pd.to_numeric(
            ranked_df[rank_column],
            errors="raise",
        ).astype(int)

        ranked_df[score_column] = pd.to_numeric(
            ranked_df[score_column],
            errors="raise",
        ).astype(float)

        top_df = (
            ranked_df[
                ranked_df[rank_column].eq(1)
            ]
            .copy()
            .reset_index(drop=True)
        )

        if len(top_df) != 1:
            raise ValueError(
                f"Expected exactly one rank-1 split for "
                f"{target_name!r}, but found {len(top_df)}. "
                f"Split indices: "
                f"{top_df['split_index'].astype(int).tolist()}"
            )

        row = top_df.iloc[0]

        train_group_ids = decode_experiment_ids(
            row["train_group_ids"]
        )
        test_group_ids = decode_experiment_ids(
            row["test_group_ids"]
        )
        train_experiment_ids = decode_experiment_ids(
            row["train_experiment_ids"]
        )
        test_experiment_ids = decode_experiment_ids(
            row["test_experiment_ids"]
        )

        if not train_group_ids or not test_group_ids:
            raise ValueError(
                f"The rank-1 split for {target_name!r} has an "
                "empty train or test group list."
            )

        if not train_experiment_ids or not test_experiment_ids:
            raise ValueError(
                f"The rank-1 split for {target_name!r} has an "
                "empty train or test experiment list."
            )

        group_overlap = set(train_group_ids).intersection(
            test_group_ids
        )
        experiment_overlap = set(
            train_experiment_ids
        ).intersection(
            test_experiment_ids
        )

        if group_overlap:
            raise ValueError(
                f"Group leakage in the rank-1 split for "
                f"{target_name!r}: {sorted(group_overlap)}"
            )

        if experiment_overlap:
            raise ValueError(
                f"Experiment leakage in the rank-1 split for "
                f"{target_name!r}: {sorted(experiment_overlap)}"
            )

        selected[target_name] = {
            "axis": columns["axis"],
            "split_index": int(row["split_index"]),
            "split_name": str(row["split_name"]),
            "qrf_rank": int(row[rank_column]),
            "qrf_score": float(row[score_column]),
            "train_group_ids": train_group_ids,
            "test_group_ids": test_group_ids,
            "train_experiment_ids": train_experiment_ids,
            "test_experiment_ids": test_experiment_ids,
        }

    return selected


def attach_test_metadata(
    prediction_df: pd.DataFrame,
    geometry_df: pd.DataFrame,
    splits_by_target: dict[str, dict],
) -> pd.DataFrame:
    prediction_df = prediction_df.copy().reset_index(drop=True)

    enriched_targets = []

    for target_name, target_predictions in prediction_df.groupby(
        "target",
        sort=False,
    ):
        target_predictions = target_predictions.reset_index(drop=True)

        split_config = splits_by_target.get(target_name)

        if split_config is None:
            st.warning(
                f"No split configuration exists for `{target_name}`."
            )
            continue

        target_test_df = geometry_df[
            geometry_df["Experiment_ID"].isin(
                split_config["test_experiment_ids"]
            )
        ].copy()

        target_test_df = target_test_df.reset_index(drop=True)

        if len(target_predictions) != len(target_test_df):
            st.warning(
                f"Cannot attach metadata for `{target_name}`: "
                f"{len(target_predictions)} prediction rows versus "
                f"{len(target_test_df)} test rows."
            )

            target_predictions["Group_ID"] = pd.NA
            target_predictions["Experiment_ID"] = pd.NA
            enriched_targets.append(target_predictions)
            continue

        target_predictions["Group_ID"] = (
            target_test_df["Group_ID"].to_numpy()
        )
        target_predictions["Experiment_ID"] = (
            target_test_df["Experiment_ID"].to_numpy()
        )

        enriched_targets.append(target_predictions)

    if not enriched_targets:
        return pd.DataFrame()

    return pd.concat(enriched_targets, ignore_index=True)


def compute_metrics(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {
            "coverage": np.nan,
            "expected_coverage": 0.90,
            "calibration_error": np.nan,
            "mean_interval_width": np.nan,
            "rmse_median": np.nan,
            "mae_median": np.nan,
            "bias": np.nan,
            "abs_bias": np.nan,
            "residual_std": np.nan,
        }

    inside = (
        (df["y_true"] >= df["y_pred_lower"])
        & (df["y_true"] <= df["y_pred_upper"])
    )
    coverage = inside.mean()
    expected_coverage = 0.90
    residual = df["y_pred_mean"] - df["y_true"]
    return {
        "coverage": coverage,
        "expected_coverage": expected_coverage,
        "calibration_error": abs(coverage - expected_coverage),
        "mean_interval_width": (df["y_pred_upper"] - df["y_pred_lower"]).mean(),
        "rmse_median": np.sqrt(np.mean(np.square(residual))),
        "mae_median": np.mean(np.abs(residual)),
        "bias": residual.mean(),
        "abs_bias": abs(residual.mean()),
        "residual_std": residual.std(ddof=0),
    }


def target_axis(target_name: str) -> str:
    for axis, configured_target in TARGETS_BY_AXIS.items():
        if configured_target == target_name:
            return axis

    raise KeyError(f"Unknown QRF target: {target_name!r}")


def find_qrf_artifact_dir(
    geometry_source: str,
    axis: str,
    split_index: int,
) -> Path:
    axis_dir = model_dir / geometry_source / axis

    if not axis_dir.exists():
        raise FileNotFoundError(
            f"Missing QRF artifact directory: {axis_dir}"
        )

    candidates = []

    for metadata_path in axis_dir.glob("*/metadata.json"):
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        if int(metadata.get("split_index", -1)) == int(split_index):
            candidates.append(
                (
                    metadata_path.parent.stat().st_mtime,
                    metadata_path.parent,
                )
            )

    if not candidates:
        raise FileNotFoundError(
            "No QRF artifact found for "
            f"source={geometry_source!r}, axis={axis!r}, "
            f"split_index={split_index}."
        )

    return max(candidates, key=lambda item: item[0])[1]


@st.cache_data
def load_qrf_metadata(metadata_path: str, metadata_mtime: float) -> dict:
    del metadata_mtime

    with Path(metadata_path).open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_resource
def load_qrf_model(model_path: str, model_mtime: float):
    del model_mtime

    return joblib.load(model_path)


@st.cache_data
def load_qrf_test_predictions(
    predictions_path: str,
    predictions_mtime: float,
) -> pd.DataFrame:
    del predictions_mtime

    return read_table(Path(predictions_path))


def load_qrf_artifact(
    geometry_source: str,
    axis: str,
    split_index: int,
) -> dict:
    artifact_dir = find_qrf_artifact_dir(
        geometry_source=geometry_source,
        axis=axis,
        split_index=split_index,
    )

    metadata_path = artifact_dir / "metadata.json"
    model_path = artifact_dir / "qrf_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing QRF model artifact: {model_path}"
        )

    metadata = load_qrf_metadata(
        str(metadata_path),
        metadata_path.stat().st_mtime,
    )

    model = load_qrf_model(
        str(model_path),
        model_path.stat().st_mtime,
    )

    return {
        "artifact_dir": artifact_dir,
        "metadata": metadata,
        "model": model,
    }


def format_prediction_frame(
    prediction_df: pd.DataFrame,
    target_name: str,
) -> pd.DataFrame:
    angle_col = "Angle[degree]ORDistance[mm]"

    return pd.DataFrame(
        {
            "angle": prediction_df[angle_col].to_numpy(),
            "y_true": prediction_df["y_true"].to_numpy(),
            "y_pred_mean": prediction_df["y_median"].to_numpy(),
            "y_pred_lower": prediction_df["y_lower"].to_numpy(),
            "y_pred_upper": prediction_df["y_upper"].to_numpy(),
            "target": target_name,
            "Group_ID": prediction_df["Group_ID"].to_numpy(),
            "Experiment_ID": prediction_df["Experiment_ID"].to_numpy(),
        }
    )


def load_test_predictions_from_artifacts(
    geometry_source: str,
    splits_by_target: dict[str, dict],
) -> pd.DataFrame:
    frames = []

    for target_name, split_config in splits_by_target.items():
        axis = target_axis(target_name)
        artifact_dir = find_qrf_artifact_dir(
            geometry_source=geometry_source,
            axis=axis,
            split_index=split_config["split_index"],
        )

        predictions_path = artifact_dir / "test_predictions.parquet"

        if not predictions_path.exists():
            predictions_path = artifact_dir / "test_predictions.csv"

        if not predictions_path.exists():
            raise FileNotFoundError(
                "Missing QRF test predictions in artifact: "
                f"{artifact_dir}"
            )

        raw_predictions = load_qrf_test_predictions(
            str(predictions_path),
            predictions_path.stat().st_mtime,
        )

        frames.append(
            format_prediction_frame(
                prediction_df=raw_predictions,
                target_name=target_name,
            )
        )

    return pd.concat(frames, ignore_index=True)


def predict_split_group(
    split_df: pd.DataFrame,
    geometry_source: str,
    splits_by_target: dict[str, dict],
) -> pd.DataFrame:
    angle_col = "Angle[degree]ORDistance[mm]"

    def predict_target(
        split_config: dict,
        target_col: str,
        target_name: str,
    ) -> pd.DataFrame:
        axis = target_axis(target_name)
        artifact = load_qrf_artifact(
            geometry_source=geometry_source,
            axis=axis,
            split_index=split_config["split_index"],
        )

        model = artifact["model"]
        metadata = artifact["metadata"]
        feature_columns = metadata["feature_columns"]
        model_config = metadata.get("model_config", {})
        lower_quantile = float(
            model_config.get("lower_quantile", 0.05)
        )
        upper_quantile = float(
            model_config.get("upper_quantile", 0.95)
        )
        X = split_df[feature_columns]

        return pd.DataFrame(
            {
                "angle": split_df[angle_col].to_numpy(),
                "y_true": split_df[target_col].to_numpy(),
                "y_pred_mean": np.asarray(model.predict(X, quantiles=0.50)).ravel(),
                "y_pred_lower": np.asarray(
                    model.predict(X, quantiles=lower_quantile)
                ).ravel(),
                "y_pred_upper": np.asarray(
                    model.predict(X, quantiles=upper_quantile)
                ).ravel(),
                "target": target_name,
                "Group_ID": split_df["Group_ID"].to_numpy(),
                "Experiment_ID": split_df["Experiment_ID"].to_numpy(),
            }
        )

    return pd.concat(
        [
            predict_target(
                splits_by_target["Main-axis [mm]"],
                "Main-axis [mm]",
                "Main-axis [mm]",
            ),
            predict_target(
                splits_by_target["Secondary-axis [mm]"],
                "Secondary-axis [mm]",
                "Secondary-axis [mm]",
            ),
        ],
        ignore_index=True,
    )



def load_target_test_predictions_from_artifact(
    geometry_source: str,
    target_name: str,
    split_config: dict,
) -> pd.DataFrame:
    axis = target_axis(target_name)

    artifact_dir = find_qrf_artifact_dir(
        geometry_source=geometry_source,
        axis=axis,
        split_index=split_config["split_index"],
    )

    predictions_path = (
        artifact_dir
        / "test_predictions.parquet"
    )

    if not predictions_path.exists():
        predictions_path = (
            artifact_dir
            / "test_predictions.csv"
        )

    if not predictions_path.exists():
        raise FileNotFoundError(
            "Missing QRF test predictions in artifact: "
            f"{artifact_dir}"
        )

    raw_predictions = load_qrf_test_predictions(
        str(predictions_path),
        predictions_path.stat().st_mtime,
    )

    return format_prediction_frame(
        prediction_df=raw_predictions,
        target_name=target_name,
    )


def predict_target_group(
    split_df: pd.DataFrame,
    geometry_source: str,
    target_name: str,
    split_config: dict,
) -> pd.DataFrame:
    axis = target_axis(target_name)

    artifact = load_qrf_artifact(
        geometry_source=geometry_source,
        axis=axis,
        split_index=split_config["split_index"],
    )

    model = artifact["model"]
    metadata = artifact["metadata"]
    feature_columns = metadata["feature_columns"]
    model_config = metadata.get(
        "model_config",
        {},
    )

    lower_quantile = float(
        model_config.get(
            "lower_quantile",
            0.05,
        )
    )
    upper_quantile = float(
        model_config.get(
            "upper_quantile",
            0.95,
        )
    )

    target_column = TARGETS_BY_AXIS[axis]
    angle_column = (
        "Angle[degree]ORDistance[mm]"
    )

    X = split_df[feature_columns]

    return pd.DataFrame(
        {
            "angle": split_df[
                angle_column
            ].to_numpy(),
            "y_true": split_df[
                target_column
            ].to_numpy(),
            "y_pred_mean": np.asarray(
                model.predict(
                    X,
                    quantiles=0.50,
                )
            ).ravel(),
            "y_pred_lower": np.asarray(
                model.predict(
                    X,
                    quantiles=lower_quantile,
                )
            ).ravel(),
            "y_pred_upper": np.asarray(
                model.predict(
                    X,
                    quantiles=upper_quantile,
                )
            ).ravel(),
            "target": target_name,
            "Group_ID": split_df[
                "Group_ID"
            ].to_numpy(),
            "Experiment_ID": split_df[
                "Experiment_ID"
            ].to_numpy(),
        }
    )


class QRFVisualizer:
    prediction_linewidth = 4.0
    experiment_signal_colors = [
        "#6F4E7C",
        "#2E7D32",
        "#C2185B",
        "#7A6F1A",
        "#6A4C3B",
        "#00897B",
        "#8E24AA",
        "#558B2F",
        "#AD1457",
        "#5D4037",
    ]

    def __init__(self, prediction_df, angle_col, plot_mode):
        self.prediction_df = prediction_df
        self.angle_col = angle_col
        self.plot_mode = plot_mode

    def prepare_data(self, target_name):
        target_predictions = self.prediction_df[
            self.prediction_df["target"] == target_name
        ].copy()

        grouped = (
            target_predictions.groupby(self.angle_col)
            .agg(
                y_true=("y_true", "mean"),
                y_pred_mean=("y_pred_mean", "mean"),
                y_pred_lower=("y_pred_lower", "mean"),
                y_pred_upper=("y_pred_upper", "mean"),
            )
            .reset_index()
            .sort_values(self.angle_col)
        )
        return grouped, target_predictions

    def plot_target(self, target_name, y_limits=None):
        df, raw_target_df = self.prepare_data(target_name)

        if df.empty:
            st.warning(f"No rows found for `{target_name}`.")
            return

        x = df[self.angle_col].values
        y_true = df["y_true"].values
        y_pred_mean = df["y_pred_mean"].values
        y_pred_lower = df["y_pred_lower"].values
        y_pred_upper = df["y_pred_upper"].values

        y_pred_lower = (
            pd.Series(y_pred_lower).rolling(window=3, center=True, min_periods=1).mean().values
        )
        y_pred_upper = (
            pd.Series(y_pred_upper).rolling(window=3, center=True, min_periods=1).mean().values
        )
        y_pred_mean = (
            pd.Series(y_pred_mean).rolling(window=3, center=True, min_periods=1).mean().values
        )

        raw_target_df["display_angle"] = (
            raw_target_df["angle"]
            + np.random.default_rng(42).normal(0, 0.08, len(raw_target_df))
        )

        raw_inside_mask = (
            (raw_target_df["y_true"] >= raw_target_df["y_pred_lower"])
            & (raw_target_df["y_true"] <= raw_target_df["y_pred_upper"])
        )
        raw_inside_df = raw_target_df[raw_inside_mask]
        raw_outside_df = raw_target_df[~raw_inside_mask]
        metrics = compute_metrics(raw_target_df)

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(14, 6))

        ax.fill_between(
            x,
            y_pred_lower,
            y_pred_upper,
            color="#4C72B0",
            alpha=0.22,
            label="Prediction Interval",
        )
        ax.plot(
            x,
            y_pred_mean,
            color="#FF8C00",
            linewidth=self.prediction_linewidth,
            label="Prediction Median",
        )

        if self.plot_mode == "Experiment signals":
            signal_ids = (
                raw_target_df["Experiment_ID"]
                .dropna()
                .drop_duplicates()
                .sort_values()
                .tolist()
            )
            for color_index, experiment_id in enumerate(signal_ids):
                signal_color = self.experiment_signal_colors[
                    color_index % len(self.experiment_signal_colors)
                ]
                signal_df = raw_target_df[
                    raw_target_df["Experiment_ID"] == experiment_id
                ].sort_values(self.angle_col)
                ax.plot(
                    signal_df[self.angle_col],
                    signal_df["y_true"],
                    color=signal_color,
                    linewidth=1.1,
                    alpha=0.35,
                    label="_nolegend_",
                    zorder=7,
                )
                ax.scatter(
                    signal_df[self.angle_col],
                    signal_df["y_true"],
                    color=signal_color,
                    s=12,
                    marker="o",
                    alpha=0.35,
                    label="_nolegend_",
                    zorder=8,
                )
            ax.plot(
                x,
                y_true,
                color="#025BFF",
                linewidth=3.2,
                linestyle="--",
                alpha=0.95,
                label="Actual Median",
                zorder=11,
            )
        else:
            ax.plot(
                x,
                y_true,
                color="#025BFF",
                linewidth=2,
                linestyle="--",
                alpha=0.9,
                label="Actual Median",
            )
            ax.scatter(
                raw_inside_df["display_angle"],
                raw_inside_df["y_true"],
                color="#3A58C3",
                s=14,
                marker="o",
                alpha=0.2,
                label="_nolegend_",
                zorder=8,
            )
            ax.scatter(
                raw_outside_df["display_angle"],
                raw_outside_df["y_true"],
                color="red",
                s=18,
                marker="o",
                alpha=0.85,
                label="_nolegend_",
                zorder=10,
            )

        ax.set_title(
            f"{target_name} - QRF Uncertainty Estimation",
            fontsize=18,
            pad=15,
            weight="bold",
        )
        ax.set_xlabel("Angle [degree]", fontsize=13)
        ax.set_ylabel(target_name, fontsize=13)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.2)

        if y_limits is not None:
            ax.set_ylim(*y_limits)

        legend_elements = [
            Line2D([0], [0], color="#4C72B0", lw=10, alpha=0.22, label="Prediction Interval"),
            Line2D([0], [0], color="#FF8C00", lw=self.prediction_linewidth, label="Prediction Median"),
        ]
        if self.plot_mode == "Experiment signals":
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color=self.experiment_signal_colors[0],
                    lw=1.1,
                    alpha=0.35,
                    label="Experiment signals",
                )
            )
            legend_elements.append(
                Line2D([0], [0], color="#025BFF", lw=3.2, linestyle="--", label="Actual Median")
            )
        else:
            legend_elements.extend(
                [
                    Line2D([0], [0], color="#025BFF", lw=2, linestyle="--", label="Actual Median"),
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor="red",
                        markersize=7,
                        label="Outside Interval",
                    ),
                ]
            )
        ax.legend(handles=legend_elements, frameon=True, facecolor="white", edgecolor="lightgray")
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("### Group Coverage")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Coverage", f"{metrics['coverage']:.3f}")
        c2.metric("Expected Coverage", f"{metrics['expected_coverage']:.3f}")
        c3.metric("Calibration Error", f"{metrics['calibration_error']:.3f}")
        c4.metric("Mean Interval Width", f"{metrics['mean_interval_width']:.3f}")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Standard Deviation", f"{metrics['residual_std']:.3f}")
        c2.metric("RMSE Median", f"{metrics['rmse_median']:.3f}")
        c3.metric("MAE Median", f"{metrics['mae_median']:.3f}")
        c4.metric("Bias", f"{metrics['bias']:.3f}")
        c5.metric("Abs Bias", f"{metrics['abs_bias']:.3f}")


ZOOM_OUT_Y_LIMITS = (
    20.8,
    23.0,
)



# ============================================================
# Shared data
# ============================================================

geometry_sources = qrf_training_geometry_sources(
    project_root
)

if not split_metadata_path.exists():
    st.error(
        "Stored split metadata was not found: "
        f"{split_metadata_path}"
    )
    st.stop()

splits_by_target = load_best_splits_by_target(
    str(split_metadata_path),
    split_metadata_path.stat().st_mtime,
)


if not bending_setup_catalog_path.exists():
    st.error(
        "Bending setup catalog was not found: "
        f"{bending_setup_catalog_path}"
    )
    st.stop()

bending_setup_df = load_bending_setup_catalog(
    str(bending_setup_catalog_path),
    bending_setup_catalog_path.stat().st_mtime,
)

def reset_axis_group_selection(
    group_widget_key: str,
) -> None:
    """
    Clear the selected group when dataset or split membership changes.
    """
    st.session_state.pop(
        group_widget_key,
        None,
    )


def render_axis_section(
    *,
    target_name: str,
    section_title: str,
    key_prefix: str,
    bending_setup_df: pd.DataFrame,
) -> dict | None:
    """
    Render one fully independent UI section for one target axis.
    """
    split_config = splits_by_target[
        target_name
    ]

    axis = split_config["axis"]

    st.markdown("---")
    st.header(section_title)

    st.caption(
        f"Independent rank-1 split for `{axis}`: "
        f"split {split_config['split_index']} — "
        f"`{split_config['split_name']}` "
        f"(score={split_config['qrf_score']:.6f})"
    )

    dataset_widget_key = (
        f"qrf_{key_prefix}_dataset"
    )
    split_widget_key = (
        f"qrf_{key_prefix}_split_membership"
    )
    plot_widget_key = (
        f"qrf_{key_prefix}_plot_mode"
    )
    scaling_widget_key = (
        f"qrf_{key_prefix}_scaling"
    )

    control_col_1, control_col_2 = st.columns(2)

    with control_col_1:
        geometry_source = st.selectbox(
            f"{section_title} dataset",
            list(geometry_sources),
            format_func=dataset_label,
            key=dataset_widget_key,
        )

        split_name = st.radio(
            f"{section_title} split membership",
            ["test", "train"],
            horizontal=True,
            key=split_widget_key,
        )

    with control_col_2:
        plot_mode = st.radio(
            f"{section_title} plot mode",
            [
                "All points",
                "Experiment signals",
            ],
            horizontal=True,
            key=plot_widget_key,
        )

        scaling = st.radio(
            f"{section_title} scaling",
            [
                "zoom_in",
                "zoom_out",
            ],
            horizontal=True,
            key=scaling_widget_key,
        )

    # Separate selectbox state for each axis and membership.
    group_widget_key = (
        f"qrf_{key_prefix}_{split_name}_group_number"
    )

    geometry_path = geometry_sources[
        geometry_source
    ]

    st.caption(
        "Using "
        f"`{geometry_path.relative_to(project_root)}`."
    )

    if not geometry_path.exists():
        st.warning(
            "Missing geometry source file for "
            f"`{geometry_source}`."
        )
        return

    geometry_df = load_preprocessed_geometry(
        str(geometry_path),
        geometry_path.stat().st_mtime,
    )

    group_id_key = (
        "test_group_ids"
        if split_name == "test"
        else "train_group_ids"
    )

    experiment_id_key = (
        "test_experiment_ids"
        if split_name == "test"
        else "train_experiment_ids"
    )

    available_groups = sorted(
        {
            int(group_id)
            for group_id
            in split_config[
                group_id_key
            ]
        }
    )

    allowed_experiment_ids = sorted(
        {
            int(experiment_id)
            for experiment_id
            in split_config[
                experiment_id_key
            ]
        }
    )

    st.caption(
        f"`{axis}` `{split_name}` membership contains "
        f"**{len(available_groups)} groups**. "
        f"Group IDs: {available_groups}"
    )

    if not available_groups:
        st.warning(
            f"No groups are available in the "
            f"`{split_name}` split for `{target_name}`."
        )
        return

    selected_group = st.selectbox(
        f"{section_title} group number",
        available_groups,
        format_func=format_group,
        key=group_widget_key,
    )

    selected_setup_df = bending_setup_df[
        bending_setup_df[
            "Group_ID"
        ].eq(int(selected_group))
    ].copy()

    st.markdown("#### Bending setup")

    if selected_setup_df.empty:
        st.warning(
            "No bending setup row was found for "
            f"{format_group(selected_group)}."
        )
    else:
        st.dataframe(
            selected_setup_df,
            use_container_width=True,
            hide_index=True,
        )

    selected_split_df = geometry_df[
        geometry_df[
            "Group_ID"
        ]
        .astype(int)
        .eq(int(selected_group))
    ].copy()

    if selected_split_df.empty:
        st.warning(
            f"No geometry rows found for "
            f"{format_group(selected_group)} "
            f"in the `{split_name}` split."
        )
        return

    if split_name == "test":
        prediction_df = (
            load_target_test_predictions_from_artifact(
                geometry_source=geometry_source,
                target_name=target_name,
                split_config=split_config,
            )
        )
    else:
        prediction_df = predict_target_group(
            split_df=selected_split_df,
            geometry_source=geometry_source,
            target_name=target_name,
            split_config=split_config,
        )

    prediction_df = prediction_df[
        (
            prediction_df[
                "Group_ID"
            ].astype("Int64")
            == int(selected_group)
        )
        & prediction_df[
            "angle"
        ].between(
            0,
            44,
            inclusive="both",
        )
    ].copy()

    if prediction_df.empty:
        st.warning(
            f"No prediction rows found for "
            f"{format_group(selected_group)} "
            f"in `{target_name}`."
        )
        return

    st.caption(
        f"Showing `{geometry_source}` | "
        f"`{split_name}` | "
        f"{format_group(selected_group)} | "
        f"split_index={split_config['split_index']}."
    )

    plot_slot = st.empty()

    return {
        "target_name": target_name,
        "prediction_df": prediction_df,
        "plot_mode": plot_mode,
        "scaling": scaling,
        "plot_slot": plot_slot,
    }


axis_sections = [
    render_axis_section(
        target_name="Main-axis [mm]",
        section_title="Main Axis",
        key_prefix="main",
        bending_setup_df=bending_setup_df,
    ),
    render_axis_section(
        target_name="Secondary-axis [mm]",
        section_title="Secondary Axis",
        key_prefix="secondary",
        bending_setup_df=bending_setup_df,
    ),
]

active_axis_sections = [
    section
    for section in axis_sections
    if section is not None
]

for section in active_axis_sections:
    with section["plot_slot"].container():
        visualizer = QRFVisualizer(
            prediction_df=section["prediction_df"],
            angle_col="angle",
            plot_mode=section["plot_mode"],
        )

        visualizer.plot_target(
            section["target_name"],
            y_limits=(
                ZOOM_OUT_Y_LIMITS
                if section["scaling"] == "zoom_out"
                else None
            ),
        )
