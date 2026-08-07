from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.lines import Line2D

from src.pipeline.ml.hgp.utils.experiments.geometry_data_preprocessor import (
    load_bending_setups,
    load_geometry_data,
    read_table,
)
from src.pipeline.ml.hgp.utils.experiments.hgp_model_trainer import (
    predict as hgp_predict,
)


st.set_page_config(
    page_title="HGP Visualization",
    layout="wide",
)

st.title("HGP Prediction Interval Visualization")

result_dir = (
    project_root
    / "src"
    / "pipeline"
    / "ml"
    / "hgp"
    / "results"
)

model_dir = result_dir / "models"

split_metadata_path = (
    project_root
    / "src"
    / "pipeline"
    / "ml"
    / "common"
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

HGP_EXCLUDED_EXPERIMENTS = [
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


def count_source_experiments(
    group_geometry_df: pd.DataFrame,
) -> int:
    """
    Count the source samples represented by the selected group.

    Augmented geometry stores the intended group sample count in
    target_group_samples. Counting filtered Experiment_ID values would
    under-report those groups because synthetic Experiment_ID values are
    not part of the original train/test split metadata.
    """
    if (
        "target_group_samples" in group_geometry_df.columns
        and group_geometry_df["target_group_samples"].notna().any()
    ):
        counts = pd.to_numeric(
            group_geometry_df["target_group_samples"],
            errors="coerce",
        ).dropna()

        if not counts.empty:
            return int(counts.max())

    return int(
        group_geometry_df[
            "Experiment_ID"
        ]
        .dropna()
        .nunique()
    )


def uses_synthetic_group_samples(
    geometry_df: pd.DataFrame,
) -> bool:
    return (
        "target_group_samples" in geometry_df.columns
        or "group_id" in geometry_df.columns
        or (
            "Synthetic" in geometry_df.columns
            and geometry_df["Synthetic"].fillna(False).astype(bool).any()
        )
    )


def hgp_training_geometry_sources(project_root: Path) -> dict[str, Path]:
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
def load_bending_setup_catalog(
    catalog_path: str,
    catalog_mtime: float,
) -> pd.DataFrame:
    del catalog_mtime

    catalog_df = read_table(Path(catalog_path)).copy()

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


@st.cache_data
def load_preprocessed_geometry(
    geometry_path: str,
    geometry_mtime: float,
) -> pd.DataFrame:
    del geometry_mtime

    bending_df = load_bending_setups(
        path=bending_setup_catalog_path,
        excluded_experiments=HGP_EXCLUDED_EXPERIMENTS,
    )

    geometry_clean = load_geometry_data(
        path=Path(geometry_path),
        bending_setups_df=bending_df,
        excluded_experiments=HGP_EXCLUDED_EXPERIMENTS,
    )

    return geometry_clean.reset_index(drop=True)


@st.cache_data
def load_best_splits_by_target(
    metadata_path: str,
    metadata_mtime: float,
) -> dict[str, dict]:
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

    missing_columns = required_columns.difference(split_df.columns)

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
        axis = columns["axis"]
        rank_column = columns["rank"]
        score_column = columns["score"]

        ranked_df = split_df.dropna(
            subset=[rank_column, score_column]
        ).copy()

        if ranked_df.empty:
            raise ValueError(
                f"No ranking exists for {target_name!r}. "
                "Run the split ranking pipeline first."
            )

        ranked_df[rank_column] = pd.to_numeric(
            ranked_df[rank_column],
            errors="raise",
        ).astype(int)

        ranked_df[score_column] = pd.to_numeric(
            ranked_df[score_column],
            errors="raise",
        ).astype(float)

        selected_df = (
            ranked_df[ranked_df[rank_column].eq(1)]
            .copy()
            .reset_index(drop=True)
        )

        if len(selected_df) != 1:
            raise ValueError(
                f"Expected exactly one rank-1 split for "
                f"{target_name!r}, but found {len(selected_df)}. "
                f"Split indices: "
                f"{selected_df['split_index'].astype(int).tolist()}"
            )

        row = selected_df.iloc[0]

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
            "axis": axis,
            "split_index": int(row["split_index"]),
            "split_name": str(row["split_name"]),
            "selection_method": "rank-1 split",
            "source_rank": int(row[rank_column]),
            "source_score": float(row[score_column]),
            "train_group_ids": train_group_ids,
            "test_group_ids": test_group_ids,
            "train_experiment_ids": train_experiment_ids,
            "test_experiment_ids": test_experiment_ids,
        }

    return selected


def compute_metrics(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {
            "coverage": np.nan,
            "expected_coverage": np.nan,
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
    if "expected_coverage" in df.columns:
        expected_coverage = float(
            df["expected_coverage"].iloc[0]
        )
    else:
        expected_coverage = 0.90
    residual = df["y_pred_mean"] - df["y_true"]

    return {
        "coverage": coverage,
        "expected_coverage": expected_coverage,
        "calibration_error": abs(coverage - expected_coverage),
        "mean_interval_width": (
            df["y_pred_upper"] - df["y_pred_lower"]
        ).mean(),
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

    raise KeyError(f"Unknown HGP target: {target_name!r}")


def hgp_model_path(
    geometry_source: str,
    axis: str,
) -> Path:
    return (
        model_dir
        / geometry_source
        / axis
        / "hgp_model.joblib"
    )


@st.cache_resource
def load_hgp_model(
    model_path: str,
    model_mtime: float,
):
    del model_mtime

    try:
        return joblib.load(model_path)
    except ValueError as error:
        if (
            "BitGenerator" in str(error)
            or "MT19937" in str(error)
        ):
            raise RuntimeError(
                "This HGP model artifact was saved with a NumPy "
                "version that is incompatible with the Python "
                f"environment running Streamlit (NumPy {np.__version__}). "
                "Run Streamlit from an environment with NumPy 2.x, "
                "or retrain/resave the HGP models with the current "
                "environment."
            ) from error

        raise


def load_hgp_artifact(
    geometry_source: str,
    axis: str,
):
    model_path = hgp_model_path(
        geometry_source=geometry_source,
        axis=axis,
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing HGP model artifact: {model_path}"
        )

    return load_hgp_model(
        str(model_path),
        model_path.stat().st_mtime,
    )


def predict_target_group(
    split_df: pd.DataFrame,
    geometry_source: str,
    target_name: str,
) -> pd.DataFrame:
    axis = target_axis(target_name)
    model_bundle = load_hgp_artifact(
        geometry_source=geometry_source,
        axis=axis,
    )

    target_column = TARGETS_BY_AXIS[axis]
    angle_column = "Angle[degree]ORDistance[mm]"

    predictions = hgp_predict(
        model_bundle=model_bundle,
        dataframe=split_df,
    )

    return pd.DataFrame(
        {
            "angle": split_df[angle_column].to_numpy(),
            "y_true": split_df[target_column].to_numpy(),
            "y_pred_mean": predictions.mean,
            "y_pred_lower": predictions.lower,
            "y_pred_upper": predictions.upper,
            "latent_std": predictions.latent_std,
            "aleatoric_std": predictions.aleatoric_std,
            "total_std": predictions.total_std,
            "expected_coverage": model_bundle.confidence_level,
            "target": target_name,
            "Group_ID": split_df["Group_ID"].to_numpy(),
            "Experiment_ID": split_df["Experiment_ID"].to_numpy(),
        }
    )


class HGPVisualizer:
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

    def __init__(
        self,
        prediction_df,
        angle_col,
        plot_mode,
        experiment_count=None,
    ):
        self.prediction_df = prediction_df
        self.angle_col = angle_col
        self.plot_mode = plot_mode
        self.experiment_count = experiment_count

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
            pd.Series(y_pred_lower)
            .rolling(window=3, center=True, min_periods=1)
            .mean()
            .values
        )
        y_pred_upper = (
            pd.Series(y_pred_upper)
            .rolling(window=3, center=True, min_periods=1)
            .mean()
            .values
        )
        y_pred_mean = (
            pd.Series(y_pred_mean)
            .rolling(window=3, center=True, min_periods=1)
            .mean()
            .values
        )

        raw_target_df["display_angle"] = (
            raw_target_df["angle"]
            + np.random.default_rng(42).normal(
                0,
                0.08,
                len(raw_target_df),
            )
        )

        raw_inside_mask = (
            (raw_target_df["y_true"] >= raw_target_df["y_pred_lower"])
            & (raw_target_df["y_true"] <= raw_target_df["y_pred_upper"])
        )
        raw_inside_df = raw_target_df[raw_inside_mask]
        raw_outside_df = raw_target_df[~raw_inside_mask]
        metrics = compute_metrics(raw_target_df)
        experiment_count = (
            int(self.experiment_count)
            if self.experiment_count is not None
            else int(
                raw_target_df[
                    "Experiment_ID"
                ]
                .dropna()
                .nunique()
            )
        )

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
            f"{target_name} - HGP Uncertainty Estimation",
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
            Line2D(
                [0],
                [0],
                color="#4C72B0",
                lw=10,
                alpha=0.22,
                label="Prediction Interval",
            ),
            Line2D(
                [0],
                [0],
                color="#FF8C00",
                lw=self.prediction_linewidth,
                label="Prediction Median",
            ),
            Line2D(
                [],
                [],
                linestyle="None",
                label=f"Source experiments: {experiment_count}",
            ),
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
                Line2D(
                    [0],
                    [0],
                    color="#025BFF",
                    lw=3.2,
                    linestyle="--",
                    label="Actual Median",
                )
            )
        else:
            legend_elements.extend(
                [
                    Line2D(
                        [0],
                        [0],
                        color="#025BFF",
                        lw=2,
                        linestyle="--",
                        label="Actual Median",
                    ),
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
        ax.legend(
            handles=legend_elements,
            frameon=True,
            facecolor="white",
            edgecolor="lightgray",
        )
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("### Group Coverage")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Coverage", f"{metrics['coverage']:.3f}")
        c2.metric(
            "Expected Coverage",
            f"{metrics['expected_coverage']:.3f}",
        )
        c3.metric(
            "Calibration Error",
            f"{metrics['calibration_error']:.3f}",
        )
        c4.metric(
            "Mean Interval Width",
            f"{metrics['mean_interval_width']:.3f}",
        )

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(
            "Standard Deviation",
            f"{metrics['residual_std']:.3f}",
        )
        c2.metric("RMSE Median", f"{metrics['rmse_median']:.3f}")
        c3.metric("MAE Median", f"{metrics['mae_median']:.3f}")
        c4.metric("Bias", f"{metrics['bias']:.3f}")
        c5.metric("Abs Bias", f"{metrics['abs_bias']:.3f}")


ZOOM_OUT_Y_LIMITS = (
    20.8,
    23.0,
)


geometry_sources = hgp_training_geometry_sources(project_root)

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


def render_axis_section(
    *,
    target_name: str,
    section_title: str,
    key_prefix: str,
    bending_setup_df: pd.DataFrame,
) -> dict | None:
    split_config = splits_by_target[target_name]
    axis = split_config["axis"]

    st.markdown("---")
    st.header(section_title)

    st.caption(
        f"Independent {split_config['selection_method']} "
        f"for `{axis}`: "
        f"split {split_config['split_index']} - "
        f"`{split_config['split_name']}` "
        f"(source score={split_config['source_score']:.6f})"
    )

    dataset_widget_key = f"hgp_{key_prefix}_dataset"
    split_widget_key = f"hgp_{key_prefix}_split_membership"
    plot_widget_key = f"hgp_{key_prefix}_plot_mode"
    scaling_widget_key = f"hgp_{key_prefix}_scaling"

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

    group_widget_key = f"hgp_{key_prefix}_{split_name}_group_number"
    geometry_path = geometry_sources[geometry_source]
    model_path = hgp_model_path(
        geometry_source=geometry_source,
        axis=axis,
    )

    st.caption(
        "Using "
        f"`{geometry_path.relative_to(project_root)}` and "
        f"`{model_path.relative_to(project_root)}`."
    )

    if not geometry_path.exists():
        st.warning(
            "Missing geometry source file for "
            f"`{geometry_source}`."
        )
        return None

    if not model_path.exists():
        st.warning(
            "Missing HGP model file for "
            f"`{geometry_source}` / `{axis}`: `{model_path}`."
        )
        return None

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
            for group_id in split_config[group_id_key]
        }
    )
    allowed_experiment_ids = sorted(
        {
            int(experiment_id)
            for experiment_id in split_config[experiment_id_key]
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
        return None

    selected_group = st.selectbox(
        f"{section_title} group number",
        available_groups,
        format_func=format_group,
        key=group_widget_key,
    )

    selected_setup_df = bending_setup_df[
        bending_setup_df["Group_ID"].eq(int(selected_group))
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
            width="stretch",
            hide_index=True,
        )

    selected_group_df = geometry_df[
        geometry_df["Group_ID"].astype(int).eq(int(selected_group))
    ].copy()

    if selected_group_df.empty:
        st.warning(
            f"No geometry rows found for "
            f"{format_group(selected_group)} "
            f"in `{dataset_label(geometry_source)}`."
        )
        return None

    selected_group_experiment_count = count_source_experiments(
        selected_group_df
    )

    if uses_synthetic_group_samples(geometry_df):
        selected_split_df = selected_group_df.copy()
    else:
        selected_split_df = selected_group_df[
            selected_group_df["Experiment_ID"].astype(int).isin(
                allowed_experiment_ids
            )
        ].copy()

    if selected_split_df.empty:
        st.warning(
            f"No geometry rows found for "
            f"{format_group(selected_group)} "
            f"in the `{split_name}` split."
        )
        return None

    try:
        prediction_df = predict_target_group(
            split_df=selected_split_df,
            geometry_source=geometry_source,
            target_name=target_name,
        )
    except RuntimeError as error:
        st.error(str(error))
        return None

    prediction_df = prediction_df[
        prediction_df["angle"].between(
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
        return None

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
        "experiment_count": selected_group_experiment_count,
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
        visualizer = HGPVisualizer(
            prediction_df=section["prediction_df"],
            angle_col="angle",
            plot_mode=section["plot_mode"],
            experiment_count=section["experiment_count"],
        )

        visualizer.plot_target(
            section["target_name"],
            y_limits=(
                ZOOM_OUT_Y_LIMITS
                if section["scaling"] == "zoom_out"
                else None
            ),
        )
