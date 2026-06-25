from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.lines import Line2D

from src.pipeline.ml.qrf.data_splittor import DataSplittor
from src.pipeline.ml.qrf.geometry_data_preprocessor import GeometryPreprocessor
from src.pipeline.ml.qrf.qrf_pipeline import geometry_source_paths
from src.pipeline.rf_augmentation.io_utils import read_table
from src.pipeline.rf_augmentation.sensor_data_augmentor import SensorDataAugmentor


st.set_page_config(
    page_title="QRF Visualization",
    layout="wide",
)

st.title("QRF Prediction Interval Visualization")

project_root = Path(__file__).resolve().parent
result_dir = project_root / "src" / "pipeline" / "ml" / "qrf" / "result"
model_dir = project_root / "src" / "pipeline" / "ml" / "model"

FEATURE_VALUE_OPTIONS = {
    "exact": "exact",
    "within-group-sampling": "within_group_sampling",
    "overall-sampling": "overall_sampling",
}


def mode_label(mode: str) -> str:
    return mode


def source_from_controls(feature_value: str, augmentation_mode: str) -> str:
    feature_suffix = FEATURE_VALUE_OPTIONS[feature_value]
    sensor_suffix = SensorDataAugmentor.mode_to_suffix(augmentation_mode)
    return f"{feature_suffix}_{sensor_suffix}"


def format_group(group_id) -> str:
    return f"Group {int(group_id)}"


@st.cache_data
def load_csv(path, file_mtime):
    return pd.read_csv(path)


@st.cache_data
def load_split_metadata(geometry_source: str, geometry_mtime: float):
    geometry_path = geometry_source_paths(project_root)[geometry_source]
    geometry_df = read_table(geometry_path)
    bending_df = pd.read_csv(
        project_root / "data" / "processed" / "processed_bending_setup.csv"
    )
    geometry_clean = GeometryPreprocessor.preprocess(
        geometry_df=geometry_df,
        bending_df=bending_df,
    )
    train_df, test_df = DataSplittor.splittor(
        geometry_df=geometry_clean,
        unique_bending_df=bending_df,
        test_size=0.2,
        random_state=42,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def attach_test_metadata(prediction_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    prediction_df = prediction_df.copy().reset_index(drop=True)
    metadata_cols = [
        "Group_ID",
        "Experiment_ID",
        "Angle[degree]ORDistance[mm]",
    ]
    metadata = test_df[metadata_cols].reset_index(drop=True)

    enriched_targets = []
    for target_name, target_predictions in prediction_df.groupby("target", sort=False):
        target_predictions = target_predictions.reset_index(drop=True)
        if len(target_predictions) != len(metadata):
            st.warning(
                f"Cannot attach group metadata for `{target_name}` because "
                f"prediction rows ({len(target_predictions)}) do not match "
                f"test rows ({len(metadata)})."
            )
            target_predictions["Group_ID"] = np.nan
            target_predictions["Experiment_ID"] = np.nan
            enriched_targets.append(target_predictions)
            continue

        enriched = pd.concat(
            [
                target_predictions,
                metadata[["Group_ID", "Experiment_ID"]],
            ],
            axis=1,
        )
        enriched_targets.append(enriched)

    return pd.concat(enriched_targets, ignore_index=True)


def compute_metrics(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {
            "coverage": np.nan,
            "expected_coverage": 0.90,
            "calibration_error": np.nan,
            "mean_interval_width": np.nan,
        }

    inside = (
        (df["y_true"] >= df["y_pred_lower"])
        & (df["y_true"] <= df["y_pred_upper"])
    )
    coverage = inside.mean()
    expected_coverage = 0.90
    return {
        "coverage": coverage,
        "expected_coverage": expected_coverage,
        "calibration_error": abs(coverage - expected_coverage),
        "mean_interval_width": (df["y_pred_upper"] - df["y_pred_lower"]).mean(),
    }


@st.cache_resource
def load_qrf_models(geometry_source: str, main_mtime: float, secondary_mtime: float):
    paper_name = f"qrf_geometry_{geometry_source}"
    main_model = joblib.load(model_dir / f"qrf_main_axis_{paper_name}.pkl")
    secondary_model = joblib.load(model_dir / f"qrf_secondary_axis_{paper_name}.pkl")
    return main_model, secondary_model


def predict_split_group(split_df: pd.DataFrame, geometry_source: str) -> pd.DataFrame:
    paper_name = f"qrf_geometry_{geometry_source}"
    main_model_path = model_dir / f"qrf_main_axis_{paper_name}.pkl"
    secondary_model_path = model_dir / f"qrf_secondary_axis_{paper_name}.pkl"

    if not main_model_path.exists() or not secondary_model_path.exists():
        st.warning(f"Missing saved QRF models for `{geometry_source}`.")
        st.stop()

    main_model, secondary_model = load_qrf_models(
        geometry_source,
        main_model_path.stat().st_mtime,
        secondary_model_path.stat().st_mtime,
    )

    angle_col = "Angle[degree]ORDistance[mm]"
    X = split_df[[angle_col]]

    def predict_target(model, target_col: str, target_name: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "angle": split_df[angle_col].to_numpy(),
                "y_true": split_df[target_col].to_numpy(),
                "y_pred_mean": np.asarray(model.predict(X, quantiles=0.50)).ravel(),
                "y_pred_lower": np.asarray(model.predict(X, quantiles=0.05)).ravel(),
                "y_pred_upper": np.asarray(model.predict(X, quantiles=0.95)).ravel(),
                "target": target_name,
                "Group_ID": split_df["Group_ID"].to_numpy(),
                "Experiment_ID": split_df["Experiment_ID"].to_numpy(),
            }
        )

    return pd.concat(
        [
            predict_target(main_model, "Main-axis [mm]", "Main-axis [mm]"),
            predict_target(
                secondary_model,
                "Secondary-axis [mm]",
                "Secondary-axis [mm]",
            ),
        ],
        ignore_index=True,
    )


st.sidebar.header("Controls")

feature_value = st.sidebar.radio(
    "Feature values",
    list(FEATURE_VALUE_OPTIONS),
)

augmentation_mode = st.sidebar.selectbox(
    "Feature augmentation",
    SensorDataAugmentor.all_augmentation_modes(),
    format_func=mode_label,
)

geometry_source = source_from_controls(
    feature_value=feature_value,
    augmentation_mode=augmentation_mode,
)

st.sidebar.markdown(
    "Showing "
    f"`{feature_value}` feature values with `{augmentation_mode}` augmentation."
)

prediction_path = result_dir / f"qrf_prediction_details_{geometry_source}.csv"
geometry_path = geometry_source_paths(project_root).get(geometry_source)

if not prediction_path.exists():
    st.warning(f"Missing QRF prediction file: {prediction_path}")
    st.stop()

if geometry_path is None or not geometry_path.exists():
    st.warning(f"Missing geometry source file for `{geometry_source}`.")
    st.stop()

prediction_df = load_csv(prediction_path, prediction_path.stat().st_mtime)
prediction_df.columns = [c.strip() for c in prediction_df.columns]

train_df, test_df = load_split_metadata(
    geometry_source,
    geometry_path.stat().st_mtime,
)

split_name = st.sidebar.radio(
    "Split",
    ["test", "train"],
    horizontal=True,
)

split_df = test_df if split_name == "test" else train_df
available_groups = sorted(split_df["Group_ID"].dropna().astype(int).unique().tolist())

selected_group = st.sidebar.selectbox(
    "Group number",
    available_groups,
    format_func=format_group,
)

st.caption(
    f"Showing `{geometry_source}` for `{split_name}` {format_group(selected_group)}."
)

selected_split_df = split_df[
    split_df["Group_ID"].astype(int) == int(selected_group)
].copy()

if split_name == "test":
    prediction_df = attach_test_metadata(prediction_df, test_df)
else:
    prediction_df = predict_split_group(selected_split_df, geometry_source)

prediction_df = prediction_df[
    (prediction_df["Group_ID"].astype("Int64") == int(selected_group))
    & (prediction_df["angle"] >= 0)
    & (prediction_df["angle"] <= 44)
].copy()

if prediction_df.empty:
    st.warning(f"No prediction rows found for {format_group(selected_group)}.")
    st.stop()

targets = [
    "Main-axis [mm]",
    "Secondary-axis [mm]",
]


class QRFVisualizer:
    def __init__(self, prediction_df, angle_col):
        self.prediction_df = prediction_df
        self.angle_col = angle_col

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

    def plot_target(self, target_name):
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
        ax.plot(x, y_pred_mean, color="#FF8C00", linewidth=2.5, label="Prediction")
        ax.plot(
            x,
            y_true,
            color="#025BFF",
            linewidth=2,
            linestyle="--",
            alpha=0.9,
            label="Actual",
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

        legend_elements = [
            Line2D([0], [0], color="#4C72B0", lw=10, alpha=0.22, label="Prediction Interval"),
            Line2D([0], [0], color="#FF8C00", lw=2.5, label="Prediction"),
            Line2D([0], [0], color="#025BFF", lw=2, linestyle="--", label="Actual"),
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
        ax.legend(handles=legend_elements, frameon=True, facecolor="white", edgecolor="lightgray")
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("### Group Coverage")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Coverage", f"{metrics['coverage']:.3f}")
        c2.metric("Expected Coverage", f"{metrics['expected_coverage']:.3f}")
        c3.metric("Calibration Error", f"{metrics['calibration_error']:.3f}")
        c4.metric("Mean Interval Width", f"{metrics['mean_interval_width']:.3f}")


visualizer = QRFVisualizer(
    prediction_df=prediction_df,
    angle_col="angle",
)

for target in targets:
    st.subheader(target)
    visualizer.plot_target(target)
