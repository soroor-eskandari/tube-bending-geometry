from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.lines import Line2D


st.set_page_config(
    page_title="Interval Prediction UI",
    layout="wide",
)

st.title("Interval Prediction UI")

project_root = Path(__file__).resolve().parent

MODEL_OPTIONS = {
    "Quantile Random Forest": {
        "key": "qrf",
        "title": "QRF Uncertainty Estimation",
    },
    "Heteroscedastic Gaussian Process": {
        "key": "hgp",
        "title": "HGP Uncertainty Estimation",
    },
    "Stochastic Kriging": {
        "key": "sk",
        "title": "Stochastic Kriging Uncertainty Estimation",
    },
}

DATA_SOURCE_OPTIONS = {
    "Measured geometry only": "real",
    "RF augmented geometry with exact feature values": "augmented_real",
    "RF augmented geometry with sampled feature values": "sampled",
}

TARGETS = [
    "Main-axis [mm]",
    "Secondary-axis [mm]",
]


st.sidebar.header("Controls")

selected_model_label = st.sidebar.radio(
    "Interval model",
    list(MODEL_OPTIONS.keys()),
)

selected_source_label = st.sidebar.radio(
    "Evaluation data source",
    list(DATA_SOURCE_OPTIONS.keys()),
)

model_key = MODEL_OPTIONS[selected_model_label]["key"]
model_title = MODEL_OPTIONS[selected_model_label]["title"]
data_source = DATA_SOURCE_OPTIONS[selected_source_label]

result_dir = (
    project_root
    / "src"
    / "pipeline"
    / "ml"
    / model_key
    / "result"
)

prediction_path = result_dir / f"{model_key}_prediction_details_{data_source}.csv"
metrics_path = result_dir / f"{model_key}_per_angle_metrics_{data_source}.csv"


@st.cache_data
def load_csv(path, file_mtime):
    return pd.read_csv(path)


if not prediction_path.exists():
    st.warning(f"Missing prediction file: {prediction_path}")
    st.stop()

if not metrics_path.exists():
    st.warning(f"Missing per-angle metrics file: {metrics_path}")
    st.stop()

prediction_df = load_csv(
    prediction_path,
    prediction_path.stat().st_mtime,
)
metrics_df = load_csv(
    metrics_path,
    metrics_path.stat().st_mtime,
)

prediction_df.columns = [column.strip() for column in prediction_df.columns]
metrics_df.columns = [column.strip() for column in metrics_df.columns]

angle_col = "angle"
if angle_col not in prediction_df.columns:
    prediction_df = prediction_df.rename(
        columns={
            "Angle[degree]": angle_col,
        }
    )

prediction_df = prediction_df[
    (prediction_df[angle_col] >= 0)
    & (prediction_df[angle_col] <= 44)
].copy()

st.caption(
    f"Model: `{selected_model_label}` | "
    f"Data source: `{selected_source_label}` | "
    "Displayed rows come from the saved 20% evaluation split."
)


class IntervalPredictionVisualizer:
    def __init__(
        self,
        prediction_df,
        metrics_df,
        angle_col,
        model_title,
    ):
        self.prediction_df = prediction_df
        self.metrics_df = metrics_df
        self.angle_col = angle_col
        self.model_title = model_title

    def prepare_data(self, target_name):
        target_predictions = self.prediction_df[
            self.prediction_df["target"] == target_name
        ].copy()

        grouped = (
            target_predictions
            .groupby(self.angle_col)
            .agg(
                y_true=("y_true", "mean"),
                y_pred_mean=("y_pred_mean", "mean"),
                y_pred_lower=("y_pred_lower", "mean"),
                y_pred_upper=("y_pred_upper", "mean"),
            )
            .reset_index()
            .sort_values(self.angle_col)
        )

        for optional_column in [
            "mean_gp_std",
            "noise_gp_variance",
            "surface_std",
            "intrinsic_variance",
            "total_std",
        ]:
            if optional_column in target_predictions.columns:
                optional_stats = (
                    target_predictions
                    .groupby(self.angle_col)[optional_column]
                    .mean()
                    .reset_index()
                )
                grouped = grouped.merge(
                    optional_stats,
                    on=self.angle_col,
                    how="left",
                )

        metrics = self.metrics_df[
            self.metrics_df["target"] == target_name
        ].copy()

        return grouped, metrics

    def plot_target(self, target_name):
        df, target_metrics = self.prepare_data(target_name)

        if df.empty:
            st.info(f"No prediction rows found for {target_name}.")
            return

        x = df[self.angle_col].values
        y_true = df["y_true"].values
        y_pred_mean = self._smooth(df["y_pred_mean"].values)
        y_pred_lower = self._smooth(df["y_pred_lower"].values)
        y_pred_upper = self._smooth(df["y_pred_upper"].values)

        raw_target_df = self.prediction_df[
            self.prediction_df["target"] == target_name
        ].copy()
        raw_target_df = raw_target_df[
            (raw_target_df[self.angle_col] >= 0)
            & (raw_target_df[self.angle_col] <= 44)
        ].copy()
        raw_target_df["display_angle"] = (
            raw_target_df[self.angle_col]
            + np.random.default_rng(42).normal(0, 0.08, len(raw_target_df))
        )

        visible_lower = np.interp(
            raw_target_df["display_angle"],
            x,
            y_pred_lower,
        )
        visible_upper = np.interp(
            raw_target_df["display_angle"],
            x,
            y_pred_upper,
        )
        inside_mask = (
            (raw_target_df["y_true"] >= visible_lower)
            & (raw_target_df["y_true"] <= visible_upper)
        )
        raw_inside_df = raw_target_df[inside_mask]
        raw_outside_df = raw_target_df[~inside_mask]

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(14, 6))

        ax.fill_between(
            x,
            y_pred_lower,
            y_pred_upper,
            color="#4C72B0",
            alpha=0.22,
        )
        ax.plot(
            x,
            y_pred_mean,
            color="#FF8C00",
            linewidth=2.5,
        )
        ax.plot(
            x,
            y_true,
            color="#025BFF",
            linewidth=2,
            linestyle="--",
            alpha=0.9,
        )
        ax.scatter(
            raw_inside_df["display_angle"],
            raw_inside_df["y_true"],
            color="#3A58C3",
            s=14,
            marker="o",
            alpha=0.2,
            zorder=8,
        )
        ax.scatter(
            raw_outside_df["display_angle"],
            raw_outside_df["y_true"],
            color="red",
            s=18,
            marker="o",
            alpha=0.85,
            zorder=10,
        )

        ax.set_title(
            f"{target_name} - {self.model_title}",
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
                lw=2.5,
                label="Prediction",
            ),
            Line2D(
                [0],
                [0],
                color="#025BFF",
                lw=2,
                linestyle="--",
                label="Actual",
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
        ax.legend(
            handles=legend_elements,
            frameon=True,
            facecolor="white",
            edgecolor="lightgray",
        )

        fig.tight_layout()
        st.pyplot(fig)

        self._show_metrics(target_metrics, df)

    @staticmethod
    def _smooth(values):
        return (
            pd.Series(values)
            .rolling(window=3, center=True, min_periods=1)
            .mean()
            .values
        )

    @staticmethod
    def _show_metrics(metrics_df, plot_df):
        if metrics_df.empty:
            st.info("No metrics found for this target.")
            return

        st.markdown("### Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Coverage", f"{metrics_df['coverage'].mean():.3f}")
        c2.metric(
            "Expected Coverage",
            f"{metrics_df['expected_coverage'].mean():.3f}",
        )
        c3.metric(
            "Calibration Error",
            f"{metrics_df['calibration_error'].mean():.3f}",
        )
        c4.metric(
            "Mean Interval Width",
            f"{metrics_df['mean_interval_width'].mean():.3f}",
        )

        extra_metrics = []
        if "mean_gp_std" in plot_df.columns:
            extra_metrics.append(
                ("Mean GP Std", plot_df["mean_gp_std"].mean())
            )
        if "noise_gp_variance" in plot_df.columns:
            extra_metrics.append(
                ("Mean Noise Std", np.sqrt(plot_df["noise_gp_variance"]).mean())
            )
        if "surface_std" in plot_df.columns:
            extra_metrics.append(
                ("Mean Surface Std", plot_df["surface_std"].mean())
            )
        if "intrinsic_variance" in plot_df.columns:
            extra_metrics.append(
                ("Mean Intrinsic Std", np.sqrt(plot_df["intrinsic_variance"]).mean())
            )
        if "total_std" in plot_df.columns:
            extra_metrics.append(
                ("Mean Total Std", plot_df["total_std"].mean())
            )

        if extra_metrics:
            columns = st.columns(len(extra_metrics))
            for column, (label, value) in zip(columns, extra_metrics):
                column.metric(label, f"{value:.4f}")


visualizer = IntervalPredictionVisualizer(
    prediction_df=prediction_df,
    metrics_df=metrics_df,
    angle_col=angle_col,
    model_title=model_title,
)

for target in TARGETS:
    st.subheader(target)
    visualizer.plot_target(target)
