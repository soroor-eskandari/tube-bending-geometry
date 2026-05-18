from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.lines import Line2D

# =========================================================
# Streamlit page
# =========================================================
st.set_page_config(
    page_title="SK Visualization",
    layout="wide",
)

st.title(
    "Stochastic Kriging Prediction Interval Visualization"
)

# =========================================================
# File paths
# =========================================================
project_root = Path(__file__).resolve().parent

result_dir = (
    project_root
    / "src"
    / "pipeline"
    / "ml"
    / "sk"
    / "result"
)

prediction_path = (
    result_dir
    / "sk_prediction_details.csv"
)

metrics_path = (
    result_dir
    / "sk_per_angle_metrics.csv"
)

# =========================================================
# Load data
# =========================================================
if not prediction_path.exists() or not metrics_path.exists():
    st.error(
        "SK result files were not found. Run "
        "`python3 3_3_run_sk_interval_predictor.py` first."
    )
    st.stop()

prediction_df = pd.read_csv(
    prediction_path
)

metrics_df = pd.read_csv(
    metrics_path
)

# =========================================================
# Clean column names
# =========================================================
prediction_df.columns = [
    c.strip()
    for c in prediction_df.columns
]

metrics_df.columns = [
    c.strip()
    for c in metrics_df.columns
]

angle_col = "angle"

if angle_col not in prediction_df.columns:
    prediction_df = prediction_df.rename(
        columns={
            "Angle[degree]": angle_col,
        }
    )

# =========================================================
# Filter angle range
# =========================================================
prediction_df = prediction_df[
    (prediction_df[angle_col] >= 0)
    &
    (prediction_df[angle_col] <= 44)
].copy()

# =========================================================
# Targets
# =========================================================
targets = [
    "Main-axis [mm]",
    "Secondary-axis [mm]",
]


# =========================================================
# Plotting class
# =========================================================
class SKVisualizer:

    def __init__(
        self,
        prediction_df,
        metrics_df,
        angle_col,
    ):

        self.prediction_df = prediction_df
        self.metrics_df = metrics_df
        self.angle_col = angle_col

    # =====================================================
    # Prepare visualization dataframe
    # =====================================================
    def prepare_data(
        self,
        target_name,
    ):

        # ---------------------------------------------
        # Prediction rows
        # ---------------------------------------------
        target_predictions = (
            self.prediction_df[
                self.prediction_df["target"]
                == target_name
            ]
            .copy()
        )

        # ---------------------------------------------
        # Aggregate by angle
        # ---------------------------------------------
        grouped = (
            target_predictions
            .groupby(self.angle_col)
            .agg(
                y_true=(
                    "y_true",
                    "mean",
                ),

                y_pred_mean=(
                    "y_pred_mean",
                    "mean",
                ),

                y_pred_lower=(
                    "y_pred_lower",
                    "mean",
                ),

                y_pred_upper=(
                    "y_pred_upper",
                    "mean",
                ),

                surface_std=(
                    "surface_std",
                    "mean",
                ),

                intrinsic_variance=(
                    "intrinsic_variance",
                    "mean",
                ),

                total_std=(
                    "total_std",
                    "mean",
                ),
            )
            .reset_index()
        )

        grouped = grouped.sort_values(
            self.angle_col
        )

        # ---------------------------------------------
        # Metrics
        # ---------------------------------------------
        metrics = (
            self.metrics_df[
                self.metrics_df["target"]
                == target_name
            ]
            .copy()
        )

        return grouped, metrics

    # =====================================================
    # Plot target
    # =====================================================
    def plot_target(
        self,
        target_name,
    ):

        df, metrics_df = self.prepare_data(
            target_name
        )

        # =================================================
        # Aggregated curves
        # =================================================
        x = df[self.angle_col].values

        y_true = (
            df["y_true"]
            .values
        )

        y_pred_mean = (
            df["y_pred_mean"]
            .values
        )

        y_pred_lower = (
            df["y_pred_lower"]
            .values
        )

        y_pred_upper = (
            df["y_pred_upper"]
            .values
        )

        # =================================================
        # Smooth curves
        # =================================================
        y_pred_lower = (
            pd.Series(y_pred_lower)
            .rolling(
                window=3,
                center=True,
                min_periods=1,
            )
            .mean()
            .values
        )

        y_pred_upper = (
            pd.Series(y_pred_upper)
            .rolling(
                window=3,
                center=True,
                min_periods=1,
            )
            .mean()
            .values
        )

        y_pred_mean = (
            pd.Series(y_pred_mean)
            .rolling(
                window=3,
                center=True,
                min_periods=1,
            )
            .mean()
            .values
        )

        # =================================================
        # RAW SAMPLE-LEVEL OUTSIDE POINTS
        # =================================================
        raw_target_df = (
            self.prediction_df[
                self.prediction_df["target"]
                == target_name
            ]
            .copy()
        )

        raw_target_df = raw_target_df[
            (raw_target_df["angle"] >= 0)
            &
            (raw_target_df["angle"] <= 44)
        ]

        # =================================================
        # Add slight horizontal jitter
        # =================================================
        raw_target_df["display_angle"] = (
            raw_target_df["angle"]
            +
            np.random.default_rng(42).normal(
                0,
                0.08,
                len(raw_target_df),
            )
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

        raw_inside_mask = (
            (
                raw_target_df["y_true"]
                >= visible_lower
            )
            &
            (
                raw_target_df["y_true"]
                <= visible_upper
            )
        )

        raw_outside_df = (
            raw_target_df[
                ~raw_inside_mask
            ]
        )

        raw_inside_df = (
            raw_target_df[
                raw_inside_mask
            ]
        )

        # =================================================
        # Styling
        # =================================================
        plt.style.use(
            "seaborn-v0_8-whitegrid"
        )

        fig, ax = plt.subplots(
            figsize=(14, 6)
        )

        # =================================================
        # Prediction interval
        # =================================================
        ax.fill_between(
            x,
            y_pred_lower,
            y_pred_upper,
            color="#4C72B0",
            alpha=0.22,
            label="Prediction Interval",
        )

        # =================================================
        # Prediction line
        # =================================================
        ax.plot(
            x,
            y_pred_mean,
            color="#FF8C00",
            linewidth=2.5,
            label="Prediction",
        )

        # =================================================
        # Actual values
        # =================================================
        ax.plot(
            x,
            y_true,
            color="#025BFF",
            linewidth=2,
            linestyle="--",
            alpha=0.9,
            label="Actual",
        )

        # =================================================
        # Inside interval geometry points
        # =================================================
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

        # =================================================
        # Outside interval points
        # =================================================
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

        # =================================================
        # Metrics
        # =================================================
        coverage = (
            metrics_df["coverage"]
            .mean()
        )

        expected_coverage = (
            metrics_df[
                "expected_coverage"
            ]
            .mean()
        )

        calibration_error = (
            metrics_df[
                "calibration_error"
            ]
            .mean()
        )

        mean_interval_width = (
            metrics_df[
                "mean_interval_width"
            ]
            .mean()
        )

        mean_surface_std = (
            df["surface_std"]
            .mean()
        )

        mean_intrinsic_std = np.sqrt(
            df["intrinsic_variance"]
        ).mean()

        mean_total_std = (
            df["total_std"]
            .mean()
        )

        # =================================================
        # Labels
        # =================================================
        ax.set_title(
            f"{target_name} - "
            f"Stochastic Kriging Uncertainty Estimation",
            fontsize=18,
            pad=15,
            weight="bold",
        )

        ax.set_xlabel(
            "Angle [degree]",
            fontsize=13,
        )

        ax.set_ylabel(
            target_name,
            fontsize=13,
        )

        # =================================================
        # Styling
        # =================================================
        ax.spines["top"].set_visible(
            False
        )

        ax.spines["right"].set_visible(
            False
        )

        ax.grid(alpha=0.2)

        # =================================================
        # Custom legend
        # =================================================
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

        plt.tight_layout()

        st.pyplot(fig)

        # =================================================
        # Streamlit metrics
        # =================================================
        st.markdown(
            "### Metrics"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Coverage",
            f"{coverage:.3f}",
        )

        c2.metric(
            "Expected Coverage",
            f"{expected_coverage:.3f}",
        )

        c3.metric(
            "Calibration Error",
            f"{calibration_error:.3f}",
        )

        c4.metric(
            "Mean Interval Width",
            f"{mean_interval_width:.3f}",
        )

        c5, c6, c7 = st.columns(3)

        c5.metric(
            "Mean Surface Std",
            f"{mean_surface_std:.4f}",
        )

        c6.metric(
            "Mean Intrinsic Std",
            f"{mean_intrinsic_std:.4f}",
        )

        c7.metric(
            "Mean Total Std",
            f"{mean_total_std:.4f}",
        )


# =========================================================
# Create visualizer
# =========================================================
visualizer = SKVisualizer(
    prediction_df=prediction_df,
    metrics_df=metrics_df,
    angle_col=angle_col,
)

# =========================================================
# Main UI
# =========================================================
for target in targets:

    st.subheader(target)

    visualizer.plot_target(target)
