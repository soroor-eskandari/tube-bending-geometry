from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.lines import Line2D


st.set_page_config(
    page_title="QRF Split Visualization",
    layout="wide",
)

st.title("QRF Split Prediction Interval Visualization")

project_root = Path(__file__).resolve().parent
prediction_path = (
    project_root / "data" / "qrf" / "split" / "various_splits.parquet"
)
legacy_prediction_path = (
    project_root / "data" / "qrf" / "split" / "various_splits.csv"
)

DATASET_LABELS = {
    "real": "Real geometry",
    "within_group_interpolation_raw": "Within group interpolation",
    "sensor_augmented_noise__time_wrapping__scaling__jittering": (
        "Sensor augmented"
    ),
}

TARGETS = [
    "Main-axis [mm]",
    "Secondary-axis [mm]",
]


def dataset_label(geometry_source: str) -> str:
    return DATASET_LABELS.get(
        geometry_source,
        geometry_source.replace("__", " + ").replace("_", " ").title(),
    )


def format_group(group_id) -> str:
    return f"Group {int(group_id)}"


def format_experiment(experiment_id) -> str:
    if experiment_id == "All test experiments":
        return experiment_id
    return f"Experiment {int(experiment_id)}"


@st.cache_data(show_spinner="Loading stored QRF split predictions...")
def load_prediction_data(path: str, file_mtime: float) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df.columns = [column.strip() for column in df.columns]
    if "y_pred_median" not in df.columns and "y_pred_mean" in df.columns:
        df = df.rename(columns={"y_pred_mean": "y_pred_median"})
    df["inside_interval"] = df["inside_interval"].astype(bool)
    return df


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


def with_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["inside_interval"] = (
        (df["y_true"] >= df["y_pred_lower"])
        & (df["y_true"] <= df["y_pred_upper"])
    )
    df["interval_width"] = df["y_pred_upper"] - df["y_pred_lower"]
    df["median_error"] = df["y_pred_median"] - df["y_true"]
    df["abs_median_error"] = df["median_error"].abs()
    df["squared_median_error"] = df["median_error"] ** 2
    df["outside_interval"] = ~df["inside_interval"]
    return df


def summarize_predictions(
    df: pd.DataFrame,
    group_columns: list[str] | None = None,
) -> pd.DataFrame:
    expected_coverage = 0.90
    df = with_error_columns(df)

    if group_columns:
        grouped = df.groupby(group_columns, dropna=False)
        summary = grouped.agg(
            n_points=("y_true", "size"),
            n_groups=("Group_ID", "nunique"),
            n_experiments=("Experiment_ID", "nunique"),
            coverage=("inside_interval", "mean"),
            mean_interval_width=("interval_width", "mean"),
            median_interval_width=("interval_width", "median"),
            mae_median=("abs_median_error", "mean"),
            mse_median=("squared_median_error", "mean"),
            bias_mean=("median_error", "mean"),
            bias_median=("median_error", "median"),
            outside_interval_count=("outside_interval", "sum"),
        ).reset_index()
    else:
        summary = pd.DataFrame(
            [
                {
                    "n_points": len(df),
                    "n_groups": df["Group_ID"].nunique(),
                    "n_experiments": df["Experiment_ID"].nunique(),
                    "coverage": df["inside_interval"].mean(),
                    "mean_interval_width": df["interval_width"].mean(),
                    "median_interval_width": df["interval_width"].median(),
                    "mae_median": df["abs_median_error"].mean(),
                    "mse_median": df["squared_median_error"].mean(),
                    "bias_mean": df["median_error"].mean(),
                    "bias_median": df["median_error"].median(),
                    "outside_interval_count": df["outside_interval"].sum(),
                }
            ]
        )

    summary["expected_coverage"] = expected_coverage
    summary["calibration_error"] = (
        summary["coverage"] - expected_coverage
    ).abs()
    summary["rmse_median"] = np.sqrt(summary["mse_median"])
    summary["outside_interval_percent"] = 1 - summary["coverage"]
    summary["diagnostic_score"] = (
        summary["rmse_median"]
        + 10 * summary["calibration_error"]
        + 0.5 * summary["mean_interval_width"]
        + summary["bias_mean"].abs()
    )
    summary = summary.drop(columns=["mse_median"])

    metric_columns = [
        "coverage",
        "expected_coverage",
        "calibration_error",
        "mean_interval_width",
        "median_interval_width",
        "rmse_median",
        "mae_median",
        "bias_mean",
        "bias_median",
        "outside_interval_percent",
        "diagnostic_score",
    ]
    for column in metric_columns:
        summary[column] = summary[column].astype(float).round(4)

    count_columns = [
        "n_points",
        "n_groups",
        "n_experiments",
        "outside_interval_count",
    ]
    for column in count_columns:
        summary[column] = summary[column].astype(int)

    return summary


def ordered_summary_columns(
    df: pd.DataFrame,
    leading_columns: list[str] | None = None,
) -> pd.DataFrame:
    leading_columns = leading_columns or []
    metric_columns = [
        "n_points",
        "n_groups",
        "n_experiments",
        "coverage",
        "expected_coverage",
        "calibration_error",
        "mean_interval_width",
        "median_interval_width",
        "rmse_median",
        "mae_median",
        "bias_mean",
        "bias_median",
        "outside_interval_count",
        "outside_interval_percent",
        "diagnostic_score",
    ]
    columns = leading_columns + [
        column for column in metric_columns if column in df.columns
    ]
    return df[columns]


def render_test_set_statistics(df: pd.DataFrame):
    st.markdown("### Test-Set Statistics")

    target_summary = summarize_predictions(
        df,
        group_columns=["target"],
    )
    target_summary = target_summary.sort_values(
        ["diagnostic_score", "calibration_error", "rmse_median"],
        ascending=[False, False, False],
    )
    st.dataframe(
        ordered_summary_columns(target_summary, ["target"]),
        use_container_width=True,
        hide_index=True,
    )

    group_tab, experiment_tab, outside_tab = st.tabs(
        ["Worst Groups", "Worst Experiments", "Outside Interval Rows"]
    )

    with group_tab:
        group_summary = summarize_predictions(
            df,
            group_columns=["target", "Group_ID"],
        )
        group_summary = group_summary.sort_values(
            ["diagnostic_score", "calibration_error", "rmse_median"],
            ascending=[False, False, False],
        )
        st.dataframe(
            ordered_summary_columns(
                group_summary.head(20),
                ["target", "Group_ID"],
            ),
            use_container_width=True,
            hide_index=True,
        )

    with experiment_tab:
        experiment_summary = summarize_predictions(
            df,
            group_columns=["target", "Group_ID", "Experiment_ID"],
        )
        experiment_summary = experiment_summary.sort_values(
            ["diagnostic_score", "calibration_error", "rmse_median"],
            ascending=[False, False, False],
        )
        st.dataframe(
            ordered_summary_columns(
                experiment_summary.head(20),
                ["target", "Group_ID", "Experiment_ID"],
            ),
            use_container_width=True,
            hide_index=True,
        )

    with outside_tab:
        outside_df = with_error_columns(df)
        outside_df = outside_df[outside_df["outside_interval"]].copy()
        if outside_df.empty:
            st.info("No rows are outside the prediction interval.")
            return

        outside_df["abs_median_error"] = outside_df[
            "abs_median_error"
        ].round(4)
        outside_df["median_error"] = outside_df["median_error"].round(4)
        outside_df["interval_width"] = outside_df["interval_width"].round(4)
        st.dataframe(
            outside_df[
                [
                    "target",
                    "Group_ID",
                    "Experiment_ID",
                    "angle",
                    "y_true",
                    "y_pred_median",
                    "y_pred_lower",
                    "y_pred_upper",
                    "median_error",
                    "abs_median_error",
                    "interval_width",
                ]
            ]
            .sort_values("abs_median_error", ascending=False)
            .head(100),
            use_container_width=True,
            hide_index=True,
        )


class QRFStoredSplitVisualizer:
    def __init__(
        self,
        prediction_df: pd.DataFrame,
        angle_col: str,
        visualization_mode: str,
    ):
        self.prediction_df = prediction_df
        self.angle_col = angle_col
        self.visualization_mode = visualization_mode

    def prepare_data(self, target_name: str):
        target_predictions = self.prediction_df[
            self.prediction_df["target"] == target_name
        ].copy()

        grouped = (
            target_predictions.groupby(self.angle_col)
            .agg(
                y_true=("y_true", "median"),
                y_pred_median=("y_pred_median", "median"),
                y_pred_lower=("y_pred_lower", "median"),
                y_pred_upper=("y_pred_upper", "median"),
            )
            .reset_index()
            .sort_values(self.angle_col)
        )

        return grouped, target_predictions

    def plot_target(self, target_name: str):
        df, raw_target_df = self.prepare_data(target_name)

        if df.empty:
            st.warning(f"No rows found for `{target_name}`.")
            return

        x = df[self.angle_col].values
        y_true = df["y_true"].values
        y_pred_median = df["y_pred_median"].values
        y_pred_lower = df["y_pred_lower"].values
        y_pred_upper = df["y_pred_upper"].values

        y_pred_lower = (
            pd.Series(y_pred_lower)
            .rolling(window=3, center=True, min_periods=1)
            .median()
            .values
        )
        y_pred_upper = (
            pd.Series(y_pred_upper)
            .rolling(window=3, center=True, min_periods=1)
            .median()
            .values
        )
        y_pred_median = (
            pd.Series(y_pred_median)
            .rolling(window=3, center=True, min_periods=1)
            .median()
            .values
        )

        raw_inside_mask = (
            (raw_target_df["y_true"] >= raw_target_df["y_pred_lower"])
            & (raw_target_df["y_true"] <= raw_target_df["y_pred_upper"])
        )
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
        ax.plot(x, y_pred_median, color="#FF8C00", linewidth=2.5, label="Prediction")

        if self.visualization_mode == "Raw points":
            self._plot_raw_points(
                ax=ax,
                x=x,
                y_true=y_true,
                raw_target_df=raw_target_df,
                raw_inside_mask=raw_inside_mask,
            )
        else:
            self._plot_experiment_curves(
                ax=ax,
                raw_target_df=raw_target_df,
                raw_inside_mask=raw_inside_mask,
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

        ax.legend(
            handles=self._legend_elements(),
            frameon=True,
            facecolor="white",
            edgecolor="lightgray",
        )
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("### Coverage")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Coverage", f"{metrics['coverage']:.3f}")
        c2.metric("Expected Coverage", f"{metrics['expected_coverage']:.3f}")
        c3.metric("Calibration Error", f"{metrics['calibration_error']:.3f}")
        c4.metric("Mean Interval Width", f"{metrics['mean_interval_width']:.3f}")
        self._render_statistics(raw_target_df)

    def _render_statistics(self, raw_target_df: pd.DataFrame):
        st.markdown("### Statistical Summary")

        global_summary = ordered_summary_columns(
            summarize_predictions(raw_target_df)
        )
        st.dataframe(global_summary, use_container_width=True, hide_index=True)

        group_tab, experiment_tab, outside_tab = st.tabs(
            ["By Group", "By Experiment", "Outside Interval"]
        )

        with group_tab:
            group_summary = summarize_predictions(
                raw_target_df,
                group_columns=["Group_ID"],
            )
            group_summary = group_summary.sort_values(
                ["diagnostic_score", "calibration_error", "rmse_median"],
                ascending=[False, False, False],
            )
            st.dataframe(
                ordered_summary_columns(group_summary, ["Group_ID"]),
                use_container_width=True,
                hide_index=True,
            )

        with experiment_tab:
            experiment_summary = summarize_predictions(
                raw_target_df,
                group_columns=["Group_ID", "Experiment_ID"],
            )
            experiment_summary = experiment_summary.sort_values(
                ["diagnostic_score", "calibration_error", "rmse_median"],
                ascending=[False, False, False],
            )
            st.dataframe(
                ordered_summary_columns(
                    experiment_summary,
                    ["Group_ID", "Experiment_ID"],
                ),
                use_container_width=True,
                hide_index=True,
            )

        with outside_tab:
            outside_df = with_error_columns(raw_target_df)
            outside_df = outside_df[outside_df["outside_interval"]].copy()
            if outside_df.empty:
                st.info("No rows are outside the prediction interval.")
            else:
                outside_df["abs_median_error"] = outside_df[
                    "abs_median_error"
                ].round(4)
                outside_df["median_error"] = outside_df["median_error"].round(4)
                outside_df["interval_width"] = outside_df[
                    "interval_width"
                ].round(4)
                st.dataframe(
                    outside_df[
                        [
                            "Group_ID",
                            "Experiment_ID",
                            "angle",
                            "y_true",
                            "y_pred_median",
                            "y_pred_lower",
                            "y_pred_upper",
                            "median_error",
                            "abs_median_error",
                            "interval_width",
                        ]
                    ].sort_values("abs_median_error", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                )

    def _plot_raw_points(
        self,
        ax,
        x,
        y_true,
        raw_target_df: pd.DataFrame,
        raw_inside_mask: pd.Series,
    ):
        raw_target_df = raw_target_df.copy()
        raw_target_df["display_angle"] = (
            raw_target_df["angle"]
            + np.random.default_rng(42).normal(0, 0.08, len(raw_target_df))
        )
        raw_inside_df = raw_target_df[raw_inside_mask]
        raw_outside_df = raw_target_df[~raw_inside_mask]

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

    def _plot_experiment_curves(
        self,
        ax,
        raw_target_df: pd.DataFrame,
        raw_inside_mask: pd.Series,
    ):
        experiment_ids = sorted(
            raw_target_df["Experiment_ID"].dropna().astype(int).unique().tolist()
        )
        color_map = plt.get_cmap("tab20")
        raw_target_df = raw_target_df.copy()
        raw_target_df["inside_interval"] = raw_inside_mask

        for color_index, experiment_id in enumerate(experiment_ids):
            experiment_df = (
                raw_target_df[
                    raw_target_df["Experiment_ID"].astype(int) == int(experiment_id)
                ]
                .sort_values(self.angle_col)
                .copy()
            )
            color = color_map(color_index % color_map.N)
            label = (
                f"Experiment {experiment_id}"
                if len(experiment_ids) <= 12
                else "_nolegend_"
            )

            ax.plot(
                experiment_df[self.angle_col],
                experiment_df["y_true"],
                color=color,
                linewidth=1.6,
                marker="o",
                markersize=3,
                alpha=0.9,
                label=label,
                zorder=8,
            )

            outside_df = experiment_df[~experiment_df["inside_interval"]]
            if not outside_df.empty:
                ax.scatter(
                    outside_df[self.angle_col],
                    outside_df["y_true"],
                    color="red",
                    s=20,
                    marker="o",
                    alpha=0.9,
                    label="_nolegend_",
                    zorder=10,
                )

        if len(experiment_ids) > 12:
            ax.text(
                0.01,
                0.98,
                f"{len(experiment_ids)} experiment curves",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=10,
                color="#333333",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "lightgray",
                    "alpha": 0.85,
                },
            )

    def _legend_elements(self):
        legend_elements = [
            Line2D(
                [0],
                [0],
                color="#4C72B0",
                lw=10,
                alpha=0.22,
                label="Prediction Interval",
            ),
            Line2D([0], [0], color="#FF8C00", lw=2.5, label="Prediction"),
        ]

        if self.visualization_mode == "Raw points":
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color="#025BFF",
                    lw=2,
                    linestyle="--",
                    label="Actual",
                )
            )
        else:
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color="#3A58C3",
                    lw=1.6,
                    marker="o",
                    markersize=4,
                    label="Actual per Experiment",
                )
            )

        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="red",
                markersize=7,
                label="Outside Interval",
            )
        )

        return legend_elements


if not prediction_path.exists() and legacy_prediction_path.exists():
    prediction_path = legacy_prediction_path

if not prediction_path.exists():
    st.warning(f"Missing stored QRF split prediction file: {prediction_path}")
    st.stop()

prediction_df = load_prediction_data(
    str(prediction_path),
    prediction_path.stat().st_mtime,
)

st.sidebar.header("Controls")

geometry_sources = sorted(prediction_df["geometry_source"].dropna().unique().tolist())
geometry_source = st.sidebar.selectbox(
    "Dataset",
    geometry_sources,
    format_func=dataset_label,
)

dataset_df = prediction_df[
    prediction_df["geometry_source"] == geometry_source
].copy()

split_options = (
    dataset_df[["split_index", "split_name", "split_column", "test_value"]]
    .drop_duplicates()
    .sort_values(["split_index"])
)

split_labels = {
    int(row.split_index): (
        f"{int(row.split_index):03d} | {row.split_name}"
    )
    for row in split_options.itertuples(index=False)
}

selected_split_index = st.sidebar.selectbox(
    "Split",
    split_options["split_index"].astype(int).tolist(),
    format_func=lambda split_index: split_labels[int(split_index)],
)

split_df = dataset_df[
    dataset_df["split_index"].astype(int) == int(selected_split_index)
].copy()

available_groups = sorted(
    split_df["Group_ID"].dropna().astype(int).unique().tolist()
)

selected_group = st.sidebar.selectbox(
    "Test group",
    available_groups,
    format_func=format_group,
)

group_df = split_df[
    split_df["Group_ID"].astype(int) == int(selected_group)
].copy()

experiment_options = ["All test experiments"] + sorted(
    group_df["Experiment_ID"].dropna().astype(int).unique().tolist()
)

selected_experiment = st.sidebar.selectbox(
    "Experiment",
    experiment_options,
    format_func=format_experiment,
)

visualization_mode = st.sidebar.radio(
    "Visualization",
    ["Raw points", "Experiment curves"],
    horizontal=False,
)

selected_df = group_df.copy()
if selected_experiment != "All test experiments":
    selected_df = selected_df[
        selected_df["Experiment_ID"].astype(int) == int(selected_experiment)
    ].copy()

selected_df = selected_df[
    (selected_df["angle"] >= 0)
    & (selected_df["angle"] <= 44)
].copy()

if selected_df.empty:
    st.warning("No prediction rows found for the selected filters.")
    st.stop()

split_meta = split_options[
    split_options["split_index"].astype(int) == int(selected_split_index)
].iloc[0]

st.caption(
    "Showing "
    f"`{dataset_label(geometry_source)}` | "
    f"`{split_meta['split_name']}` | "
    f"{format_group(selected_group)}"
)

metadata_cols = st.columns(4)
metadata_cols[0].metric("Rows", f"{len(selected_df):,}")
metadata_cols[1].metric(
    "Experiments",
    selected_df["Experiment_ID"].nunique(),
)
metadata_cols[2].metric(
    "Test Groups",
    split_df["Group_ID"].nunique(),
)
metadata_cols[3].metric(
    "Test Experiments",
    split_df["Experiment_ID"].nunique(),
)

with st.expander("Split Metadata"):
    split_metadata = (
        split_df[
            [
                "split_index",
                "split_name",
                "split_column",
                "test_value",
                "test_group_ids",
                "test_experiment_ids",
                "train_group_ids",
                "train_experiment_ids",
                "test_experiment_percent",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    st.dataframe(split_metadata, use_container_width=True)

split_statistics_df = split_df[
    (split_df["angle"] >= 0)
    & (split_df["angle"] <= 44)
].copy()
with st.expander("Test-Set Statistics", expanded=True):
    render_test_set_statistics(split_statistics_df)

visualizer = QRFStoredSplitVisualizer(
    prediction_df=selected_df,
    angle_col="angle",
    visualization_mode=visualization_mode,
)

for target in TARGETS:
    st.subheader(target)
    visualizer.plot_target(target)
