from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components


ROOT = Path(__file__).resolve().parent
ANGLE_COL = "Angle[degree]ORDistance[mm]"
GROUP_COL = "Group_ID"
EXPERIMENT_COL = "Experiment_ID"
MODEL_ROOTS = {
    "QRF": ROOT / "src" / "pipeline" / "ml" / "qrf" / "results" / "models",
    "HGP": ROOT / "src" / "pipeline" / "ml" / "hgp" / "results" / "models",
}
# From the exported split ranking metadata:
# qrf_rank_main == 1 -> split_index 1
# qrf_rank_secondary == 1 -> split_index 8
RANK_ONE_SPLIT_BY_AXIS = {
    "main": 1,
    "secondary": 8,
}
TARGET_ANGLE_DEGREES = 44
MAX_PLOT_ANGLE = 44
GEOMETRY_PATH = ROOT / "data" / "processed" / "geometry.csv"
BENDING_SETUP_PATH = ROOT / "data" / "processed" / "processed_bending_setup.csv"
AXIS_COLUMNS = {
    "Main Axis": "Main-axis [mm]",
    "Secondary Axis": "Secondary-axis [mm]",
}
AXIS_KEYS = {
    "Main Axis": "main",
    "Secondary Axis": "secondary",
}
DATA_SOURCE_LABELS = {
    "real": "Real Experimental Geometry",
    "within_group_interpolation_raw": "Within-group interpolation",
    "sensor_augmented_noise__time_wrapping__scaling__jittering": (
        "Sensor-augmented geometry"
    ),
}
SETUP_COLUMNS = [
    "Collet boost",
    "Pressure-die distance",
    "Mandrel retraction timing",
    "Pressure-die boost",
    "Clamp-die lateral position",
    "Mandrel position",
    "Pressure-die lateral position",
    "Target-angle",
    "Outer-diameter",
    "Wall-thickness",
]
METRIC_COLUMNS = [
    "coverage_percent",
    "mae",
    "rmse",
    "mean_interval_width",
    "median_interval_width",
    "interval_score",
    "coverage_error_percent",
    "negative_log_predictive_density",
    "r2",
    "qrf_score",
    "source_score",
]
METRIC_INFO = {
    "coverage_percent": (
        "Coverage",
        "%",
        "Observed values contained within the exported prediction interval.",
    ),
    "mae": (
        "MAE",
        "mm",
        "Mean absolute error between observed geometry and central prediction.",
    ),
    "rmse": (
        "RMSE",
        "mm",
        "Root mean squared prediction error.",
    ),
    "mean_interval_width": (
        "Mean Interval Width",
        "mm",
        "Average width between the exported lower and upper prediction bounds.",
    ),
    "median_interval_width": (
        "Median Interval Width",
        "mm",
        "Median width between the exported lower and upper prediction bounds.",
    ),
    "interval_score": (
        "Interval Score",
        "",
        "Exported interval diagnostic from the thesis pipeline.",
    ),
    "coverage_error_percent": (
        "Coverage Error",
        "%",
        "Absolute deviation from the exported nominal coverage target.",
    ),
    "negative_log_predictive_density": (
        "Negative Log Predictive Density",
        "",
        "HGP probabilistic diagnostic exported by the thesis pipeline.",
    ),
    "r2": ("R2", "", "Coefficient of determination exported by the pipeline."),
    "qrf_score": ("QRF Score", "", "QRF split score exported by the pipeline."),
    "source_score": (
        "Source Score",
        "",
        "Source split score used by the exported HGP run.",
    ),
}
PRIMARY_METRICS = [
    "coverage_percent",
    "mae",
    "rmse",
    "mean_interval_width",
]
MODEL_COLORS = {
    "Observed": "rgb(35, 35, 35)",
    "QRF": "rgb(37, 99, 235)",
    "HGP": "rgb(190, 88, 44)",
}
PROCESS_PARAMETER_COLUMNS = [
    "Collet boost",
    "Pressure-die distance",
    "Mandrel retraction timing",
    "Pressure-die boost",
    "Clamp-die lateral position",
    "Mandrel position",
    "Pressure-die lateral position",
]
TUBE_PARAMETER_COLUMNS = [
    "Target-angle",
    "Outer-diameter",
    "Wall-thickness",
]


st.set_page_config(
    page_title="Rotary Tube Bending Interval Prediction",
    layout="wide",
)


def make_selectboxes_selection_only() -> None:
    st.markdown(
        """
        <style>
        div[data-baseweb="select"] input {
            caret-color: transparent;
            pointer-events: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    components.html(
        """
        <script>
        const doc = window.parent.document;
        const allowedKeys = new Set([
            "Tab",
            "Escape",
            "Enter",
            "ArrowUp",
            "ArrowDown",
            "ArrowLeft",
            "ArrowRight",
            "Home",
            "End",
        ]);

        function lockSelectboxInputs() {
            doc.querySelectorAll('div[data-baseweb="select"] input').forEach((input) => {
                input.readOnly = true;
                input.setAttribute("inputmode", "none");
                input.setAttribute("autocomplete", "off");

                if (input.dataset.selectionOnly === "true") {
                    return;
                }
                input.dataset.selectionOnly = "true";

                input.addEventListener(
                    "keydown",
                    (event) => {
                        if (!allowedKeys.has(event.key)) {
                            event.preventDefault();
                        }
                    },
                    true,
                );
                input.addEventListener("beforeinput", (event) => event.preventDefault(), true);
                input.addEventListener("paste", (event) => event.preventDefault(), true);
            });
        }

        lockSelectboxInputs();
        new MutationObserver(lockSelectboxInputs).observe(doc.body, {
            childList: true,
            subtree: true,
        });
        </script>
        """,
        height=0,
    )


make_selectboxes_selection_only()


def display_source(source: str) -> str:
    return DATA_SOURCE_LABELS.get(
        source,
        source.replace("__", " + ").replace("_", " ").title(),
    )


def display_axis(axis_col: str) -> str:
    return axis_col.replace("-axis", " Axis").replace(" [", " [")


def metric_label(metric: str) -> str:
    return METRIC_INFO.get(
        metric,
        (metric.replace("_", " ").title(), "", ""),
    )[0]


def metric_unit(metric: str) -> str:
    return METRIC_INFO.get(metric, ("", "", ""))[1]


def metric_description(metric: str) -> str:
    return METRIC_INFO.get(metric, ("", "", ""))[2]


def format_setup_value(value: object) -> str:
    if pd.isna(value):
        return "n/a"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def parse_list(value: object) -> list[int]:
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        items = ast.literal_eval(value)
    else:
        items = []
    return [int(item) for item in items]


def validate_columns(df: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns.difference(df.columns))
    if missing:
        st.error(f"{name} is missing required columns: {', '.join(missing)}")
        st.stop()


@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def read_json(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner="Loading thesis data exports...")
def load_data() -> dict[str, pd.DataFrame]:
    if not GEOMETRY_PATH.exists():
        st.error(f"Missing observed geometry file: {GEOMETRY_PATH.relative_to(ROOT)}")
        st.stop()
    if not BENDING_SETUP_PATH.exists():
        st.error(
            f"Missing bending setup file: {BENDING_SETUP_PATH.relative_to(ROOT)}"
        )
        st.stop()

    geometry = read_csv(str(GEOMETRY_PATH))
    setup = read_csv(str(BENDING_SETUP_PATH))
    validate_columns(
        geometry,
        {EXPERIMENT_COL, ANGLE_COL, "Main-axis [mm]", "Secondary-axis [mm]"},
        "Observed geometry",
    )
    validate_columns(setup, {GROUP_COL, "Experiment_Number"}, "Bending setup")

    experiment_to_group = []
    setup = setup.copy()
    for _, row in setup.iterrows():
        for experiment_id in parse_list(row["Experiment_Number"]):
            experiment_to_group.append(
                {EXPERIMENT_COL: experiment_id, GROUP_COL: int(row[GROUP_COL])}
            )
    mapping = pd.DataFrame(experiment_to_group)
    observed = geometry.merge(mapping, on=EXPERIMENT_COL, how="inner")

    predictions = []
    metrics = []
    metadata = []
    for model, root in MODEL_ROOTS.items():
        if not root.exists():
            continue
        prediction_paths = [
            *root.glob("**/test_predictions.csv"),
            *root.glob("**/all_group_predictions.csv"),
        ]
        for prediction_path in prediction_paths:
            pred = read_csv(str(prediction_path))
            required = {
                GROUP_COL,
                EXPERIMENT_COL,
                ANGLE_COL,
                "geometry_source",
                "target_axis",
                "split_index",
                "split_name",
                "y_true",
                "y_lower",
                "y_upper",
            }
            validate_columns(pred, required, f"{model} predictions")
            pred = pred.copy()
            pred["Model"] = model
            pred["prediction_file"] = str(prediction_path.relative_to(ROOT))
            pred["prediction_scope"] = (
                "All groups"
                if prediction_path.name == "all_group_predictions.csv"
                else "Test groups"
            )
            if "y_median" not in pred.columns and "y_mean" in pred.columns:
                pred["y_median"] = pred["y_mean"]
            predictions.append(pred)

            metric_path = prediction_path.with_name("metrics.csv")
            if metric_path.exists():
                metric = read_csv(str(metric_path)).copy()
                metric["Model"] = model
                metric["prediction_file"] = str(
                    prediction_path.relative_to(ROOT)
                )
                metrics.append(metric)

            metadata_path = prediction_path.with_name("metadata.json")
            if metadata_path.exists():
                meta = read_json(str(metadata_path))
                row = {
                    "Model": model,
                    "geometry_source": meta.get("geometry_source"),
                    "target_axis": meta.get("target_axis"),
                    "split_index": meta.get("split_index"),
                    "split_name": meta.get("split_name"),
                    "model_config": meta.get("model_config", {}),
                    "feature_columns": meta.get("feature_columns", []),
                }
                metadata.append(row)

    if not predictions:
        st.error("No static prediction exports were found under src/pipeline/ml/*/results/models.")
        st.stop()

    predictions_df = pd.concat(predictions, ignore_index=True)
    metrics_df = pd.concat(metrics, ignore_index=True) if metrics else pd.DataFrame()
    metadata_df = pd.DataFrame(metadata)

    return {
        "observed": observed,
        "setup": setup,
        "predictions": predictions_df,
        "metrics": metrics_df,
        "metadata": metadata_df,
    }


def keep_rank_one_splits(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "target_axis" not in df.columns or "split_index" not in df.columns:
        return df

    rank_one_parts = []
    for axis_key, split_index in RANK_ONE_SPLIT_BY_AXIS.items():
        rank_one_parts.append(
            df[
                (df["target_axis"] == axis_key)
                & (pd.to_numeric(df["split_index"], errors="coerce") == split_index)
            ]
        )
    if not rank_one_parts:
        return df.iloc[0:0].copy()
    return pd.concat(rank_one_parts, ignore_index=True)


def selected_prediction_slice(
    predictions: pd.DataFrame,
    model: str,
    source: str,
    axis_key: str,
    group_id: int,
) -> pd.DataFrame:
    required_split = RANK_ONE_SPLIT_BY_AXIS[axis_key]
    data = predictions[
        (predictions["Model"] == model)
        & (predictions["geometry_source"] == source)
        & (predictions["target_axis"] == axis_key)
        & (pd.to_numeric(predictions["split_index"], errors="coerce") == required_split)
        & (predictions["prediction_scope"] == "Test groups")
        & (predictions[GROUP_COL] == group_id)
    ].copy()
    if data.empty:
        return data

    selected_file = sorted(data["prediction_file"].dropna().unique())[0]
    data = data[data["prediction_file"] == selected_file].copy()
    data = data[pd.to_numeric(data[ANGLE_COL], errors="coerce") <= MAX_PLOT_ANGLE]
    return data.sort_values([EXPERIMENT_COL, ANGLE_COL])


def valid_test_groups(predictions: pd.DataFrame, axis_key: str) -> list[int]:
    required_split = RANK_ONE_SPLIT_BY_AXIS[axis_key]
    data = predictions[
        (predictions["target_axis"] == axis_key)
        & (pd.to_numeric(predictions["split_index"], errors="coerce") == required_split)
        & (predictions["prediction_scope"] == "Test groups")
    ]
    return sorted(data[GROUP_COL].dropna().astype(int).unique())


def split_membership(
    predictions: pd.DataFrame,
    model: str,
    source: str,
    axis_key: str,
    group_id: int,
) -> tuple[str, str]:
    required_split = RANK_ONE_SPLIT_BY_AXIS[axis_key]
    split_rows = predictions[
        (predictions["Model"] == model)
        & (predictions["geometry_source"] == source)
        & (predictions["target_axis"] == axis_key)
        & (pd.to_numeric(predictions["split_index"], errors="coerce") == required_split)
        & (predictions["prediction_scope"] == "Test groups")
    ]
    if split_rows.empty:
        return "Unavailable", "No exported rank-1 split"

    test_groups = set(split_rows[GROUP_COL].dropna().astype(int).unique())
    split_names = split_rows["split_name"].dropna().astype(str).unique()
    split_name = split_names[0] if len(split_names) else "rank-1 split"
    status = "Test" if group_id in test_groups else "Train"
    return status, split_name


def observed_geometry_stats(observed: pd.DataFrame, axis_col: str) -> pd.DataFrame:
    columns = [
        ANGLE_COL,
        "observed_min",
        "observed_mean",
        "observed_max",
        "observed_range",
    ]
    if observed.empty:
        return pd.DataFrame(columns=columns)
    stats = (
        observed.groupby(ANGLE_COL, as_index=False)
        .agg(
            observed_min=(axis_col, "min"),
            observed_mean=(axis_col, "mean"),
            observed_max=(axis_col, "max"),
        )
        .sort_values(ANGLE_COL)
    )
    stats["observed_range"] = stats["observed_max"] - stats["observed_min"]
    return stats[columns]


def prediction_summary(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    pred = (
        data.groupby(ANGLE_COL, as_index=False)
        .agg(
            y_median=("y_median", "mean"),
            y_lower=("y_lower", "mean"),
            y_upper=("y_upper", "mean"),
        )
        .sort_values(ANGLE_COL)
    )
    pred["interval_width"] = pred["y_upper"] - pred["y_lower"]
    return pred


def add_interval(fig: go.Figure, data: pd.DataFrame, name: str, color: str) -> None:
    interval = prediction_summary(data)
    if interval.empty:
        return
    x = pd.concat([interval[ANGLE_COL], interval[ANGLE_COL].iloc[::-1]])
    y = pd.concat([interval["y_upper"], interval["y_lower"].iloc[::-1]])
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            fill="toself",
            fillcolor=color.replace("1)", "0.16)"),
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name=name,
            showlegend=True,
        )
    )


def add_prediction_line(
    fig: go.Figure,
    data: pd.DataFrame,
    name: str,
    color: str,
    dash: str = "solid",
) -> None:
    if data.empty:
        return
    pred = prediction_summary(data)
    fig.add_trace(
        go.Scatter(
            x=pred[ANGLE_COL],
            y=pred["y_median"],
            mode="lines",
            name=name,
            line=dict(color=color, width=3, dash=dash),
            customdata=pred[["y_lower", "y_upper", "interval_width"]],
            hovertemplate=(
                "Angle: %{x:.1f} deg<br>"
                "Central prediction: %{y:.3f} mm<br>"
                "Lower bound: %{customdata[0]:.3f} mm<br>"
                "Upper bound: %{customdata[1]:.3f} mm<br>"
                "Interval width: %{customdata[2]:.3f} mm<extra></extra>"
            ),
        )
    )


def add_observed_band(fig: go.Figure, stats: pd.DataFrame) -> None:
    if stats.empty:
        return
    x = pd.concat([stats[ANGLE_COL], stats[ANGLE_COL].iloc[::-1]])
    y = pd.concat([stats["observed_max"], stats["observed_min"].iloc[::-1]])
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            fill="toself",
            fillcolor="rgba(92, 92, 92, 0.18)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="Observed min-max band",
        )
    )


def add_observed_individuals(
    fig: go.Figure,
    observed: pd.DataFrame,
    axis_col: str,
) -> None:
    for experiment_id, experiment in observed.groupby(EXPERIMENT_COL):
        fig.add_trace(
            go.Scatter(
                x=experiment[ANGLE_COL],
                y=experiment[axis_col],
                mode="lines",
                name=f"Experiment {int(experiment_id)}",
                line=dict(color="rgba(90, 90, 90, 0.24)", width=1),
                legendgroup="observed_individual",
                showlegend=False,
                hovertemplate=(
                    "Experiment %{customdata}<br>"
                    "Angle: %{x:.1f} deg<br>"
                    "Observed: %{y:.3f} mm<extra></extra>"
                ),
                customdata=np.full(len(experiment), int(experiment_id)),
            )
        )


def add_observed_mean(fig: go.Figure, stats: pd.DataFrame) -> None:
    if stats.empty:
        return
    fig.add_trace(
        go.Scatter(
            x=stats[ANGLE_COL],
            y=stats["observed_mean"],
            mode="lines",
            name="Observed group mean",
            line=dict(color=MODEL_COLORS["Observed"], width=3),
            hovertemplate=(
                "Angle: %{x:.1f} deg<br>"
                "Observed group mean: %{y:.3f} mm<extra></extra>"
            ),
        )
    )


def shared_y_range(*frames: pd.DataFrame, observed: pd.DataFrame, axis_col: str) -> list[float] | None:
    values = []
    if not observed.empty and axis_col in observed:
        values.append(observed[axis_col])
    for frame in frames:
        if frame.empty:
            continue
        for col in ("y_true", "y_lower", "y_median", "y_upper"):
            if col in frame.columns:
                values.append(frame[col])
    if not values:
        return None
    combined = pd.concat(values)
    combined = pd.to_numeric(combined, errors="coerce").dropna()
    if combined.empty:
        return None
    low = float(combined.min())
    high = float(combined.max())
    padding = max((high - low) * 0.08, 0.02)
    return [low - padding, high + padding]


def prediction_plot(
    observed: pd.DataFrame,
    observed_stats: pd.DataFrame,
    qrf: pd.DataFrame,
    hgp: pd.DataFrame,
    axis_col: str,
    view: str,
    show_individual: bool,
    show_interval: bool,
    synchronize_axes: bool,
) -> go.Figure:
    fig = go.Figure()
    add_observed_band(fig, observed_stats)
    if show_individual:
        add_observed_individuals(fig, observed, axis_col)

    if view in ("QRF", "Compare") and not qrf.empty:
        if show_interval:
            add_interval(fig, qrf, "QRF Prediction Interval", "rgba(37, 99, 235, 1)")
        add_prediction_line(fig, qrf, "QRF Central Prediction", MODEL_COLORS["QRF"])

    if view in ("HGP", "Compare") and not hgp.empty:
        if show_interval:
            add_interval(fig, hgp, "HGP Prediction Interval", "rgba(190, 88, 44, 1)")
        add_prediction_line(
            fig,
            hgp,
            "HGP Central Prediction",
            MODEL_COLORS["HGP"],
            dash="dash" if view == "Compare" else "solid",
        )
    add_observed_mean(fig, observed_stats)

    y_range = (
        shared_y_range(qrf, hgp, observed=observed, axis_col=axis_col)
        if synchronize_axes
        else None
    )
    fig.update_layout(
        template="plotly_white",
        height=620,
        margin=dict(l=20, r=20, t=24, b=20),
        xaxis_title="Bending Angle [deg]",
        yaxis_title=display_axis(axis_col),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(range=[0, MAX_PLOT_ANGLE])
    fig.update_yaxes(range=y_range)
    return fig


def variability_plot(
    observed_stats: pd.DataFrame,
    qrf: pd.DataFrame,
    hgp: pd.DataFrame,
    view: str,
) -> go.Figure:
    fig = go.Figure()
    if not observed_stats.empty:
        fig.add_trace(
            go.Scatter(
                x=observed_stats[ANGLE_COL],
                y=observed_stats["observed_range"],
                mode="lines",
                name="Observed Experimental Range",
                line=dict(color="rgba(35, 35, 35, 0.85)", width=2),
                hovertemplate=(
                    "Angle: %{x:.1f} deg<br>"
                    "Observed range: %{y:.3f} mm<extra></extra>"
                ),
            )
        )
    for model, frame in (("QRF", qrf), ("HGP", hgp)):
        if view not in (model, "Compare") or frame.empty:
            continue
        pred = prediction_summary(frame)
        fig.add_trace(
            go.Scatter(
                x=pred[ANGLE_COL],
                y=pred["interval_width"],
                mode="lines",
                name=f"{model} Prediction Interval Width",
                line=dict(
                    color=MODEL_COLORS[model],
                    width=2,
                    dash="dash" if model == "HGP" and view == "Compare" else "solid",
                ),
                hovertemplate=(
                    "Angle: %{x:.1f} deg<br>"
                    "Interval width: %{y:.3f} mm<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        template="plotly_white",
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Bending Angle [deg]",
        yaxis_title="Width [mm]",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(range=[0, MAX_PLOT_ANGLE])
    return fig


def metric_format(value: object, metric: str) -> str:
    if pd.isna(value):
        return ""
    value = float(value)
    if "coverage" in metric:
        return f"{value:.1f}%"
    if metric_unit(metric) == "mm":
        return f"{value:.3f}"
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.4f}"


def selected_metric_rows(
    metrics: pd.DataFrame,
    qrf: pd.DataFrame,
    hgp: pd.DataFrame,
) -> dict[str, pd.Series]:
    if metrics.empty:
        return {}
    selected = {}
    for model, frame in (("QRF", qrf), ("HGP", hgp)):
        if frame.empty:
            continue
        split_files = frame["prediction_file"].dropna().unique()
        model_metrics = metrics[
            (metrics["Model"] == model)
            & (metrics["prediction_file"].isin(split_files))
        ]
        if model_metrics.empty:
            continue
        selected[model] = model_metrics.iloc[0]
    return selected


def raw_metric_table(metrics: pd.DataFrame, qrf: pd.DataFrame, hgp: pd.DataFrame) -> pd.DataFrame:
    metric_rows = selected_metric_rows(metrics, qrf, hgp)
    rows = []
    for model, row in metric_rows.items():
        for metric in METRIC_COLUMNS:
            if metric in row.index and pd.notna(row[metric]):
                rows.append(
                    {
                        "Model": model,
                        "Metric": metric_label(metric),
                        "Raw Field": metric,
                        "Value": metric_format(row[metric], metric),
                        "Unit": metric_unit(metric),
                    }
                )
    return pd.DataFrame(rows)


def render_setup_summary(setup_row: pd.Series, observed: pd.DataFrame) -> None:
    with st.expander("Bending Setup Parameters", expanded=True):
        process, tube = st.columns([2, 1])

        with process:
            st.caption("PROCESS PARAMETERS")
            cols = st.columns(3)
            for index, column in enumerate(PROCESS_PARAMETER_COLUMNS):
                if column not in setup_row.index or pd.isna(setup_row[column]):
                    continue
                label = column.replace("-", " ")
                cols[index % 3].markdown(
                    f"**{label}**  \n{format_setup_value(setup_row[column])}"
                )

        with tube:
            st.caption("TUBE / EXPERIMENT INFORMATION")
            experiment_count = observed[EXPERIMENT_COL].nunique()
            tube_rows = [
                ("Target Angle", f"{format_setup_value(setup_row.get('Target-angle'))} deg"),
                ("Outer Diameter", f"{format_setup_value(setup_row.get('Outer-diameter'))} mm"),
                ("Wall Thickness", f"{format_setup_value(setup_row.get('Wall-thickness'))} mm"),
                ("Observed Experiments", str(experiment_count)),
            ]
            for label, value in tube_rows:
                st.markdown(f"**{label}**  \n{value}")


def render_performance_summary(
    metrics: pd.DataFrame,
    qrf: pd.DataFrame,
    hgp: pd.DataFrame,
    view: str,
) -> None:
    metric_rows = selected_metric_rows(metrics, qrf, hgp)
    visible_models = [
        model
        for model in ("QRF", "HGP")
        if view in (model, "Compare") and model in metric_rows
    ]
    if not visible_models:
        if not qrf.empty or not hgp.empty:
            st.caption("No exported performance metrics are available for this selection.")
        return

    st.subheader("Model Performance")
    for model in visible_models:
        if view == "Compare":
            st.caption(model)
        cols = st.columns(4)
        row = metric_rows[model]
        for col, metric in zip(cols, PRIMARY_METRICS):
            value = metric_format(row[metric], metric) if metric in row.index else ""
            unit = metric_unit(metric)
            col.metric(metric_label(metric), f"{value} {unit}".strip() if value else "n/a")


def nominal_coverage_percent(
    metadata: pd.DataFrame,
    source: str,
    axis_key: str,
) -> float | None:
    if metadata.empty:
        return None
    candidates = metadata[
        (metadata["geometry_source"] == source)
        & (metadata["target_axis"] == axis_key)
    ]
    coverages = []
    for _, row in candidates.iterrows():
        config = row.get("model_config", {})
        if not isinstance(config, dict):
            continue
        if "confidence_level" in config:
            coverages.append(float(config["confidence_level"]) * 100)
        if "lower_quantile" in config and "upper_quantile" in config:
            coverages.append(
                (float(config["upper_quantile"]) - float(config["lower_quantile"]))
                * 100
            )
    unique_coverages = sorted({round(value, 6) for value in coverages})
    if len(unique_coverages) == 1:
        return unique_coverages[0]
    return None


def membership_table(
    qrf_status: str,
    qrf_split: str,
    qrf_has_prediction: bool,
    hgp_status: str,
    hgp_split: str,
    hgp_has_prediction: bool,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Model": "QRF",
                "Rank-1 role": qrf_status,
                "Interval export": "Available" if qrf_has_prediction else "Missing",
                "Rank-1 split": qrf_split,
            },
            {
                "Model": "HGP",
                "Rank-1 role": hgp_status,
                "Interval export": "Available" if hgp_has_prediction else "Missing",
                "Rank-1 split": hgp_split,
            },
        ]
    )


data = load_data()
observed_df = data["observed"]
setup_df = data["setup"]
predictions_df = data["predictions"]
metrics_df = data["metrics"]
metadata_df = data["metadata"]

predictions_df = predictions_df.copy()
rank_predictions_df = keep_rank_one_splits(
    predictions_df[predictions_df["prediction_scope"] == "Test groups"]
)
sources = sorted(predictions_df["geometry_source"].dropna().unique())
default_source = "real" if "real" in sources else sources[0]

st.markdown("### Rotary Tube Bending")
st.markdown("## Interval Prediction of Geometric Quality Metrics")
st.caption(
    "Interactive comparison of QRF and HGP prediction intervals on exported bending setups."
)

source = default_source
control_cols = st.columns([1, 1.25])
axis_label = control_cols[0].selectbox(
    "Target Geometry",
    list(AXIS_COLUMNS),
    index=0,
)
axis_key = AXIS_KEYS[axis_label]
axis_col = AXIS_COLUMNS[axis_label]
test_groups = valid_test_groups(rank_predictions_df, axis_key)
if st.session_state.get("test_group") not in test_groups:
    st.session_state["test_group"] = None
group_id = control_cols[1].selectbox(
    "Test Bending Setup",
    test_groups,
    index=None,
    placeholder="Select a test group...",
    format_func=lambda group: f"Group {int(group)}",
    key="test_group",
)

if group_id is None:
    st.info(
        "Select a test bending setup to inspect its experimental geometry "
        "and prediction uncertainty."
    )
else:
    group_id = int(group_id)
    group_observed = observed_df[
        (observed_df[GROUP_COL] == group_id)
        & (pd.to_numeric(observed_df[ANGLE_COL], errors="coerce") <= MAX_PLOT_ANGLE)
    ].sort_values([EXPERIMENT_COL, ANGLE_COL])
    observed_stats = observed_geometry_stats(group_observed, axis_col)

    qrf_status, qrf_split = split_membership(
        rank_predictions_df, "QRF", source, axis_key, group_id
    )
    hgp_status, hgp_split = split_membership(
        rank_predictions_df, "HGP", source, axis_key, group_id
    )
    qrf_data = selected_prediction_slice(
        predictions_df, "QRF", source, axis_key, group_id
    )
    hgp_data = selected_prediction_slice(
        predictions_df, "HGP", source, axis_key, group_id
    )
    has_prediction = not qrf_data.empty or not hgp_data.empty
    coverage = nominal_coverage_percent(metadata_df, source, axis_key)
    coverage_label = (
        f"{coverage:.0f}% prediction interval"
        if coverage is not None
        else "exported prediction interval"
    )

    st.markdown(f"### Group {group_id}")
    st.caption(
        "TEST SET · UNSEEN BENDING SETUP  \n"
        f"{axis_label} · {group_observed[EXPERIMENT_COL].nunique()} observed experiments · {coverage_label}"
    )

    setup_row = setup_df[setup_df[GROUP_COL] == group_id]
    if setup_row.empty:
        st.warning("No bending setup row was found for the selected Group_ID.")
    else:
        render_setup_summary(setup_row.iloc[0], group_observed)

    view = st.segmented_control(
        "Model View",
        ["Compare", "QRF", "HGP"],
        default="Compare",
    )
    with st.expander("Plot Settings", expanded=False):
        option_cols = st.columns(2)
        show_individual = option_cols[0].checkbox(
            "Show individual experiments",
            value=False,
        )
        show_interval = option_cols[1].checkbox(
            "Show prediction interval",
            value=True,
        )
    synchronize_axes = True

    if not has_prediction:
        st.warning(
            "No exported rank-1 test prediction is available for this group. "
            "The dashboard only reads static thesis exports and does not run inference."
        )
    elif view == "Compare" and qrf_data.empty != hgp_data.empty:
        missing_model = "QRF" if qrf_data.empty else "HGP"
        st.info(f"{missing_model} prediction is unavailable for this rank-1 test selection.")
    elif view == "QRF" and qrf_data.empty:
        st.warning("No QRF prediction export is available for this rank-1 test selection.")
    elif view == "HGP" and hgp_data.empty:
        st.warning("No HGP prediction export is available for this rank-1 test selection.")

    fig = prediction_plot(
        group_observed,
        observed_stats,
        qrf_data,
        hgp_data,
        axis_col,
        view,
        show_individual,
        show_interval,
        synchronize_axes,
    )
    st.plotly_chart(fig, width="stretch")

    st.plotly_chart(
        variability_plot(observed_stats, qrf_data, hgp_data, view),
        width="stretch",
    )

    render_performance_summary(metrics_df, qrf_data, hgp_data, view)

    detailed = raw_metric_table(metrics_df, qrf_data, hgp_data)
    with st.expander("Advanced / Reproducibility Details", expanded=False):
        st.caption(f"Displayed prediction source: {display_source(source)}")
        st.dataframe(
            membership_table(
                qrf_status,
                qrf_split,
                not qrf_data.empty,
                hgp_status,
                hgp_split,
                not hgp_data.empty,
            ),
            hide_index=True,
            width="stretch",
        )
        if not detailed.empty:
            st.markdown("**Detailed Model Metrics**")
            st.dataframe(detailed, hide_index=True, width="stretch")
        exported_files = []
        for model, frame in (("QRF", qrf_data), ("HGP", hgp_data)):
            if not frame.empty:
                exported_files.append(
                    {
                        "Model": model,
                        "Prediction File": frame["prediction_file"].iloc[0],
                        "Split": int(frame["split_index"].iloc[0]),
                        "Split Description": frame["split_name"].iloc[0],
                    }
                )
        if exported_files:
            st.markdown("**Selected Prediction Exports**")
            st.dataframe(pd.DataFrame(exported_files), hide_index=True, width="stretch")
