import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.pipeline.rf_augmentation.io_utils import existing_table_path, read_table
from src.pipeline.rf_augmentation.sensor_data_augmentor import SensorDataAugmentor


st.set_page_config(page_title="RF Generated Geometry", layout="wide")
st.title("RF Generated Geometry by Group")

project_root = Path(__file__).resolve().parent
data_dir = project_root / "data" / "rf_augmented"
group_setup_path = project_root / "data" / "raw" / "unique_bending_setups.csv"

ANGLE_COL = "Angle[degree]ORDistance[mm]"
MAIN_COL = "Main-axis [mm]"
SEC_COL = "Secondary-axis [mm]"
ID_COL = "Experiment_ID"
GROUP_COL = "group_id"

Y_AXIS_MIN_MM = 20
Y_AXIS_MAX_MM = 23
Y_AXIS_TICK_STEP_MM = 0.1

METHODS = {
    "random-within-group": {
        "title": "Random Within Group",
        "suffix": "random_within_group_raw",
    },
    "within-group-interpolation": {
        "title": "Within Group Interpolation",
        "suffix": "within_group_interpolation_raw",
    },
}


def sensor_mode_suffix(mode: str) -> str:
    return SensorDataAugmentor.mode_to_suffix(mode)


def table_path(prefix: str, suffix: str) -> Path:
    return existing_table_path(data_dir / f"{prefix}_{suffix}.parquet")


def final_geometry_path(suffix: str) -> Path:
    return table_path("final_geometry", suffix)


def selection_values_path(suffix: str) -> Path:
    return table_path("signal_feature_selected_values", suffix)


@st.cache_data
def load_table(path: Path, file_mtime: float) -> pd.DataFrame:
    return read_table(path)


@st.cache_data
def load_group_setup(path: Path, file_mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


def load_existing_table(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return load_table(path, path.stat().st_mtime)


def available_augmented_sensor_modes() -> list[str]:
    modes = []
    for mode in SensorDataAugmentor.all_augmentation_modes():
        if mode == "raw":
            continue
        suffix = f"sensor_augmented_{sensor_mode_suffix(mode)}"
        if final_geometry_path(suffix).exists():
            modes.append(mode)
    return modes


def available_group_ids() -> list[int]:
    group_ids = set()
    suffixes = [config["suffix"] for config in METHODS.values()]
    suffixes.extend(
        f"sensor_augmented_{sensor_mode_suffix(mode)}"
        for mode in available_augmented_sensor_modes()
    )

    for suffix in suffixes:
        path = final_geometry_path(suffix)
        if not path.exists():
            continue
        df = load_table(path, path.stat().st_mtime)
        if GROUP_COL in df.columns:
            group_ids.update(
                int(group_id)
                for group_id in df[GROUP_COL].dropna().unique()
            )

    return sorted(group_ids)


def parse_experiment_ids(value) -> list[int]:
    if pd.isna(value):
        return []
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if isinstance(value, (list, tuple, set)):
        return [int(item) for item in value]
    return [int(value)]


def set_fixed_y_axis(ax):
    ax.set_ylim(Y_AXIS_MIN_MM, Y_AXIS_MAX_MM)
    ax.set_yticks(
        np.arange(
            Y_AXIS_MIN_MM,
            Y_AXIS_MAX_MM + Y_AXIS_TICK_STEP_MM,
            Y_AXIS_TICK_STEP_MM,
        )
    )


def set_zoomed_y_axis(ax, df_group: pd.DataFrame, value_col: str):
    values = pd.to_numeric(df_group[value_col], errors="coerce").dropna()
    if values.empty:
        return

    y_min = float(values.min())
    y_max = float(values.max())
    padding = (y_max - y_min) * 0.08
    if padding == 0:
        padding = max(abs(y_min), 1.0) * 0.02

    ax.set_ylim(y_min - padding, y_max + padding)


def set_plot_y_axis(ax, df_group: pd.DataFrame, value_col: str):
    if plot_scale == "Zoom in":
        set_zoomed_y_axis(ax, df_group, value_col)
    else:
        set_fixed_y_axis(ax)


def plot_axis(df_group: pd.DataFrame, value_col: str, title: str):
    df_real_group = df_group[df_group["Synthetic"] == False]
    df_fake_group = df_group[df_group["Synthetic"] == True]
    real_ids = df_real_group[ID_COL].dropna().unique()
    fake_ids = df_fake_group[ID_COL].dropna().unique()

    fig, ax = plt.subplots(figsize=(12, 5))

    for real_id in real_ids:
        temp = df_real_group[df_real_group[ID_COL] == real_id].sort_values(ANGLE_COL)
        ax.plot(
            temp[ANGLE_COL],
            temp[value_col],
            linewidth=2,
            alpha=0.9,
            label=f"Real {int(real_id)}",
        )

    for synthetic_id in fake_ids:
        temp = df_fake_group[df_fake_group[ID_COL] == synthetic_id].sort_values(
            ANGLE_COL
        )
        ax.plot(
            temp[ANGLE_COL],
            temp[value_col],
            linestyle="--",
            linewidth=2,
            alpha=0.9,
            label=f"Synthetic {int(synthetic_id)}",
        )

    ax.set_title(title)
    ax.set_xlabel("Angle")
    ax.set_ylabel(value_col)
    set_plot_y_axis(ax, df_group, value_col)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    st.pyplot(fig)


def short_feature_label(row) -> str:
    signal_name = row.get("signal_name")
    feature_type = row.get("feature_type")
    feature_name = row.get("feature_name")

    if pd.notna(signal_name) and str(signal_name).strip():
        signal = str(signal_name)
        if len(signal) > 32:
            signal = signal[:29] + "..."
        return f"{signal}_{feature_type}"

    return str(feature_name)


def show_feature_values(selection_group: pd.DataFrame, method_label: str):
    if selection_group is None or selection_group.empty:
        st.info("No feature-value records found for this group.")
        return

    model_inputs = sorted(selection_group["model_input"].dropna().unique())
    synthetic_ids = sorted(
        int(synthetic_id)
        for synthetic_id in selection_group["synthetic_experiment_id"].dropna().unique()
    )
    if not model_inputs or not synthetic_ids:
        st.info("No generated feature values found for this group.")
        return

    left_col, right_col = st.columns(2)
    model_input = left_col.selectbox(
        "Model input",
        model_inputs,
        key=f"{method_label}_model_input",
    )
    synthetic_id = right_col.selectbox(
        "Synthetic experiment",
        synthetic_ids,
        key=f"{method_label}_synthetic_id",
    )

    temp = selection_group[
        (selection_group["model_input"] == model_input)
        & (selection_group["synthetic_experiment_id"] == synthetic_id)
        & (selection_group["input_source"] == "signal_feature")
    ].copy()
    if temp.empty:
        st.info("No signal-feature records found for this generated sample.")
        return

    temp["feature"] = temp.apply(short_feature_label, axis=1)
    temp["selected_minus_base"] = temp["selected_value"] - temp["actual_value"]
    changed_count = int(temp["selected_minus_base"].abs().gt(1e-12).sum())
    unchanged_count = int(len(temp) - changed_count)

    caption_parts = [
        f"Base experiment: {temp['base_experiment_id'].iloc[0]}",
        f"Changed features: {changed_count}",
        f"Unchanged features: {unchanged_count}",
    ]
    paired_id = temp["paired_base_experiment_id"].iloc[0]
    if pd.notna(paired_id) and str(paired_id).strip():
        caption_parts.append(f"Paired experiment: {paired_id}")
    alpha = temp["interpolation_weight"].iloc[0]
    if pd.notna(alpha) and str(alpha).strip():
        caption_parts.append(f"Alpha: {float(alpha):.3f}")

    st.caption(" | ".join(caption_parts))
    st.dataframe(
        temp[
            [
                "feature",
                "actual_value",
                "paired_actual_value",
                "selected_value",
                "selected_minus_base",
                "group_min_value",
                "group_max_value",
                "selection_method",
            ]
        ],
        width="stretch",
        hide_index=True,
    )


def render_method_section(
    *,
    title: str,
    suffix: str,
    selected_group: int,
    section_key: str,
):
    geometry_path = final_geometry_path(suffix)

    geometry_df = load_existing_table(geometry_path)
    if geometry_df is None:
        st.warning(f"Missing generated geometry file: `{geometry_path}`")
        return

    geometry_df[ANGLE_COL] = pd.to_numeric(geometry_df[ANGLE_COL], errors="coerce")
    group_df = geometry_df[geometry_df[GROUP_COL] == selected_group].copy()
    if group_df.empty:
        st.info(f"No generated geometry found for group `{selected_group}`.")
        return

    if metric in ("Both", "Main"):
        plot_axis(group_df, MAIN_COL, f"{title} - Main Axis")
    if metric in ("Both", "Secondary"):
        plot_axis(group_df, SEC_COL, f"{title} - Secondary Axis")


group_ids = available_group_ids()
if not group_ids:
    st.warning(
        "No generated geometry files found. Run "
        "`2_1_4_run_data_augmented_generator.py` first."
    )
    st.stop()

st.sidebar.header("Controls")
selected_group = st.sidebar.selectbox("Group ID", group_ids)
metric = st.sidebar.selectbox("Axis", ["Both", "Main", "Secondary"])
plot_scale = st.sidebar.radio("Plot scale", ["Zoom in", "Zoom out (fixed y-axis)"])
augmented_modes = available_augmented_sensor_modes()
selected_augmented_modes = st.sidebar.multiselect(
    "Sensor augmented modes",
    augmented_modes,
    default=augmented_modes[:1],
)

if group_setup_path.exists():
    group_setup_df = load_group_setup(group_setup_path, group_setup_path.stat().st_mtime)
    setup_row_idx = int(selected_group) - 1
    if 0 <= setup_row_idx < len(group_setup_df):
        st.write("### Selected Group Setup")
        st.dataframe(
            group_setup_df.iloc[[setup_row_idx]],
            width="stretch",
            hide_index=False,
        )

method_1_col, method_2_col, method_3_col = st.columns(3)

with method_1_col:
    config = METHODS["random-within-group"]
    st.write("## Random Within Group")
    render_method_section(
        title=config["title"],
        suffix=config["suffix"],
        selected_group=int(selected_group),
        section_key="random-within-group",
    )

with method_2_col:
    config = METHODS["within-group-interpolation"]
    st.write("## Interpolated Within Group")
    render_method_section(
        title=config["title"],
        suffix=config["suffix"],
        selected_group=int(selected_group),
        section_key="within-group-interpolation",
    )

with method_3_col:
    st.write("## Sensor Signals Augmented")
    if not augmented_modes:
        st.info("No sensor-augmented generated geometry files found.")
    else:
        if not selected_augmented_modes:
            st.info("Select one or more sensor augmentation modes.")

        for mode in selected_augmented_modes:
            suffix = f"sensor_augmented_{sensor_mode_suffix(mode)}"
            render_method_section(
                title=f"Sensor Signals Augmented ({mode})",
            suffix=suffix,
            selected_group=int(selected_group),
            section_key=f"sensor_augmented_{sensor_mode_suffix(mode)}",
        )
