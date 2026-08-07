import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="RF Generated Geometry", layout="wide")
st.title("RF Generated Geometry by Group")

project_root = Path(__file__).resolve().parent
ui_data_dir = project_root / "data" / "rf_augmented" / "ui_data"
augmented_data_dir = project_root / "data" / "rf_augmented"
manifest_path = ui_data_dir / "manifest.json"

ANGLE_COL = "Angle[degree]ORDistance[mm]"
MAIN_COL = "Main-axis [mm]"
SEC_COL = "Secondary-axis [mm]"
ID_COL = "Experiment_ID"
GROUP_COL = "group_id"

Y_AXIS_MIN_MM = 20
Y_AXIS_MAX_MM = 23
Y_AXIS_TICK_STEP_MM = 0.1

@st.cache_data
def load_manifest(path: Path, file_mtime: float) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_table(path: Path, file_mtime: float) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)

    return pd.read_csv(path)


@st.cache_data
def load_group_setup(path: Path, file_mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


def load_existing_table(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return load_table(path, path.stat().st_mtime)


def generated_geometry_path(file_name: str) -> Path:
    source_overrides = {
        "final_geometry_sensor_augmented_noise__time_wrapping__scaling__jittering.csv": (
            augmented_data_dir
            / "final_geometry_sensor_augmented_noise__time_wrapping__scaling__jittering.parquet"
        ),
        "final_geometry_within_group_interpolation_raw.csv": (
            augmented_data_dir
            / "final_geometry_within_group_interpolation_raw.parquet"
        ),
    }

    return source_overrides.get(
        file_name,
        ui_data_dir / file_name,
    )


if not manifest_path.exists():
    st.warning(
        "Missing deployment UI data. Run "
        "`2_1_4_run_data_augmented_generator.py` to export "
        "`data/rf_augmented/ui_data/manifest.json`."
    )
    st.stop()

manifest = load_manifest(manifest_path, manifest_path.stat().st_mtime)
METHODS = {
    method["key"]: method
    for method in manifest.get("methods", [])
}
group_setup_csv = manifest.get("group_setup_csv")
group_setup_path = ui_data_dir / group_setup_csv if group_setup_csv else None


def available_group_ids() -> list[int]:
    group_ids = set()

    for config in METHODS.values():
        path = generated_geometry_path(
            config["csv"]
        )
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
    csv: str,
    selected_group: int,
    section_key: str,
):
    geometry_path = generated_geometry_path(csv)

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

if group_setup_path is not None and group_setup_path.exists():
    group_setup_df = load_group_setup(group_setup_path, group_setup_path.stat().st_mtime)
    setup_row_idx = int(selected_group) - 1
    if 0 <= setup_row_idx < len(group_setup_df):
        st.write("### Selected Group Setup")
        st.dataframe(
            group_setup_df.iloc[[setup_row_idx]],
            width="stretch",
            hide_index=False,
        )

method_1_col, method_2_col = st.columns(2)

with method_1_col:
    config = METHODS["within-group-interpolation"]
    st.write("## Interpolated Within Group")
    render_method_section(
        title=config["title"],
        csv=config["csv"],
        selected_group=int(selected_group),
        section_key="within-group-interpolation",
    )

with method_2_col:
    config = METHODS["sensor-augmented"]
    st.write("## Sensor Signals Augmented")
    render_method_section(
        title=config["title"],
        csv=config["csv"],
        selected_group=int(selected_group),
        section_key="sensor-augmented",
    )
