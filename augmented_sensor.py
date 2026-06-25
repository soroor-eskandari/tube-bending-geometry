import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.pipeline.rf_augmentation.io_utils import existing_table_path, read_table
from src.pipeline.rf_augmentation.rf_preprocessor import RFPreprocessor
from src.pipeline.rf_augmentation.sensor_data_augmentor import SensorDataAugmentor


st.set_page_config(page_title="Raw Sensor Augmentation", layout="wide")
st.title("Augmented Sensor Signals by Group")

project_root = Path(__file__).resolve().parent
raw_sensor_path = project_root / "data" / "processed" / "machine_and_movement.csv"
bending_path = project_root / "data" / "processed" / "bending.csv"
group_setup_path = project_root / "data" / "raw" / "unique_bending_setups.csv"
sensor_data_dir = project_root / "data" / "rf_augmented" / "sensor_data"
group_metrics_path = (
    project_root / "data" / "rf_augmented" / "sensor_augmentation_group_metrics.parquet"
)

ID_COL = "Experiment_ID"
TIME_COL = "Time_[s]"
RAW_SIGNAL_COLOR = "#111111"
AUGMENTED_SIGNAL_COLOR = "#d62728"
ZOOM_COLOR = "#ff7f0e"


@st.cache_data
def load_csv(path, file_mtime):
    return read_table(path)


@st.cache_data
def load_cleaned_raw_sensor_data(raw_path, raw_mtime, bending_path, bending_mtime):
    raw_sensor_df = pd.read_csv(raw_path)
    bending_df = pd.read_csv(bending_path)
    cleaned_sensor_df, _ = RFPreprocessor.preprocess_data(
        machine_movement_df=raw_sensor_df,
        bending_df=bending_df,
    )
    return cleaned_sensor_df


def parse_experiment_ids(value):
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if isinstance(value, (list, tuple, set)):
        return [int(item) for item in value]
    return [int(value)]


def mode_suffix(mode):
    return SensorDataAugmentor.mode_to_suffix(mode)


def mode_from_suffix(suffix):
    if suffix == "raw":
        return "raw"
    return suffix.replace("__", "+").replace("_", "-")


def discover_augmented_sensor_modes():
    modes_by_suffix = {}
    for pattern in ("machine_movement_*.parquet", "machine_movement_*.csv"):
        for path in sensor_data_dir.glob(pattern):
            suffix = path.stem.replace("machine_movement_", "", 1)
            if suffix == "raw":
                continue
            modes_by_suffix[suffix] = mode_from_suffix(suffix)

    return [
        modes_by_suffix[suffix]
        for suffix in sorted(modes_by_suffix)
    ]


def load_sensor_mode(mode):
    if mode == "raw":
        path = raw_sensor_path
        if "raw_df" in globals():
            return raw_df, path
    else:
        path = existing_table_path(
            sensor_data_dir / f"machine_movement_{mode_suffix(mode)}.parquet"
        )

    if not path.exists():
        return None, path

    return load_csv(path, path.stat().st_mtime), path


def numeric_signal_columns(df):
    return [
        column
        for column in df.select_dtypes(include=[np.number]).columns
        if column not in {ID_COL, TIME_COL}
    ]


def available_signal_columns_for_modes(modes):
    raw_columns = set(numeric_signal_columns(raw_df))
    common_columns = set(raw_columns)

    for mode in modes:
        sensor_df, _ = load_sensor_mode(mode)
        if sensor_df is None:
            continue
        common_columns &= set(numeric_signal_columns(sensor_df))

    return [
        column
        for column in numeric_signal_columns(raw_df)
        if column in common_columns
    ]


def axis_limits(raw_values, augmented_values, padding_fraction=0.08):
    values = pd.concat([raw_values, augmented_values], ignore_index=True).dropna()
    if values.empty:
        return None

    min_value = float(values.min())
    max_value = float(values.max())
    padding = (max_value - min_value) * padding_fraction
    if padding == 0:
        padding = max(abs(min_value), 1.0) * 0.05
    return min_value - padding, max_value + padding


def input_signal_nrmse(raw_df, augmented_df, signal_col):
    merged = raw_df[[ID_COL, TIME_COL, signal_col]].merge(
        augmented_df[[ID_COL, TIME_COL, signal_col]],
        on=[ID_COL, TIME_COL],
        how="inner",
        suffixes=("_raw", "_augmented"),
    )
    if merged.empty:
        return np.nan

    raw_values = merged[f"{signal_col}_raw"].to_numpy(dtype=float)
    augmented_values = merged[f"{signal_col}_augmented"].to_numpy(dtype=float)
    rmse = np.sqrt(np.mean((raw_values - augmented_values) ** 2))
    raw_std = np.std(raw_values)
    return rmse / raw_std if raw_std > 0 else np.nan


def metric_row(metrics_df, group_id, mode, split):
    if metrics_df.empty:
        return None

    mode_metrics = metrics_df[
        (metrics_df["group_id"] == group_id)
        & (metrics_df["sensor_mode_suffix"] == mode_suffix(mode))
    ].copy()
    if mode_metrics.empty:
        return None

    split_metrics = mode_metrics[mode_metrics["split"] == split]
    if not split_metrics.empty:
        return split_metrics.iloc[0]

    return mode_metrics.iloc[0]


def format_metric(value):
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.4f}"


def default_zoom_window_for_difference(
    *,
    raw_experiment_df,
    selected_augmented_modes,
    selected_signal,
    selected_experiments,
    time_min,
    time_max,
):
    fallback_width = (time_max - time_min) * 0.05
    fallback_start = time_min + (time_max - time_min) * 0.45

    if not selected_augmented_modes or not selected_signal:
        return fallback_start, fallback_start + fallback_width

    augmented_df, _ = load_sensor_mode(selected_augmented_modes[0])
    if augmented_df is None or selected_signal not in augmented_df.columns:
        return fallback_start, fallback_start + fallback_width

    augmented_experiment_df = augmented_df[
        augmented_df[ID_COL].isin(selected_experiments)
    ].copy()
    merged = raw_experiment_df[[ID_COL, TIME_COL, selected_signal]].merge(
        augmented_experiment_df[[ID_COL, TIME_COL, selected_signal]],
        on=[ID_COL, TIME_COL],
        how="inner",
        suffixes=("_raw", "_augmented"),
    )
    if merged.empty:
        return fallback_start, fallback_start + fallback_width

    difference = (
        merged[f"{selected_signal}_raw"] - merged[f"{selected_signal}_augmented"]
    ).abs()
    if difference.dropna().empty:
        return fallback_start, fallback_start + fallback_width

    center_time = float(merged.loc[difference.idxmax(), TIME_COL])
    window_width = max((time_max - time_min) * 0.04, 1.0)
    zoom_start = max(time_min, center_time - window_width / 2)
    zoom_end = min(time_max, center_time + window_width / 2)
    if zoom_end - zoom_start < window_width:
        if zoom_start == time_min:
            zoom_end = min(time_max, zoom_start + window_width)
        else:
            zoom_start = max(time_min, zoom_end - window_width)

    return zoom_start, zoom_end


def plot_signal_groups(
    ax,
    df,
    signal_col,
    color,
    label,
    linewidth,
    first_alpha,
    linestyle="-",
):
    for idx, (_, experiment_df) in enumerate(df.groupby(ID_COL)):
        ax.plot(
            experiment_df[TIME_COL],
            experiment_df[signal_col],
            color=color,
            linewidth=linewidth,
            alpha=first_alpha if idx == 0 else 0.45,
            label=label if idx == 0 else None,
            linestyle=linestyle,
        )


def add_zoom_inset(
    *,
    ax,
    raw_plot,
    augmented_plot,
    signal_col,
    zoom_window,
):
    if zoom_window is None:
        return

    zoom_start, zoom_end = zoom_window
    if zoom_end <= zoom_start:
        return

    raw_zoom = raw_plot[
        raw_plot[TIME_COL].between(zoom_start, zoom_end, inclusive="both")
    ]
    augmented_zoom = augmented_plot[
        augmented_plot[TIME_COL].between(zoom_start, zoom_end, inclusive="both")
    ]
    if raw_zoom.empty and augmented_zoom.empty:
        return

    zoom_limits = axis_limits(
        raw_zoom[signal_col] if not raw_zoom.empty else pd.Series(dtype=float),
        (
            augmented_zoom[signal_col]
            if not augmented_zoom.empty
            else pd.Series(dtype=float)
        ),
        padding_fraction=0.015,
    )
    if zoom_limits is None:
        return

    inset_ax = ax.inset_axes([0.43, 0.08, 0.54, 0.46])
    plot_signal_groups(
        inset_ax,
        raw_zoom,
        signal_col,
        RAW_SIGNAL_COLOR,
        "raw",
        1.5,
        0.9,
    )
    plot_signal_groups(
        inset_ax,
        augmented_zoom,
        signal_col,
        AUGMENTED_SIGNAL_COLOR,
        "augmented",
        1.3,
        0.9,
    )
    inset_ax.set_xlim(zoom_start, zoom_end)
    inset_ax.set_ylim(*zoom_limits)
    inset_ax.tick_params(axis="both", labelsize=6)
    inset_ax.grid(True, alpha=0.18)
    for spine in inset_ax.spines.values():
        spine.set_edgecolor(ZOOM_COLOR)
        spine.set_linewidth(1.4)

    ax.indicate_inset_zoom(inset_ax, edgecolor=ZOOM_COLOR, linewidth=1.0, alpha=0.85)


def plot_mode_card(
    *,
    mode,
    raw_experiment_df,
    augmented_experiment_df,
    signal_col,
    metrics,
    zoom_window,
):
    fig, ax = plt.subplots(figsize=(5.8, 3.2))

    raw_plot = raw_experiment_df.sort_values([ID_COL, TIME_COL])
    augmented_plot = augmented_experiment_df.sort_values([ID_COL, TIME_COL])

    plot_signal_groups(
        ax,
        raw_plot,
        signal_col,
        RAW_SIGNAL_COLOR,
        "raw",
        1.5,
        0.8,
    )

    if mode != "raw":
        plot_signal_groups(
            ax,
            augmented_plot,
            signal_col,
            AUGMENTED_SIGNAL_COLOR,
            "augmented",
            1.3,
            0.85,
        )

    limits = axis_limits(raw_plot[signal_col], augmented_plot[signal_col])
    if limits is not None:
        ax.set_ylim(*limits)

    if mode != "raw":
        add_zoom_inset(
            ax=ax,
            raw_plot=raw_plot,
            augmented_plot=augmented_plot,
            signal_col=signal_col,
            zoom_window=zoom_window,
        )

    signal_nrmse = input_signal_nrmse(
        raw_plot,
        augmented_plot,
        signal_col,
    )

    main_nrmse = metrics["nrmse_main"] if metrics is not None else np.nan
    secondary_nrmse = metrics["nrmse_secondary"] if metrics is not None else np.nan
    mean_nrmse = metrics["mean_nrmse"] if metrics is not None else np.nan

    ax.set_title(mode, fontsize=9)
    ax.set_xlabel("Time [s]", fontsize=8)
    ax.set_ylabel(signal_col, fontsize=7)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()

    st.pyplot(fig)
    st.caption(f"Input signal NRMSE vs raw: `{format_metric(signal_nrmse)}`")


def build_mode_summary(
    modes,
    signals,
    experiments,
    raw_experiment_df,
    metrics_df,
    group_id,
    split,
):
    rows = []

    for mode in modes:
        sensor_df, _ = load_sensor_mode(mode)
        metrics = metric_row(metrics_df, group_id, mode, split)

        input_nrmse_values = []
        if sensor_df is not None:
            augmented_experiment_df = sensor_df[
                sensor_df[ID_COL].isin(experiments)
            ].copy()

            for signal in signals:
                if signal not in augmented_experiment_df.columns:
                    continue
                input_nrmse_values.append(
                    input_signal_nrmse(
                        raw_experiment_df,
                        augmented_experiment_df,
                        signal,
                    )
                )

        input_nrmse_values = [
            value for value in input_nrmse_values if not pd.isna(value)
        ]

        rows.append({
            "augmentation_mode": mode,
            "nrmse_main": metrics["nrmse_main"] if metrics is not None else np.nan,
            "nrmse_secondary": (
                metrics["nrmse_secondary"] if metrics is not None else np.nan
            ),
            "mean_nrmse": metrics["mean_nrmse"] if metrics is not None else np.nan,
            "avg_input_signal_nrmse": (
                float(np.mean(input_nrmse_values))
                if input_nrmse_values
                else np.nan
            ),
            "rf_nrmse_available": metrics is not None,
            "signals_used": len(input_nrmse_values),
        })

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        sort_columns = (
            ["mean_nrmse", "avg_input_signal_nrmse"]
            if summary_df["mean_nrmse"].notna().any()
            else ["avg_input_signal_nrmse"]
        )
        summary_df = summary_df.sort_values(
            sort_columns,
            na_position="last",
        ).reset_index(drop=True)
        summary_df.insert(0, "rank", np.arange(1, len(summary_df) + 1))

    return summary_df


def highlight_best_row(row):
    if row.name == 0:
        return [
            "font-weight: 700; background-color: #e8f5e9"
            for _ in row
        ]
    return ["" for _ in row]


if not raw_sensor_path.exists():
    st.warning(f"Missing raw sensor file: {raw_sensor_path}")
    st.stop()

if not bending_path.exists():
    st.warning(f"Missing bending file needed for cleaning: {bending_path}")
    st.stop()

if not group_setup_path.exists():
    st.warning(f"Missing group setup file: {group_setup_path}")
    st.stop()

raw_df = load_cleaned_raw_sensor_data(
    raw_sensor_path,
    raw_sensor_path.stat().st_mtime,
    bending_path,
    bending_path.stat().st_mtime,
)
group_setup_df = load_csv(group_setup_path, group_setup_path.stat().st_mtime)
resolved_group_metrics_path = existing_table_path(group_metrics_path)
group_metrics_df = (
    load_csv(resolved_group_metrics_path, resolved_group_metrics_path.stat().st_mtime)
    if resolved_group_metrics_path.exists()
    else pd.DataFrame()
)

available_augmented_modes = discover_augmented_sensor_modes()

if not available_augmented_modes:
    st.warning(
        "No augmented sensor data files were found in "
        f"`{sensor_data_dir}`. Run `2_1_3_run_data_rf_augmentator.py` first."
    )
    st.stop()

st.sidebar.header("Controls")

group_ids = list(range(1, len(group_setup_df) + 1))
selected_group = st.sidebar.selectbox("Group", group_ids)

group_exp_ids = parse_experiment_ids(
    group_setup_df.iloc[selected_group - 1]["Experiment_Number"]
)
available_exp_ids = sorted(
    set(group_exp_ids).intersection(set(raw_df[ID_COL].dropna().astype(int)))
)

if not available_exp_ids:
    st.warning(f"No raw sensor experiments found for group {selected_group}.")
    st.stop()

selected_augmented_modes = st.sidebar.multiselect(
    "Augmented sensor files",
    available_augmented_modes,
    default=available_augmented_modes,
)

if not selected_augmented_modes:
    st.warning("Select at least one existing augmented sensor file.")
    st.stop()

mode_filter = ["raw", *selected_augmented_modes]
st.sidebar.caption(
    f"Using {len(selected_augmented_modes)} existing augmented sensor file(s)."
)

selected_experiments = st.sidebar.multiselect(
    "Experiments",
    available_exp_ids,
    default=available_exp_ids,
)

if not selected_experiments:
    st.warning("Select at least one experiment.")
    st.stop()

signal_cols = available_signal_columns_for_modes(mode_filter)

if not signal_cols:
    st.warning(
        "No common numeric input signals found for the selected modes. "
        "Reduce the displayed modes."
    )
    st.stop()

default_signals = signal_cols[: min(2, len(signal_cols))]
selected_signals = st.sidebar.multiselect(
    "Input signals",
    signal_cols,
    default=default_signals,
)

if not selected_signals:
    st.warning("Select at least one input signal.")
    st.stop()

metric_split = "test"

st.write("### Group Setup")
st.dataframe(
    group_setup_df.iloc[[selected_group - 1]],
    width="stretch",
    hide_index=False,
)

if group_metrics_df.empty:
    st.info(
        "No group metric file found. NRMSE values will show as `n/a` until "
        "`data/rf_augmented/sensor_augmentation_group_metrics.parquet` exists."
    )

selected_experiment_label = ", ".join(str(item) for item in selected_experiments)
st.write(f"### Group {selected_group} | Experiments {selected_experiment_label}")

raw_experiment_df = raw_df[raw_df[ID_COL].isin(selected_experiments)].copy()

if raw_experiment_df.empty:
    st.warning("Selected experiments have no raw sensor rows.")
    st.stop()

time_min = float(raw_experiment_df[TIME_COL].min())
time_max = float(raw_experiment_df[TIME_COL].max())
default_zoom_start, default_zoom_end = default_zoom_window_for_difference(
    raw_experiment_df=raw_experiment_df,
    selected_augmented_modes=selected_augmented_modes,
    selected_signal=selected_signals[0],
    selected_experiments=selected_experiments,
    time_min=time_min,
    time_max=time_max,
)
zoom_window = st.sidebar.slider(
    "Zoom time window [s]",
    min_value=time_min,
    max_value=time_max,
    value=(default_zoom_start, default_zoom_end),
    help=(
        "Defaults to a narrow window around the largest raw-vs-augmented "
        "difference for the first selected signal and augmented file."
    ),
)

summary_df = build_mode_summary(
    modes=selected_augmented_modes,
    signals=selected_signals,
    experiments=selected_experiments,
    raw_experiment_df=raw_experiment_df,
    metrics_df=group_metrics_df,
    group_id=selected_group,
    split=metric_split,
)

if not summary_df.empty:
    st.write("### Selected Augmentation Metrics")
    best_row = summary_df.iloc[0]
    has_rf_nrmse = summary_df["mean_nrmse"].notna().any()
    if has_rf_nrmse:
        st.caption(
            "`mean_nrmse` compares RF geometry prediction error for each selected "
            "sensor augmentation mode. Lower is better."
        )
        st.markdown(
            "**Best selected augmented mode:** "
            f"`{best_row['augmentation_mode']}` "
            f"with mean NRMSE `{format_metric(best_row['mean_nrmse'])}` "
            f"(Main `{format_metric(best_row['nrmse_main'])}`, "
            f"Secondary `{format_metric(best_row['nrmse_secondary'])}`)."
        )
    else:
        st.caption(
            "`avg_input_signal_nrmse` compares the selected augmented signals "
            "against the cleaned raw signals. Lower means closer to raw."
        )
        st.markdown(
            "**Closest selected augmented signal:** "
            f"`{best_row['augmentation_mode']}` "
            f"with input-signal NRMSE "
            f"`{format_metric(best_row['avg_input_signal_nrmse'])}`."
        )
    display_summary_df = summary_df.drop(
        columns=["rf_nrmse_available", "signals_used"],
        errors="ignore",
    ).dropna(axis=1, how="all")
    st.dataframe(
        display_summary_df.style.apply(highlight_best_row, axis=1).format({
            "nrmse_main": "{:.4f}",
            "nrmse_secondary": "{:.4f}",
            "mean_nrmse": "{:.4f}",
            "avg_input_signal_nrmse": "{:.4f}",
        }, na_rep="n/a"),
        width="stretch",
        hide_index=True,
    )

columns_per_row = 4
for selected_signal in selected_signals:
    st.write(f"#### Signal `{selected_signal}`")

    unavailable_modes = []
    for start in range(0, len(mode_filter), columns_per_row):
        columns = st.columns(columns_per_row)
        for column, sensor_mode in zip(
            columns,
            mode_filter[start:start + columns_per_row],
        ):
            with column:
                sensor_df, sensor_path = load_sensor_mode(sensor_mode)
                if sensor_df is None:
                    st.warning(f"Missing `{sensor_path.name}`")
                    continue

                if selected_signal not in sensor_df.columns:
                    unavailable_modes.append(sensor_mode)
                    continue

                augmented_experiment_df = sensor_df[
                    sensor_df[ID_COL].isin(selected_experiments)
                ].copy()
                if augmented_experiment_df.empty:
                    st.warning(
                        f"No rows for selected experiments "
                        f"in `{sensor_path.name}`."
                    )
                    continue

                metrics = metric_row(
                    group_metrics_df,
                    selected_group,
                    sensor_mode,
                    metric_split,
                )
                plot_mode_card(
                    mode=sensor_mode,
                    raw_experiment_df=raw_experiment_df,
                    augmented_experiment_df=augmented_experiment_df,
                    signal_col=selected_signal,
                    metrics=metrics,
                    zoom_window=zoom_window,
                )

    if unavailable_modes:
        st.info(
            f"`{selected_signal}` was skipped for modes where it is not saved: "
            + ", ".join(unavailable_modes)
        )
