# ============================================================
# QRF: all test groups
#
# Output:
#   3 geometry sources × 2 axes = 6 figures
#
# Each figure:
#   - one subplot per test Group_ID
#   - actual group mean
#   - QRF prediction median
#   - QRF prediction interval
#   - group-level metrics below each subplot
#   - model configuration in the figure header
# ============================================================

from pathlib import Path
import ast
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# Configuration
# ============================================================

def find_project_root(start: Path) -> Path:
    current = start.resolve()

    while True:
        if (
            (current / "data").exists()
            and (
                current
                / "src"
                / "pipeline"
                / "ml"
                / "qrf"
            ).exists()
        ):
            return current

        if current == current.parent:
            raise RuntimeError(
                "Could not locate the project root."
            )

        current = current.parent


PROJECT_ROOT = find_project_root(Path.cwd())

QRF_ROOT = (
    PROJECT_ROOT
    / "src"
    / "pipeline"
    / "ml"
    / "qrf"
)

MODEL_RESULTS_DIR = (
    QRF_ROOT
    / "results"
    / "models"
)

SPLIT_METADATA_PATH = (
    QRF_ROOT
    / "data"
    / "various_splits.parquet"
)

DATA_SOURCES = {
    "real": "Real",
    (
        "sensor_augmented_noise__"
        "time_wrapping__scaling__jittering"
    ): "Augmented",
    "within_group_interpolation_raw": "Interpolation",
}

AXIS_CONFIG = {
    "main": {
        "target": "Main-axis [mm]",
    },
    "secondary": {
        "target": "Secondary-axis [mm]",
    },
}

ANGLE_COLUMN = "Angle[degree]ORDistance[mm]"

# Five columns gives 4 rows when there are 20 test groups.
SUBPLOT_COLUMNS = 5

# Increase this when labels or metrics look crowded.
SUBPLOT_WIDTH = 5.1
SUBPLOT_HEIGHT = 5.4

# Fixed y-axis range for every subplot.
Y_AXIS_LIMITS = (20, 25)

# True applies a small rolling average only to plotted curves.
# Metrics are always calculated from the original unsmoothed rows.
SMOOTH_PLOTTED_CURVES = False
SMOOTHING_WINDOW = 3

SAVE_FIGURES = False

FIGURE_OUTPUT_DIR = (
    QRF_ROOT
    / "results"
    / "all_test_group_figures"
)

if SAVE_FIGURES:
    FIGURE_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

print("Project root:", PROJECT_ROOT)
print("Model directory:", MODEL_RESULTS_DIR)


# ============================================================
# Data helpers
# ============================================================

def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(
        f"Unsupported table format: {path}"
    )


def decode_integer_list(value) -> list[int]:
    if isinstance(value, str):
        value = ast.literal_eval(value)

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, pd.Series):
        value = value.tolist()

    if isinstance(value, (list, tuple, set)):
        return sorted(
            {
                int(item)
                for item in value
                if pd.notna(item)
            }
        )

    if pd.isna(value):
        return []

    return [int(value)]


def load_split_config_from_artifact_metadata(
    axis: str,
    metadata: dict,
) -> dict:
    """
    Resolve split membership from the artifact metadata.

    The artifact is the source of truth for which split was
    actually trained. various_splits.parquet is used only to
    recover the stored test group IDs for that split_index.
    """
    if axis not in AXIS_CONFIG:
        raise KeyError(
            f"Unknown axis: {axis!r}"
        )

    split_index = int(
        metadata["split_index"]
    )
    rank_column = f"qrf_rank_{axis}"
    score_column = f"qrf_score_{axis}"

    split_df = pd.read_parquet(
        SPLIT_METADATA_PATH
    )

    required_columns = {
        "split_index",
        "split_name",
        "test_group_ids",
        rank_column,
    }

    missing = required_columns.difference(
        split_df.columns
    )

    if missing:
        raise KeyError(
            "Split metadata is missing columns: "
            f"{sorted(missing)}"
        )

    matches = split_df[
        pd.to_numeric(
            split_df["split_index"],
            errors="raise",
        ).astype(int).eq(split_index)
    ].copy()

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one split with "
            f"split_index={split_index}, "
            f"but found {len(matches)}."
        )

    row = matches.iloc[0]

    return {
        "axis": axis,
        "split_index": split_index,
        "split_name": str(
            metadata.get(
                "split_name",
                row["split_name"],
            )
        ),
        "test_group_ids": decode_integer_list(
            row["test_group_ids"]
        ),
        "rank": (
            int(metadata["qrf_rank"])
            if pd.notna(metadata.get("qrf_rank"))
            else (
                int(row[rank_column])
                if pd.notna(row[rank_column])
                else None
            )
        ),
        "score": (
            float(metadata["qrf_score"])
            if pd.notna(metadata.get("qrf_score"))
            else (
                float(row[score_column])
                if (
                    score_column in row.index
                    and pd.notna(row[score_column])
                )
                else None
            )
        ),
        "split_selection_mode": metadata.get(
            "split_selection_mode",
            "unknown",
        ),
    }


# ============================================================
# Artifact loading
# ============================================================

def find_latest_qrf_artifact(
    geometry_source: str,
    axis: str,
    split_index: int | None = None,
) -> tuple[Path, dict]:
    """
    Find the newest artifact matching source and axis.

    When split_index is provided, restrict the search to that
    split. Otherwise the newest artifact metadata is used as
    the source of truth for the selected split.
    """
    axis_directory = (
        MODEL_RESULTS_DIR
        / geometry_source
        / axis
    )

    if not axis_directory.exists():
        raise FileNotFoundError(
            f"Missing QRF axis directory: "
            f"{axis_directory}"
        )

    candidates = []

    for metadata_path in axis_directory.glob(
        "*/metadata.json"
    ):
        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)

        metadata_split_index = metadata.get(
            "split_index"
        )

        if metadata_split_index is None:
            continue

        if (
            split_index is not None
            and int(metadata_split_index) != int(split_index)
        ):
            continue

        artifact_directory = metadata_path.parent

        candidates.append(
            {
                "modified_time": (
                    artifact_directory
                    .stat()
                    .st_mtime
                ),
                "artifact_directory": artifact_directory,
                "metadata": metadata,
            }
        )

    if not candidates:
        raise FileNotFoundError(
            "No matching QRF artifact found for "
            f"source={geometry_source!r}, "
            f"axis={axis!r}, "
            f"split_index={split_index}."
        )

    selected = max(
        candidates,
        key=lambda item: item["modified_time"],
    )

    return (
        selected["artifact_directory"],
        selected["metadata"],
    )


def load_test_predictions(
    geometry_source: str,
    axis: str,
) -> tuple[pd.DataFrame, dict, Path, dict]:
    artifact_directory, metadata = (
        find_latest_qrf_artifact(
            geometry_source=geometry_source,
            axis=axis,
        )
    )

    split_config = load_split_config_from_artifact_metadata(
        axis=axis,
        metadata=metadata,
    )

    prediction_path = (
        artifact_directory
        / "test_predictions.parquet"
    )

    if not prediction_path.exists():
        prediction_path = (
            artifact_directory
            / "test_predictions.csv"
        )

    if not prediction_path.exists():
        raise FileNotFoundError(
            "No test_predictions.parquet or "
            f"test_predictions.csv found in "
            f"{artifact_directory}"
        )

    prediction_df = read_table(
        prediction_path
    ).copy()

    required_columns = {
        ANGLE_COLUMN,
        "Group_ID",
        "Experiment_ID",
        "y_true",
        "y_median",
        "y_lower",
        "y_upper",
    }

    missing = required_columns.difference(
        prediction_df.columns
    )

    if missing:
        raise KeyError(
            f"{prediction_path} is missing columns: "
            f"{sorted(missing)}"
        )

    numeric_columns = [
        ANGLE_COLUMN,
        "Group_ID",
        "Experiment_ID",
        "y_true",
        "y_median",
        "y_lower",
        "y_upper",
    ]

    for column in numeric_columns:
        prediction_df[column] = pd.to_numeric(
            prediction_df[column],
            errors="coerce",
        )

    prediction_df = prediction_df.dropna(
        subset=[
            ANGLE_COLUMN,
            "Group_ID",
            "y_true",
            "y_median",
            "y_lower",
            "y_upper",
        ]
    ).copy()

    prediction_df["Group_ID"] = (
        prediction_df["Group_ID"]
        .astype(int)
    )

    prediction_df = prediction_df[
        prediction_df[ANGLE_COLUMN].between(
            0,
            44,
            inclusive="both",
        )
    ].copy()

    expected_groups = set(
        split_config["test_group_ids"]
    )

    prediction_df = prediction_df[
        prediction_df["Group_ID"].isin(
            expected_groups
        )
    ].copy()

    return (
        prediction_df,
        metadata,
        artifact_directory,
        split_config,
    )


# ============================================================
# Metrics
# ============================================================

def resolve_quantiles(
    metadata: dict,
) -> tuple[float, float]:
    model_config = metadata.get(
        "model_config",
        {},
    )

    lower_quantile = float(
        model_config.get(
            "lower_quantile",
            metadata.get(
                "lower_quantile",
                0.05,
            ),
        )
    )

    upper_quantile = float(
        model_config.get(
            "upper_quantile",
            metadata.get(
                "upper_quantile",
                0.95,
            ),
        )
    )

    return (
        lower_quantile,
        upper_quantile,
    )


def compute_group_metrics(
    group_df: pd.DataFrame,
    expected_coverage: float,
) -> dict[str, float]:
    """
    Match the metric definitions currently used by qrf_ui.

    Metrics are calculated on all raw test rows belonging
    to the selected Group_ID.
    """
    y_true = group_df["y_true"].to_numpy(
        dtype=float
    )

    y_median = group_df["y_median"].to_numpy(
        dtype=float
    )

    y_lower = group_df["y_lower"].to_numpy(
        dtype=float
    )

    y_upper = group_df["y_upper"].to_numpy(
        dtype=float
    )

    finite_mask = (
        np.isfinite(y_true)
        & np.isfinite(y_median)
        & np.isfinite(y_lower)
        & np.isfinite(y_upper)
    )

    y_true = y_true[finite_mask]
    y_median = y_median[finite_mask]
    y_lower = y_lower[finite_mask]
    y_upper = y_upper[finite_mask]

    if len(y_true) == 0:
        return {
            "coverage": np.nan,
            "expected_coverage": expected_coverage,
            "calibration_error": np.nan,
            "mean_interval_width": np.nan,
            "residual_std": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "bias": np.nan,
            "abs_bias": np.nan,
        }

    inside_interval = (
        (y_true >= y_lower)
        & (y_true <= y_upper)
    )

    residual = y_median - y_true

    coverage = float(
        inside_interval.mean()
    )

    rmse = float(
        np.sqrt(
            np.mean(
                np.square(residual)
            )
        )
    )

    mae = float(
        np.mean(
            np.abs(residual)
        )
    )

    bias = float(
        np.mean(residual)
    )

    return {
        "coverage": coverage,
        "expected_coverage": expected_coverage,
        "calibration_error": abs(
            coverage - expected_coverage
        ),
        "mean_interval_width": float(
            np.mean(y_upper - y_lower)
        ),
        "residual_std": float(
            np.std(
                residual,
                ddof=0,
            )
        ),
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "abs_bias": abs(bias),
    }


# ============================================================
# Plot preparation
# ============================================================

def prepare_group_curve(
    group_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Average experiments inside one Group_ID at each angle.

    This removes individual experiment signals and leaves:
      - actual group curve
      - median prediction curve
      - prediction interval
    """
    curve_df = (
        group_df
        .groupby(
            ANGLE_COLUMN,
            as_index=False,
        )
        .agg(
            y_true=("y_true", "mean"),
            y_median=("y_median", "mean"),
            y_lower=("y_lower", "mean"),
            y_upper=("y_upper", "mean"),
        )
        .sort_values(
            ANGLE_COLUMN
        )
        .reset_index(drop=True)
    )

    if (
        SMOOTH_PLOTTED_CURVES
        and len(curve_df) >= 2
    ):
        for column in [
            "y_true",
            "y_median",
            "y_lower",
            "y_upper",
        ]:
            curve_df[column] = (
                curve_df[column]
                .rolling(
                    window=SMOOTHING_WINDOW,
                    center=True,
                    min_periods=1,
                )
                .mean()
            )

    return curve_df


def format_model_config(
    metadata: dict,
) -> str:
    model_config = metadata.get(
        "model_config",
        {},
    )

    parameter_names = [
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "min_samples_split",
        "max_features",
        "bootstrap",
        "max_samples",
        "lower_quantile",
        "upper_quantile",
        "random_state",
    ]

    parts = []

    for parameter_name in parameter_names:
        if parameter_name in model_config:
            parts.append(
                f"{parameter_name}="
                f"{model_config[parameter_name]}"
            )

    if not parts:
        return "Model configuration unavailable"

    # Split the config over multiple lines.
    lines = []

    for start in range(
        0,
        len(parts),
        4,
    ):
        lines.append(
            " | ".join(
                parts[start:start + 4]
            )
        )

    return "\n".join(lines)


def format_metric_text(
    metrics: dict[str, float],
) -> str:
    return (
        f"Coverage: {metrics['coverage']:.3f}    "
        f"Expected: {metrics['expected_coverage']:.3f}    "
        f"Calibration error: "
        f"{metrics['calibration_error']:.3f}\n"
        f"Mean width: "
        f"{metrics['mean_interval_width']:.4f}    "
        f"Residual std: "
        f"{metrics['residual_std']:.4f}    "
        f"RMSE: {metrics['rmse']:.4f}\n"
        f"MAE: {metrics['mae']:.4f}    "
        f"Bias: {metrics['bias']:.4f}    "
        f"Abs bias: {metrics['abs_bias']:.4f}"
    )


# ============================================================
# Main plotting function
# ============================================================

def plot_all_test_groups(
    geometry_source: str,
    source_label: str,
    axis: str,
):
    (
        prediction_df,
        metadata,
        artifact_directory,
        split_config,
    ) = load_test_predictions(
        geometry_source=geometry_source,
        axis=axis,
    )

    target_name = AXIS_CONFIG[axis]["target"]

    lower_quantile, upper_quantile = (
        resolve_quantiles(metadata)
    )

    expected_coverage = (
        upper_quantile
        - lower_quantile
    )

    expected_group_ids = (
        split_config["test_group_ids"]
    )

    present_group_ids = sorted(
        prediction_df["Group_ID"]
        .unique()
        .tolist()
    )

    missing_group_ids = sorted(
        set(expected_group_ids)
        - set(present_group_ids)
    )

    if missing_group_ids:
        print(
            f"Missing prediction groups for "
            f"{source_label} / {axis}: "
            f"{missing_group_ids}"
        )

    group_ids = [
        group_id
        for group_id in expected_group_ids
        if group_id in present_group_ids
    ]

    if not group_ids:
        raise ValueError(
            f"No test groups found for "
            f"{source_label} / {axis}."
        )

    number_of_groups = len(group_ids)

    number_of_columns = min(
        SUBPLOT_COLUMNS,
        number_of_groups,
    )

    number_of_rows = math.ceil(
        number_of_groups
        / number_of_columns
    )

    fig, axes = plt.subplots(
        nrows=number_of_rows,
        ncols=number_of_columns,
        figsize=(
            SUBPLOT_WIDTH * number_of_columns,
            SUBPLOT_HEIGHT * number_of_rows,
        ),
        squeeze=False,
        sharex=True,
    )

    flat_axes = axes.ravel()

    for plot_index, group_id in enumerate(
        group_ids
    ):
        ax = flat_axes[plot_index]

        group_df = prediction_df[
            prediction_df["Group_ID"].eq(
                int(group_id)
            )
        ].copy()

        group_df = group_df.sort_values(
            [
                ANGLE_COLUMN,
                "Experiment_ID",
            ]
        )

        curve_df = prepare_group_curve(
            group_df
        )

        metrics = compute_group_metrics(
            group_df=group_df,
            expected_coverage=expected_coverage,
        )

        x = curve_df[
            ANGLE_COLUMN
        ].to_numpy()

        ax.fill_between(
            x,
            curve_df["y_lower"].to_numpy(),
            curve_df["y_upper"].to_numpy(),
            alpha=0.22,
            label=(
                f"Prediction interval "
                f"[{lower_quantile:.3f}, "
                f"{upper_quantile:.3f}]"
            ),
        )

        ax.plot(
            x,
            curve_df["y_median"].to_numpy(),
            linewidth=2.7,
            label="Prediction median",
        )

        ax.plot(
            x,
            curve_df["y_true"].to_numpy(),
            linewidth=2.2,
            linestyle="--",
            label="Actual",
        )

        number_of_experiments = int(
            group_df["Experiment_ID"]
            .dropna()
            .nunique()
        )

        ax.set_title(
            (
                f"Group {group_id}\n"
                f"{number_of_experiments} test experiment"
                f"{'s' if number_of_experiments != 1 else ''}"
            ),
            fontsize=11,
            pad=8,
        )

        ax.set_xlabel(
            "Angle [degree]",
            fontsize=9,
        )

        ax.set_ylabel(
            target_name,
            fontsize=9,
        )

        ax.set_ylim(*Y_AXIS_LIMITS)

        ax.tick_params(
            axis="both",
            labelsize=8,
        )

        ax.grid(
            True,
            alpha=0.20,
        )

        ax.spines["top"].set_visible(
            False
        )

        ax.spines["right"].set_visible(
            False
        )

        # Metrics directly below each subplot.
        ax.text(
            0.5,
            -0.31,
            format_metric_text(metrics),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7.4,
            linespacing=1.35,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "0.80",
                "alpha": 0.95,
            },
            clip_on=False,
        )

    # Hide unused subplot locations.
    for unused_index in range(
        number_of_groups,
        len(flat_axes),
    ):
        flat_axes[unused_index].axis(
            "off"
        )

    # One common legend because the line meanings are identical.
    legend_handles = [
        Line2D(
            [0],
            [0],
            linewidth=7,
            alpha=0.22,
            label=(
                f"Prediction interval "
                f"[{lower_quantile:.3f}, "
                f"{upper_quantile:.3f}]"
            ),
        ),
        Line2D(
            [0],
            [0],
            linewidth=2.7,
            label="Prediction median",
        ),
        Line2D(
            [0],
            [0],
            linewidth=2.2,
            linestyle="--",
            label="Actual",
        ),
    ]

    model_config_text = format_model_config(
        metadata
    )

    rank_text = (
        f"rank={split_config['rank']}"
        if split_config["rank"] is not None
        else "rank=unknown"
    )

    score_text = (
        f"score={split_config['score']:.6f}"
        if split_config["score"] is not None
        else "score=unknown"
    )

    selection_text = (
        f"selection={split_config['split_selection_mode']}"
    )


    figure_title = (
        f"QRF test-group predictions | "
        f"{source_label} | {axis.upper()}\n"
        f"Target: {target_name} | "
        f"split_index={split_config['split_index']} | "
        f"{split_config['split_name']} | "
        f"{selection_text} | "
        f"{rank_text} | {score_text} | "
        f"groups={number_of_groups}\n"
        f"{model_config_text}"
    )

    fig.suptitle(
        figure_title,
        fontsize=14,
        y=0.995,
    )

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            0.935,
        ),
        ncol=3,
        frameon=True,
    )

    # Extra vertical space is required for the metric boxes.
    fig.subplots_adjust(
        top=0.86,
        bottom=0.08,
        hspace=0.78,
        wspace=0.28,
    )

    if SAVE_FIGURES:
        output_path = (
            FIGURE_OUTPUT_DIR
            / (
                f"qrf_all_test_groups__"
                f"{geometry_source}__"
                f"{axis}.png"
            )
        )

        fig.savefig(
            output_path,
            dpi=180,
            bbox_inches="tight",
        )

        print(
            "Saved:",
            output_path,
        )

    plt.show()

    return {
        "source": geometry_source,
        "axis": axis,
        "artifact_directory": artifact_directory,
        "number_of_groups": number_of_groups,
        "missing_group_ids": missing_group_ids,
        "metadata": metadata,
    }


# ============================================================
# Generate all six figures
# ============================================================

plot_results = []

for geometry_source, source_label in DATA_SOURCES.items():
    for axis in [
        "main",
        "secondary",
    ]:
        print(
            "\n"
            + "=" * 90
        )
        print(
            f"Plotting {source_label} / {axis}"
        )
        print(
            "=" * 90
        )

        result = plot_all_test_groups(
            geometry_source=geometry_source,
            source_label=source_label,
            axis=axis,
        )

        plot_results.append(result)


# Optional execution summary
plot_summary_df = pd.DataFrame(
    [
        {
            "geometry_source": result["source"],
            "axis": result["axis"],
            "split_selection_mode": result["metadata"].get(
                "split_selection_mode",
                "unknown",
            ),
            "split_index": result["metadata"].get(
                "split_index"
            ),
            "split_name": result["metadata"].get(
                "split_name"
            ),
            "qrf_rank": result["metadata"].get(
                "qrf_rank"
            ),
            "qrf_score": result["metadata"].get(
                "qrf_score"
            ),
            "number_of_groups": result[
                "number_of_groups"
            ],
            "missing_group_ids": result[
                "missing_group_ids"
            ],
            "artifact_directory": str(
                result["artifact_directory"]
            ),
        }
        for result in plot_results
    ]
)

display(plot_summary_df)