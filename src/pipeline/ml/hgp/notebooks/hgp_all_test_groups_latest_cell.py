# ============================================================
# HGP: all test groups
#
# Output:
#   3 geometry sources x 2 axes = 6 figures
#
# Each figure:
#   - one subplot per test Group_ID
#   - actual group mean
#   - HGP prediction median/mean
#   - HGP prediction interval
#   - group-level metrics below each subplot
#   - model configuration in the figure header
#   - y-axis fixed between 20 and 25
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
                / "hgp"
            ).exists()
        ):
            return current

        if current == current.parent:
            raise RuntimeError(
                "Could not locate the project root."
            )

        current = current.parent


PROJECT_ROOT = find_project_root(Path.cwd())

HGP_ROOT = (
    PROJECT_ROOT
    / "src"
    / "pipeline"
    / "ml"
    / "hgp"
)

MODEL_RESULTS_DIR = (
    HGP_ROOT
    / "results"
    / "models"
)

SPLIT_METADATA_PATH = (
    PROJECT_ROOT
    / "src"
    / "pipeline"
    / "ml"
    / "common"
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

GEOMETRY_SOURCE_PATHS = {
    "real": (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "geometry.csv"
    ),
    (
        "sensor_augmented_noise__"
        "time_wrapping__scaling__jittering"
    ): (
        PROJECT_ROOT
        / "data"
        / "rf_augmented"
        / (
            "final_geometry_sensor_augmented_noise__"
            "time_wrapping__scaling__jittering.parquet"
        )
    ),
    "within_group_interpolation_raw": (
        PROJECT_ROOT
        / "data"
        / "rf_augmented"
        / (
            "final_geometry_within_group_"
            "interpolation_raw.parquet"
        )
    ),
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

SUBPLOT_COLUMNS = 5
SUBPLOT_WIDTH = 5.1
SUBPLOT_HEIGHT = 5.4

Y_AXIS_LIMITS = (20.5, 22.5)

SMOOTH_PLOTTED_CURVES = False
SMOOTHING_WINDOW = 3

SAVE_FIGURES = False

FIGURE_OUTPUT_DIR = (
    HGP_ROOT
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


_GEOMETRY_SOURCE_CACHE: dict[str, pd.DataFrame] = {}


def load_geometry_source_table(
    geometry_source: str,
) -> pd.DataFrame:
    if geometry_source not in _GEOMETRY_SOURCE_CACHE:
        if geometry_source not in GEOMETRY_SOURCE_PATHS:
            raise KeyError(
                f"Unknown geometry source: {geometry_source!r}"
            )

        geometry_df = read_table(
            GEOMETRY_SOURCE_PATHS[geometry_source]
        ).copy()

        geometry_df.columns = (
            geometry_df.columns.str.strip()
        )

        if (
            "Group_ID" not in geometry_df.columns
            and "group_id" in geometry_df.columns
        ):
            geometry_df["Group_ID"] = pd.to_numeric(
                geometry_df["group_id"],
                errors="raise",
            ).astype(int)

        _GEOMETRY_SOURCE_CACHE[geometry_source] = (
            geometry_df
        )

    return _GEOMETRY_SOURCE_CACHE[
        geometry_source
    ]


def count_source_experiments_for_group(
    geometry_source: str,
    group_id: int,
    fallback_group_df: pd.DataFrame,
) -> int:
    """
    Count source samples represented by the selected group.

    HGP test_predictions only contain the original split Experiment_IDs.
    For augmented/interpolation sources, the full source file stores the
    represented group size in target_group_samples, which is the number
    that should appear in the subplot title.
    """
    try:
        geometry_df = load_geometry_source_table(
            geometry_source
        )
    except (FileNotFoundError, KeyError):
        geometry_df = pd.DataFrame()

    if "Group_ID" in geometry_df.columns:
        source_group_df = geometry_df[
            geometry_df["Group_ID"]
            .astype(int)
            .eq(int(group_id))
        ].copy()
    else:
        source_group_df = pd.DataFrame()

    if not source_group_df.empty:
        if (
            "target_group_samples"
            in source_group_df.columns
            and source_group_df[
                "target_group_samples"
            ].notna().any()
        ):
            counts = pd.to_numeric(
                source_group_df[
                    "target_group_samples"
                ],
                errors="coerce",
            ).dropna()

            if not counts.empty:
                return int(counts.max())

        if "Experiment_ID" in source_group_df.columns:
            return int(
                source_group_df[
                    "Experiment_ID"
                ]
                .dropna()
                .nunique()
            )

    return int(
        fallback_group_df[
            "Experiment_ID"
        ]
        .dropna()
        .nunique()
    )


def load_split_config_from_artifact_metadata(
    axis: str,
    metadata: dict,
) -> dict:
    """
    Resolve split membership from the artifact metadata.

    The artifact is the source of truth for which split was
    actually trained. various_splits.parquet is used only to
    recover stored test Group_IDs for that split_index.
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
        score_column,
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
            int(metadata["hgp_rank"])
            if pd.notna(metadata.get("hgp_rank"))
            else int(row[rank_column])
        ),
        "score": (
            float(metadata["source_score"])
            if pd.notna(metadata.get("source_score"))
            else float(row[score_column])
        ),
        "source_rank_column": metadata.get(
            "source_rank_column",
            rank_column,
        ),
    }


# ============================================================
# Artifact loading
# ============================================================

def find_hgp_artifact(
    geometry_source: str,
    axis: str,
) -> tuple[Path, dict]:
    """
    Find the HGP artifact for source and axis.

    HGP stores stable artifact directories:
        results/models/<source>/<axis>/
    """
    artifact_directory = (
        MODEL_RESULTS_DIR
        / geometry_source
        / axis
    )

    metadata_path = (
        artifact_directory
        / "metadata.json"
    )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing HGP metadata: {metadata_path}"
        )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    return artifact_directory, metadata


def load_test_predictions(
    geometry_source: str,
    axis: str,
) -> tuple[pd.DataFrame, dict, Path, dict]:
    artifact_directory, metadata = find_hgp_artifact(
        geometry_source=geometry_source,
        axis=axis,
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
        "y_lower",
        "y_upper",
    }

    if "y_median" not in prediction_df.columns:
        if "y_mean" not in prediction_df.columns:
            required_columns.add("y_median")
        else:
            prediction_df["y_median"] = (
                prediction_df["y_mean"]
            )

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

def resolve_expected_coverage(
    metadata: dict,
) -> float:
    model_config = metadata.get(
        "model_config",
        {},
    )

    return float(
        model_config.get(
            "confidence_level",
            metadata.get(
                "confidence_level",
                0.90,
            ),
        )
    )


def compute_group_metrics(
    group_df: pd.DataFrame,
    expected_coverage: float,
) -> dict[str, float]:
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

    mean_params = model_config.get(
        "mean_kernel_params",
        {},
    )
    noise_params = model_config.get(
        "noise_kernel_params",
        {},
    )

    parts = [
        f"mean_kernel={model_config.get('mean_kernel', 'unknown')}",
        f"noise_kernel={model_config.get('noise_kernel', 'unknown')}",
        f"mean_gp_alpha={model_config.get('mean_gp_alpha', 'unknown')}",
        f"noise_gp_alpha={model_config.get('noise_gp_alpha', 'unknown')}",
        f"confidence_level={model_config.get('confidence_level', 'unknown')}",
        (
            "mean_length_scale="
            f"{mean_params.get('initial_length_scale', 'unknown')}"
        ),
        (
            "mean_constant="
            f"{mean_params.get('constant_value', 'unknown')}"
        ),
        (
            "noise_length_scale="
            f"{noise_params.get('initial_length_scale', 'unknown')}"
        ),
        (
            "noise_constant="
            f"{noise_params.get('constant_value', 'unknown')}"
        ),
        f"random_state={model_config.get('random_state', 'unknown')}",
    ]

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
    expected_coverage = resolve_expected_coverage(
        metadata
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
            color="#4C72B0",
            alpha=0.34,
            label=(
                f"Prediction interval "
                f"[confidence={expected_coverage:.3f}]"
            ),
            zorder=1,
        )

        ax.plot(
            x,
            curve_df["y_lower"].to_numpy(),
            color="#4C72B0",
            linewidth=0.75,
            alpha=0.85,
            linestyle=":",
            label="Prediction interval lower",
            zorder=2,
        )

        ax.plot(
            x,
            curve_df["y_upper"].to_numpy(),
            color="#4C72B0",
            linewidth=0.75,
            alpha=0.85,
            linestyle=":",
            label="Prediction interval upper",
            zorder=2,
        )

        ax.plot(
            x,
            curve_df["y_median"].to_numpy(),
            color="#FF8C00",
            linewidth=0.85,
            label="Prediction median",
            zorder=4,
        )

        ax.plot(
            x,
            curve_df["y_true"].to_numpy(),
            color="#025BFF",
            linewidth=0.85,
            linestyle="--",
            label="Actual",
            zorder=5,
        )

        ax.set_ylim(*Y_AXIS_LIMITS)

        number_of_experiments = (
            count_source_experiments_for_group(
                geometry_source=geometry_source,
                group_id=int(group_id),
                fallback_group_df=group_df,
            )
        )

        ax.set_title(
            (
                f"Group {group_id}\n"
                f"{number_of_experiments} source experiment"
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

    for unused_index in range(
        number_of_groups,
        len(flat_axes),
    ):
        flat_axes[unused_index].axis(
            "off"
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="#4C72B0",
            linewidth=7,
            alpha=0.34,
            label=(
                f"Prediction interval "
                f"[confidence={expected_coverage:.3f}]"
            ),
        ),
        Line2D(
            [0],
            [0],
            color="#4C72B0",
            linewidth=0.75,
            alpha=0.85,
            linestyle=":",
            label="Prediction interval bounds",
        ),
        Line2D(
            [0],
            [0],
            color="#FF8C00",
            linewidth=0.85,
            label="Prediction median",
        ),
        Line2D(
            [0],
            [0],
            color="#025BFF",
            linewidth=0.85,
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

    figure_title = (
        f"HGP test-group predictions | "
        f"{source_label} | {axis.upper()}\n"
        f"Target: {target_name} | "
        f"split_index={split_config['split_index']} | "
        f"{split_config['split_name']} | "
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
                f"hgp_all_test_groups__"
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


plot_summary_df = pd.DataFrame(
    [
        {
            "geometry_source": result["source"],
            "axis": result["axis"],
            "split_index": result["metadata"].get(
                "split_index"
            ),
            "split_name": result["metadata"].get(
                "split_name"
            ),
            "hgp_rank": result["metadata"].get(
                "hgp_rank"
            ),
            "source_score": result["metadata"].get(
                "source_score"
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
