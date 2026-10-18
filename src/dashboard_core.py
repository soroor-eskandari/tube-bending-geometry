from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ANGLE_COL = "Angle[degree]ORDistance[mm]"
GROUP_COL = "Group_ID"
EXPERIMENT_COL = "Experiment_ID"
MAIN_AXIS = "Main-axis [mm]"
SECONDARY_AXIS = "Secondary-axis [mm]"


def parse_int_list(value: object) -> list[int]:
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str):
        try:
            raw_items = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            raw_items = []
    else:
        raw_items = []

    return [int(item) for item in raw_items]


def experiment_group_mapping(setup: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, int]] = []
    for _, row in setup.iterrows():
        group_id = int(row[GROUP_COL])
        for experiment_id in parse_int_list(row["Experiment_Number"]):
            rows.append({EXPERIMENT_COL: experiment_id, GROUP_COL: group_id})
    return pd.DataFrame(rows)


def attach_groups(geometry: pd.DataFrame, setup: pd.DataFrame) -> pd.DataFrame:
    mapping = experiment_group_mapping(setup)
    return geometry.merge(mapping, on=EXPERIMENT_COL, how="inner")


def axis_column(axis_key: str) -> str:
    return MAIN_AXIS if axis_key == "main" else SECONDARY_AXIS


def axis_label(axis_key: str) -> str:
    return "Main-axis" if axis_key == "main" else "Secondary-axis"


def group_experiments(setup: pd.DataFrame, group_id: int) -> list[int]:
    rows = setup[setup[GROUP_COL].astype(int).eq(int(group_id))]
    if rows.empty:
        return []
    return parse_int_list(rows.iloc[0]["Experiment_Number"])


def geometry_summary(geometry: pd.DataFrame, axis_col: str) -> pd.DataFrame:
    if geometry.empty:
        return pd.DataFrame(
            columns=[ANGLE_COL, "mean", "min", "max", "std", "range"]
        )
    summary = (
        geometry.groupby(ANGLE_COL, as_index=False)
        .agg(
            mean=(axis_col, "mean"),
            min=(axis_col, "min"),
            max=(axis_col, "max"),
            std=(axis_col, "std"),
        )
        .sort_values(ANGLE_COL)
    )
    summary["range"] = summary["max"] - summary["min"]
    return summary


def mean_pairwise_curve_distance(
    group_geometry: pd.DataFrame,
    axis_col: str,
    experiment_col: str = EXPERIMENT_COL,
    angle_col: str = ANGLE_COL,
) -> float | None:
    if group_geometry[experiment_col].nunique() < 2:
        return None

    curves = (
        group_geometry.pivot_table(
            index=experiment_col,
            columns=angle_col,
            values=axis_col,
            aggfunc="mean",
        )
        .dropna(axis=1, how="any")
        .dropna(axis=0, how="any")
    )
    if len(curves) < 2 or curves.shape[1] == 0:
        return None

    values = curves.to_numpy(dtype=float)
    distances = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            distances.append(float(np.linalg.norm(values[i] - values[j])))
    return float(np.mean(distances)) if distances else None


def variability_by_group(
    observed: pd.DataFrame,
    setup: pd.DataFrame,
    axis_col: str,
    parameter: str,
) -> pd.DataFrame:
    rows = []
    for group_id, group_df in observed.groupby(GROUP_COL):
        distance = mean_pairwise_curve_distance(group_df, axis_col)
        setup_row = setup[setup[GROUP_COL].astype(int).eq(int(group_id))]
        parameter_value = (
            setup_row.iloc[0][parameter]
            if not setup_row.empty and parameter in setup_row.columns
            else np.nan
        )
        rows.append(
            {
                GROUP_COL: int(group_id),
                "n_experiments": int(group_df[EXPERIMENT_COL].nunique()),
                "mean_pairwise_distance": distance,
                parameter: parameter_value,
            }
        )
    return pd.DataFrame(rows)


def prediction_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    center = "y_median" if "y_median" in predictions.columns else "y_mean"
    return (
        predictions.groupby(ANGLE_COL, as_index=False)
        .agg(
            y_true=("y_true", "mean"),
            y_lower=("y_lower", "mean"),
            y_center=(center, "mean"),
            y_upper=("y_upper", "mean"),
        )
        .sort_values(ANGLE_COL)
    )


def metric_value(metrics: pd.DataFrame, metric: str) -> float | None:
    if metrics.empty or metric not in metrics.columns:
        return None
    value = metrics.iloc[0][metric]
    return None if pd.isna(value) else float(value)


def find_files(root: Path, names: Iterable[str]) -> list[Path]:
    wanted = set(names)
    return sorted(path for path in root.rglob("*") if path.name in wanted)
