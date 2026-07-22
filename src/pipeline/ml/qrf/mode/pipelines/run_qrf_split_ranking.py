from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TARGET_TO_AXIS = {
    "Main-axis [mm]": "main",
    "Secondary-axis [mm]": "secondary",
}


def find_repo_root(start: Path) -> Path:
    current = start.resolve()

    while current.name != "tube-bending-geometry":
        if current == current.parent:
            raise RuntimeError(
                "Could not find repository root named "
                "'tube-bending-geometry'."
            )
        current = current.parent

    return current


SCRIPT_PATH = Path(__file__).resolve()

QRF_ROOT = SCRIPT_PATH.parent.parent.parent
REPO_ROOT = find_repo_root(QRF_ROOT)

for import_root in (REPO_ROOT, QRF_ROOT):
    value = str(import_root)
    if value not in sys.path:
        sys.path.insert(0, value)

from src.pipeline.ml.qrf.mode.experiments.qrf_pipeline import (
    load_config,
    run_experiments,
)


def min_max_scale(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(
        series,
        errors="raise",
    ).astype(float)

    minimum = values.min()
    maximum = values.max()

    if np.isclose(minimum, maximum):
        return pd.Series(
            np.zeros(len(values), dtype=float),
            index=values.index,
        )

    return (values - minimum) / (maximum - minimum)


def validate_columns(
    frame: pd.DataFrame,
    required_columns: Iterable[str],
    frame_name: str,
) -> None:
    missing = set(required_columns).difference(frame.columns)

    if missing:
        raise KeyError(
            f"{frame_name} is missing required columns: "
            f"{sorted(missing)}"
        )


def unique_split_catalog(
    split_metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "split_index",
        "split_name",
        "train_group_ids",
        "test_group_ids",
    }
    validate_columns(
        split_metadata_df,
        required,
        "Split metadata",
    )

    catalog = (
        split_metadata_df[
            [
                "split_index",
                "split_name",
                "train_group_ids",
                "test_group_ids",
            ]
        ]
        .drop_duplicates(subset=["split_index"])
        .sort_values("split_index")
        .reset_index(drop=True)
    )

    duplicated_indices = (
        split_metadata_df[
            ["split_index", "split_name"]
        ]
        .drop_duplicates()
        .groupby("split_index")
        .size()
    )

    ambiguous = duplicated_indices[
        duplicated_indices.gt(1)
    ]

    if not ambiguous.empty:
        raise ValueError(
            "Some split_index values refer to more than one "
            "split_name. Ambiguous indices: "
            f"{ambiguous.index.tolist()}"
        )

    return catalog


def build_axis_ranking(
    metrics_df: pd.DataFrame,
    *,
    axis: str,
    lower_quantile: float,
    upper_quantile: float,
    n_estimators: int,
    coverage_weight: float,
    rmse_weight: float,
    mae_weight: float,
    interval_width_weight: float,
) -> pd.DataFrame:
    required = {
        "split_index",
        "axis",
        "n_estimators",
        "coverage_percent",
        "rmse",
        "mae",
        "mean_interval_width",
    }
    validate_columns(metrics_df, required, "Metrics")

    axis_metrics = metrics_df[
        metrics_df["axis"].eq(axis)
        & pd.to_numeric(
            metrics_df["n_estimators"],
            errors="coerce",
        ).eq(int(n_estimators))
    ].copy()

    if axis_metrics.empty:
        raise ValueError(
            f"No metrics found for axis={axis!r} and "
            f"n_estimators={n_estimators}."
        )

    if "split_name" not in axis_metrics.columns:
        axis_metrics["split_name"] = (
            "split "
            + axis_metrics["split_index"].astype(str)
        )

    target_coverage = (
        float(upper_quantile)
        - float(lower_quantile)
    ) * 100.0

    axis_metrics["coverage_error"] = (
        axis_metrics["coverage_percent"].astype(float)
        - target_coverage
    ).abs()

    ranking = (
        axis_metrics
        .groupby(
            ["split_index", "split_name"],
            as_index=False,
            dropna=False,
        )
        .agg(
            coverage_percent=("coverage_percent", "mean"),
            coverage_error=("coverage_error", "mean"),
            rmse=("rmse", "mean"),
            mae=("mae", "mean"),
            mean_interval_width=(
                "mean_interval_width",
                "mean",
            ),
        )
    )

    weights = np.asarray(
        [
            coverage_weight,
            rmse_weight,
            mae_weight,
            interval_width_weight,
        ],
        dtype=float,
    )

    if np.any(weights < 0):
        raise ValueError(
            "Ranking weights must be non-negative."
        )

    if np.isclose(weights.sum(), 0.0):
        raise ValueError(
            "At least one ranking weight must be positive."
        )

    weights = weights / weights.sum()

    ranking["coverage_error_scaled"] = min_max_scale(
        ranking["coverage_error"]
    )
    ranking["rmse_scaled"] = min_max_scale(
        ranking["rmse"]
    )
    ranking["mae_scaled"] = min_max_scale(
        ranking["mae"]
    )
    ranking["interval_width_scaled"] = min_max_scale(
        ranking["mean_interval_width"]
    )

    ranking["qrf_score"] = (
        ranking["coverage_error_scaled"] * weights[0]
        + ranking["rmse_scaled"] * weights[1]
        + ranking["mae_scaled"] * weights[2]
        + ranking["interval_width_scaled"] * weights[3]
    )

    ranking["axis"] = axis
    ranking["target_coverage_percent"] = (
        target_coverage
    )

    ranking = (
        ranking
        .sort_values(
            [
                "qrf_score",
                "coverage_error",
                "mean_interval_width",
                "rmse",
                "mae",
            ],
            ascending=True,
        )
        .reset_index(drop=True)
    )

    ranking.insert(
        0,
        "qrf_rank",
        range(1, len(ranking) + 1),
    )

    return ranking


def attach_rankings_to_metadata(
    split_metadata_df: pd.DataFrame,
    main_ranking_df: pd.DataFrame,
    secondary_ranking_df: pd.DataFrame,
) -> pd.DataFrame:
    output_df = split_metadata_df.copy()

    main_columns = {
        "qrf_rank": "qrf_rank_main",
        "qrf_score": "qrf_score_main",
        "coverage_percent": "qrf_coverage_main",
        "coverage_error": "qrf_coverage_error_main",
        "rmse": "qrf_rmse_main",
        "mae": "qrf_mae_main",
        "mean_interval_width": (
            "qrf_mean_interval_width_main"
        ),
    }

    secondary_columns = {
        "qrf_rank": "qrf_rank_secondary",
        "qrf_score": "qrf_score_secondary",
        "coverage_percent": "qrf_coverage_secondary",
        "coverage_error": (
            "qrf_coverage_error_secondary"
        ),
        "rmse": "qrf_rmse_secondary",
        "mae": "qrf_mae_secondary",
        "mean_interval_width": (
            "qrf_mean_interval_width_secondary"
        ),
    }

    main_lookup = (
        main_ranking_df[
            ["split_index", *main_columns.keys()]
        ]
        .rename(columns=main_columns)
    )

    secondary_lookup = (
        secondary_ranking_df[
            [
                "split_index",
                *secondary_columns.keys(),
            ]
        ]
        .rename(columns=secondary_columns)
    )

    columns_to_replace = set(
        main_columns.values()
    ).union(secondary_columns.values())

    existing_replaced_columns = [
        column
        for column in columns_to_replace
        if column in output_df.columns
    ]

    if existing_replaced_columns:
        output_df = output_df.drop(
            columns=existing_replaced_columns
        )

    output_df = output_df.merge(
        main_lookup,
        on="split_index",
        how="left",
        validate="many_to_one",
    )

    output_df = output_df.merge(
        secondary_lookup,
        on="split_index",
        how="left",
        validate="many_to_one",
    )

    if "target" in output_df.columns:
        output_df["qrf_axis"] = (
            output_df["target"].map(TARGET_TO_AXIS)
        )

        output_df["qrf_rank"] = np.where(
            output_df["qrf_axis"].eq("main"),
            output_df["qrf_rank_main"],
            np.where(
                output_df["qrf_axis"].eq("secondary"),
                output_df["qrf_rank_secondary"],
                np.nan,
            ),
        )

        output_df["qrf_score"] = np.where(
            output_df["qrf_axis"].eq("main"),
            output_df["qrf_score_main"],
            np.where(
                output_df["qrf_axis"].eq("secondary"),
                output_df["qrf_score_secondary"],
                np.nan,
            ),
        )

    return output_df


def backup_file(path: Path) -> Path:
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    backup_path = path.with_name(
        f"{path.stem}__backup_{timestamp}"
        f"{path.suffix}"
    )
    shutil.copy2(path, backup_path)
    return backup_path


def run_qrf_ranking_pipeline(
    config_path: Path,
    *,
    n_estimators: int = 200,
    coverage_weight: float = 0.15,
    rmse_weight: float = 0.25,
    mae_weight: float = 0.35,
    interval_width_weight: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    config = load_config(
        config_path=config_path,
        project_root=REPO_ROOT,
    )

    split_metadata_path = (
        REPO_ROOT
        / config["paths"]["split_metadata"]
    ).resolve()

    if not split_metadata_path.exists():
        raise FileNotFoundError(
            "Split metadata does not exist: "
            f"{split_metadata_path}"
        )

    original_metadata_df = pd.read_parquet(
        split_metadata_path
    )

    split_catalog_df = unique_split_catalog(
        original_metadata_df
    )

    split_indices = (
        split_catalog_df["split_index"]
        .astype(int)
        .tolist()
    )

    ranking_config = dict(config)
    ranking_config["search"] = dict(
        config["search"]
    )
    ranking_config["output"] = dict(
        config.get("output", {})
    )

    ranking_config["search"]["split_indices"] = (
        split_indices
    )
    ranking_config["search"]["n_estimators"] = [
        int(n_estimators)
    ]
    ranking_config["search"]["axes"] = [
        "main",
        "secondary",
    ]
    ranking_config["output"]["save_predictions"] = (
        False
    )

    print(
        f"Evaluating {len(split_indices)} splits "
        f"with {n_estimators} trees..."
    )

    metrics_df, _, run_dir = run_experiments(
        ranking_config
    )

    lower_quantile = ranking_config["model"][
        "lower_quantile"
    ]
    upper_quantile = ranking_config["model"][
        "upper_quantile"
    ]

    main_ranking_df = build_axis_ranking(
        metrics_df=metrics_df,
        axis="main",
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        n_estimators=n_estimators,
        coverage_weight=coverage_weight,
        rmse_weight=rmse_weight,
        mae_weight=mae_weight,
        interval_width_weight=(
            interval_width_weight
        ),
    )

    secondary_ranking_df = build_axis_ranking(
        metrics_df=metrics_df,
        axis="secondary",
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        n_estimators=n_estimators,
        coverage_weight=coverage_weight,
        rmse_weight=rmse_weight,
        mae_weight=mae_weight,
        interval_width_weight=(
            interval_width_weight
        ),
    )

    updated_metadata_df = (
        attach_rankings_to_metadata(
            split_metadata_df=original_metadata_df,
            main_ranking_df=main_ranking_df,
            secondary_ranking_df=(
                secondary_ranking_df
            ),
        )
    )

    backup_path = backup_file(
        split_metadata_path
    )

    updated_metadata_df.to_parquet(
        split_metadata_path,
        index=False,
    )

    ranking_output_dir = (
        run_dir / "axis_rankings"
    )
    ranking_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    main_ranking_df.to_csv(
        ranking_output_dir
        / "qrf_ranking_main.csv",
        index=False,
    )
    main_ranking_df.to_parquet(
        ranking_output_dir
        / "qrf_ranking_main.parquet",
        index=False,
    )

    secondary_ranking_df.to_csv(
        ranking_output_dir
        / "qrf_ranking_secondary.csv",
        index=False,
    )
    secondary_ranking_df.to_parquet(
        ranking_output_dir
        / "qrf_ranking_secondary.parquet",
        index=False,
    )

    print()
    print("Updated split metadata:")
    print(split_metadata_path)
    print()
    print("Backup:")
    print(backup_path)
    print()
    print("Detailed ranking tables:")
    print(ranking_output_dir)

    return (
        main_ranking_df,
        secondary_ranking_df,
        split_metadata_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate every split using QRF, rank "
            "main and secondary axes separately, and "
            "overwrite various_splits.parquet after "
            "creating a timestamped backup."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=(
            QRF_ROOT
            / "configs"
            / "qrf_split_ranking.yaml"
        ),
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--coverage-weight",
        type=float,
        default=0.60,
    )
    parser.add_argument(
        "--rmse-weight",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--mae-weight",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--interval-width-weight",
        type=float,
        default=0.10,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    main_ranking_df, secondary_ranking_df, _ = (
        run_qrf_ranking_pipeline(
            config_path=args.config,
            n_estimators=args.n_estimators,
            coverage_weight=args.coverage_weight,
            rmse_weight=args.rmse_weight,
            mae_weight=args.mae_weight,
            interval_width_weight=(
                args.interval_width_weight
            ),
        )
    )

    print()
    print("Top 3 — main axis")
    print(
        main_ranking_df[
            [
                "qrf_rank",
                "split_index",
                "split_name",
                "qrf_score",
                "coverage_percent",
                "coverage_error",
                "rmse",
                "mae",
                "mean_interval_width",
            ]
        ].head(3).to_string(index=False)
    )

    print()
    print("Worst 3 — main axis")
    print(
        main_ranking_df[
            [
                "qrf_rank",
                "split_index",
                "split_name",
                "qrf_score",
                "coverage_percent",
                "coverage_error",
                "rmse",
                "mae",
                "mean_interval_width",
            ]
        ].tail(3).to_string(index=False)
    )

    print()
    print("Top 3 — secondary axis")
    print(
        secondary_ranking_df[
            [
                "qrf_rank",
                "split_index",
                "split_name",
                "qrf_score",
                "coverage_percent",
                "coverage_error",
                "rmse",
                "mae",
                "mean_interval_width",
            ]
        ].head(3).to_string(index=False)
    )

    print()
    print("Worst 3 — secondary axis")
    print(
        secondary_ranking_df[
            [
                "qrf_rank",
                "split_index",
                "split_name",
                "qrf_score",
                "coverage_percent",
                "coverage_error",
                "rmse",
                "mae",
                "mean_interval_width",
            ]
        ].tail(3).to_string(index=False)
    )


if __name__ == "__main__":
    main()
