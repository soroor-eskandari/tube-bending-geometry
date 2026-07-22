import logging
import ast
from pathlib import Path

import pandas as pd

from src.pipeline.ml.qrf.mode.experiments.qrf_pipeline import (
    run,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


QRF_FEATURE_COLUMNS = [
    "Group_ID",
    "Angle[degree]ORDistance[mm]",
]

QRF_EXCLUDED_EXPERIMENTS = [
    1,
    48,
    166,
]

STORED_QRF_RANKING_SOURCE = (
    "sensor_augmented_noise__time_wrapping__scaling__jittering"
)

BEST_QRF_PARAMS_BY_DATASET = {
    "real": {
        "n_estimators": 300,
        "max_depth": 5,
        "lower_quantile": 0.05,
        "upper_quantile": 0.95,
    },
    "sensor_augmented_noise__time_wrapping__scaling__jittering": {
        "n_estimators": 300,
        "max_depth": 5,
        "lower_quantile": 0.05,
        "upper_quantile": 0.95,
    },
    "within_group_interpolation_raw": {
        "n_estimators": 300,
        "max_depth": 5,
        "lower_quantile": 0.05,
        "upper_quantile": 0.95,
    },
}


STORED_QRF_SPLIT_PATH = Path(
    "src/pipeline/ml/qrf/data/various_splits.parquet"
)
STORED_QRF_MODEL_DIR = Path("src/pipeline/ml/qrf/result/models")


def load_top_qrf_splits_by_axis(
    project_root: Path,
    ranking_source: str,
) -> dict[str, dict]:
    """
    Load the independently ranked Top-1 split for the main and
    secondary axes of one geometry source.
    """
    split_path = (
        project_root
        / STORED_QRF_SPLIT_PATH
    ).resolve()

    if not split_path.exists():
        raise FileNotFoundError(
            f"Stored QRF split file was not found: "
            f"{split_path}"
        )

    required_columns = [
        "split_index",
        "geometry_source",
        "split_name",
        "train_experiment_ids",
        "test_experiment_ids",
        "qrf_rank_main",
        "qrf_score_main",
        "qrf_rank_secondary",
        "qrf_score_secondary",
    ]

    split_df = pd.read_parquet(
        split_path,
        columns=required_columns,
    )

    source_df = split_df[
        split_df["geometry_source"].eq(
            ranking_source
        )
    ].copy()

    if source_df.empty:
        available_sources = sorted(
            split_df["geometry_source"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "No stored QRF ranking was found for "
            f"geometry_source={ranking_source!r}. "
            f"Available sources: {available_sources}"
        )

    # The parquet has multiple prediction rows for each split.
    split_catalog_df = (
        source_df
        .drop_duplicates(
            subset=["split_index"]
        )
        .reset_index(drop=True)
    )

    axis_columns = {
        "main": {
            "rank": "qrf_rank_main",
            "score": "qrf_score_main",
            "target": "Main-axis [mm]",
        },
        "secondary": {
            "rank": "qrf_rank_secondary",
            "score": "qrf_score_secondary",
            "target": "Secondary-axis [mm]",
        },
    }

    selected: dict[str, dict] = {}

    for axis, columns in axis_columns.items():
        rank_column = columns["rank"]
        score_column = columns["score"]

        ranked_df = split_catalog_df.dropna(
            subset=[rank_column]
        ).copy()

        ranked_df[rank_column] = pd.to_numeric(
            ranked_df[rank_column],
            errors="raise",
        ).astype(int)

        top_df = ranked_df[
            ranked_df[rank_column].eq(1)
        ].copy()

        if top_df.empty:
            raise ValueError(
                f"No Top-1 QRF split was found for "
                f"source={ranking_source!r}, "
                f"axis={axis!r}."
            )

        unique_top_df = top_df.drop_duplicates(
            subset=["split_index"]
        )

        if len(unique_top_df) != 1:
            raise ValueError(
                f"Expected exactly one Top-1 split for "
                f"source={ranking_source!r}, "
                f"axis={axis!r}, but found "
                f"{len(unique_top_df)}: "
                f"{unique_top_df['split_index'].tolist()}"
            )

        row = unique_top_df.iloc[0]

        selected[axis] = {
            "axis": axis,
            "target_column": columns["target"],
            "split_index": int(
                row["split_index"]
            ),
            "split_name": str(
                row["split_name"]
            ),
            "qrf_rank": int(
                row[rank_column]
            ),
            "qrf_score": float(
                row[score_column]
            ),
            "train_exp": (
                _normalize_stored_experiment_ids(
                    row["train_experiment_ids"]
                )
            ),
            "test_exp": (
                _normalize_stored_experiment_ids(
                    row["test_experiment_ids"]
                )
            ),
        }

    return selected


def _normalize_stored_experiment_ids(value) -> list[int]:
    if isinstance(value, str):
        value = ast.literal_eval(value)

    return sorted(set(map(int, value)))


def qrf_training_geometry_sources(
    project_root: Path,
) -> dict[str, Path]:
    return {
        "real": (
            project_root
            / "data"
            / "processed"
            / "geometry.csv"
        ),
        "sensor_augmented_noise__time_wrapping__scaling__jittering": (
            project_root
            / "data"
            / "rf_augmented"
            / "ui_data"
            / (
                "final_geometry_sensor_augmented_noise__"
                "time_wrapping__scaling__jittering.csv"
            )
        ),
        "within_group_interpolation_raw": (
            project_root
            / "data"
            / "rf_augmented"
            / "ui_data"
            / (
                "final_geometry_within_group_"
                "interpolation_raw.csv"
            )
        ),
    }


if __name__ == "__main__":
    project_root = Path(
        __file__
    ).resolve().parent

    use_best_qrf_params = True

    geometry_sources = (
        qrf_training_geometry_sources(
            project_root
        )
    )

    logger.info(
        "Available geometry sources: %s",
        sorted(geometry_sources),
    )

    shared_top_splits_by_axis = (
        load_top_qrf_splits_by_axis(
            project_root=project_root,
            ranking_source=(
                STORED_QRF_RANKING_SOURCE
            ),
        )
    )

    for axis, split_config in (
        shared_top_splits_by_axis.items()
    ):
        logger.info(
            "Selected shared Top-1 split | "
            "ranking_source=%s | axis=%s | "
            "split_index=%s | score=%.6f | "
            "train_experiments=%s | "
            "test_experiments=%s | "
            "split=%s",
            STORED_QRF_RANKING_SOURCE,
            axis,
            split_config["split_index"],
            split_config["qrf_score"],
            len(split_config["train_exp"]),
            len(split_config["test_exp"]),
            split_config["split_name"],
        )

    selected_splits_by_source = {
        geometry_source: (
            shared_top_splits_by_axis
        )
        for geometry_source in geometry_sources
    }

    model_params_by_source = (
        BEST_QRF_PARAMS_BY_DATASET
    )

    results = run(
        project_root=project_root,
        geometry_sources=geometry_sources,
        splits_by_source=(
            selected_splits_by_source
        ),
        feature_columns=(
            QRF_FEATURE_COLUMNS
        ),
        excluded_experiments=(
            QRF_EXCLUDED_EXPERIMENTS
        ),
        model_params_by_source=(
            model_params_by_source
        ),
        model_root=(
            project_root
            / STORED_QRF_MODEL_DIR
        ),
        bending_setups_path=(
            project_root
            / "data"
            / "rf_augmented"
            / "ui_data"
            / "unique_bending_setups.csv"
        ),
    )

    for geometry_source, source_results in (
        results.items()
    ):
        for target_axis, result in (
            source_results.items()
        ):
            logger.info(
                "Stored model | "
                "source=%s | axis=%s | "
                "model=%s | metrics=%s",
                geometry_source,
                target_axis,
                result["model_path"],
                result["metrics"],
            )