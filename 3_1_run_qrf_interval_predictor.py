import logging
import ast
from pathlib import Path

import pandas as pd

from pipeline.ml.qrf.mode.experiments.qrf_pipeline import (
    QRFPipeline,
    qrf_training_geometry_sources,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


BEST_QRF_PARAMS_BY_DATASET = {
    "real": {
        "n_estimators": 800,
        "max_depth": 5,
        "lower_quantile": 0.05,
        "upper_quantile": 0.95,
    },
    "sensor_augmented_noise__time_wrapping__scaling__jittering": {
        "n_estimators": 800,
        "max_depth": 5,
        "lower_quantile": 0.05,
        "upper_quantile": 0.95,
    },
    "within_group_interpolation_raw": {
        "n_estimators": 800,
        "max_depth": 5,
        "lower_quantile": 0.05,
        "upper_quantile": 0.95,
    },
}


STORED_QRF_SPLIT_PATH = Path("exploration/data/qrf/split/various_splits.parquet")
STORED_QRF_MODEL_DIR = Path("exploration/data/qrf/split/models")
STORED_QRF_GEOMETRY_SOURCE = (
    "sensor_augmented_noise__time_wrapping__scaling__jittering"
)
STORED_QRF_SPLIT_INDEX = 1
STORED_QRF_PAPER_NAME = (
    "qrf_geometry_sensor_augmented_noise__time_wrapping__scaling__jittering_"
    "001_split_single__Collet_boost__train__0.85_0.87_0.9_0.95___test__0.92"
)


def load_stored_qrf_split(
    project_root: Path,
    geometry_source: str,
    split_index: int,
) -> dict:
    split_path = project_root / STORED_QRF_SPLIT_PATH

    split_df = pd.read_parquet(
        split_path,
        columns=[
            "split_index",
            "geometry_source",
            "split_name",
            "train_experiment_ids",
            "test_experiment_ids",
        ],
    )

    split_rows = split_df[
        (split_df["split_index"].astype(int) == int(split_index))
        & (split_df["geometry_source"] == geometry_source)
    ]

    if split_rows.empty:
        raise ValueError(
            "Stored QRF split was not found: "
            f"geometry_source={geometry_source}, split_index={split_index}"
        )

    split_row = split_rows.iloc[0]

    return {
        "split_name": split_row["split_name"],
        "train_exp": _normalize_stored_experiment_ids(
            split_row["train_experiment_ids"]
        ),
        "test_exp": _normalize_stored_experiment_ids(
            split_row["test_experiment_ids"]
        ),
    }


def _normalize_stored_experiment_ids(value) -> list[int]:
    if isinstance(value, str):
        value = ast.literal_eval(value)

    return sorted(set(map(int, value)))


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parent
    use_mlflow = False
    use_best_qrf_params = False
    use_default_split = False
    use_experiment_split = False
    train_exp = None
    test_exp = None
    geometry_sources = qrf_training_geometry_sources(project_root)

    if not use_default_split:
        stored_split = load_stored_qrf_split(
            project_root=project_root,
            geometry_source=STORED_QRF_GEOMETRY_SOURCE,
            split_index=STORED_QRF_SPLIT_INDEX,
        )
        geometry_sources = {
            STORED_QRF_GEOMETRY_SOURCE:
                geometry_sources[STORED_QRF_GEOMETRY_SOURCE],
        }
        use_experiment_split = True
        train_exp = stored_split["train_exp"]
        test_exp = stored_split["test_exp"]
        logger.info(
            "Using stored QRF split | source=%s | split_index=%s | %s | "
            "train_experiments=%s | test_experiments=%s",
            STORED_QRF_GEOMETRY_SOURCE,
            STORED_QRF_SPLIT_INDEX,
            stored_split["split_name"],
            len(train_exp),
            len(test_exp),
        )

    # ---------------------------------------
    # Run the QRF interval prediction workflow for the real geometry
    # and the augmented datasets exported for the UI.
    # ---------------------------------------
    for geometry_source, geometry_path in geometry_sources.items():
        qrf_params = (
            BEST_QRF_PARAMS_BY_DATASET[geometry_source]
            if use_best_qrf_params
            else {}
        )
        logger.info(
            "Running QRF interval prediction for %s from %s with %s",
            geometry_source,
            geometry_path,
            qrf_params,
        )
        QRFPipeline.run(
            project_root=project_root,
            geometry_source=geometry_source,
            geometry_path=geometry_path,
            model_dir=(
                STORED_QRF_MODEL_DIR
                if not use_default_split
                else None
            ),
            paper_name=(
                STORED_QRF_PAPER_NAME
                if not use_default_split
                else None
            ),
            qrf_params=qrf_params,
            use_mlflow=use_mlflow,
            use_experiment_split=use_experiment_split,
            train_exp=train_exp,
            test_exp=test_exp,
        )
