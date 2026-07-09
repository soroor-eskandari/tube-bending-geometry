from pathlib import Path
import logging
import pandas as pd
import ast
import json

from src.logging.log_utils import log_function
from src.pipeline.rf_augmentation.io_utils import read_table
from src.pipeline.rf_augmentation.sensor_data_augmentor import SensorDataAugmentor
from src.pipeline.ml.qrf.geometry_data_preprocessor import (
    GeometryPreprocessor,
)
from src.pipeline.ml.qrf.data_splittor import (
    DataSplittor,
)
from src.pipeline.ml.qrf.qrf_model_trainer import (
    QRFModelTrainer,
)
from src.pipeline.ml.qrf.qrf_evaluator import (
    QRFModelEvaluator,
)

logger = logging.getLogger(__name__)


FEATURE_SAMPLING_SOURCE_SUFFIXES = {
    "exact": "exact",
    "within-group-sampling": "within_group_sampling",
    "overall-sampling": "overall_sampling",
}


def generated_geometry_sources() -> dict[str, Path]:
    geometry_sources = {}
    for feature_suffix in FEATURE_SAMPLING_SOURCE_SUFFIXES.values():
        for sensor_mode in SensorDataAugmentor.all_augmentation_modes():
            sensor_suffix = SensorDataAugmentor.mode_to_suffix(sensor_mode)
            geometry_source = f"{feature_suffix}_{sensor_suffix}"
            geometry_sources[geometry_source] = (
                Path("data")
                / "rf_augmented"
                / f"final_geometry_{geometry_source}.parquet"
            )
    return geometry_sources


def geometry_source_paths(project_root) -> dict[str, Path]:
    project_root = Path(project_root)
    paths = {
        "real": project_root / "data" / "processed" / "geometry.csv",
    }
    paths.update(
        {
            source: project_root / relative_path
            for source, relative_path in generated_geometry_sources().items()
        }
    )
    paths.update(
        {
            "augmented_real": paths["exact_raw"],
            "sampled": paths["within_group_sampling_raw"],
        }
    )
    return paths


def qrf_training_geometry_sources(project_root) -> dict[str, Path]:
    project_root = Path(project_root)
    ui_data_dir = project_root / "data" / "rf_augmented" / "ui_data"
    manifest_path = ui_data_dir / "manifest.json"

    paths = {
        "real": project_root / "data" / "processed" / "geometry.csv",
    }

    with manifest_path.open() as manifest_file:
        manifest = json.load(manifest_file)

    paths.update(
        {
            method["suffix"]: ui_data_dir / method["csv"]
            for method in manifest["methods"]
        }
    )

    return paths


class QRFPipeline:

    @staticmethod
    @log_function
    def run(
        project_root,
        geometry_source: str = "real",
        geometry_path=None,
        qrf_params: dict = None,
        use_mlflow: bool = False,
        mlflow_tracking_uri: str = None,
        mlflow_experiment: str = "QRF_Geometry_Model",
        use_experiment_split: bool = False,
        train_exp: list[int] | None = None,
        test_exp: list[int] | None = None,
    ):
        project_root = Path(project_root)
        qrf_params = qrf_params or {}

        if geometry_path is None:
            geometry_paths = geometry_source_paths(project_root)

            if geometry_source not in geometry_paths:
                raise ValueError(
                    "geometry_source must be one of: "
                    f"{', '.join(geometry_paths.keys())}. Got: {geometry_source}"
                )

            geometry_path = geometry_paths[geometry_source]
        else:
            geometry_path = Path(geometry_path)
            if not geometry_path.is_absolute():
                geometry_path = project_root / geometry_path

        # ============================================================
        # LOAD DATA
        # ============================================================

        geometry = read_table(geometry_path)

        bending = pd.read_csv(
            project_root / "data" / "processed" / "processed_bending_setup.csv"
        )
        result_dir = (
            project_root / "src" / "pipeline" / "ml" / "qrf" / "result"
        )

        model_dir = (
            project_root/ "src"/ "pipeline" / "ml" / "model"
        )

        result_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ============================================================
        # PREPROCESS GEOMETRY DATA
        # ============================================================

        logger.info(
            "Preprocessing geometry data"
        )

        geometry_clean = (
            GeometryPreprocessor.preprocess(
                geometry_df=geometry,
                bending_df=bending,
            )
        )

        # ============================================================
        # SPLIT DATA
        # ============================================================

        train_df, test_df = (
            DataSplittor.splittor(
                geometry_df=geometry_clean,
                unique_bending_df=bending,
                test_size=0.2,
                random_state=42,
                use_experiment_split=use_experiment_split,
                train_exp=train_exp,
                test_exp=test_exp,
            )
        )

        # ============================================================
        # TRAIN QRF MODEL
        # ============================================================

        logger.info(
            "Training QRF models"
        )

        training_result = (
            QRFModelTrainer.train(
                train_df=train_df,
                test_df=test_df,
                model_dir=model_dir,
                paper_name=f"qrf_geometry_{geometry_source}",
                **qrf_params,
                use_mlflow=use_mlflow,
                mlflow_tracking_uri=mlflow_tracking_uri,
                mlflow_experiment=mlflow_experiment,
                mlflow_tags={
                    "Dataset": geometry_source,
                    "geometry_source": geometry_source,
                },
            )
        )

        # ============================================================
        # EVALUATE MAIN TARGET
        # ============================================================

        logger.info(
            "Evaluating Main-axis model"
        )

        main_evaluation = (
            QRFModelEvaluator.evaluate(

                y_true=(
                    training_result["main"]["y_true"]
                ),

                y_pred_median=(
                    training_result["main"]["y_pred_median"]
                ),

                y_pred_lower=(
                    training_result["main"]["y_pred_lower"]
                ),

                y_pred_upper=(
                    training_result["main"]["y_pred_upper"]
                ),

                angle_values=(
                    training_result["main"]
                    ["prediction_df"]
                    ["Angle[degree]"]
                    .values
                ),

                target_name="Main-axis [mm]",
            )
        )

        # ============================================================
        # EVALUATE SECONDARY TARGET
        # ============================================================

        logger.info(
            "Evaluating Secondary-axis model"
        )

        secondary_evaluation = (
            QRFModelEvaluator.evaluate(

                y_true=(
                    training_result["secondary"]["y_true"]
                ),

                y_pred_median=(
                    training_result["secondary"]["y_pred_median"]
                ),

                y_pred_lower=(
                    training_result["secondary"]["y_pred_lower"]
                ),

                y_pred_upper=(
                    training_result["secondary"]["y_pred_upper"]
                ),

                angle_values=(
                    training_result["secondary"]
                    ["prediction_df"]
                    ["Angle[degree]"]
                    .values
                ),

                target_name="Secondary-axis [mm]",
            )
        )

        # ============================================================
        # GLOBAL METRICS
        # ============================================================

        logger.info(
            "Saving global metrics"
        )

        global_evaluation_df = pd.DataFrame(
            [
                main_evaluation[
                    "global_metrics"
                ],

                secondary_evaluation[
                    "global_metrics"
                ],
            ]
        )

        global_evaluation_df.to_csv(
            result_dir
            / f"qrf_global_metrics_{geometry_source}.csv",
            index=False,
        )

        # ============================================================
        # PER-ANGLE METRICS
        # ============================================================

        logger.info(
            "Saving per-angle metrics"
        )

        per_angle_df = pd.concat(
            [
                main_evaluation[
                    "per_angle_metrics"
                ],

                secondary_evaluation[
                    "per_angle_metrics"
                ],
            ],
            ignore_index=True,
        )

        per_angle_df.to_csv(
            result_dir
            / f"qrf_per_angle_metrics_{geometry_source}.csv",
            index=False,
        )

        # ============================================================
        # RAW PREDICTIONS
        # ============================================================

        logger.info(
            "Saving prediction details"
        )

        main_prediction_details = (
            main_evaluation[
                "prediction_details"
            ].copy()
        )

        main_prediction_details[
            "target"
        ] = "Main-axis [mm]"

        secondary_prediction_details = (
            secondary_evaluation[
                "prediction_details"
            ].copy()
        )

        secondary_prediction_details[
            "target"
        ] = "Secondary-axis [mm]"

        prediction_details_df = pd.concat(
            [
                main_prediction_details,
                secondary_prediction_details,
            ],
            ignore_index=True,
        )

        prediction_details_df.to_csv(
            result_dir
            / f"qrf_prediction_details_{geometry_source}.csv",
            index=False,
        )

        # ============================================================
        # RETURN RESULTS
        # ============================================================

        logger.info(
            "QRF pipeline completed successfully"
        )

        return {

            "global_metrics":
                global_evaluation_df,

            "per_angle_metrics":
                per_angle_df,

            "prediction_details":
                prediction_details_df,

            "geometry_source":
                geometry_source,

            "geometry_path":
                geometry_path,
        }
