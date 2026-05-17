import numpy as np
from pathlib import Path
import logging
import pandas as pd
import ast

from src.logging.log_utils import log_function
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


class QRFPipeline:

    @staticmethod
    @log_function
    def run(project_root):

        # ============================================================
        # LOAD DATA
        # ============================================================

        geometry = pd.read_csv(
            project_root / "data" / "processed" / "geometry.csv"
        )

        bending = pd.read_csv(
            project_root / "data" / "raw" / "unique_bending_setups.csv"
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
                test_size=0.2,
                random_state=42,
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

                y_pred_mean=(
                    training_result["main"]["y_pred_mean"]
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

                y_pred_mean=(
                    training_result["secondary"]["y_pred_mean"]
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
            / "qrf_global_metrics.csv",
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
            / "qrf_per_angle_metrics.csv",
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
            / "qrf_prediction_details.csv",
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
        }