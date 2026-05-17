import numpy as np
from pathlib import Path
import logging
import pandas as pd
import ast

from src.logging.log_utils import log_function
from src.pipeline.ml.hgp.geometry_data_preprocessor import (
    GeometryPreprocessor,
)
from src.pipeline.ml.hgp.data_splitter import (
    DataSplitter,
)
from src.pipeline.ml.hgp.group_wise_aggregator import (
    GroupWiseAggregator,
)
from src.pipeline.ml.hgp.gp_dataset_builder import (
    GPDatasetBuilder,
)
from src.pipeline.ml.hgp.mean_gp_model_trainer import (
    MeanGPModelTrainer,
)
from src.pipeline.ml.hgp.noise_gp_model_trainer import (
    NoiseGPModelTrainer,
)
from src.pipeline.ml.hgp.hgp_predictor import (
    HGPPredictor,
)
from src.pipeline.ml.hgp.hgp_evaluator import (
    HGPModelEvaluator,
)


logger = logging.getLogger(__name__)


class HGPipeline:

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
            project_root / "src" / "pipeline" / "ml" / "hgp" / "result"
        )

        model_dir = (
            project_root / "artifacts" / "models" / "hgp"
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
            DataSplitter.splittor(
                geometry_df=geometry_clean,
                test_size=0.2,
                random_state=42,
            )
        )

        # ============================================================
        # AGGREGATE REPEATED TRAINING INPUTS
        # ============================================================

        logger.info(
            "Aggregating repeated training inputs"
        )

        train_group_statistics = (
            GroupWiseAggregator.aggregate(
                train_df=train_df,
            )
        )

        # ============================================================
        # BUILD GP TRAINING DATASET
        # ============================================================

        gp_dataset, gp_dataset_columns = (
            GPDatasetBuilder.build(
                train_df=train_df,
                train_group_statistics=train_group_statistics,
            )
        )

        # ============================================================
        # TRAIN MEAN GP MODEL
        # ============================================================

        logger.info(
            "Training mean GP models"
        )

        mean_gp_training_result = (
            MeanGPModelTrainer.train(
                gp_dataset=gp_dataset,
                model_dir=model_dir,
            )
        )

        # ============================================================
        # BUILD NOISE GP TRAINING DATASET
        # ============================================================

        noise_gp_dataset, noise_gp_dataset_columns = (
            GPDatasetBuilder.build_noise_gp_dataset(
                train_group_statistics=train_group_statistics,
            )
        )

        # ============================================================
        # TRAIN NOISE GP MODEL
        # ============================================================

        logger.info(
            "Training noise GP models"
        )

        noise_gp_training_result = (
            NoiseGPModelTrainer.train(
                noise_gp_dataset=noise_gp_dataset,
                model_dir=model_dir,
            )
        )

        # ============================================================
        # PREDICT HGP INTERVALS
        # ============================================================

        logger.info(
            "Predicting HGP intervals"
        )

        hgp_prediction_result = (
            HGPPredictor.predict(
                test_df=test_df,
                mean_gp_training_result=mean_gp_training_result,
                noise_gp_training_result=noise_gp_training_result,
            )
        )

        # ============================================================
        # EVALUATE MAIN TARGET
        # ============================================================

        logger.info(
            "Evaluating HGP Main-axis model"
        )

        main_evaluation = (
            HGPModelEvaluator.evaluate(
                prediction_df=(
                    hgp_prediction_result["main"]
                    ["prediction_df"]
                ),
                target_name="Main-axis [mm]",
                lower_quantile=(
                    hgp_prediction_result["lower_quantile"]
                ),
                upper_quantile=(
                    hgp_prediction_result["upper_quantile"]
                ),
            )
        )

        # ============================================================
        # EVALUATE SECONDARY TARGET
        # ============================================================

        logger.info(
            "Evaluating HGP Secondary-axis model"
        )

        secondary_evaluation = (
            HGPModelEvaluator.evaluate(
                prediction_df=(
                    hgp_prediction_result["secondary"]
                    ["prediction_df"]
                ),
                target_name="Secondary-axis [mm]",
                lower_quantile=(
                    hgp_prediction_result["lower_quantile"]
                ),
                upper_quantile=(
                    hgp_prediction_result["upper_quantile"]
                ),
            )
        )

        # ============================================================
        # GLOBAL METRICS
        # ============================================================

        logger.info(
            "Saving HGP global metrics"
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
            / "hgp_global_metrics.csv",
            index=False,
        )

        # ============================================================
        # PER-ANGLE METRICS
        # ============================================================

        logger.info(
            "Saving HGP per-angle metrics"
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
            / "hgp_per_angle_metrics.csv",
            index=False,
        )

        # ============================================================
        # RAW PREDICTIONS
        # ============================================================

        logger.info(
            "Saving HGP prediction details"
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
            / "hgp_prediction_details.csv",
            index=False,
        )

        # ============================================================
        # RETURN RESULTS
        # ============================================================

        logger.info(
            "HGP pipeline completed successfully"
        )

        return {

            "train_group_statistics":
                train_group_statistics,

            "gp_dataset":
                gp_dataset,

            "gp_dataset_columns":
                gp_dataset_columns,

            "noise_gp_dataset":
                noise_gp_dataset,

            "noise_gp_dataset_columns":
                noise_gp_dataset_columns,

            "mean_gp_training_result":
                mean_gp_training_result,

            "noise_gp_training_result":
                noise_gp_training_result,

            "hgp_prediction_result":
                hgp_prediction_result,

            "global_metrics":
                global_evaluation_df,

            "per_angle_metrics":
                per_angle_df,

            "prediction_details":
                prediction_details_df,
        }
