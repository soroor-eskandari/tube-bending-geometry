from pathlib import Path
import logging
import pandas as pd

from src.logging.log_utils import log_function
from src.pipeline.ml.sk.geometry_data_preprocessor import (
    GeometryPreprocessor,
)
from src.pipeline.ml.sk.data_splitter import (
    DataSplitter,
)
from src.pipeline.ml.sk.group_wise_aggregator import (
    GroupWiseAggregator,
)
from src.pipeline.ml.sk.sk_dataset_builder import (
    SKDatasetBuilder,
)
from src.pipeline.ml.sk.sk_model_trainer import (
    SKModelTrainer,
)
from src.pipeline.ml.sk.sk_predictor import (
    SKPredictor,
)
from src.pipeline.ml.sk.sk_evaluator import (
    SKModelEvaluator,
)


logger = logging.getLogger(__name__)


class SKPipeline:

    @staticmethod
    @log_function
    def run(project_root, geometry_source: str = "real"):

        project_root = Path(project_root)
        geometry_paths = {
            "real": project_root / "data" / "processed" / "geometry.csv",
            "augmented_real": (
                project_root / "data" / "rf_augmented" / "final_geometry_exact.csv"
            ),
            "sampled": (
                project_root / "data" / "rf_augmented" / "final_geometry_within_group_sampling.csv"
            ),
        }

        if geometry_source not in geometry_paths:
            raise ValueError(
                "geometry_source must be one of: "
                f"{', '.join(geometry_paths.keys())}. Got: {geometry_source}"
            )

        geometry_path = geometry_paths[geometry_source]

        # ============================================================
        # LOAD DATA
        # ============================================================
        geometry = pd.read_csv(geometry_path)

        bending = pd.read_csv(
            project_root / "data" / "processed" / "processed_bending_setups.csv"
        )

        result_dir = (
            project_root / "src" / "pipeline" / "ml" / "sk" / "result"
        )

        model_dir = (
            project_root / "artifacts" / "models" / "sk"
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
        # BUILD STOCHASTIC KRIGING DATASET
        # ============================================================
        sk_dataset, sk_dataset_columns = (
            SKDatasetBuilder.build(
                train_group_statistics=train_group_statistics,
            )
        )

        # ============================================================
        # TRAIN STOCHASTIC KRIGING MODELS
        # ============================================================
        logger.info(
            "Training stochastic kriging models"
        )

        sk_training_result = (
            SKModelTrainer.train(
                sk_dataset=sk_dataset,
                model_dir=model_dir,
                paper_name=f"sk_geometry_{geometry_source}",
            )
        )

        # ============================================================
        # PREDICT SK INTERVALS
        # ============================================================
        logger.info(
            "Predicting stochastic kriging intervals"
        )

        sk_prediction_result = (
            SKPredictor.predict(
                test_df=test_df,
                sk_training_result=sk_training_result,
            )
        )

        # ============================================================
        # EVALUATE MAIN TARGET
        # ============================================================
        logger.info(
            "Evaluating SK Main-axis model"
        )

        main_evaluation = (
            SKModelEvaluator.evaluate(
                prediction_df=(
                    sk_prediction_result["main"]
                    ["prediction_df"]
                ),
                target_name="Main-axis [mm]",
                lower_quantile=(
                    sk_prediction_result["lower_quantile"]
                ),
                upper_quantile=(
                    sk_prediction_result["upper_quantile"]
                ),
            )
        )

        # ============================================================
        # EVALUATE SECONDARY TARGET
        # ============================================================
        logger.info(
            "Evaluating SK Secondary-axis model"
        )

        secondary_evaluation = (
            SKModelEvaluator.evaluate(
                prediction_df=(
                    sk_prediction_result["secondary"]
                    ["prediction_df"]
                ),
                target_name="Secondary-axis [mm]",
                lower_quantile=(
                    sk_prediction_result["lower_quantile"]
                ),
                upper_quantile=(
                    sk_prediction_result["upper_quantile"]
                ),
            )
        )

        # ============================================================
        # GLOBAL METRICS
        # ============================================================
        logger.info(
            "Saving SK global metrics"
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
            / f"sk_global_metrics_{geometry_source}.csv",
            index=False,
        )

        # ============================================================
        # PER-ANGLE METRICS
        # ============================================================
        logger.info(
            "Saving SK per-angle metrics"
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
            / f"sk_per_angle_metrics_{geometry_source}.csv",
            index=False,
        )

        # ============================================================
        # RAW PREDICTIONS
        # ============================================================
        logger.info(
            "Saving SK prediction details"
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
            / f"sk_prediction_details_{geometry_source}.csv",
            index=False,
        )

        # ============================================================
        # RETURN RESULTS
        # ============================================================
        logger.info(
            "SK pipeline completed successfully"
        )

        return {
            "train_group_statistics": train_group_statistics,
            "sk_dataset": sk_dataset,
            "sk_dataset_columns": sk_dataset_columns,
            "sk_training_result": sk_training_result,
            "sk_prediction_result": sk_prediction_result,
            "global_metrics": global_evaluation_df,
            "per_angle_metrics": per_angle_df,
            "prediction_details": prediction_details_df,
            "geometry_source": geometry_source,
            "geometry_path": geometry_path,
        }
