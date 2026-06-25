import pandas as pd
import numpy as np
from pathlib import Path
import logging

from src.logging.log_utils import log_function
from src.pipeline.rf_augmentation.rf_preprocessor import RFPreprocessor
from src.pipeline.rf_augmentation.rf_feature_extractor import TimeSeriesFeatureExtractor
from src.pipeline.rf_augmentation.rf_feature_cleaner import RFFeatureCleaner
from src.pipeline.rf_augmentation.rf_feature_ranking_dataset_builder import RFFeatureRankerDatasetBuilder
from src.pipeline.rf_augmentation.rf_feature_type_ranker import RFFeatureTypeRanker


logger = logging.getLogger(__name__)


class RFRankerPipeline:

    @staticmethod
    @log_function
    def run(project_root: Path):

        # -------------------------
        # Load data
        # -------------------------
        logger.info("Loading input datasets...")

        machine_movement = pd.read_csv(
            project_root / "data" / "processed" / "machine_and_movement.csv"
        )
        bending = pd.read_csv(
            project_root / "data" / "processed" / "bending.csv"
        )
        geometry = pd.read_csv(
            project_root / "data" / "processed" / "geometry.csv"
        )

        logger.info(f"Machine & movement: {machine_movement.shape}")
        logger.info(f"Bending: {bending.shape}")
        logger.info(f"Geometry: {geometry.shape}")

        # -------------------------
        # Preprocessing
        # -------------------------
        logger.info("Preprocessing data")

        machine_movement_clean, bending_clean = RFPreprocessor.preprocess_data(
            machine_movement_df=machine_movement,
            bending_df=bending,
        )

        logger.info(f"machine_movement_clean: {machine_movement_clean.shape}")
        logger.info(f"bending_clean: {bending_clean.shape}")

        # -------------------------
        # Feature Extraction 
        # -------------------------
        logger.info("Extracting time-series features")

        signal_cols_machine = [
            col for col in machine_movement_clean.columns
            if col not in ["Experiment_ID", "Time_[s]"]
        ]

        machine_movement_features = TimeSeriesFeatureExtractor.extract_features(
            df=machine_movement_clean,
            signal_cols=signal_cols_machine,
            tsfel_included = False
        )

        logger.info(f"machine_movement_features: {machine_movement_features.shape}")

        # -------------------------
        # Feature Cleaning 
        # -------------------------        

        logger.info("Cleaning features")

        machine_movement_features_clean = RFFeatureCleaner.clean_features(
            df=machine_movement_features,
            target=(geometry["Main-axis [mm]"] + geometry["Secondary-axis [mm]"]) / 2,
            corr_threshold=0.95,
        )

        logger.info(f"Cleaned features shape: {machine_movement_features_clean.shape}")

        # -------------------------
        # Feature Ranking Dataset
        # -------------------------
        logger.info("Building dataset for feature ranking")

        X_rf, y_main, y_secondary, feature_names = RFFeatureRankerDatasetBuilder.build(
            machine_movement_df=machine_movement_features_clean,
            bending_df=bending_clean,
            geometry_df=geometry,
            use_bending=True,   
        )

        logger.info(f"X_rf shape: {X_rf.shape}")
        logger.info(f"y_main shape: {y_main.shape}")
        logger.info(f"y_secondary shape: {y_secondary.shape}")


        # -------------------------
        # Feature Type Ranking 
        # -------------------------
        logger.info("Ranking feature types for MAIN target")

        # ouput model directory
        model_dir = project_root / "src" / "pipeline" / "rf_augmentation" / "model"

        df_main_types, df_secondary_types, rf_main, rf_secondary = (
            RFFeatureTypeRanker.rank_feature_types_dual(
                X=X_rf,
                y_main=y_main,
                y_secondary=y_secondary,
                feature_names=feature_names,   # only needed if X_rf is ndarray
                model_output_dir=model_dir
            )
        )

        # -------------------------
        # Save feature type ranks
        # -------------------------
        result_dir = project_root / "src" / "pipeline" / "rf_augmentation" / "result"
        result_dir.mkdir(parents=True, exist_ok=True)

        main_types_path = result_dir / "feature_type_rank_main.csv"
        secondary_types_path = result_dir / "feature_type_rank_secondary.csv"

        df_main_types.to_csv(main_types_path, index=False)
        df_secondary_types.to_csv(secondary_types_path, index=False)

        logger.info(f"Saved main feature type ranks to: {main_types_path}")
        logger.info(f"Saved secondary feature type ranks to: {secondary_types_path}")
