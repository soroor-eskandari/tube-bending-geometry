import pandas as pd
import numpy as np
from pathlib import Path
import logging

from src.logging.log_utils import log_function
from src.pipeline.rf_augmentation.rf_preprocessor import RFPreprocessor
from src.pipeline.rf_augmentation.rf_feature_extractor import TimeSeriesFeatureExtractor
from src.pipeline.rf_augmentation.rf_dataset_builder import RFTrainingDatasetBuilder
from src.pipeline.rf_augmentation.rf_model_trainer import RFModelTrainer
from src.pipeline.rf_augmentation.rf_augmentation_generator import RFAugmentationGenerator
from src.pipeline.rf_augmentation.geometry_rebuilder import GeometryRebuilder
from src.pipeline.rf_augmentation.rf_model_evaluator import RFModelEvaluator
from src.pipeline.rf_augmentation.rf_feature_cleaner import RFFeatureCleaner
from src.pipeline.rf_augmentation.rf_feature_ranking_dataset_builder import RFFeatureRankerDatasetBuilder
from src.pipeline.rf_augmentation.rf_feature_type_ranker import RFFeatureTypeRanker


logger = logging.getLogger(__name__)


class RFAugmentationPipeline:

    @staticmethod
    @log_function
    def run(project_root: Path, expected_per_group: int, output_dir: Path):

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
        # Infer angle values
        # -------------------------
        logger.info("Inferring angle configuration")

        angle_counts = geometry.groupby("Experiment_ID")["Angle[degree]ORDistance[mm]"].count()

        if angle_counts.empty:
            raise ValueError("geometry.csv has no angle records")

        if expected_per_group is not None and expected_per_group in angle_counts.values:
            expected_angles = int(expected_per_group)
        else:
            logger.warning(
                "Expected group size %s not found, using most frequent angle count",
                expected_per_group,
            )
            expected_angles = int(angle_counts.mode().iloc[0])

        valid_id = angle_counts[angle_counts == expected_angles].index[0]

        angle_values = (
            geometry[geometry["Experiment_ID"] == valid_id]
            .sort_values("Angle[degree]ORDistance[mm]")["Angle[degree]ORDistance[mm]"]
            .to_numpy()
        )

        logger.info(f"Using {expected_angles} angles per experiment")

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
            signal_cols=signal_cols_machine
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
        # Feature Type Ranking (MAIN)
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

        logger.info(f"Top feature types (main): {df_main_types.head(10)['feature_type'].tolist()}")
        logger.info(f"Top feature types (secondary): {df_secondary_types.head(10)['feature_type'].tolist()}")

        # Show top types (pipeline decides, not class)
        top_feature_types_main = df_main_types["feature_type"].head(10).tolist()
        top_feature_types_secondary = df_secondary_types["feature_type"].head(10).tolist()

        selected_feature_types = list(
            set(top_feature_types_main) |
            set(top_feature_types_secondary)
        )

        logger.info(f"Selected feature types (union): {selected_feature_types}")

        # -------------------------
        # Dataset
        # -------------------------
        logger.info("Building dataset")

        X_rf, y_main, y_secondary = RFTrainingDatasetBuilder.build(
            machine_movement_average_df=machine_movement_features,
            bending_features_df=bending_clean,
            geometry_df=geometry,
            expected_angles=expected_angles,
        )

        logger.info(f"X_rf: {X_rf.shape}")
        logger.info(f"y_main: {y_main.shape}")
        logger.info(f"y_secondary: {y_secondary.shape}")

        # -------------------------
        # Train
        # -------------------------
        logger.info("Training Random Forest models")

        trainer = RFModelTrainer()
        train_out = trainer.train(X_rf, y_main, y_secondary)

        logger.info("Training completed")

        # -------------------------
        # Save models
        # -------------------------
        model_dir = project_root / "ml" / "rf_augmentation"
        model_dir.mkdir(parents=True, exist_ok=True)

        trainer.save(train_out, model_dir)

        # -------------------------
        # Predict
        # -------------------------
        logger.info("Running predictions on test split")

        preds = trainer.predict(train_out, train_out["X_test"])

        # -------------------------
        # Evaluate
        # -------------------------
        logger.info("Evaluating models")

        evaluator = RFModelEvaluator()

        main_results = evaluator.evaluate(
            train_out["y_main_test"],
            preds["y_main_pred"]
        )

        sec_results = evaluator.evaluate(
            train_out["y_sec_test"],
            preds["y_sec_pred"]
        )

        logger.info(f"Main R2: {main_results['r2_global']:.4f}")
        logger.info(f"Secondary R2: {sec_results['r2_global']:.4f}")

        # -------------------------
        # Augmentation
        # -------------------------
        logger.info("Generating augmented data")

        X_new, y_main_new, y_secondary_new = RFAugmentationGenerator.generate(
            X_rf=X_rf,
            model_main=train_out["model_main"],
            model_secondary=train_out["model_secondary"],
            n_samples=1000,
            noise_scale=0.01
        )

        # -------------------------
        # Combine
        # -------------------------
        X_aug = np.vstack([X_rf, X_new])
        y_main_aug = np.vstack([y_main, y_main_new])
        y_secondary_aug = np.vstack([y_secondary, y_secondary_new])

        # -------------------------
        # Rebuild geometry
        # -------------------------
        logger.info("Rebuilding geometry")

        geometry_aug_df = GeometryRebuilder.build(
            y_main=y_main_aug,
            y_secondary=y_secondary_aug,
            angle_values=angle_values,
            n_original=y_main.shape[0],
        )

        # -------------------------
        # Save
        # -------------------------
        output_dir.mkdir(parents=True, exist_ok=True)
        geometry_aug_df.to_csv(output_dir / "geometry_augmented.csv", index=False)

        logger.info(f"Saved augmented dataset to: {output_dir}")

        return {
            "main_metrics": main_results,
            "secondary_metrics": sec_results,
            "model_dir": model_dir,
            "output_dir": output_dir,
        }