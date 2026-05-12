import pandas as pd
import numpy as np
from pathlib import Path
import logging
import ast

from src.logging.log_utils import log_function
from src.pipeline.rf_augmentation.rf_preprocessor import RFPreprocessor
from src.pipeline.rf_augmentation.rf_dataset_builder import RFTrainingDatasetBuilder
from src.pipeline.rf_augmentation.rf_best_model_trainer import RFModelTrainer
from src.pipeline.rf_augmentation.rf_augmentation_generator import RFAugmentationGenerator
from src.pipeline.rf_augmentation.geometry_rebuilder import GeometryRebuilder

logger = logging.getLogger(__name__)


class RFAugmentationPipeline:

    @staticmethod
    @log_function
    def run(project_root, output_dir):

        # ============================================================
        # LOAD DATA
        # ============================================================
        logger.info("Reading data")

        machine_movement = pd.read_csv(
            project_root / "data" / "processed" / "machine_and_movement.csv"
        )
        bending = pd.read_csv(
            project_root / "data" / "processed" / "bending.csv"
        )
        geometry = pd.read_csv(
            project_root / "data" / "processed" / "geometry.csv"
        )

        result_dir = project_root / "src" / "pipeline" / "rf_augmentation" / "result"
        model_dir = project_root / "src" / "pipeline" / "rf_augmentation" / "model"

        output_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        # ============================================================
        # FEATURE TYPE SELECTION
        # ============================================================
                
        best_subset_combo = pd.read_csv(
            result_dir / "greedy_search_results.csv"
        )

        # --- 5th BEST MAIN ---
        sorted_main = best_subset_combo.sort_values(
            by="r2_main_best", ascending=False
        )
        best_main_row = sorted_main.iloc[9]  # 5th place

        # --- 5th BEST SECONDARY ---
        sorted_secondary = best_subset_combo.sort_values(
            by="r2_secondary_best", ascending=False
        )
        best_secondary_row = sorted_secondary.iloc[-1]  # 5th place

        main_top_features = ast.literal_eval(best_main_row["main_subset"])
        secondary_top_features = ast.literal_eval(best_secondary_row["secondary_subset"])

        logger.info(
            f"MAIN 5th-best features row: {best_main_row.to_dict()}"
        )
        logger.info(
            f"SECONDARY 5th-best features row: {best_secondary_row.to_dict()}"
        )

        # ============================================================
        # PREPROCESS
        # ============================================================
        logger.info("Preprocessing data")

        machine_movement_clean, bending_clean = RFPreprocessor.preprocess_data(
            machine_movement_df=machine_movement,
            bending_df=bending,
        )

        # ============================================================
        # BUILD DATASET
        # ============================================================
        logger.info("Building dataset")

        X_main, X_secc, Y_main, Y_sec, feature_names_main, feature_names_secondary = RFTrainingDatasetBuilder.build(
            machine_movement__df=machine_movement_clean,
            geometry_df=geometry,
            bending_df=bending_clean,
            main_selected_features=main_top_features,
            secondary_selected_features=secondary_top_features,
        )

        # ============================================================
        # TRAIN MODEL (FINAL BEST CONFIG)
        # ============================================================
        logger.info("Training final models (MAIN + SECONDARY)")

        models = RFModelTrainer.train(
            X_main=X_main,
            X_secondary=X_secc,
            y_main=Y_main,
            y_secondary=Y_sec,
            model_dir=model_dir,

            # Naming
            paper_name="rf_best_model", 

            # Model params
            n_estimators=1100,
            random_state=42,

            # MLflow
            use_mlflow=True,
            mlflow_tracking_uri=None,  
            mlflow_experiment="rf_final_models",
            mlflow_run_name=None,      

            # Metadata (optional but recommended)
            feature_names_main=feature_names_main,
            feature_names_secondary=feature_names_secondary,
        )

        # ============================================================
        # SAVE FINAL RESULTS SUMMARY
        # ============================================================
        logger.info("Saving final results summary")

        results_summary = {
            "paper_name": "rf_best_model",

            # metrics
            "r2_main": models["metrics"]["r2_main"],
            "r2_secondary": models["metrics"]["r2_secondary"],
            "mse_main": models["metrics"]["mse_main"],
            "mse_secondary": models["metrics"]["mse_secondary"],

            # feature info
            "main_features": str(main_top_features),
            "secondary_features": str(secondary_top_features),

            # feature counts
            "n_features_main": len(feature_names_main),
            "n_features_secondary": len(feature_names_secondary),

            # dataset info
            "n_samples": X_main.shape[0],

            # paths
            "model_main_path": str(models["model_main_path"]),
            "model_secondary_path": str(models["model_secondary_path"]),
        }

        results_df = pd.DataFrame([results_summary])

        results_path = output_dir / "final_model_results.csv"
        results_df.to_csv(results_path, index=False)

        logger.info(f"Saved final results → {results_path}")

       