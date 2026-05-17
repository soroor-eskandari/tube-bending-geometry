import pandas as pd
import numpy as np
from pathlib import Path
import logging
import ast

from src.logging.log_utils import log_function
from src.pipeline.rf_augmentation.rf_preprocessor import RFPreprocessor
from src.pipeline.rf_augmentation.data_splitter import DataSplittor
from src.pipeline.rf_augmentation.rf_dataset_builder import RFTrainingDatasetBuilder
from src.pipeline.rf_augmentation.rf_best_model_trainer import RFModelTrainer
from src.pipeline.rf_augmentation.rf_model_evaluator import RFModelEvaluator

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
        unique_bending = pd.read_csv(
            project_root / "data" / "raw" / "unique_bending_setups.csv"
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

        # --- BEST MAIN ---
        sorted_main = best_subset_combo.sort_values(
            by="r2_main_best", ascending=False
        )
        best_main_row = sorted_main.iloc[0]  

        # --- BEST SECONDARY ---
        sorted_secondary = best_subset_combo.sort_values(
            by="r2_secondary_best", ascending=False
        )
        best_secondary_row = sorted_secondary.iloc[0]  

        main_top_features = ast.literal_eval(best_main_row["main_subset"])
        secondary_top_features = ast.literal_eval(best_secondary_row["secondary_subset"])

        logger.info(
            f"MAIN best features row: {best_main_row.to_dict()}"
        )
        logger.info(
            f"SECONDARY best features row: {best_secondary_row.to_dict()}"
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
        # SPLIT DATA
        # ============================================================
        logger.info("Splitting geometry data")

        train_df, test_df = DataSplittor.splittor(
            geometry_df=geometry,
            unique_bending_df=unique_bending,
            test_size=0.2,
            random_state=42,
        )

        logger.info(
            "Geometry split | train_rows=%s | test_rows=%s | "
            "train_groups=%s | test_groups=%s",
            len(train_df),
            len(test_df),
            train_df["Group_ID"].nunique(),
            test_df["Group_ID"].nunique(),
        )

        # ============================================================
        # BUILD DATASET
        # ============================================================
        logger.info("Building dataset")

        X_main_train, X_sec_train, Y_main_train, Y_sec_train, feature_names_main, feature_names_secondary = RFTrainingDatasetBuilder.build(
            machine_movement__df=machine_movement_clean,
            geometry_df=train_df,
            bending_df=bending_clean,
            main_selected_features=main_top_features,
            secondary_selected_features=secondary_top_features,
        )

        # ============================================================
        # TRAIN MODEL (FINAL BEST CONFIG)
        # ============================================================
        logger.info("Training final models (MAIN + SECONDARY)")

        models = RFModelTrainer.train(
            X_main=X_main_train,
            X_secondary=X_sec_train,
            y_main=Y_main_train,
            y_secondary=Y_sec_train,
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
        # EVALUATE TRAINING DATA
        # ============================================================
        logger.info("Evaluating final models on training data")

        evaluation_dir = output_dir / "train_evaluation"

        train_evaluation = RFModelEvaluator.evaluate_trained_model_on_train_data(
            models=models,
            X_main_train=X_main_train,
            X_secondary_train=X_sec_train,
            y_main_train=Y_main_train,
            y_secondary_train=Y_sec_train,
            output_dir=evaluation_dir,
            save_outputs=True,
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
            "n_samples": X_main_train.shape[0],
            "n_train_geometry_rows": len(train_df),
            "n_test_geometry_rows": len(test_df),
            "n_train_groups": train_df["Group_ID"].nunique(),
            "n_test_groups": test_df["Group_ID"].nunique(),

            # train evaluation
            "train_eval_main_r2": train_evaluation["main"]["metrics"]["r2_global"],
            "train_eval_secondary_r2": (
                train_evaluation["secondary"]["metrics"]["r2_global"]
            ),
            "train_eval_main_mse": train_evaluation["main"]["metrics"]["mse_global"],
            "train_eval_secondary_mse": (
                train_evaluation["secondary"]["metrics"]["mse_global"]
            ),
            "train_eval_dir": str(evaluation_dir),

            # paths
            "model_main_path": str(models["model_main_path"]),
            "model_secondary_path": str(models["model_secondary_path"]),
        }

        results_df = pd.DataFrame([results_summary])

        results_path = output_dir / "final_model_results.csv"
        results_df.to_csv(results_path, index=False)

        logger.info(f"Saved final results → {results_path}")

       
