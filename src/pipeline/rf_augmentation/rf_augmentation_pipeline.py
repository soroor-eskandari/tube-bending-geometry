import pandas as pd
import numpy as np
from pathlib import Path
import logging
import ast

from src.logging.log_utils import log_function
from src.pipeline.rf_augmentation.rf_preprocessor import RFPreprocessor
from src.pipeline.rf_augmentation.sensor_data_augmentor import SensorDataAugmentor
from src.pipeline.rf_augmentation.data_splitter import DataSplittor
from src.pipeline.rf_augmentation.rf_dataset_builder import RFTrainingDatasetBuilder
from src.pipeline.rf_augmentation.rf_best_model_trainer import RFModelTrainer
from src.pipeline.rf_augmentation.rf_model_evaluator import RFModelEvaluator
from src.pipeline.rf_augmentation.io_utils import existing_table_path, read_table, write_table
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)


class RFAugmentationPipeline:

    @staticmethod
    @log_function
    def run(
        project_root,
        output_dir,
        include_all_features: bool = False,
        sensor_augmentation_mode: str = "raw",
        use_mlflow: bool = False,
    ):
        if sensor_augmentation_mode == "all":
            sensor_augmentation_mode = "+".join(
                SensorDataAugmentor.BASE_AUGMENTATION_MODES
            )

        sensor_augmentation_modes = SensorDataAugmentor.AUGMENTATION_MODES
        if sensor_augmentation_mode not in sensor_augmentation_modes:
            raise ValueError(
                "sensor_augmentation_mode must be one of "
                f"{sorted(sensor_augmentation_modes)}. "
                f"Got: {sensor_augmentation_mode}"
            )

        feature_mode_suffix = (
            "all_features" if include_all_features else "ranked_features"
        )
        sensor_mode_suffix = SensorDataAugmentor.mode_to_suffix(
            sensor_augmentation_mode
        )
        paper_name = f"rf_best_model_{feature_mode_suffix}_{sensor_mode_suffix}"

        # ============================================================
        # LOAD DATA
        # ============================================================
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
            project_root / "data" / "processed" / "processed_bending_setup.csv"
        )

        result_dir = project_root / "src" / "pipeline" / "rf_augmentation" / "result"
        model_dir = project_root / "src" / "pipeline" / "rf_augmentation" / "model"

        output_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        # ============================================================
        # FEATURE TYPE SELECTION
        # ============================================================
        if include_all_features:
            best_main_row = None
            best_secondary_row = None
            main_top_features = None
            secondary_top_features = None
        else:
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

        # ============================================================
        # PREPROCESS
        # ============================================================
        raw_machine_movement_clean, bending_clean = RFPreprocessor.preprocess_data(
            machine_movement_df=machine_movement,
            bending_df=bending,
        )

        if sensor_augmentation_mode != "raw":
            machine_movement_clean = SensorDataAugmentor.run(
                machine_movement_df=raw_machine_movement_clean,
                output_dir=project_root / "data" / "rf_augmented" / "sensor_data",
                augmentation_mode=sensor_augmentation_mode,
                random_state=42,
            )
        else:
            machine_movement_clean = raw_machine_movement_clean
            logger.info("Using raw cleaned sensor data; skipping SensorDataAugmentor")

        # ============================================================
        # SPLIT DATA
        # ============================================================
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
        X_main_train, X_sec_train, Y_main_train, Y_sec_train, feature_names_main, feature_names_secondary, train_experiment_ids = RFTrainingDatasetBuilder.build(
            machine_movement__df=machine_movement_clean,
            geometry_df=train_df,
            bending_df=bending_clean,
            main_selected_features=main_top_features,
            secondary_selected_features=secondary_top_features,
            return_experiment_ids=True,
        )

        X_main_test, X_sec_test, Y_main_test, Y_sec_test, _, _, test_experiment_ids = RFTrainingDatasetBuilder.build(
            machine_movement__df=raw_machine_movement_clean,
            geometry_df=test_df,
            bending_df=bending_clean,
            main_selected_features=main_top_features,
            secondary_selected_features=secondary_top_features,
            return_experiment_ids=True,
        )

        # ============================================================
        # TRAIN MODEL (FINAL BEST CONFIG)
        # ============================================================

        models = RFModelTrainer.train(
            X_main=X_main_train,
            X_secondary=X_sec_train,
            y_main=Y_main_train,
            y_secondary=Y_sec_train,
            model_dir=model_dir,

            # Naming
            paper_name=paper_name,

            # Model params
            n_estimators=1100,
            random_state=42,

            # MLflow
            use_mlflow=use_mlflow,
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

        evaluation_dir = output_dir / f"train_evaluation_{sensor_mode_suffix}"

        train_evaluation = RFModelEvaluator.evaluate_trained_model_on_train_data(
            models=models,
            X_main_train=X_main_train,
            X_secondary_train=X_sec_train,
            y_main_train=Y_main_train,
            y_secondary_train=Y_sec_train,
            output_dir=evaluation_dir,
            save_outputs=True,
        )


        y_main_test_pred = models["model_main"].predict(X_main_test)
        y_sec_test_pred = models["model_secondary"].predict(X_sec_test)

        test_evaluation_dir = output_dir / f"test_evaluation_{sensor_mode_suffix}"
        test_main_results = RFModelEvaluator.evaluate(Y_main_test, y_main_test_pred)
        test_secondary_results = RFModelEvaluator.evaluate(
            Y_sec_test,
            y_sec_test_pred,
        )
        RFModelEvaluator.save_results(
            results=test_main_results,
            output_dir=test_evaluation_dir,
            prefix="test_main",
        )
        RFModelEvaluator.save_results(
            results=test_secondary_results,
            output_dir=test_evaluation_dir,
            prefix="test_secondary",
        )

        train_metric_summary = RFAugmentationPipeline._combined_metric_summary(
            Y_main_train,
            train_evaluation["main"]["y_pred"],
            Y_sec_train,
            train_evaluation["secondary"]["y_pred"],
        )
        test_metric_summary = RFAugmentationPipeline._combined_metric_summary(
            Y_main_test,
            y_main_test_pred,
            Y_sec_test,
            y_sec_test_pred,
        )

        experiment_to_group = (
            pd.concat([train_df, test_df], ignore_index=True)
            .drop_duplicates("Experiment_ID")
            .set_index("Experiment_ID")["Group_ID"]
            .to_dict()
        )
        group_metrics_df = pd.concat(
            [
                RFAugmentationPipeline._build_group_metrics(
                    sensor_augmentation_mode=sensor_augmentation_mode,
                    sensor_mode_suffix=sensor_mode_suffix,
                    feature_mode=feature_mode_suffix,
                    include_all_features=include_all_features,
                    split="train",
                    experiment_ids=train_experiment_ids,
                    experiment_to_group=experiment_to_group,
                    y_main_true=Y_main_train,
                    y_main_pred=train_evaluation["main"]["y_pred"],
                    y_secondary_true=Y_sec_train,
                    y_secondary_pred=train_evaluation["secondary"]["y_pred"],
                ),
                RFAugmentationPipeline._build_group_metrics(
                    sensor_augmentation_mode=sensor_augmentation_mode,
                    sensor_mode_suffix=sensor_mode_suffix,
                    feature_mode=feature_mode_suffix,
                    include_all_features=include_all_features,
                    split="test",
                    experiment_ids=test_experiment_ids,
                    experiment_to_group=experiment_to_group,
                    y_main_true=Y_main_test,
                    y_main_pred=y_main_test_pred,
                    y_secondary_true=Y_sec_test,
                    y_secondary_pred=y_sec_test_pred,
                ),
            ],
            ignore_index=True,
        )

        # ============================================================
        # SAVE FINAL RESULTS SUMMARY
        # ============================================================
        results_summary = {
            "paper_name": paper_name,
            "include_all_features": include_all_features,
            "feature_mode": feature_mode_suffix,
            "sensor_augmentation_mode": sensor_augmentation_mode,
            "sensor_mode_suffix": sensor_mode_suffix,

            # metrics
            "r2_main": models["metrics"]["r2_main"],
            "r2_secondary": models["metrics"]["r2_secondary"],
            "mse_main": models["metrics"]["mse_main"],
            "mse_secondary": models["metrics"]["mse_secondary"],
            "rmse_main": models["metrics"]["rmse_main"],
            "rmse_secondary": models["metrics"]["rmse_secondary"],
            "mae_main": models["metrics"]["mae_main"],
            "mae_secondary": models["metrics"]["mae_secondary"],
            "train_nrmse_main": train_metric_summary["nrmse_main"],
            "train_nrmse_secondary": train_metric_summary["nrmse_secondary"],
            "train_mean_nrmse": train_metric_summary["mean_nrmse"],
            "test_r2_main": test_metric_summary["r2_main"],
            "test_r2_secondary": test_metric_summary["r2_secondary"],
            "test_mse_main": test_metric_summary["mse_main"],
            "test_mse_secondary": test_metric_summary["mse_secondary"],
            "test_rmse_main": test_metric_summary["rmse_main"],
            "test_rmse_secondary": test_metric_summary["rmse_secondary"],
            "test_mae_main": test_metric_summary["mae_main"],
            "test_mae_secondary": test_metric_summary["mae_secondary"],
            "test_nrmse_main": test_metric_summary["nrmse_main"],
            "test_nrmse_secondary": test_metric_summary["nrmse_secondary"],
            "mean_test_nrmse": test_metric_summary["mean_nrmse"],
            "train_test_gap_main": (
                test_metric_summary["nrmse_main"]
                - train_metric_summary["nrmse_main"]
            ),
            "train_test_gap_secondary": (
                test_metric_summary["nrmse_secondary"]
                - train_metric_summary["nrmse_secondary"]
            ),

            # feature info
            "main_features": "ALL" if include_all_features else str(main_top_features),
            "secondary_features": (
                "ALL" if include_all_features else str(secondary_top_features)
            ),

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
            "test_eval_dir": str(test_evaluation_dir),

            # paths
            "model_main_path": str(models["model_main_path"]),
            "model_secondary_path": str(models["model_secondary_path"]),
        }

        results_df = pd.DataFrame([results_summary])

        results_path = write_table(
            results_df,
            output_dir / f"final_model_results_{sensor_mode_suffix}.parquet",
            index=False,
        )

        comparison_path = output_dir / "sensor_augmentation_comparison.parquet"
        RFAugmentationPipeline._upsert_csv(
            path=comparison_path,
            new_rows=results_df,
            key_columns=[
                "sensor_mode_suffix",
                "feature_mode",
                "include_all_features",
            ],
        )

        group_metrics_path = output_dir / "sensor_augmentation_group_metrics.parquet"
        RFAugmentationPipeline._upsert_csv(
            path=group_metrics_path,
            new_rows=group_metrics_df,
            key_columns=[
                "sensor_mode_suffix",
                "feature_mode",
                "include_all_features",
                "split",
                "group_id",
            ],
        )

       
    @staticmethod
    def _combined_metric_summary(
        y_main_true,
        y_main_pred,
        y_secondary_true,
        y_secondary_pred,
    ):
        main = RFAugmentationPipeline._metric_summary(y_main_true, y_main_pred)
        secondary = RFAugmentationPipeline._metric_summary(
            y_secondary_true,
            y_secondary_pred,
        )
        return {
            "r2_main": main["r2"],
            "r2_secondary": secondary["r2"],
            "mse_main": main["mse"],
            "mse_secondary": secondary["mse"],
            "rmse_main": main["rmse"],
            "rmse_secondary": secondary["rmse"],
            "mae_main": main["mae"],
            "mae_secondary": secondary["mae"],
            "nrmse_main": main["nrmse"],
            "nrmse_secondary": secondary["nrmse"],
            "mean_nrmse": np.nanmean([main["nrmse"], secondary["nrmse"]]),
        }

    @staticmethod
    def _metric_summary(y_true, y_pred):
        y_true_flat = np.asarray(y_true).ravel()
        y_pred_flat = np.asarray(y_pred).ravel()
        mse = mean_squared_error(y_true_flat, y_pred_flat)
        rmse = np.sqrt(mse)
        y_std = np.std(y_true_flat)
        return {
            "r2": r2_score(y_true_flat, y_pred_flat),
            "mse": mse,
            "rmse": rmse,
            "mae": mean_absolute_error(y_true_flat, y_pred_flat),
            "nrmse": rmse / y_std if y_std > 0 else np.nan,
        }

    @staticmethod
    def _build_group_metrics(
        *,
        sensor_augmentation_mode,
        sensor_mode_suffix,
        feature_mode,
        include_all_features,
        split,
        experiment_ids,
        experiment_to_group,
        y_main_true,
        y_main_pred,
        y_secondary_true,
        y_secondary_pred,
    ):
        rows = []
        group_ids = [
            int(experiment_to_group[int(experiment_id)])
            for experiment_id in experiment_ids
        ]
        group_ids_array = np.asarray(group_ids)

        for group_id in sorted(set(group_ids)):
            mask = group_ids_array == group_id
            metrics = RFAugmentationPipeline._combined_metric_summary(
                y_main_true[mask],
                y_main_pred[mask],
                y_secondary_true[mask],
                y_secondary_pred[mask],
            )
            rows.append({
                "sensor_augmentation_mode": sensor_augmentation_mode,
                "sensor_mode_suffix": sensor_mode_suffix,
                "feature_mode": feature_mode,
                "include_all_features": include_all_features,
                "split": split,
                "group_id": group_id,
                "n_experiments": int(mask.sum()),
                "r2_main": metrics["r2_main"],
                "r2_secondary": metrics["r2_secondary"],
                "rmse_main": metrics["rmse_main"],
                "rmse_secondary": metrics["rmse_secondary"],
                "nrmse_main": metrics["nrmse_main"],
                "nrmse_secondary": metrics["nrmse_secondary"],
                "mean_nrmse": metrics["mean_nrmse"],
            })

        return pd.DataFrame(rows)

    @staticmethod
    def _upsert_csv(path: Path, new_rows: pd.DataFrame, key_columns: list[str]):
        path = Path(path)
        existing_path = existing_table_path(path)
        if existing_path.exists():
            existing = read_table(existing_path)
            for column in key_columns:
                if column not in existing.columns:
                    existing[column] = np.nan
            keep_mask = pd.Series(True, index=existing.index)
            for _, row in new_rows[key_columns].drop_duplicates().iterrows():
                row_mask = pd.Series(True, index=existing.index)
                for column in key_columns:
                    row_mask &= existing[column].astype(str) == str(row[column])
                keep_mask &= ~row_mask
            output = pd.concat([existing[keep_mask], new_rows], ignore_index=True)
        else:
            output = new_rows

        write_table(output, path, index=False)
