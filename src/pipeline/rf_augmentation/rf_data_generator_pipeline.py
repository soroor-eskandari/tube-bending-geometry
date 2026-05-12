import ast
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.logging.log_utils import log_function
from src.pipeline.rf_augmentation.rf_preprocessor import RFPreprocessor
from src.pipeline.rf_augmentation.rf_dataset_builder import RFTrainingDatasetBuilder
from src.pipeline.rf_augmentation.rf_augmentation_generator import RFAugmentationGenerator
from src.pipeline.rf_augmentation.geometry_rebuilder import GeometryRebuilder

logger = logging.getLogger(__name__)


class RFDataGeneratorPipeline:

    @staticmethod
    def _parse_experiment_ids(value):
        """
        Convert values like:
            "[1, 2, 3]"
            "[55]"
            55
            [1, 2, 3]
        into a clean list of integers.
        """
        if isinstance(value, list):
            return [int(x) for x in value]

        if isinstance(value, str):
            value = value.strip()
            parsed = ast.literal_eval(value)

            if isinstance(parsed, list):
                return [int(x) for x in parsed]

            return [int(parsed)]

        return [int(value)]

    @staticmethod
    @log_function
    def run(project_root, n_new_samples, output_dir):

        project_root = Path(project_root)
        output_dir = Path(output_dir)

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

        group = pd.read_csv(
            project_root / "data" / "raw" / "unique_bending_setups.csv"
        )

        result_dir = project_root / "src" / "pipeline" / "rf_augmentation" / "result"
        model_dir = project_root / "src" / "pipeline" / "rf_augmentation" / "model"

        output_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        # ============================================================
        # LOAD TRAINED MODELS
        # ============================================================
        logger.info("Loading trained models")

        model_main_path = model_dir / "rf_main_rf_best_model.pkl"
        model_secondary_path = model_dir / "rf_secondary_rf_best_model.pkl"

        model_main = joblib.load(model_main_path)
        model_secondary = joblib.load(model_secondary_path)

        logger.info(
            f"Loaded models:\n"
            f"MAIN → {model_main_path}\n"
            f"SECONDARY → {model_secondary_path}"
        )

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
        best_main_row = sorted_main.iloc[9]

        # --- 5th BEST SECONDARY ---
        sorted_secondary = best_subset_combo.sort_values(
            by="r2_secondary_best", ascending=False
        )
        best_secondary_row = sorted_secondary.iloc[-1]

        main_top_features = ast.literal_eval(best_main_row["main_subset"])
        secondary_top_features = ast.literal_eval(
            best_secondary_row["secondary_subset"]
        )

        logger.info(f"MAIN best features row: {best_main_row.to_dict()}")
        logger.info(f"SECONDARY best features row: {best_secondary_row.to_dict()}")

        # ============================================================
        # PREPROCESS
        # ============================================================
        logger.info("Preprocessing data")

        machine_movement_clean, bending_clean = RFPreprocessor.preprocess_data(
            machine_movement_df=machine_movement,
            bending_df=bending,
        )

        # ============================================================
        # GROUP-WISE GENERATION
        # ============================================================
        logger.info("Starting group-wise augmentation")

        all_group_results = []

        for group_idx, group_row in group.iterrows():

            exp_ids = RFDataGeneratorPipeline._parse_experiment_ids(
                group_row["Experiment_Number"]
            )

            logger.info(
                f"Processing group {group_idx} | "
                f"Experiment IDs: {exp_ids}"
            )

            # --------------------------------------------------------
            # Filter data for current group
            # --------------------------------------------------------
            machine_group = machine_movement_clean[
                machine_movement_clean["Experiment_ID"].isin(exp_ids)
            ].copy()

            bending_group = bending_clean[
                bending_clean["Experiment_ID"].isin(exp_ids)
            ].copy()

            geometry_group = geometry[
                geometry["Experiment_ID"].isin(exp_ids)
            ].copy()

            if machine_group.empty or bending_group.empty or geometry_group.empty:
                logger.warning(
                    f"Skipping group {group_idx} because one or more filtered "
                    f"datasets are empty."
                )
                continue

            # --------------------------------------------------------
            # BUILD DATASET FOR CURRENT GROUP
            # --------------------------------------------------------
            try:
                (
                    X_main,
                    X_secc,
                    Y_main,
                    Y_sec,
                    feature_names_main,
                    feature_names_secondary,
                ) = RFTrainingDatasetBuilder.build(
                    machine_movement__df=machine_group,
                    geometry_df=geometry_group,
                    bending_df=bending_group,
                    main_selected_features=main_top_features,
                    secondary_selected_features=secondary_top_features,
                )

            except Exception as exc:
                logger.warning(
                    f"Skipping group {group_idx} because dataset building failed: {exc}"
                )
                continue

            n_existing = X_main.shape[0]
            deficit = n_new_samples - n_existing

            logger.info(
                f"Group {group_idx} | existing={n_existing} | "
                f"target={n_new_samples} | deficit={deficit}"
            )

            # --------------------------------------------------------
            # ORIGINAL PREDICTIONS
            # --------------------------------------------------------
            y_main_original = model_main.predict(X_main)
            y_sec_original = model_secondary.predict(X_secc)

            # --------------------------------------------------------
            # GENERATE ONLY MISSING SAMPLES
            # --------------------------------------------------------
            if deficit > 0:
                (
                    X_main_aug,
                    X_sec_aug,
                    y_main_new,
                    y_sec_new,
                ) = RFAugmentationGenerator.generate(
                    X_main=X_main,
                    X_secondary=X_secc,
                    model_main=model_main,
                    model_secondary=model_secondary,
                    n_new_samples=deficit,
                )

                y_main_all = np.vstack([y_main_original, y_main_new])
                y_sec_all = np.vstack([y_sec_original, y_sec_new])

                n_generated = len(y_main_new)

            else:
                y_main_all = y_main_original
                y_sec_all = y_sec_original
                n_generated = 0

            # --------------------------------------------------------
            # REBUILD GEOMETRY FOR CURRENT GROUP
            # --------------------------------------------------------
            n_points = y_main_all.shape[1]
            angle_values = np.arange(n_points)

            final_geometry_group_df = GeometryRebuilder.build(
                y_main=y_main_all,
                y_secondary=y_sec_all,
                angle_values=angle_values,
                n_original=n_existing,
            )

            final_geometry_group_df["group_id"] = group_idx + 1
            final_geometry_group_df["source_experiment_ids"] = str(exp_ids)
            final_geometry_group_df["n_real_group_samples"] = n_existing
            final_geometry_group_df["n_synthetic_group_samples"] = n_generated
            final_geometry_group_df["target_group_samples"] = n_new_samples

            all_group_results.append(final_geometry_group_df)

        # ============================================================
        # CONCAT ALL GROUP RESULTS
        # ============================================================
        if not all_group_results:
            raise ValueError("No group-wise augmented geometry was generated.")

        final_geometry_df = pd.concat(all_group_results, ignore_index=True)

        # ============================================================
        # SAVE FINAL OUTPUT
        # ============================================================
        final_geometry_path = output_dir / "final_geometry.csv"
        final_geometry_df.to_csv(final_geometry_path, index=False)

        logger.info(
            f"Group-wise augmentation pipeline finished | "
            f"Groups processed: {len(all_group_results)} | "
            f"Final geometry rows: {len(final_geometry_df)} | "
            f"Saved to: {final_geometry_path}"
        )