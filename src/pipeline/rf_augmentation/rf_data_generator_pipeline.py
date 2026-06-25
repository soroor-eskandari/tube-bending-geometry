import ast
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.logging.log_utils import log_function
from src.pipeline.rf_augmentation.rf_preprocessor import RFPreprocessor
from src.pipeline.rf_augmentation.rf_dataset_builder_generated import RFDatasetBuilder
from src.pipeline.rf_augmentation.rf_augmentation_generator import RFAugmentationGenerator
from src.pipeline.rf_augmentation.geometry_rebuilder import GeometryRebuilder
from src.pipeline.rf_augmentation.rf_signal_selection_recorder import RFSignalSelectionRecorder
from src.pipeline.rf_augmentation.sensor_data_augmentor import SensorDataAugmentor
from src.pipeline.rf_augmentation.io_utils import existing_table_path, read_table, write_table

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
    def _non_raw_sensor_augmentation_modes() -> list[str]:
        return [
            mode
            for mode in SensorDataAugmentor.all_augmentation_modes()
            if mode != "raw"
        ]

    @staticmethod
    @log_function
    def run(
        project_root,
        n_new_samples,
        output_dir,
        feature_sampling_mode: str = "within-group-interpolation",
        include_all_features: bool = False,
        sensor_augmentation_mode: str = "raw",
        regenerate_sensor_data: bool = False,
        sensor_noise_snr_db: float = 40.0,
    ):

        project_root = Path(project_root)
        output_dir = Path(output_dir)
        feature_sampling_suffixes = {
            "random-within-group": "random_within_group",
            "within-group-interpolation": "within_group_interpolation",
            "sensor-augmented": "sensor_augmented",
        }
        if feature_sampling_mode not in feature_sampling_suffixes:
            raise ValueError(
                "feature_sampling_mode must be one of "
                f"{sorted(feature_sampling_suffixes)}. "
                f"Got: {feature_sampling_mode}"
            )
        if sensor_augmentation_mode != "raw":
            raise ValueError(
                "RFDataGeneratorPipeline only supports raw sensor data. "
                f"Got: {sensor_augmentation_mode}"
            )
        sensor_mode_suffix = "raw"
        output_suffix = f"{feature_sampling_suffixes[feature_sampling_mode]}_{sensor_mode_suffix}"

        # ============================================================
        # LOAD DATA
        # ============================================================
        machine_movement_path = (
            project_root / "data" / "processed" / "machine_and_movement.csv"
        )

        if not machine_movement_path.exists():
            raise FileNotFoundError(
                "Missing machine/movement data for "
                f"sensor_augmentation_mode={sensor_augmentation_mode}: "
                f"{machine_movement_path}"
            )

        machine_movement = read_table(machine_movement_path)

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
        feature_mode_suffix = (
            "all_features" if include_all_features else "ranked_features"
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        # ============================================================
        # LOAD TRAINED MODELS
        # ============================================================
        if include_all_features:
            model_main_path = (
                model_dir
                / f"rf_main_rf_best_model_all_features_{sensor_mode_suffix}.pkl"
            )
            model_secondary_path = (
                model_dir
                / f"rf_secondary_rf_best_model_all_features_{sensor_mode_suffix}.pkl"
            )
        else:
            model_main_path = (
                model_dir
                / f"rf_main_rf_best_model_ranked_features_{sensor_mode_suffix}.pkl"
            )
            model_secondary_path = (
                model_dir
                / f"rf_secondary_rf_best_model_ranked_features_{sensor_mode_suffix}.pkl"
            )

        if not model_main_path.exists() or not model_secondary_path.exists():
            raise FileNotFoundError(
                "Missing trained RF model for "
                f"sensor_augmentation_mode={sensor_augmentation_mode}. "
                f"Expected: {model_main_path} and {model_secondary_path}"
            )

        model_main = joblib.load(model_main_path)
        model_secondary = joblib.load(model_secondary_path)

        # ============================================================
        # FEATURE TYPE SELECTION
        # ============================================================
        if include_all_features:
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
            secondary_top_features = ast.literal_eval(
                best_secondary_row["secondary_subset"]
            )

        # ============================================================
        # PREPROCESS
        # ============================================================
        machine_movement_clean, bending_clean = RFPreprocessor.preprocess_data(
            machine_movement_df=machine_movement,
            bending_df=bending,
        )

        (
            X_main_all,
            X_sec_all,
            _,
            _,
            _,
            _,
            aligned_ids_all,
        ) = RFDatasetBuilder.build(
            machine_movement__df=machine_movement_clean,
            geometry_df=geometry,
            bending_df=bending_clean,
            main_selected_features=main_top_features,
            secondary_selected_features=secondary_top_features,
            return_experiment_ids=True,
        )
        if feature_sampling_mode == "sensor-augmented":
            sensor_modes = RFDataGeneratorPipeline._non_raw_sensor_augmentation_modes()

            for augmented_sensor_mode in sensor_modes:
                augmented_sensor_suffix = SensorDataAugmentor.mode_to_suffix(
                    augmented_sensor_mode
                )
                augmented_machine_movement_path = existing_table_path(
                    project_root
                    / "data"
                    / "rf_augmented"
                    / "sensor_data"
                    / f"machine_movement_{augmented_sensor_suffix}.parquet"
                )

                if (
                    augmented_machine_movement_path.exists()
                    and not regenerate_sensor_data
                ):
                    augmented_machine_movement = read_table(
                        augmented_machine_movement_path
                    )
                else:
                    if augmented_machine_movement_path.exists():
                        logger.info(
                            "Regenerating augmented sensor data for %s.",
                            augmented_sensor_mode,
                        )
                    else:
                        logger.info(
                            "Missing augmented sensor data for %s; generating it now.",
                            augmented_sensor_mode,
                        )
                    augmented_machine_movement = SensorDataAugmentor.run(
                        machine_movement_df=machine_movement_clean,
                        output_dir=project_root
                        / "data"
                        / "rf_augmented"
                        / "sensor_data",
                        augmentation_mode=augmented_sensor_mode,
                        random_state=42,
                        noise_target_snr_db=sensor_noise_snr_db,
                    )

                augmented_machine_movement_clean, _ = (
                    RFPreprocessor.preprocess_data(
                        machine_movement_df=augmented_machine_movement,
                        bending_df=bending,
                    )
                )

                all_group_results = []
                all_selection_records = []

                for group_idx, group_row in group.iterrows():
                    exp_ids = RFDataGeneratorPipeline._parse_experiment_ids(
                        group_row["Experiment_Number"]
                    )

                    machine_group = machine_movement_clean[
                        machine_movement_clean["Experiment_ID"].isin(exp_ids)
                    ].copy()
                    augmented_machine_group = augmented_machine_movement_clean[
                        augmented_machine_movement_clean["Experiment_ID"].isin(exp_ids)
                    ].copy()
                    bending_group = bending_clean[
                        bending_clean["Experiment_ID"].isin(exp_ids)
                    ].copy()
                    geometry_group = geometry[
                        geometry["Experiment_ID"].isin(exp_ids)
                    ].copy()

                    if (
                        machine_group.empty
                        or augmented_machine_group.empty
                        or bending_group.empty
                        or geometry_group.empty
                    ):
                        logger.warning(
                            f"Skipping group {group_idx} for {augmented_sensor_mode} "
                            f"because one or more filtered datasets are empty."
                        )
                        continue

                    try:
                        (
                            X_main,
                            X_secc,
                            Y_main,
                            Y_sec,
                            feature_names_main,
                            feature_names_secondary,
                            aligned_ids,
                        ) = RFDatasetBuilder.build(
                            machine_movement__df=machine_group,
                            geometry_df=geometry_group,
                            bending_df=bending_group,
                            main_selected_features=main_top_features,
                            secondary_selected_features=secondary_top_features,
                            return_experiment_ids=True,
                        )
                        (
                            X_main_augmented,
                            X_sec_augmented,
                            _,
                            _,
                            _,
                            _,
                            augmented_aligned_ids,
                        ) = RFDatasetBuilder.build(
                            machine_movement__df=augmented_machine_group,
                            geometry_df=geometry_group,
                            bending_df=bending_group,
                            main_selected_features=main_top_features,
                            secondary_selected_features=secondary_top_features,
                            return_experiment_ids=True,
                        )

                    except Exception as exc:
                        logger.warning(
                            f"Skipping group {group_idx} for {augmented_sensor_mode} "
                            f"because dataset building failed: {exc}"
                        )
                        continue

                    id_to_raw_idx = {
                        experiment_id: idx
                        for idx, experiment_id in enumerate(aligned_ids)
                    }
                    raw_indices = [
                        id_to_raw_idx[experiment_id]
                        for experiment_id in augmented_aligned_ids
                        if experiment_id in id_to_raw_idx
                    ]
                    augmented_indices = [
                        idx
                        for idx, experiment_id in enumerate(augmented_aligned_ids)
                        if experiment_id in id_to_raw_idx
                    ]

                    if not raw_indices:
                        logger.warning(
                            f"Skipping group {group_idx} for {augmented_sensor_mode} "
                            "because raw and augmented experiment IDs do not overlap."
                        )
                        continue

                    X_main_base = X_main[raw_indices]
                    X_sec_base = X_secc[raw_indices]
                    aligned_augmented_ids = [
                        augmented_aligned_ids[idx] for idx in augmented_indices
                    ]

                    n_existing = X_main.shape[0]
                    deficit = n_new_samples - n_existing
                    if deficit <= 0:
                        y_main_all = Y_main
                        y_sec_all = Y_sec
                        n_generated = 0
                        synthetic_experiment_ids = []
                        selection_details = None
                    else:
                        rng = np.random.default_rng(42 + group_idx)
                        candidate_indices = np.arange(len(augmented_indices))
                        repeats = int(np.ceil(deficit / len(candidate_indices)))
                        sampled_candidate_indices = np.tile(
                            candidate_indices,
                            repeats,
                        )[:deficit]
                        rng.shuffle(sampled_candidate_indices)

                        X_main_actual = X_main_base[sampled_candidate_indices]
                        X_sec_actual = X_sec_base[sampled_candidate_indices]
                        X_main_selected = X_main_augmented[augmented_indices][
                            sampled_candidate_indices
                        ]
                        X_sec_selected = X_sec_augmented[augmented_indices][
                            sampled_candidate_indices
                        ]

                        y_main_new = model_main.predict(X_main_selected)
                        y_sec_new = model_secondary.predict(X_sec_selected)
                        n_generated = len(y_main_new)
                        synthetic_experiment_ids = list(range(1, n_generated + 1))

                        y_main_all = np.vstack([Y_main, y_main_new])
                        y_sec_all = np.vstack([Y_sec, y_sec_new])

                        selection_details = {
                            "sampled_indices": np.array(raw_indices)[
                                sampled_candidate_indices
                            ],
                            "X_main_actual": X_main_actual,
                            "X_main_selected": X_main_selected,
                            "X_main_group_min": np.min(X_main_selected, axis=0),
                            "X_main_group_max": np.max(X_main_selected, axis=0),
                            "X_secondary_actual": X_sec_actual,
                            "X_secondary_selected": X_sec_selected,
                            "X_secondary_group_min": np.min(X_sec_selected, axis=0),
                            "X_secondary_group_max": np.max(X_sec_selected, axis=0),
                            "selection_method": "sensor_augmented_features",
                            "feature_sampling_mode": feature_sampling_mode,
                        }
                        all_selection_records.extend(
                            RFSignalSelectionRecorder.build_records(
                                group_id=group_idx + 1,
                                exp_ids=exp_ids,
                                aligned_ids=aligned_ids,
                                feature_names_main=feature_names_main,
                                feature_names_secondary=feature_names_secondary,
                                selection_details=selection_details,
                                synthetic_id_offset=n_existing,
                                synthetic_experiment_ids=synthetic_experiment_ids,
                            )
                        )

                    n_points = y_main_all.shape[1]
                    angle_values = np.arange(n_points)
                    final_geometry_group_df = GeometryRebuilder.build(
                        y_main=y_main_all,
                        y_secondary=y_sec_all,
                        angle_values=angle_values,
                        n_original=n_existing,
                        experiment_ids=list(aligned_ids) + synthetic_experiment_ids,
                    )

                    final_geometry_group_df["group_id"] = group_idx + 1
                    final_geometry_group_df["source_experiment_ids"] = str(exp_ids)
                    final_geometry_group_df["augmented_source_experiment_ids"] = str(
                        aligned_augmented_ids
                    )
                    final_geometry_group_df["n_real_group_samples"] = n_existing
                    final_geometry_group_df["n_synthetic_group_samples"] = n_generated
                    final_geometry_group_df["target_group_samples"] = n_new_samples
                    final_geometry_group_df["feature_value_mode"] = (
                        feature_sampling_mode
                    )
                    final_geometry_group_df["feature_sampling_mode"] = (
                        feature_sampling_mode
                    )
                    final_geometry_group_df["feature_training_mode"] = (
                        feature_mode_suffix
                    )
                    final_geometry_group_df["sensor_augmentation_mode"] = (
                        augmented_sensor_mode
                    )
                    final_geometry_group_df["sensor_mode_suffix"] = (
                        augmented_sensor_suffix
                    )
                    final_geometry_group_df["sample_alpha_min"] = 0.0
                    final_geometry_group_df["sample_alpha_max"] = 0.0

                    all_group_results.append(final_geometry_group_df)

                if not all_group_results:
                    raise ValueError(
                        "No sensor-augmented geometry was generated for "
                        f"{augmented_sensor_mode}."
                    )

                final_geometry_df = pd.concat(
                    all_group_results,
                    ignore_index=True,
                )
                augmented_output_suffix = (
                    f"{feature_sampling_suffixes[feature_sampling_mode]}_"
                    f"{augmented_sensor_suffix}"
                )
                final_geometry_path = write_table(
                    final_geometry_df,
                    output_dir / f"final_geometry_{augmented_output_suffix}.parquet",
                    index=False,
                )
                selection_values_path = (
                    output_dir
                    / f"signal_feature_selected_values_{augmented_output_suffix}.parquet"
                )
                selection_values_path = RFSignalSelectionRecorder.save(
                    all_selection_records,
                    selection_values_path,
                )
                logger.info(
                    f"Sensor-augmented RF generation finished | "
                    f"sensor_mode={augmented_sensor_mode} | "
                    f"Groups processed: {len(all_group_results)} | "
                    f"Final geometry rows: {len(final_geometry_df)} | "
                    f"Saved to: {final_geometry_path} | "
                    f"Selection values saved to: {selection_values_path}"
                )

            return

        # ============================================================
        # GROUP-WISE GENERATION
        # ============================================================
        all_group_results = []
        all_selection_records = []

        for group_idx, group_row in group.iterrows():

            exp_ids = RFDataGeneratorPipeline._parse_experiment_ids(
                group_row["Experiment_Number"]
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
                    aligned_ids,
                ) = RFDatasetBuilder.build(
                    machine_movement__df=machine_group,
                    geometry_df=geometry_group,
                    bending_df=bending_group,
                    main_selected_features=main_top_features,
                    secondary_selected_features=secondary_top_features,
                    return_experiment_ids=True,
                )

            except Exception as exc:
                logger.warning(
                    f"Skipping group {group_idx} because dataset building failed: {exc}"
                )
                continue

            n_existing = X_main.shape[0]
            deficit = n_new_samples - n_existing
            synthetic_experiment_ids = []


            # --------------------------------------------------------
            # ORIGINAL PREDICTIONS
            # --------------------------------------------------------
            y_main_original = Y_main
            y_sec_original = Y_sec

            # --------------------------------------------------------
            # GENERATE ONLY MISSING SAMPLES
            # --------------------------------------------------------
            if deficit > 0:
                reference_mask = np.array(
                    [experiment_id not in set(aligned_ids) for experiment_id in aligned_ids_all],
                    dtype=bool,
                )

                (
                    X_main_aug,
                    X_sec_aug,
                    y_main_new,
                    y_sec_new,
                    selection_details,
                ) = RFAugmentationGenerator.generate(
                    X_main=X_main,
                    X_secondary=X_secc,
                    X_main_reference=X_main_all[reference_mask],
                    X_secondary_reference=X_sec_all[reference_mask],
                    reference_experiment_ids=[
                        experiment_id
                        for experiment_id, keep in zip(aligned_ids_all, reference_mask)
                        if keep
                    ],
                    model_main=model_main,
                    model_secondary=model_secondary,
                    n_new_samples=deficit,
                    feature_sampling_mode=feature_sampling_mode,
                    return_selection_details=True,
                    random_state=42 + group_idx,
                )

                y_main_all = np.vstack([y_main_original, y_main_new])
                y_sec_all = np.vstack([y_sec_original, y_sec_new])

                n_generated = len(y_main_new)
                synthetic_experiment_ids = list(range(1, n_generated + 1))
                all_selection_records.extend(
                    RFSignalSelectionRecorder.build_records(
                        group_id=group_idx + 1,
                        exp_ids=exp_ids,
                        aligned_ids=aligned_ids,
                        feature_names_main=feature_names_main,
                        feature_names_secondary=feature_names_secondary,
                        selection_details=selection_details,
                        synthetic_id_offset=n_existing,
                        synthetic_experiment_ids=synthetic_experiment_ids,
                    )
                )

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
                experiment_ids=list(aligned_ids) + synthetic_experiment_ids,
            )

            final_geometry_group_df["group_id"] = group_idx + 1
            final_geometry_group_df["source_experiment_ids"] = str(exp_ids)
            final_geometry_group_df["n_real_group_samples"] = n_existing
            final_geometry_group_df["n_synthetic_group_samples"] = n_generated
            final_geometry_group_df["target_group_samples"] = n_new_samples
            final_geometry_group_df["feature_value_mode"] = feature_sampling_mode
            final_geometry_group_df["feature_sampling_mode"] = feature_sampling_mode
            final_geometry_group_df["feature_training_mode"] = feature_mode_suffix
            final_geometry_group_df["sensor_augmentation_mode"] = (
                sensor_augmentation_mode
            )
            final_geometry_group_df["sensor_mode_suffix"] = sensor_mode_suffix
            final_geometry_group_df["sample_alpha_min"] = (
                selection_details.get("alpha_min", 0.0) if deficit > 0 else 0.0
            )
            final_geometry_group_df["sample_alpha_max"] = (
                selection_details.get("alpha_max", 0.0) if deficit > 0 else 0.0
            )

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
        final_geometry_path = write_table(
            final_geometry_df,
            output_dir / f"final_geometry_{output_suffix}.parquet",
            index=False,
        )

        selection_values_path = (
            output_dir / f"signal_feature_selected_values_{output_suffix}.parquet"
        )
        selection_values_path = RFSignalSelectionRecorder.save(
            all_selection_records,
            selection_values_path,
        )

        logger.info(
            f"Group-wise augmentation pipeline finished | "
            f"Groups processed: {len(all_group_results)} | "
            f"Final geometry rows: {len(final_geometry_df)} | "
            f"Saved to: {final_geometry_path} | "
            f"Selection values saved to: {selection_values_path}"
        )
