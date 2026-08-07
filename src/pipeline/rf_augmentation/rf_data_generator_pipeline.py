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
from src.pipeline.rf_augmentation.io_utils import read_table, write_table

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
    def _synthetic_experiment_ids(
        group_idx: int,
        n_generated: int,
    ) -> list[int]:
        """
        Create synthetic Experiment_ID values that cannot collide with
        real experiment IDs or synthetic IDs from other groups.
        """
        synthetic_id_base = (
            int(group_idx) + 1
        ) * 1_000_000

        return [
            synthetic_id_base + offset
            for offset in range(
                1,
                int(n_generated) + 1,
            )
        ]

    @staticmethod
    def _non_raw_sensor_augmentation_modes() -> list[str]:
        return [SensorDataAugmentor.ALL_METHODS_MODE]

    @staticmethod
    def _values_equal(left, right) -> bool:
        if pd.isna(left) and pd.isna(right):
            return True

        try:
            return bool(np.isclose(float(left), float(right), rtol=1e-9, atol=1e-12))
        except (TypeError, ValueError):
            return str(left) == str(right)

    @staticmethod
    def _load_bending_feature_rank_order(
        result_dir: Path,
        group: pd.DataFrame,
    ) -> tuple[list[str], dict[str, float]]:
        rank_path = result_dir / "bending_feature_rank_mean_axis.csv"
        if not rank_path.exists():
            logger.warning(
                "Bending feature rank file not found at %s. "
                "Sparse groups will use the previous reference behavior.",
                rank_path,
            )
            return [], {}

        rank_df = pd.read_csv(rank_path)
        if "bending_feature" not in rank_df.columns:
            raise ValueError(
                "bending_feature_rank_mean_axis.csv must contain "
                "'bending_feature'."
            )

        if "mean_importance" in rank_df.columns:
            rank_df = rank_df.sort_values("mean_importance", ascending=True)
        elif "rank" in rank_df.columns:
            rank_df = rank_df.sort_values("rank", ascending=False)

        available_features = [
            feature
            for feature in rank_df["bending_feature"].tolist()
            if feature in group.columns
        ]
        importance_by_feature = {
            row["bending_feature"]: float(row.get("mean_importance", 1.0))
            for _, row in rank_df.iterrows()
            if row["bending_feature"] in available_features
        }

        if not available_features:
            logger.warning(
                "No ranked bending features from %s were found in the group setup table.",
                rank_path,
            )

        return available_features, importance_by_feature

    @staticmethod
    def _feature_distance(left, right, scale: float) -> float:
        if RFDataGeneratorPipeline._values_equal(left, right):
            return 0.0

        try:
            return abs(float(left) - float(right)) / scale
        except (TypeError, ValueError):
            return 1.0

    @staticmethod
    def _find_sparse_reference_group(
        *,
        group: pd.DataFrame,
        group_idx: int,
        ranked_bending_features_priority: list[str],
        bending_feature_importance: dict[str, float],
        max_reference_groups: int = 5,
    ) -> dict | None:
        if not ranked_bending_features_priority:
            return None

        target_row = group.loc[group_idx]
        numeric_scales = {
            feature: max(
                float(pd.to_numeric(group[feature], errors="coerce").std()),
                1e-8,
            )
            for feature in ranked_bending_features_priority
        }
        candidates = []

        for candidate_idx, candidate_row in group.iterrows():
            if candidate_idx == group_idx:
                continue

            changed_features = [
                feature
                for feature in ranked_bending_features_priority
                if not RFDataGeneratorPipeline._values_equal(
                    target_row[feature],
                    candidate_row[feature],
                )
            ]
            if not changed_features:
                continue

            distance = sum(
                bending_feature_importance.get(feature, 1.0)
                * RFDataGeneratorPipeline._feature_distance(
                    target_row[feature],
                    candidate_row[feature],
                    numeric_scales[feature],
                )
                for feature in ranked_bending_features_priority
            )

            candidate_exp_ids = RFDataGeneratorPipeline._parse_experiment_ids(
                candidate_row["Experiment_Number"]
            )
            candidates.append(
                {
                    "group_idx": int(candidate_idx),
                    "group_id": int(candidate_idx) + 1,
                    "experiment_ids": candidate_exp_ids,
                    "changed_features": changed_features,
                    "distance": float(distance),
                }
            )

        if candidates:
            candidates.sort(
                key=lambda candidate: (
                    candidate["distance"],
                    abs(candidate["group_idx"] - group_idx),
                    len(candidate["changed_features"]),
                    candidate["group_idx"],
                )
            )
            candidates = candidates[:max_reference_groups]
            experiment_ids = []
            for candidate in candidates:
                experiment_ids.extend(candidate["experiment_ids"])

            return {
                "group_ids": [candidate["group_id"] for candidate in candidates],
                "experiment_ids": experiment_ids,
                "changed_features": sorted(
                    {
                        feature
                        for candidate in candidates
                        for feature in candidate["changed_features"]
                    }
                ),
                "candidate_count": len(candidates),
                "reference_selection_method": "closest_weighted_bending_setup",
                "max_reference_groups": max_reference_groups,
            }

        return None

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
        (
            ranked_bending_features_priority,
            bending_feature_importance,
        ) = (
            RFDataGeneratorPipeline._load_bending_feature_rank_order(
                result_dir=result_dir,
                group=group,
            )
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
                all_group_results = []
                all_selection_records = []
                all_augmented_sensor_groups = []

                for group_idx, group_row in group.iterrows():
                    exp_ids = RFDataGeneratorPipeline._parse_experiment_ids(
                        group_row["Experiment_Number"]
                    )

                    machine_group = machine_movement_clean[
                        machine_movement_clean["Experiment_ID"].isin(exp_ids)
                    ].copy()
                    bending_group = bending_clean[
                        bending_clean["Experiment_ID"].isin(exp_ids)
                    ].copy()
                    geometry_group = geometry[
                        geometry["Experiment_ID"].isin(exp_ids)
                    ].copy()

                    if (
                        machine_group.empty
                        or bending_group.empty
                        or geometry_group.empty
                    ):
                        logger.warning(
                            f"Skipping group {group_idx} for {augmented_sensor_mode} "
                            f"because one or more filtered datasets are empty."
                        )
                        continue

                    sensor_rng = np.random.default_rng(42000 + group_idx)
                    sparse_sensor_reference_group = None
                    sensor_reference_machine_group = machine_group
                    n_real_group_samples = bending_group["Experiment_ID"].nunique()

                    if n_real_group_samples <= 2:
                        sparse_sensor_reference_group = (
                            RFDataGeneratorPipeline._find_sparse_reference_group(
                                group=group,
                                group_idx=group_idx,
                                ranked_bending_features_priority=(
                                    ranked_bending_features_priority
                                ),
                                bending_feature_importance=bending_feature_importance,
                            )
                        )
                        if sparse_sensor_reference_group is not None:
                            reference_exp_ids = set(
                                sparse_sensor_reference_group["experiment_ids"]
                            )
                            reference_machine_group = machine_movement_clean[
                                machine_movement_clean["Experiment_ID"].isin(
                                    reference_exp_ids
                                )
                            ].copy()
                            if not reference_machine_group.empty:
                                sensor_reference_machine_group = pd.concat(
                                    [machine_group, reference_machine_group],
                                    ignore_index=True,
                                )
                                logger.info(
                                    "Using sparse sensor range reference groups %s "
                                    "for group %s. Changed ranked bending features: %s",
                                    sparse_sensor_reference_group["group_ids"],
                                    group_idx + 1,
                                    sparse_sensor_reference_group["changed_features"],
                                )
                            else:
                                logger.warning(
                                    "Matched sparse sensor reference groups %s for "
                                    "group %s, but no reference sensor rows were found.",
                                    sparse_sensor_reference_group["group_ids"],
                                    group_idx + 1,
                                )
                                sparse_sensor_reference_group = None

                    augmented_machine_groups = []
                    sensor_range_details = None
                    n_sensor_variants = 1
                    if n_real_group_samples <= 2:
                        n_sensor_variants = max(1, n_new_samples - n_real_group_samples)

                    for variant_idx in range(n_sensor_variants):
                        variant_rng = np.random.default_rng(
                            42000 + group_idx * 100 + variant_idx
                        )
                        if sparse_sensor_reference_group is not None:
                            (
                                augmented_variant,
                                sensor_range_details,
                            ) = SensorDataAugmentor.apply_all_methods_from_reference_ranges(
                                machine_group,
                                sensor_reference_machine_group,
                                variant_rng,
                            )
                        else:
                            (
                                augmented_variant,
                                sensor_range_details,
                            ) = SensorDataAugmentor.apply_all_methods_from_group_ranges(
                                machine_group,
                                variant_rng,
                            )
                        augmented_variant = augmented_variant.copy()
                        if n_sensor_variants > 1:
                            augmented_variant["Experiment_ID"] = (
                                augmented_variant["Experiment_ID"].astype(int)
                                * 1000
                                + variant_idx
                                + 1
                            )
                        augmented_machine_groups.append(augmented_variant)

                    augmented_machine_group = pd.concat(
                        augmented_machine_groups,
                        ignore_index=True,
                    )
                    augmented_machine_group_for_output = augmented_machine_group.copy()
                    augmented_machine_group_for_output["sensor_augmentation_mode"] = (
                        augmented_sensor_mode
                    )
                    augmented_machine_group_for_output["sensor_mode_suffix"] = (
                        augmented_sensor_suffix
                    )
                    augmented_machine_group_for_output["group_id"] = group_idx + 1
                    all_augmented_sensor_groups.append(
                        augmented_machine_group_for_output
                    )

                    augmented_bending_group = bending_group.copy()
                    augmented_geometry_group = geometry_group.copy()
                    if n_sensor_variants > 1:
                        augmented_bending_groups = []
                        augmented_geometry_groups = []
                        for variant_idx in range(n_sensor_variants):
                            bending_variant = bending_group.copy()
                            geometry_variant = geometry_group.copy()
                            bending_variant["Experiment_ID"] = (
                                bending_variant["Experiment_ID"].astype(int)
                                * 1000
                                + variant_idx
                                + 1
                            )
                            geometry_variant["Experiment_ID"] = (
                                geometry_variant["Experiment_ID"].astype(int)
                                * 1000
                                + variant_idx
                                + 1
                            )
                            augmented_bending_groups.append(bending_variant)
                            augmented_geometry_groups.append(geometry_variant)

                        augmented_bending_group = pd.concat(
                            augmented_bending_groups,
                            ignore_index=True,
                        )
                        augmented_geometry_group = pd.concat(
                            augmented_geometry_groups,
                            ignore_index=True,
                        )

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
                            geometry_df=augmented_geometry_group,
                            bending_df=augmented_bending_group,
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
                    augmented_to_raw_id = {
                        experiment_id: (
                            int((experiment_id - 1) // 1000)
                            if n_sensor_variants > 1
                            else experiment_id
                        )
                        for experiment_id in augmented_aligned_ids
                    }
                    raw_indices = [
                        id_to_raw_idx[augmented_to_raw_id[experiment_id]]
                        for experiment_id in augmented_aligned_ids
                        if augmented_to_raw_id[experiment_id] in id_to_raw_idx
                    ]
                    augmented_indices = [
                        idx
                        for idx, experiment_id in enumerate(augmented_aligned_ids)
                        if augmented_to_raw_id[experiment_id] in id_to_raw_idx
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
                        synthetic_experiment_ids = (
                            RFDataGeneratorPipeline
                            ._synthetic_experiment_ids(
                                group_idx=group_idx,
                                n_generated=n_generated,
                            )
                        )

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
                            "sensor_range_details": sensor_range_details,
                        }
                        if sparse_sensor_reference_group is not None:
                            selection_details["sparse_reference_group_id"] = str(
                                sparse_sensor_reference_group["group_ids"]
                            )
                            selection_details["sparse_reference_experiment_ids"] = str(
                                sparse_sensor_reference_group["experiment_ids"]
                            )
                            selection_details["sparse_reference_changed_feature"] = str(
                                sparse_sensor_reference_group["changed_features"]
                            )
                            selection_details["sparse_reference_target_value"] = (
                                sparse_sensor_reference_group[
                                    "reference_selection_method"
                                ]
                            )
                            selection_details["sparse_reference_value"] = str(
                                sparse_sensor_reference_group["candidate_count"]
                            )
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
                    final_geometry_group_df["sparse_reference_group_id"] = (
                        str(sparse_sensor_reference_group["group_ids"])
                        if sparse_sensor_reference_group is not None
                        else ""
                    )
                    final_geometry_group_df["sparse_reference_experiment_ids"] = (
                        str(sparse_sensor_reference_group["experiment_ids"])
                        if sparse_sensor_reference_group is not None
                        else ""
                    )
                    final_geometry_group_df["sparse_reference_changed_feature"] = (
                        str(sparse_sensor_reference_group["changed_features"])
                        if sparse_sensor_reference_group is not None
                        else ""
                    )
                    final_geometry_group_df["sparse_reference_target_value"] = (
                        sparse_sensor_reference_group["reference_selection_method"]
                        if sparse_sensor_reference_group is not None
                        else ""
                    )
                    final_geometry_group_df["sparse_reference_value"] = (
                        str(sparse_sensor_reference_group["candidate_count"])
                        if sparse_sensor_reference_group is not None
                        else ""
                    )

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
                if all_augmented_sensor_groups:
                    augmented_sensor_df = pd.concat(
                        all_augmented_sensor_groups,
                        ignore_index=True,
                    )
                    write_table(
                        augmented_sensor_df,
                        project_root
                        / "data"
                        / "rf_augmented"
                        / "sensor_data"
                        / f"machine_movement_{augmented_sensor_suffix}.parquet",
                        index=False,
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
            sparse_reference_group = None


            # --------------------------------------------------------
            # ORIGINAL PREDICTIONS
            # --------------------------------------------------------
            y_main_original = Y_main
            y_sec_original = Y_sec

            # --------------------------------------------------------
            # GENERATE ONLY MISSING SAMPLES
            # --------------------------------------------------------
            if deficit > 0:
                reference_experiment_ids = None
                X_main_reference = None
                X_secondary_reference = None

                if n_existing <= 2:
                    sparse_reference_group = (
                        RFDataGeneratorPipeline._find_sparse_reference_group(
                            group=group,
                            group_idx=group_idx,
                            ranked_bending_features_priority=(
                                ranked_bending_features_priority
                            ),
                            bending_feature_importance=bending_feature_importance,
                        )
                    )

                if sparse_reference_group is not None:
                    reference_id_set = set(sparse_reference_group["experiment_ids"])
                    reference_mask = np.array(
                        [
                            experiment_id in reference_id_set
                            for experiment_id in aligned_ids_all
                        ],
                        dtype=bool,
                    )
                    reference_experiment_ids = [
                        experiment_id
                        for experiment_id, keep in zip(aligned_ids_all, reference_mask)
                        if keep
                    ]
                    if reference_experiment_ids:
                        X_main_reference = X_main_all[reference_mask]
                        X_secondary_reference = X_sec_all[reference_mask]
                        logger.info(
                            "Using sparse reference groups %s for group %s. "
                            "Changed ranked bending features: %s",
                            sparse_reference_group["group_ids"],
                            group_idx + 1,
                            sparse_reference_group["changed_features"],
                        )
                    else:
                        logger.warning(
                            "Matched sparse reference groups %s for group %s, "
                            "but none of its experiments are aligned with RF features.",
                            sparse_reference_group["group_ids"],
                            group_idx + 1,
                        )
                        sparse_reference_group = None
                elif n_existing == 1:
                    reference_mask = np.array(
                        [
                            experiment_id not in set(aligned_ids)
                            for experiment_id in aligned_ids_all
                        ],
                        dtype=bool,
                    )
                    X_main_reference = X_main_all[reference_mask]
                    X_secondary_reference = X_sec_all[reference_mask]
                    reference_experiment_ids = [
                        experiment_id
                        for experiment_id, keep in zip(aligned_ids_all, reference_mask)
                        if keep
                    ]

                (
                    X_main_aug,
                    X_sec_aug,
                    y_main_new,
                    y_sec_new,
                    selection_details,
                ) = RFAugmentationGenerator.generate(
                    X_main=X_main,
                    X_secondary=X_secc,
                    X_main_reference=X_main_reference,
                    X_secondary_reference=X_secondary_reference,
                    reference_experiment_ids=reference_experiment_ids,
                    model_main=model_main,
                    model_secondary=model_secondary,
                    n_new_samples=deficit,
                    feature_sampling_mode=feature_sampling_mode,
                    return_selection_details=True,
                    random_state=42 + group_idx,
                )
                if sparse_reference_group is not None:
                    selection_details["sparse_reference_group_id"] = (
                        str(sparse_reference_group["group_ids"])
                    )
                    selection_details["sparse_reference_experiment_ids"] = str(
                        sparse_reference_group["experiment_ids"]
                    )
                    selection_details["sparse_reference_changed_feature"] = (
                        str(sparse_reference_group["changed_features"])
                    )
                    selection_details["sparse_reference_target_value"] = (
                        sparse_reference_group["reference_selection_method"]
                    )
                    selection_details["sparse_reference_value"] = (
                        str(sparse_reference_group["candidate_count"])
                    )

                y_main_all = np.vstack([y_main_original, y_main_new])
                y_sec_all = np.vstack([y_sec_original, y_sec_new])

                n_generated = len(y_main_new)
                synthetic_experiment_ids = (
                    RFDataGeneratorPipeline
                    ._synthetic_experiment_ids(
                        group_idx=group_idx,
                        n_generated=n_generated,
                    )
                )
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
            final_geometry_group_df["sparse_reference_group_id"] = (
                str(sparse_reference_group["group_ids"])
                if sparse_reference_group is not None
                else ""
            )
            final_geometry_group_df["sparse_reference_experiment_ids"] = (
                str(sparse_reference_group["experiment_ids"])
                if sparse_reference_group is not None
                else ""
            )
            final_geometry_group_df["sparse_reference_changed_feature"] = (
                str(sparse_reference_group["changed_features"])
                if sparse_reference_group is not None
                else ""
            )
            final_geometry_group_df["sparse_reference_target_value"] = (
                sparse_reference_group["reference_selection_method"]
                if sparse_reference_group is not None
                else ""
            )
            final_geometry_group_df["sparse_reference_value"] = (
                str(sparse_reference_group["candidate_count"])
                if sparse_reference_group is not None
                else ""
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
