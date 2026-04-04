import pandas as pd
import numpy as np
import logging

from src.logging.log_utils import log_function

logger = logging.getLogger(__name__)


class RFTrainingDatasetBuilder:

    @staticmethod
    @log_function
    def build(
        machine_movement_average_df: pd.DataFrame,
        bending_features_df: pd.DataFrame,
        geometry_df: pd.DataFrame,
        padding_value: float = 0.0,
        expected_angles: int = 45,
    ):
        """
        Main entry point:
        - builds RF feature matrix
        - prepares geometry targets
        - aligns both

        Returns:
            X_rf      → (n_experiments, n_features)
            y_main    → (n_experiments, 45)
            y_secondary → (n_experiments, 45)
        """

        logger.info("Starting full dataset build")

        # -------------------------
        # Step 1: Features
        # -------------------------
        X_rf, experiment_ids = RFTrainingDatasetBuilder._build_features(
            machine_movement_average_df,
            bending_features_df,
            padding_value,
        )

        # -------------------------
        # Step 2: Geometry
        # -------------------------
        y_main, y_secondary, aligned_ids = RFTrainingDatasetBuilder._prepare_geometry_targets(
            geometry_df,
            experiment_ids,
            expected_angles,
        )

        # -------------------------
        # Step 3: Align X with geometry
        # -------------------------
        logger.info("Aligning features with valid geometry experiments")

        id_to_index = {eid: i for i, eid in enumerate(experiment_ids)}

        valid_indices = [id_to_index[eid] for eid in aligned_ids]

        X_rf_aligned = X_rf[valid_indices]



        logger.info("Dataset build completed")

        return X_rf_aligned, y_main, y_secondary


    # ============================================================
    # INTERNAL HELPERS
    # ============================================================

    @staticmethod
    def _build_features(
        machine_movement_average_df: pd.DataFrame,
        bending_features_df: pd.DataFrame,
        padding_value: float,
    ):
        """
        Builds flattened feature matrix.
        """

        logger.info("Building RF features")

        df = machine_movement_average_df.copy()
        bending = bending_features_df.copy()

        if "Experiment_ID" not in df.columns:
            raise ValueError("machine_movement_average_df must contain 'Experiment_ID'")

        if "Experiment_ID" not in bending.columns:
            raise ValueError("bending_features_df must contain 'Experiment_ID'")

        df = df.sort_values(["Experiment_ID"])
        grouped = df.groupby("Experiment_ID")

        sequences = []
        experiment_ids = []

        for exp_id, group in grouped:
            seq = group["avg_signal"].to_numpy(dtype=np.float32)
            sequences.append(seq)
            experiment_ids.append(exp_id)

        max_len = max(len(seq) for seq in sequences)

        padded = np.full((len(sequences), max_len), padding_value, dtype=np.float32)

        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = seq

        signal_df = pd.DataFrame(padded)
        signal_df["Experiment_ID"] = experiment_ids

        final_df = pd.merge(
            signal_df,
            bending,
            on="Experiment_ID",
            how="inner",
            validate="one_to_one",
        )

        experiment_ids = final_df["Experiment_ID"].tolist()
        final_df = final_df.drop(columns=["Experiment_ID"])

        X_rf = final_df.to_numpy(dtype=np.float32)

        logger.info(f"Feature matrix shape: {X_rf.shape}")

        return X_rf, experiment_ids


    @staticmethod
    def _prepare_geometry_targets(
        geometry_df: pd.DataFrame,
        experiment_ids: list,
        expected_angles: int,
    ):
        """
        Builds geometry targets aligned by Experiment_ID.
        """

        logger.info("Preparing geometry targets")

        df = geometry_df.copy()

        required_cols = ["Experiment_ID", "Angle[degree]ORDistance[mm]", "Main-axis [mm]", "Secondary-axis [mm]"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing column in geometry_df: {col}")

        df = df[df["Experiment_ID"].isin(experiment_ids)]
        df = df.sort_values(["Experiment_ID", "Angle[degree]ORDistance[mm]"])

        grouped = df.groupby("Experiment_ID")

        y_main = []
        y_secondary = []
        aligned_ids = []

        for exp_id in experiment_ids:

            if exp_id not in grouped.groups:
                continue

            group = grouped.get_group(exp_id).sort_values("Angle[degree]ORDistance[mm]")

            group = group.head(expected_angles)

            if len(group) < expected_angles:
                continue

            y_main.append(group["Main-axis [mm]"].to_numpy(dtype=np.float32))
            y_secondary.append(group["Secondary-axis [mm]"].to_numpy(dtype=np.float32))
            aligned_ids.append(exp_id)

        y_main = np.vstack(y_main)
        y_secondary = np.vstack(y_secondary)

        logger.info(f"y_main shape: {y_main.shape}")
        logger.info(f"y_secondary shape: {y_secondary.shape}")

        return y_main, y_secondary, aligned_ids