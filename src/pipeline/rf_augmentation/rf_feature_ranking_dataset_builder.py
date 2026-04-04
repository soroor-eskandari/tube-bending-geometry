import pandas as pd
import numpy as np
import logging

from src.logging.log_utils import log_function

logger = logging.getLogger(__name__)


class RFFeatureRankerDatasetBuilder:

    @staticmethod
    @log_function
    def build(
        machine_movement_df: pd.DataFrame,
        bending_df: pd.DataFrame,
        geometry_df: pd.DataFrame,
        expected_angles: int = 45,
        use_bending: bool = True,
    ):
        """
        Builds dataset for RF training.

        Key behavior:
        - Each experiment may have 45–47 geometry rows
        - We STANDARDIZE to `expected_angles` (default=45):
            → sort by angle
            → take first 45
            → drop only if <45

        Args:
            use_bending:
                True  → use machine + bending features
                False → only machine movement features

        Returns:
            X_rf        → (n_experiments, n_features)
            y_main      → (n_experiments, 45)
            y_secondary → (n_experiments, 45)
            feature_names → list
        """

        logger.info(f"Starting dataset build | use_bending={use_bending}")

        # -------------------------
        # Step 1: Feature matrix
        # -------------------------
        X_df, experiment_ids = RFFeatureRankerDatasetBuilder._build_features(
            machine_movement_df,
            bending_df,
            use_bending,
        )

        # -------------------------
        # Step 2: Geometry targets
        # -------------------------
        y_main, y_secondary, aligned_ids = RFFeatureRankerDatasetBuilder._prepare_geometry_targets(
            geometry_df,
            experiment_ids,
            expected_angles,
        )

        # -------------------------
        # Step 3: Align X with geometry
        # -------------------------
        logger.info("Aligning features with geometry")

        id_to_index = {eid: i for i, eid in enumerate(experiment_ids)}
        valid_indices = [id_to_index[eid] for eid in aligned_ids]

        X_aligned = X_df.iloc[valid_indices]

        # -------------------------
        # Step 4: Convert to numpy
        # -------------------------
        X_rf = X_aligned.to_numpy(dtype=np.float32)

        logger.info(f"Final dataset shape: X={X_rf.shape}, y={y_main.shape}")

        return X_rf, y_main, y_secondary, X_aligned.columns.tolist()


    # ============================================================
    # FEATURE BUILDING
    # ============================================================

    @staticmethod
    def _build_features(
        machine_movement_df: pd.DataFrame,
        bending_df: pd.DataFrame,
        use_bending: bool,
    ):
        """
        Builds feature matrix per experiment.

        - One row per Experiment_ID
        - Optionally merges bending features
        """

        logger.info("Building feature matrix")

        df = machine_movement_df.copy()

        if "Experiment_ID" not in df.columns:
            raise ValueError("machine_movement_df must contain 'Experiment_ID'")

        # -------------------------
        # WITH bending
        # -------------------------
        if use_bending:
            logger.info("Including bending features")

            bending = bending_df.copy()

            if "Experiment_ID" not in bending.columns:
                raise ValueError("bending_df must contain 'Experiment_ID'")

            merged = pd.merge(
                df,
                bending,
                on="Experiment_ID",
                how="inner",
                validate="one_to_one",
            )

        # -------------------------
        # WITHOUT bending
        # -------------------------
        else:
            logger.info("Using only machine movement features")
            merged = df

        logger.info(f"Feature shape before cleaning: {merged.shape}")

        # -------------------------
        # Extract IDs
        # -------------------------
        experiment_ids = merged["Experiment_ID"].tolist()

        X_df = merged.drop(columns=["Experiment_ID"])

        # -------------------------
        # Drop all-NaN columns
        # -------------------------
        nan_cols = X_df.columns[X_df.isna().all()].tolist()
        if nan_cols:
            logger.warning(f"Dropping {len(nan_cols)} all-NaN columns")
            X_df = X_df.drop(columns=nan_cols)

        logger.info(f"Final feature shape: {X_df.shape}")

        return X_df, experiment_ids


    # ============================================================
    # GEOMETRY TARGET PREPARATION (FIXED LOGIC)
    # ============================================================

    @staticmethod
    def _prepare_geometry_targets(
        geometry_df: pd.DataFrame,
        experiment_ids: list,
        expected_angles: int,
    ):
        """
        Standardizes geometry targets.

        For each experiment:
        - sort by angle
        - if >= expected_angles → take first 45
        - if < expected_angles → drop experiment

        Returns aligned targets.
        """

        logger.info("Preparing geometry targets (45-angle standardization)")

        df = geometry_df.copy()

        required_cols = [
            "Experiment_ID",
            "Angle[degree]ORDistance[mm]",
            "Main-axis [mm]",
            "Secondary-axis [mm]",
        ]

        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing column in geometry_df: {col}")

        df = df[df["Experiment_ID"].isin(experiment_ids)]
        df = df.sort_values(["Experiment_ID", "Angle[degree]ORDistance[mm]"])

        grouped = df.groupby("Experiment_ID")

        y_main = []
        y_secondary = []
        aligned_ids = []

        dropped_due_to_short = 0

        for exp_id in experiment_ids:

            if exp_id not in grouped.groups:
                continue

            group = grouped.get_group(exp_id)

            # Sort + truncate to 45
            group = group.sort_values("Angle[degree]ORDistance[mm]")

            if len(group) < expected_angles:
                dropped_due_to_short += 1
                continue

            group = group.head(expected_angles)

            y_main.append(group["Main-axis [mm]"].to_numpy(dtype=np.float32))
            y_secondary.append(group["Secondary-axis [mm]"].to_numpy(dtype=np.float32))
            aligned_ids.append(exp_id)

        if dropped_due_to_short > 0:
            logger.warning(f"Dropped {dropped_due_to_short} experiments (<{expected_angles} angles)")

        y_main = np.vstack(y_main)
        y_secondary = np.vstack(y_secondary)

        logger.info(f"y_main shape: {y_main.shape}")
        logger.info(f"y_secondary shape: {y_secondary.shape}")

        return y_main, y_secondary, aligned_ids