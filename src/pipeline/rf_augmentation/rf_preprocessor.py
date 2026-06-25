import pandas as pd
from src.logging.log_utils import log_function


class RFPreprocessor:

    @staticmethod
    def preprocess_data(
        machine_movement_df: pd.DataFrame,
        bending_df: pd.DataFrame,
    ):

        machine_movement = machine_movement_df.copy()
        bending = bending_df.copy()

        # =========================
        # MACHINE + MOVEMENT CLEANING
        # =========================

        # 1. Remove constant features
        constant_machine_movement_cols = [
            "COLLET_ROTATING_Movement_[mm]",
            "PRESSURE-DIE_LATERAL_Movement_[mm]",
        ]
        machine_movement = machine_movement.drop(
            columns=[c for c in constant_machine_movement_cols if c in machine_movement.columns],
            errors="ignore"
        )

        # 2. Remove highly correlated features
        if (
            "CLAMP-DIE_LATERAL_Movement_[mm]" in machine_movement.columns
            and "PRESSURE-DIE_LEFT_AXIAL_Movement_[mm]" in machine_movement.columns
        ):
            machine_movement = machine_movement.drop(
                columns=["PRESSURE-DIE_LEFT_AXIAL_Movement_[mm]"]
            )

        # 3. Remove noisy features
        noisy_machine_movement_cols = [
            "BEND-DIE_VERTICAL_Movement_[mm]",
            "MACHINE_PRESSURE-DIE_AXIAL_Max_Torque_[%]",
        ]
        machine_movement = machine_movement.drop(
            columns=[c for c in noisy_machine_movement_cols if c in machine_movement.columns],
            errors="ignore"
        )

        # 4. Remove unnecessary signals
        unnecessary_machine_columns = [
            "MACHINE_BEND-DIE_VERTICAL_Max_Torque_[%]",
            "MACHINE_COLLET_ROTATING_Max_Torque_[%]",
            "MACHINE_PRESSURE-DIE_LEFT_AXIAL_Max_Torque_[%]",
        ]
        machine_movement = machine_movement.drop(
            columns=[c for c in unnecessary_machine_columns if c in machine_movement.columns],
            errors="ignore"
        )

        # =========================
        # BENDING CLEANING
        # =========================

        constant_bending_cols = [
            "Tube",
            "Outer-diameter",
            "Wall-thickness",
            "Wiper-die shortening",
            "Target-angle",
            "Pressure-die lateral position",
        ]
        bending = bending.drop(
            columns=[c for c in constant_bending_cols if c in bending.columns],
            errors="ignore",
        )

        # =========================
        # FINAL OUTPUT
        # =========================

        return machine_movement, bending
