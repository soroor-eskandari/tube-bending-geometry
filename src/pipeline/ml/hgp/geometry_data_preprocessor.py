import ast
import pandas as pd
from src.logging.log_utils import log_function


class GeometryPreprocessor:

    @staticmethod
    @log_function
    def preprocess(
        geometry_df: pd.DataFrame,
        bending_df: pd.DataFrame,
    ) -> pd.DataFrame:

        geometry = geometry_df.copy()
        bending = bending_df.copy()

        # =========================
        # COLUMN NAMES
        # =========================
        angle_col = "Angle[degree]ORDistance[mm]"

        # =========================
        # CREATE GROUP MAPPING
        # =========================
        experiment_to_group = {}

        for group_id, experiments in enumerate(
            bending["Experiment_Number"],
            start=1,
        ):

            # Convert string representation of list to actual list
            if isinstance(experiments, str):
                experiments = ast.literal_eval(experiments)

            for exp_id in experiments:
                experiment_to_group[exp_id] = group_id

        # =========================
        # ADD GROUP ID TO GEOMETRY
        # =========================
        geometry["Group_ID"] = geometry["Experiment_ID"].map(
            experiment_to_group
        )

        # =========================
        # FILTER ANGLES
        # Keep only angles between 0 and 45 degrees (inclusive)
        # =========================
        geometry = geometry[
            (geometry[angle_col] >= 0)
            & (geometry[angle_col] <= 44)
        ]

        # =========================
        # KEEP REQUIRED COLUMNS
        # =========================
        geometry = geometry[
            [
                "Group_ID",
                "Experiment_ID",
                angle_col,
                "Secondary-axis [mm]",
                "Main-axis [mm]",
            ]
        ]

        # =========================
        # RESET INDEX
        # =========================
        geometry = geometry.reset_index(drop=True)

        # =========================
        # FINAL OUTPUT
        # =========================
        return geometry