import ast

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.logging.log_utils import log_function


class DataSplittor:

    @staticmethod
    @log_function
    def splittor(
        geometry_df: pd.DataFrame,
        unique_bending_df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

        geometry = geometry_df.copy()
        unique_bending = unique_bending_df.copy()

        # =========================
        # VALIDATION
        # =========================
        required_geometry_columns = [
            "Experiment_ID",
        ]

        missing_geometry_columns = [
            col for col in required_geometry_columns
            if col not in geometry.columns
        ]

        if missing_geometry_columns:
            raise ValueError(
                f"Missing required columns in geometry_df: {missing_geometry_columns}"
            )

        required_unique_bending_columns = [
            "Experiment_Number",
        ]

        missing_unique_bending_columns = [
            col for col in required_unique_bending_columns
            if col not in unique_bending.columns
        ]

        if missing_unique_bending_columns:
            raise ValueError(
                "Missing required columns in unique_bending_df: "
                f"{missing_unique_bending_columns}"
            )

        # =========================
        # ADD GROUP ID
        # unique_bending rows define the groups.
        # Experiment_Number values such as "[1, 2, 3]"
        # mean experiments 1, 2, and 3 share Group_ID = row index + 1.
        # =========================
        experiment_to_group = DataSplittor._build_experiment_to_group_mapping(
            unique_bending
        )

        experiment_ids = pd.to_numeric(
            geometry["Experiment_ID"],
            errors="coerce",
        )

        geometry["Group_ID"] = experiment_ids.map(experiment_to_group)

        if geometry["Group_ID"].isna().any():
            missing_experiment_ids = sorted(
                geometry.loc[
                    geometry["Group_ID"].isna(),
                    "Experiment_ID",
                ]
                .dropna()
                .unique()
                .tolist()
            )

            raise ValueError(
                "Some geometry rows could not be assigned a Group_ID. "
                f"Missing Experiment_ID values: {missing_experiment_ids}"
            )

        geometry["Group_ID"] = geometry["Group_ID"].astype(int)
        geometry = DataSplittor._move_group_id_after_experiment_id(geometry)

        # =========================
        # GROUP-AWARE TRAIN/TEST SPLIT
        # Keep all experiments from the same group
        # either in train or in test
        # =========================
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state,
        )

        groups = geometry["Group_ID"]

        train_idx, test_idx = next(
            splitter.split(
                geometry,
                groups=groups,
            )
        )

        train_df = geometry.iloc[train_idx].reset_index(drop=True)
        test_df = geometry.iloc[test_idx].reset_index(drop=True)

        # =========================
        # FINAL OUTPUT
        # =========================
        return train_df, test_df

    @staticmethod
    def _build_experiment_to_group_mapping(
        unique_bending: pd.DataFrame,
    ) -> dict[int, int]:

        experiment_to_group = {}

        for group_id, experiments in enumerate(
            unique_bending["Experiment_Number"],
            start=1,
        ):
            if isinstance(experiments, str):
                experiments = ast.literal_eval(experiments)

            if not isinstance(experiments, (list, tuple, set)):
                experiments = [experiments]

            for exp_id in experiments:
                experiment_to_group[int(exp_id)] = group_id

        return experiment_to_group

    @staticmethod
    def _move_group_id_after_experiment_id(
        geometry: pd.DataFrame,
    ) -> pd.DataFrame:

        columns = geometry.columns.tolist()
        columns.remove("Group_ID")

        experiment_index = columns.index("Experiment_ID")
        columns.insert(experiment_index + 1, "Group_ID")

        return geometry[columns]
