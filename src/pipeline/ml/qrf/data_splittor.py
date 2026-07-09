import ast

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.logging.log_utils import log_function


class DataSplittor:

    @staticmethod
    def splittor(
        geometry_df: pd.DataFrame,
        unique_bending_df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
        use_experiment_split: bool = False,
        train_exp: list[int] | None = None,
        test_exp: list[int] | None = None,
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
        # ADD GROUP ID ONLY IF MISSING
        # If geometry already has Group_ID, use it as the source of truth.
        # Otherwise, derive it from unique_bending rows.
        # =========================
        if "Group_ID" not in geometry.columns:
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

        if use_experiment_split:
            return DataSplittor._split_by_experiment(
                geometry=geometry,
                train_exp=train_exp,
                test_exp=test_exp,
            )

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
    def _split_by_experiment(
        geometry: pd.DataFrame,
        train_exp: list[int] | None,
        test_exp: list[int] | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

        if train_exp is None:
            raise ValueError(
                "train_exp must be provided when use_experiment_split=True"
            )

        experiment_ids = pd.to_numeric(
            geometry["Experiment_ID"],
            errors="coerce",
        )

        if experiment_ids.isna().any():
            invalid_experiment_ids = sorted(
                geometry.loc[
                    experiment_ids.isna(),
                    "Experiment_ID",
                ]
                .dropna()
                .unique()
                .tolist()
            )

            raise ValueError(
                "Experiment_ID must be numeric for experiment-based split. "
                f"Invalid values: {invalid_experiment_ids}"
            )

        experiment_ids = experiment_ids.astype(int)

        train_experiments = DataSplittor._normalize_experiment_list(
            experiments=train_exp,
            argument_name="train_exp",
        )

        available_experiments = set(
            experiment_ids.unique().tolist()
        )

        missing_train_experiments = sorted(
            train_experiments - available_experiments
        )

        if missing_train_experiments:
            raise ValueError(
                "train_exp contains Experiment_ID values not present in "
                f"geometry_df: {missing_train_experiments}"
            )

        train_mask = experiment_ids.isin(train_experiments)

        if test_exp is None:
            test_mask = ~train_mask
        else:
            test_experiments = DataSplittor._normalize_experiment_list(
                experiments=test_exp,
                argument_name="test_exp",
            )

            overlapping_experiments = sorted(
                train_experiments & test_experiments
            )

            if overlapping_experiments:
                raise ValueError(
                    "train_exp and test_exp must not overlap. "
                    f"Overlapping Experiment_ID values: {overlapping_experiments}"
                )

            missing_test_experiments = sorted(
                test_experiments - available_experiments
            )

            if missing_test_experiments:
                raise ValueError(
                    "test_exp contains Experiment_ID values not present in "
                    f"geometry_df: {missing_test_experiments}"
                )

            test_mask = experiment_ids.isin(test_experiments)

        train_df = geometry.loc[train_mask].reset_index(drop=True)
        test_df = geometry.loc[test_mask].reset_index(drop=True)

        if train_df.empty:
            raise ValueError(
                "Experiment-based split produced an empty train_df"
            )

        if test_df.empty:
            raise ValueError(
                "Experiment-based split produced an empty test_df"
            )

        return train_df, test_df

    @staticmethod
    def _normalize_experiment_list(
        experiments: list[int],
        argument_name: str,
    ) -> set[int]:

        if not isinstance(experiments, (list, tuple, set)):
            raise ValueError(
                f"{argument_name} must be a list, tuple, or set of "
                "Experiment_ID values"
            )

        try:
            normalized = {
                int(exp_id)
                for exp_id in experiments
            }
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{argument_name} must contain only numeric Experiment_ID values"
            ) from exc

        if not normalized:
            raise ValueError(
                f"{argument_name} must not be empty"
            )

        return normalized

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
