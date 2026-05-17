import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.logging.log_utils import log_function


class DataSplitter:

    @staticmethod
    @log_function
    def splittor(
        geometry_df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

        geometry = geometry_df.copy()

        # =========================
        # VALIDATION
        # =========================
        required_columns = [
            "Group_ID",
            "Experiment_ID",
        ]

        missing_columns = [
            col for col in required_columns
            if col not in geometry.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
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