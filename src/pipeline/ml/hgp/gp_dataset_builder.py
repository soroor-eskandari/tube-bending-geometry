import numpy as np
import pandas as pd

from src.logging.log_utils import log_function


class GPDatasetBuilder:

    @staticmethod
    @log_function
    def build(
        train_df: pd.DataFrame,
        train_group_statistics: pd.DataFrame,
    ) -> tuple[np.ndarray, list[str]]:

        train_data = train_df.copy()
        group_statistics = train_group_statistics.copy()

        # =========================
        # COLUMN NAMES
        # =========================
        angle_column = "Angle[degree]ORDistance[mm]"

        join_columns = [
            "Group_ID",
            angle_column,
        ]

        main_target = "Main-axis [mm]"
        secondary_target = "Secondary-axis [mm]"

        main_mean_column = f"{main_target}_Mean"
        secondary_mean_column = f"{secondary_target}_Mean"

        output_columns = [
            "Experiment_ID",
            "Group_ID",
            angle_column,
            secondary_mean_column,
            main_mean_column,
        ]

        # =========================
        # VALIDATION
        # =========================
        required_train_columns = [
            "Experiment_ID",
            "Group_ID",
            angle_column,
            secondary_target,
            main_target,
        ]

        required_statistics_columns = (
            join_columns
            + [
                secondary_mean_column,
                main_mean_column,
            ]
        )

        missing_train_columns = [
            col for col in required_train_columns
            if col not in train_data.columns
        ]

        if missing_train_columns:
            raise ValueError(
                "Missing required columns in train_df: "
                f"{missing_train_columns}"
            )

        missing_statistics_columns = [
            col for col in required_statistics_columns
            if col not in group_statistics.columns
        ]

        if missing_statistics_columns:
            raise ValueError(
                "Missing required columns in train_group_statistics: "
                f"{missing_statistics_columns}"
            )

        # =========================
        # MERGE GROUP STATISTICS
        # =========================
        dataset_df = train_data.merge(
            group_statistics[
                required_statistics_columns
            ],
            on=join_columns,
            how="left",
        )

        # =========================
        # FALLBACK TO ACTUAL VALUES
        # Use the mean when a repeated group-angle statistic exists.
        # Otherwise keep the original measured value.
        # =========================
        dataset_df[secondary_mean_column] = (
            dataset_df[secondary_mean_column]
            .fillna(dataset_df[secondary_target])
        )

        dataset_df[main_mean_column] = (
            dataset_df[main_mean_column]
            .fillna(dataset_df[main_target])
        )

        dataset_df = dataset_df[
            output_columns
        ]

        # =========================
        # FINAL OUTPUT
        # =========================
        return (
            dataset_df.to_numpy(),
            output_columns,
        )

    @staticmethod
    @log_function
    def build_noise_gp_dataset(
        train_group_statistics: pd.DataFrame,
    ) -> tuple[np.ndarray, list[str]]:

        group_statistics = train_group_statistics.copy()

        # =========================
        # COLUMN NAMES
        # =========================
        angle_column = "Angle[degree]ORDistance[mm]"

        main_variance_column = (
            "Main-axis [mm]_Local_Variance"
        )

        secondary_variance_column = (
            "Secondary-axis [mm]_Local_Variance"
        )

        output_columns = [
            "Group_ID",
            angle_column,
            secondary_variance_column,
            main_variance_column,
        ]

        # =========================
        # VALIDATION
        # =========================
        missing_columns = [
            col for col in output_columns
            if col not in group_statistics.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing required columns in train_group_statistics: "
                f"{missing_columns}"
            )

        # =========================
        # KEEP ONLY OBSERVED LOCAL VARIANCE
        # =========================
        dataset_df = group_statistics[
            output_columns
        ].copy()

        dataset_df = dataset_df.dropna(
            subset=[
                secondary_variance_column,
                main_variance_column,
            ]
        )

        dataset_df = dataset_df[
            (
                dataset_df[secondary_variance_column]
                >= 0
            )
            & (
                dataset_df[main_variance_column]
                >= 0
            )
        ].reset_index(drop=True)

        # =========================
        # FINAL OUTPUT
        # =========================
        return (
            dataset_df.to_numpy(),
            output_columns,
        )
