import numpy as np
import pandas as pd

from src.logging.log_utils import log_function


class SKDatasetBuilder:

    @staticmethod
    @log_function
    def build(
        train_group_statistics: pd.DataFrame,
    ) -> tuple[np.ndarray, list[str]]:

        group_statistics = train_group_statistics.copy()

        # =========================
        # COLUMN NAMES
        # =========================
        angle_column = "Angle[degree]ORDistance[mm]"

        main_target = "Main-axis [mm]"
        secondary_target = "Secondary-axis [mm]"

        main_mean_column = f"{main_target}_Mean"
        secondary_mean_column = f"{secondary_target}_Mean"
        main_variance_column = f"{main_target}_Local_Variance"
        secondary_variance_column = f"{secondary_target}_Local_Variance"

        output_columns = [
            "Group_ID",
            angle_column,
            secondary_mean_column,
            main_mean_column,
            secondary_variance_column,
            main_variance_column,
            "Repeat_Count",
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
        # UNIQUE SK DESIGN DATASET
        # Stochastic kriging fits sample means at unique design
        # points with alpha = local_variance / repeat_count.
        # =========================
        dataset_df = group_statistics[
            output_columns
        ].copy()

        dataset_df = dataset_df.dropna(
            subset=[
                secondary_mean_column,
                main_mean_column,
                secondary_variance_column,
                main_variance_column,
                "Repeat_Count",
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
            & (
                dataset_df["Repeat_Count"]
                > 0
            )
        ].reset_index(drop=True)

        if dataset_df.empty:
            raise ValueError(
                "No repeated design points with valid local variance "
                "were found for stochastic kriging."
            )

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


GPDatasetBuilder = SKDatasetBuilder
