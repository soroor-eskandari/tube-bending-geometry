import pandas as pd

from src.logging.log_utils import log_function


class GroupWiseAggregator:

    @staticmethod
    @log_function
    def aggregate(
        train_df: pd.DataFrame,
        output_columns: list[str] = None,
        variance_ddof: int = 0,
    ) -> pd.DataFrame:

        train_data = train_df.copy()

        # =========================
        # COLUMN NAMES
        # =========================
        angle_column = "Angle[degree]ORDistance[mm]"

        group_columns = [
            "Group_ID",
            angle_column,
        ]

        if output_columns is None:
            output_columns = [
                "Main-axis [mm]",
                "Secondary-axis [mm]",
            ]

        # =========================
        # VALIDATION
        # =========================
        required_columns = [
            "Group_ID",
            "Experiment_ID",
            angle_column,
        ] + output_columns

        missing_columns = [
            col for col in required_columns
            if col not in train_data.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

        # =========================
        # GROUP-WISE AGGREGATION
        # For each Group_ID and angle, calculate output statistics
        # only when that Group_ID contains more than one unique
        # experiment.
        # =========================
        repeated_group_ids = (
            train_data
            .groupby(
                "Group_ID",
                dropna=False,
                sort=True,
            )["Experiment_ID"]
            .nunique()
        )

        repeated_group_ids = (
            repeated_group_ids[
                repeated_group_ids > 1
            ]
            .index
        )

        train_data = train_data[
            train_data["Group_ID"].isin(
                repeated_group_ids
            )
        ].copy()

        grouped = train_data.groupby(
            group_columns,
            dropna=False,
            sort=True,
        )

        aggregated = grouped.size().rename(
            "Repeat_Count"
        ).reset_index()

        for output_column in output_columns:
            output_stats = (
                grouped[output_column]
                .agg(
                    **{
                        f"{output_column}_Mean": "mean",
                        f"{output_column}_Local_Variance": (
                            lambda values: values.var(
                                ddof=variance_ddof
                            )
                        ),
                    }
                )
                .reset_index()
            )

            aggregated = aggregated.merge(
                output_stats,
                on=group_columns,
                how="left",
            )

        statistic_columns = [
            statistic_column
            for output_column in output_columns
            for statistic_column in [
                f"{output_column}_Mean",
                f"{output_column}_Local_Variance",
            ]
        ]

        output_columns_order = [
            "Group_ID",
            angle_column,
            "Repeat_Count",
        ] + statistic_columns

        # =========================
        # FINAL OUTPUT
        # =========================
        return aggregated[
            output_columns_order
        ].reset_index(drop=True)
