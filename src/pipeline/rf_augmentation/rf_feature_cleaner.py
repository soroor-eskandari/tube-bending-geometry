import pandas as pd
import numpy as np
import logging

from src.logging.log_utils import log_function

logger = logging.getLogger(__name__)


class RFFeatureCleaner:

    @staticmethod
    @log_function
    def clean_features(
        df: pd.DataFrame,
        target: pd.Series = None,
        corr_threshold: float = 0.95,
    ):
        """
        Cleans feature dataframe while preserving Experiment_ID.

        Steps:
        1. Preserve Experiment_ID
        2. Drop Time column if exists
        3. Remove all-NaN columns
        4. Remove constant columns
        5. Fill NaNs
        6. Remove highly correlated features
        7. Return dataframe with Experiment_ID as first column
        """

        logger.info("Starting feature cleaning")

        df_clean = df.copy()

        # -------------------------
        # 0. Preserve Experiment_ID
        # -------------------------
        if "Experiment_ID" not in df_clean.columns:
            raise ValueError("Experiment_ID column is required")

        experiment_ids = df_clean["Experiment_ID"]
        df_clean = df_clean.drop(columns=["Experiment_ID"])

        # -------------------------
        # Drop Time column if exists
        # -------------------------
        if "Time_[s]" in df_clean.columns:
            df_clean = df_clean.drop(columns=["Time_[s]"])
            logger.info("Dropped Time_[s] column")

        # -------------------------
        # 1. Remove all-NaN columns
        # -------------------------
        all_nan_cols = df_clean.columns[df_clean.isna().all()]
        df_clean = df_clean.drop(columns=all_nan_cols)

        logger.info(f"Removed {len(all_nan_cols)} all-NaN columns")

        # -------------------------
        # 2. Remove constant columns
        # -------------------------
        nunique = df_clean.nunique()
        constant_cols = nunique[nunique <= 1].index

        df_clean = df_clean.drop(columns=constant_cols)

        logger.info(f"Removed {len(constant_cols)} constant columns")

        # -------------------------
        # 3. Fill NaNs
        # -------------------------
        nan_before = df_clean.isna().sum().sum()

        df_clean = df_clean.fillna(df_clean.median())

        nan_after = df_clean.isna().sum().sum()

        logger.info(f"NaNs before fill: {nan_before}, after fill: {nan_after}")

        # -------------------------
        # 4. Correlation filtering
        # -------------------------
        logger.info("Computing correlation matrix")

        corr_matrix = df_clean.corr().abs()

        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = set()

        logger.info("Applying correlation filtering")

        for col in upper_triangle.columns:
            high_corr = upper_triangle[col][upper_triangle[col] > corr_threshold]

            for row in high_corr.index:
                if row in to_drop or col in to_drop:
                    continue

                if target is not None:
                    # Safe correlation (handle NaNs in target)
                    corr_row = abs(df_clean[row].corr(target))
                    corr_col = abs(df_clean[col].corr(target))

                    if np.isnan(corr_row):
                        corr_row = 0
                    if np.isnan(corr_col):
                        corr_col = 0

                    if corr_row >= corr_col:
                        to_drop.add(col)
                    else:
                        to_drop.add(row)
                else:
                    to_drop.add(col)

        df_clean = df_clean.drop(columns=list(to_drop))

        logger.info(f"Removed {len(to_drop)} highly correlated features")

        # -------------------------
        # 5. Reattach Experiment_ID
        # -------------------------
        df_clean.insert(0, "Experiment_ID", experiment_ids)

        logger.info(f"Final feature shape: {df_clean.shape}")

        return df_clean