import logging
import pandas as pd

from src.logging.log_utils import log_function

logger = logging.getLogger(__name__)


class GroupStatisticsBuilder:
    """
    Computes per-group statistics (mean/std) for each feature per angle.
    """

    def __init__(self, angle_col: str, feature_cols: list[str]):
        self.angle_col = angle_col
        self.feature_cols = feature_cols

    @log_function
    def build(self, df: pd.DataFrame) -> pd.DataFrame:

        logger.info("Building group statistics")
        logger.info("Input rows: %s", len(df))

        stats = (
            df
            .groupby(["Group_ID", self.angle_col])[self.feature_cols]
            .agg(["mean", "std"])
            .reset_index()
        )

        stats.columns = [
            "_".join(col).strip("_") if isinstance(col, tuple) else col
            for col in stats.columns
        ]

        logger.info("Generated statistics rows: %s", len(stats))

        return stats