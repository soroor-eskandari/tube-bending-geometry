from __future__ import annotations

import ast
import logging
from pathlib import Path

import pandas as pd

from src.logging.log_utils import log_function

logger = logging.getLogger(__name__)


class AugmentationDatasetBuilder:
    """
    Constructs the dataset used for augmentation.

    1. Read processed geometry measurements.
    2. Build Group_ID mapping from unique bending setups.
    3. Attach Group_ID to geometry observations.
    """

    TARGET_COLUMNS = [
        "Secondary-axis [mm]",
        "Main-axis [mm]",
        "Out-of-roundness [-]",
        "Collapse [mm]",
    ]

    def __init__(self, path_root: Path) -> None:

        self.path_root = path_root

        self.geometry_path = (
            self.path_root / "data" / "processed" / "geometry_data.csv"
        )

        self.unique_setup = (
            self.path_root / "data" / "raw" / "unique_bending_setups.csv"
        )

    @log_function
    def build_dataset(self) -> pd.DataFrame:

        logger.info("Starting augmentation dataset preparation")

        geometry_df = self._read_geometry_data()

        group_mapping = self._build_group_mapping()

        logger.info("Adding Group_ID column")

        geometry_df["Group_ID"] = geometry_df["Experiment_ID"].map(group_mapping)

        if geometry_df["Group_ID"].isna().any():
            raise ValueError("Some geometry rows could not be assigned a Group_ID")

        geometry_df["Group_ID"] = geometry_df["Group_ID"].astype(int)

        geometry_df = self._reorder_df(geometry_df)

        logger.info(
            "Dataset ready for augmentation | rows=%s | groups=%s",
            len(geometry_df),
            geometry_df["Group_ID"].nunique(),
        )

        return geometry_df

    @log_function
    def _read_geometry_data(self) -> pd.DataFrame:

        logger.info("Reading geometry dataset from %s", self.geometry_path)

        geometry_df = pd.read_csv(self.geometry_path)

        required_columns = {
            "Experiment_ID",
            "Angle[degree]ORDistance[mm]",
            *self.TARGET_COLUMNS,
        }

        self._validate_columns(geometry_df, required_columns, "geometry_df")

        logger.info("Geometry data loaded | shape=%s", geometry_df.shape)

        return geometry_df[
            [
                "Experiment_ID",
                "Angle[degree]ORDistance[mm]",
                "Main-axis [mm]",
                "Out-of-roundness [-]",
                "Secondary-axis [mm]",
                "Collapse [mm]",
            ]
        ]

    @log_function
    def _build_group_mapping(self) -> dict:

        logger.info("Building Experiment_ID → Group_ID mapping")

        unique_df = pd.read_csv(self.unique_setup)

        mapping = {}

        for group_id, row in enumerate(unique_df.itertuples(), start=1):

            exp_ids = ast.literal_eval(row.Experiment_Number)

            for exp_id in exp_ids:
                mapping[int(exp_id)] = group_id

        logger.info("Generated mapping for %s experiments", len(mapping))

        return mapping

    @log_function
    def _reorder_df(self, df: pd.DataFrame) -> pd.DataFrame:

        cols = df.columns.tolist()

        cols.remove("Group_ID")

        exp_index = cols.index("Experiment_ID")

        cols.insert(exp_index + 1, "Group_ID")

        return df[cols]

    @staticmethod
    def _validate_columns(
        df: pd.DataFrame,
        required_columns: set[str],
        df_name: str,
    ):

        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise ValueError(f"Missing columns in {df_name}: {sorted(missing_columns)}")