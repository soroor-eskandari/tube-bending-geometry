import logging
from pathlib import Path
import pandas as pd

from src.logging.log_utils import log_function
from .augmentation_data_builder import AugmentationDatasetBuilder
from .stats_builder import GroupStatisticsBuilder
from .synthetic_generator import SyntheticExperimentGenerator


logger = logging.getLogger(__name__)


class GeometryAugmentationPipeline:

    @staticmethod
    @log_function
    def run(
        project_root: Path,
        expected_per_group: int,
        output_dir: Path,
    ):

        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting geometry augmentation pipeline")

        # -------------------------------------------------
        # Build dataset
        # -------------------------------------------------

        dataset_builder = AugmentationDatasetBuilder(project_root)

        geometry_df = dataset_builder.build_dataset()

        # mark original data
        geometry_df["Synthetic"] = False

        # -------------------------------------------------
        # Augmentation configuration
        # -------------------------------------------------

        angle_col = "Angle[degree]ORDistance[mm]"

        feature_cols = [
            "Angle[degree]ORDistance[mm]",
            "Secondary-axis [mm]",
            "Main-axis [mm]",
            "Out-of-roundness [-]",
            "Collapse [mm]",
        ]

        # -------------------------------------------------
        # Compute statistics
        # -------------------------------------------------

        logger.info("Building statistics")

        stats_builder = GroupStatisticsBuilder(
            angle_col=angle_col,
            feature_cols=feature_cols,
        )

        stats = stats_builder.build(geometry_df)

        # -------------------------------------------------
        # Generate synthetic experiments
        # -------------------------------------------------

        logger.info("Generating synthetic experiments")

        generator = SyntheticExperimentGenerator(
            angle_col=angle_col,
            feature_cols=feature_cols,
            expected_per_group=expected_per_group,
        )

        synthetic_df = generator.generate(
            geometry_df,
            stats,
        )

        # mark synthetic rows
        synthetic_df["Synthetic"] = True

        # -------------------------------------------------
        # Merge datasets
        # -------------------------------------------------

        augmented_df = pd.concat(
            [geometry_df, synthetic_df],
            ignore_index=True,
        )

        # store augmentation parameter in dataset
        augmented_df["Expected_Per_Group"] = expected_per_group

        # -------------------------------------------------
        # Save dataset
        # -------------------------------------------------

        augmented_path = output_dir / f"augmented_geometry_{expected_per_group}.csv"

        augmented_df.to_csv(augmented_path, index=False)

        logger.info("Original rows: %s", len(geometry_df))
        logger.info("Synthetic rows: %s", len(synthetic_df))
        logger.info("Final rows: %s", len(augmented_df))

        logger.info("Saved augmented dataset to %s", augmented_path)

        return augmented_df