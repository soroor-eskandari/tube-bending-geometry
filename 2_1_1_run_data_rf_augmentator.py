import logging
from pathlib import Path

from src.pipeline.rf_augmentation.rf_augmentation_pipeline import RFAugmentationPipeline

logger = logging.getLogger(__name__)


if __name__ == "__main__":

    project_root = Path(".")

    output_dir = project_root / "data" / "rf_augmented"

    logger.info("Starting RF-based augmentation run")

    RFAugmentationPipeline.run(
        project_root=project_root,
        expected_per_group=40,
        output_dir=output_dir,
    )

    logger.info("RF geometry augmentation finished")