import logging
from pathlib import Path

from src.pipeline.augmentation.augmentation_pipeline import GeometryAugmentationPipeline

logger = logging.getLogger(__name__)


if __name__ == "__main__":

    project_root = Path(".")

    output_dir = project_root / "data" / "augmented"

    logger.info("Starting augmentation run")

    GeometryAugmentationPipeline.run(
        project_root=project_root,
        expected_per_group=40,
        output_dir=output_dir,
    )

    logger.info("Geometry augmentation finished")