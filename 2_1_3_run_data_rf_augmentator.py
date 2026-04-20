import logging
from pathlib import Path

from src.pipeline.rf_augmentation.rf_augmentation_pipeline import RFAugmentationPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    project_root = Path(".")
    output_dir = project_root / "data" / "rf_augmented"

    logger.info("Starting RF-based augmentation run")

    # ---------------------------------------
    # Candidate pool sizes (NOT final features)
    # ---------------------------------------


    logger.info("Running pipeline with candidate top_k_feature=10")


    RFAugmentationPipeline.run(
        project_root=project_root,
        n_new_samples=100,
        output_dir=output_dir,
    )

    logger.info("RF geometry augmentation finished")

