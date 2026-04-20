import logging
from pathlib import Path

from src.pipeline.rf_augmentation.rf_feature_rank_finder_pipeline import RFRankerPipeline

logger = logging.getLogger(__name__)


if __name__ == "__main__":

    project_root = Path(".")

    output_dir = project_root / "data" / "rf_augmented"

    logger.info("Starting RF-based feature rank finding run")

    RFRankerPipeline.run(
        project_root=project_root,
    )

    logger.info("Finding rank of features finished")