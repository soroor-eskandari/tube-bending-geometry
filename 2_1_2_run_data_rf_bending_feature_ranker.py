import logging
from pathlib import Path

from src.pipeline.rf_augmentation.rf_bending_feature_ranker_pipeline import (
    RFBendingFeatureRankerPipeline,
)

logger = logging.getLogger(__name__)


if __name__ == "__main__":

    project_root = Path(".")

    logger.info("Starting RF-based bending feature rank finding run")

    RFBendingFeatureRankerPipeline.run(
        project_root=project_root,
    )

    logger.info("Finding rank of bending features finished")
