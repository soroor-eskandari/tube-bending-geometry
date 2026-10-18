import logging
from pathlib import Path

from src.pipeline.rf_augmentation.rf_best_subset_pipeline import RFSubsetFinderPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    project_root = Path(".")
    output_dir = project_root / "src" / "pipeline" / "rf_augmentation" / "result"

    logger.info("Starting RF greedy feature subset search")

    RFSubsetFinderPipeline.run(
        project_root=project_root,
        top_k_feature=23,
        output_dir=output_dir,
    )

    logger.info("RF greedy feature subset search finished")
