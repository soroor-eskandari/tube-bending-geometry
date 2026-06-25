import logging
from pathlib import Path

from src.pipeline.rf_augmentation.rf_augmentation_pipeline import RFAugmentationPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    project_root = Path(".")
    output_dir = project_root / "data" / "rf_augmented"
    include_all_features = False
    use_mlflow = False
    sensor_augmentation_modes = ["raw"]

    """
    sensor_augmentation_modes options:
        ["raw"] for raw cleaned sensor data only
    """

    logger.info("Starting RF-based augmentation run")

    # ---------------------------------------
    # Candidate pool sizes (NOT final features)
    # ---------------------------------------


    logger.info("Running pipeline with candidate top_k_feature=10")


    for sensor_augmentation_mode in sensor_augmentation_modes:
        logger.info(
            "Running RF augmentation training for sensor mode: %s",
            sensor_augmentation_mode,
        )
        RFAugmentationPipeline.run(
            project_root=project_root,
            output_dir=output_dir,
            include_all_features=include_all_features,
            sensor_augmentation_mode=sensor_augmentation_mode,
            use_mlflow=use_mlflow,
        )

    logger.info("RF geometry augmentation finished")
