import logging
from pathlib import Path

from src.pipeline.rf_augmentation.rf_data_generator_pipeline import RFDataGeneratorPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    project_root = Path(".")
    output_dir = project_root / "data" / "rf_augmented"
    feature_sampling_modes = [
        "random-within-group",
        "within-group-interpolation",
        "sensor-augmented",
    ]
    '''
    Options for feature_sampling_modes:
        "random-within-group"
        "within-group-interpolation"
        "sensor-augmented"
    '''
    include_all_features = False
    sensor_augmentation_modes = ["raw"]

    '''
    sensor_augmentation_modes options:
        ["raw"] for raw cleaned sensor data only
    '''
    sensor_noise_snr_db = 40.0

    # ---------------------------------------
    # Generate new samples using the trained RF model
    # ---------------------------------------
    for sensor_augmentation_mode in sensor_augmentation_modes:
        for feature_sampling_mode in feature_sampling_modes:
            logger.info(
                "Generating RF data | sensor mode=%s | feature source=%s",
                sensor_augmentation_mode,
                feature_sampling_mode,
            )
            RFDataGeneratorPipeline.run(
                project_root=project_root,
                n_new_samples=10,
                output_dir=output_dir,
                feature_sampling_mode=feature_sampling_mode,
                include_all_features=include_all_features,
                sensor_augmentation_mode=sensor_augmentation_mode,
                regenerate_sensor_data=True,
                sensor_noise_snr_db=sensor_noise_snr_db,
            )
