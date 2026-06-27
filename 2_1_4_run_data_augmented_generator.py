import json
import logging
from pathlib import Path

import pandas as pd

from src.pipeline.rf_augmentation.rf_data_generator_pipeline import RFDataGeneratorPipeline
from src.pipeline.rf_augmentation.io_utils import existing_table_path, read_table
from src.pipeline.rf_augmentation.sensor_data_augmentor import SensorDataAugmentor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_streamlit_ui_data(project_root: Path, output_dir: Path) -> None:
    ui_output_dir = output_dir / "ui_data"
    ui_output_dir.mkdir(parents=True, exist_ok=True)

    methods = [
        {
            "key": "within-group-interpolation",
            "title": "Within Group Interpolation",
            "heading": "Interpolated Within Group",
            "suffix": "within_group_interpolation_raw",
            "csv": "final_geometry_within_group_interpolation_raw.csv",
        },
        {
            "key": "sensor-augmented",
            "title": "Sensor Signals Augmented",
            "heading": "Sensor Signals Augmented",
            "suffix": (
                "sensor_augmented_"
                f"{SensorDataAugmentor.mode_to_suffix(SensorDataAugmentor.ALL_METHODS_MODE)}"
            ),
            "csv": (
                "final_geometry_sensor_augmented_"
                f"{SensorDataAugmentor.mode_to_suffix(SensorDataAugmentor.ALL_METHODS_MODE)}"
                ".csv"
            ),
        },
    ]

    for method in methods:
        source_path = existing_table_path(
            output_dir / f"final_geometry_{method['suffix']}.parquet"
        )
        if not source_path.exists():
            raise FileNotFoundError(
                "Missing generated geometry for Streamlit export: "
                f"{source_path}"
            )

        geometry_df = read_table(source_path)
        geometry_df.to_csv(ui_output_dir / method["csv"], index=False)

    group_setup_path = project_root / "data" / "raw" / "unique_bending_setups.csv"
    group_setup_csv = None
    if group_setup_path.exists():
        group_setup_df = pd.read_csv(group_setup_path)
        group_setup_csv = "unique_bending_setups.csv"
        group_setup_df.to_csv(ui_output_dir / group_setup_csv, index=False)

    manifest = {
        "methods": methods,
        "group_setup_csv": group_setup_csv,
    }
    with (ui_output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    logger.info("Exported Streamlit UI data to %s", ui_output_dir)


if __name__ == "__main__":

    project_root = Path(".")
    output_dir = project_root / "data" / "rf_augmented"
    n_new_samples = 6
    generation_modes = (
        "within-group-interpolation",
        "sensor-augmented",
    )
    '''
    Options for generation_modes:
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
        for feature_sampling_mode in generation_modes:
            logger.info(
                "Generating RF data | sensor mode=%s | feature source=%s",
                sensor_augmentation_mode,
                feature_sampling_mode,
            )
            RFDataGeneratorPipeline.run(
                project_root=project_root,
                n_new_samples=n_new_samples,
                output_dir=output_dir,
                feature_sampling_mode=feature_sampling_mode,
                include_all_features=include_all_features,
                sensor_augmentation_mode=sensor_augmentation_mode,
                regenerate_sensor_data=True,
                sensor_noise_snr_db=sensor_noise_snr_db,
            )

    export_streamlit_ui_data(project_root=project_root, output_dir=output_dir)
