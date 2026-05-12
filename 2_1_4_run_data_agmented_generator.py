import logging
from pathlib import Path

from src.pipeline.rf_augmentation.rf_data_generator_pipeline import RFDataGeneratorPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    project_root = Path(".")
    output_dir = project_root / "data" / "rf_augmented"

    # ---------------------------------------
    # Generate new samples using the trained RF model
    # ---------------------------------------
    RFDataGeneratorPipeline.run(
        project_root=project_root,
        n_new_samples=10,
        output_dir=output_dir,
    )

