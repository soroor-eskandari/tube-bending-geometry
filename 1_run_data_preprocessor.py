# src/pipeline/preprocessing/run_preprocessor.py

import logging
from pathlib import Path

from src.pipeline.preprocessing.preprocessor import DataPreprocessPipeline


logger = logging.getLogger(__name__)


if __name__ == "__main__":

    data_dir = Path("data")
    output_dir = data_dir / "processed"

    failed_experiment = [1, 48, 166]

    logger.info("Starting preprocessing run")

    DataPreprocessPipeline.run(
        failed_experiment=failed_experiment,
        output_dir=output_dir,
        eliminated_columns=None,
        normalized_tables=None,
        nan_handler=True,
    )

    logger.info("Preprocessing finished. CSVs saved to %s", output_dir)
