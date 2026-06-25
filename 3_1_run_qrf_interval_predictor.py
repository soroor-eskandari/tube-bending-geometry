import logging
from pathlib import Path

from src.pipeline.ml.qrf.qrf_pipeline import (
    QRFPipeline,
    generated_geometry_sources,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    project_root = Path(".")
    use_mlflow = False
    geometry_sources = list(generated_geometry_sources())

    # ---------------------------------------
    # Run the QRF interval prediction workflow for every generated
    # geometry source: 3 feature sampling modes x 16 sensor modes.
    # ---------------------------------------
    for geometry_source in geometry_sources:
        logger.info("Running QRF interval prediction for %s", geometry_source)
        QRFPipeline.run(
            project_root=project_root,
            geometry_source=geometry_source,
            use_mlflow=use_mlflow,
        )
