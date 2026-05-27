import logging
from pathlib import Path

from src.pipeline.ml.qrf.qrf_pipeline import QRFPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    project_root = Path(".")

    # ---------------------------------------
    # Run the QRF interval prediction workflow.
    # ---------------------------------------
    QRFPipeline.run(
        project_root=project_root,
    )
