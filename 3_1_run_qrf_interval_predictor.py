import logging
from pathlib import Path

from src.pipeline.ml.qrf.qrf_pipeline import QRFPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    project_root = Path(".")

    # ---------------------------------------
    # Generate new samples using the trained RF model
    # ---------------------------------------
    QRFPipeline.run(
        project_root=project_root,
    )
