import logging
from pathlib import Path

from src.pipeline.ml.hgp.hgp_pipeline import HGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    project_root = Path(".")

    # ---------------------------------------
    # Generate new samples using the trained RF model
    # ---------------------------------------
    HGPipeline.run(
        project_root=project_root,
    )
