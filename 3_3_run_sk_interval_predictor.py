import logging
from pathlib import Path

from src.pipeline.ml.sk.sk_pipeline import SKPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    project_root = Path(".")

    # ---------------------------------------
    # Generate new samples using the trained SK model
    # ---------------------------------------
    SKPipeline.run(
        project_root=project_root,
    )
