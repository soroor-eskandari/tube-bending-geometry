import logging
from pathlib import Path

from src.pipeline.ml.hgp.hgp_pipeline import HGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    project_root = Path(".")
    geometry_source = "sampled"
    # options:
    # real
    # augmented_real
    # sampled

    # ---------------------------------------
    # Generate new samples using the trained HGP model
    # ---------------------------------------
    HGPipeline.run(
        project_root=project_root,
        geometry_source=geometry_source,
    )
