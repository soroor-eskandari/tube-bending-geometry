# src/pipeline/preprocessing/run_preprocessor.py

import hashlib
import logging
import shutil
import urllib.request
from pathlib import Path

from src.pipeline.preprocessing.preprocessor import DataPreprocessPipeline


logger = logging.getLogger(__name__)


# Raw experiment data published with the tube bending dataset
# (https://github.com/zeyneddinoz/tubebend), stored there with Git LFS.
RAW_PKL_URL = (
    "https://media.githubusercontent.com/media/zeyneddinoz/tubebend"
    "/main/experiments_process_and_results.pkl"
)
RAW_PKL_SHA256 = "38fbd22241ead82c0cea33aee84d3c8bc3b1de24e61551f60416c8b59dba1ade"

# The bending setup groups are committed with the UI data export.
COMMITTED_BENDING_SETUPS_PATH = Path(
    "data/rf_augmented/ui_data/unique_bending_setups.csv"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def download_raw_pkl(target: Path) -> None:
    """Download the raw experiment pickle if it is not present yet."""
    if target.exists():
        logger.info("Raw data already present: %s", target)
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    print(f"Downloading raw data (~800 MB) from {RAW_PKL_URL}")

    with urllib.request.urlopen(RAW_PKL_URL) as response, partial.open("wb") as file:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        while chunk := response.read(1 << 20):
            file.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / total:6.1%}", end="", flush=True)
    print()

    if _sha256(partial) != RAW_PKL_SHA256:
        partial.unlink()
        raise RuntimeError(
            "Downloaded raw data does not match the expected checksum. "
            "Run this script again."
        )

    partial.rename(target)
    logger.info("Raw data saved to %s", target)


def ensure_bending_setups(target: Path) -> None:
    """Copy the committed bending setup groups into data/raw/ if missing."""
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(COMMITTED_BENDING_SETUPS_PATH, target)
    logger.info("Copied %s to %s", COMMITTED_BENDING_SETUPS_PATH, target)


if __name__ == "__main__":

    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    output_dir = data_dir / "processed"

    download_raw_pkl(raw_dir / "experiments_process_and_results.pkl")
    ensure_bending_setups(raw_dir / "unique_bending_setups.csv")

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
