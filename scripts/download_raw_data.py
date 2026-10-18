"""Download the raw thesis input data into data/raw/.

The raw files are too large for git, so they are attached to a GitHub
release instead. Run from the project root:

    uv run python scripts/download_raw_data.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

RELEASE_URL = (
    "https://github.com/soroor-eskandari/tube-bending-geometry"
    "/releases/download/data-v1"
)

RAW_FILES = (
    "experiments_process_and_results.pkl",
    "unique_bending_setups.csv",
)

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def download(file_name: str) -> None:
    target = RAW_DIR / file_name

    if target.exists():
        print(f"Already present, skipping: {target}")
        return

    url = f"{RELEASE_URL}/{file_name}"
    partial = target.with_suffix(target.suffix + ".part")
    print(f"Downloading {url}")

    with urllib.request.urlopen(url) as response, partial.open("wb") as file:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        while chunk := response.read(1 << 20):
            file.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / total:6.1%}", end="", flush=True)

    print()
    partial.rename(target)


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    try:
        for name in RAW_FILES:
            download(name)
    except Exception as error:
        sys.exit(f"Download failed: {error}")
    print(f"Raw data ready in {RAW_DIR}")
