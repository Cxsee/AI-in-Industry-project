"""Locations of datasets and experiment outputs, overridable through the environment."""

import os
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return Path(os.environ.get("MOVIEREC_DATA_DIR", REPO_ROOT / "data"))


def results_dir() -> Path:
    return Path(os.environ.get("MOVIEREC_RESULTS_DIR", REPO_ROOT / "results"))


def replace_file(
    source: Path, destination: Path, *, attempts: int = 20, delay: float = 0.1
) -> None:
    """Atomically replace a file, tolerating brief Windows reader locks.

    Persistent permission failures still propagate. Neither file is deleted to
    work around a lock, so readers always see a complete old or new artifact.
    """
    if attempts < 1 or delay < 0:
        raise ValueError("attempts must be positive and delay non-negative.")
    for attempt in range(attempts):
        try:
            source.replace(destination)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(delay)
