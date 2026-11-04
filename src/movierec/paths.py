"""Locations of datasets and experiment outputs, overridable through the environment."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return Path(os.environ.get("MOVIEREC_DATA_DIR", REPO_ROOT / "data"))


def results_dir() -> Path:
    return Path(os.environ.get("MOVIEREC_RESULTS_DIR", REPO_ROOT / "results"))
