"""Central configuration for PathSnap.

* constants below: data-pipeline thresholds (pathsnap.data.*)
* load_config(): reads config.yaml for the model / HMM / API code

Every threshold used during cleaning, candidate generation and labelling lives
here so that the exact same values are applied to training, student-test and
(at demo time) evaluator trips. Values were chosen from the EDA on the
*training* partition only (see docs/DATA.md).
"""
from __future__ import annotations

import os
import logging
import random
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# Raw data location. Override with the PATHSNAP_DATA environment variable.
DATA_ROOT = Path(os.environ.get("PATHSNAP_DATA", REPO_ROOT / "Student folder"))
PROCESSED_DIR = REPO_ROOT / "data_processed"
RESULTS_DIR = REPO_ROOT / "results"

PARTITIONS = ("train", "student_test")

# Mean Earth radius (IUGG), metres.
EARTH_RADIUS_M = 6_371_008.8

# --- Cleaning -----------------------------------------------------------------
# A time step larger than this starts a new trajectory segment (GPS dropout).
GAP_SECONDS = 10.0
# A spatial jump larger than this also starts a new segment.
GAP_METRES = 300.0
# Speeds above this (m/s, ~250 km/h) are physically implausible for road vehicles.
MAX_SPEED_MPS = 70.0
# Points whose step from the previous point is below this are "stationary".
STATIONARY_METRES = 0.5

# --- Candidate generation -----------------------------------------------------
# Road segments within this distance of a GPS point become candidates.
CANDIDATE_RADIUS_M = 60.0
# Keep at most this many candidates per GPS point (closest first).
MAX_CANDIDATES = 8
# Arc geometry is densified at this spacing for the KD-tree index.
DENSIFY_STEP_M = 10.0
# Half-window (points) used to estimate a stable GPS heading.
HEADING_HALF_WINDOW = 2

# --- Labelling (uses the reference route; never used at inference time) --------
# Points whose aligned reference arc is farther than this get label_valid=False.
LABEL_MAX_DIST_M = 50.0


# --- config.yaml loader + reproducibility helpers (models / HMM / API) ---------


def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path) if path else REPO_ROOT / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(REPO_ROOT)
    return cfg


def resolve(*parts: str) -> Path:
    """Resolve a path relative to the repo root (absolute paths pass through)."""
    p = Path(*parts)
    return p if p.is_absolute() else REPO_ROOT / p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(name: str = "pathsnap") -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
    return log
