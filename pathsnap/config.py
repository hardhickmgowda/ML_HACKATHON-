"""Config loading + reproducibility helpers."""
from __future__ import annotations
import logging, os, random
from pathlib import Path
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


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
