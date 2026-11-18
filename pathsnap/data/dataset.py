"""Model-ready data access for the modelling team.

    from pathsnap.data.dataset import load_split, cv_folds
    tr = load_split("train")            # labelled, reliable rows only
    for fold, (fit_idx, val_idx) in enumerate(cv_folds(tr)):
        model.fit(tr.X.iloc[fit_idx], tr.y.iloc[fit_idx])
        ...

Rules baked in:
* only rows with ``label_valid`` are returned for fitting/scoring (unless
  ``reliable_only=False``);
* folds are grouped by ``trip_id`` (no trip is split across folds);
* fitted preprocessing (e.g. scalers) must be fit inside each fold / on
  the train partition only - this module returns raw features.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .. import config
from .features import FEATURE_COLUMNS

RANDOM_SEED = 42


@dataclass
class Split:
    name: str
    frame: pd.DataFrame        # full candidate rows (ids + features + label)

    @property
    def X(self) -> pd.DataFrame:
        return self.frame[FEATURE_COLUMNS]

    @property
    def y(self) -> pd.Series:
        return self.frame["label"]

    @property
    def groups(self) -> pd.Series:
        return self.frame["trip_id"]

    @property
    def point_key(self) -> pd.Series:
        """Unique observation id (trip_id, point_pos) -> used for Top-1 metrics."""
        return self.frame["trip_id"] + ":" + self.frame["point_pos"].astype(str)

    def __len__(self) -> int:
        return len(self.frame)


def read_processed(partition: str, table: str) -> pd.DataFrame:
    """Read data_processed/<partition>/<table>.csv. trip_id is kept as a
    zero-padded string ('00000001') so it still matches the data folders."""
    path = config.PROCESSED_DIR / partition / f"{table}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python scripts/build_dataset.py` first")
    return pd.read_csv(path, dtype={"trip_id": str})


def load_split(partition: str, reliable_only: bool = True) -> Split:
    df = read_processed(partition, "candidates")
    if reliable_only and "label_valid" in df:
        df = df[df["label_valid"]]
    df = df.sort_values(["trip_id", "point_pos", "dist_rank"]).reset_index(drop=True)
    return Split(partition, df)


def cv_folds(split: Split, n_splits: int = 5, seed: int = RANDOM_SEED):
    """Group-aware folds over trips. Trips are shuffled with a fixed seed so
    folds are reproducible but not ordered by trip id."""
    trips = split.groups.unique()
    rng = np.random.default_rng(seed)
    order = {t: i for i, t in enumerate(rng.permutation(trips))}
    g = split.groups.map(order).to_numpy()
    return list(GroupKFold(n_splits=n_splits).split(split.X, split.y, groups=g))
