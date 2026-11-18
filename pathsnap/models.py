"""The five model families + sampling helpers (Debasditya)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = ["logreg", "knn", "rf", "hgb", "mlp"]
PRETTY = {"logreg": "Logistic Regression", "knn": "K-Nearest Neighbors", "rf": "Random Forest",
          "hgb": "Hist. Gradient Boosting", "mlp": "MLP (Neural Net)"}
FAMILY = {"logreg": "Linear / probabilistic", "knn": "Instance-based", "rf": "Bagging ensemble",
          "hgb": "Gradient boosting", "mlp": "Neural network"}


def make_model(name: str, seed: int = 42, params: dict | None = None):
    p = dict(params or {})
    if name == "logreg":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegression(C=p.get("C", 1.0), max_iter=1000, random_state=seed))])
    if name == "knn":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", KNeighborsClassifier(n_neighbors=p.get("n_neighbors", 40), weights="uniform", n_jobs=1))])
    if name == "rf":
        return RandomForestClassifier(n_estimators=p.get("n_estimators", 100), max_depth=p.get("max_depth", 14),
                                      min_samples_leaf=p.get("min_samples_leaf", 5), max_features=p.get("max_features", 0.5),
                                      n_jobs=1, random_state=seed)
    if name == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=p.get("max_iter", 300), learning_rate=p.get("learning_rate", 0.08),
            max_leaf_nodes=p.get("max_leaf_nodes", 31), min_samples_leaf=p.get("min_samples_leaf", 40),
            l2_regularization=p.get("l2_regularization", 1.0), max_depth=p.get("max_depth", None),
            early_stopping=False, random_state=seed)
    if name == "mlp":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", MLPClassifier(hidden_layer_sizes=tuple(p.get("hidden", (64, 32))),
                                               alpha=p.get("alpha", 1e-4), learning_rate_init=p.get("lr", 1e-3),
                                               max_iter=60, early_stopping=True, n_iter_no_change=6,
                                               random_state=seed))])
    raise ValueError(name)


def sample_points(df: pd.DataFrame, per_trip: int, seed: int, max_rows: int | None = None) -> pd.DataFrame:
    """Randomly keep <= per_trip labelled GPS points per trip (all candidate rows of a kept point stay together)."""
    rng = np.random.default_rng(seed)
    keep_keys = []
    for tid, g in df[df["labeled"]].groupby("trip_id", sort=False):
        pts = g["pt"].unique()
        if len(pts) > per_trip:
            pts = rng.choice(pts, per_trip, replace=False)
        keep_keys.append(pd.MultiIndex.from_arrays([np.full(len(pts), tid), pts]))
    keys = keep_keys[0].append(keep_keys[1:]) if len(keep_keys) > 1 else keep_keys[0]
    out = df[pd.MultiIndex.from_frame(df[["trip_id", "pt"]]).isin(keys)]
    if max_rows and len(out) > max_rows:                 # cap by dropping whole points
        frac = max_rows / len(out)
        k = pd.MultiIndex.from_frame(out[["trip_id", "pt"]].drop_duplicates())
        k = k[rng.random(len(k)) < frac]
        out = out[pd.MultiIndex.from_frame(out[["trip_id", "pt"]]).isin(k)]
    return out.reset_index(drop=True)
