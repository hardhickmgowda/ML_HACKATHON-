"""Shared evaluation metrics so every model is scored identically.

Two levels:

* candidate level (binary classifier):  accuracy, precision, recall, macro F1,
  log loss, confusion matrix - computed on (point, candidate) rows.
* observation level (map-matching):     Top-1 accuracy, candidate miss rate,
  route continuity - computed per GPS point / per trip.

``scores`` are the model's positive-class scores/probabilities, aligned with
the rows of the candidate frame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)


def candidate_metrics(y_true, scores, threshold: float = 0.5, is_proba: bool = True) -> dict:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores >= threshold).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    out["log_loss"] = (log_loss(y_true, np.clip(scores, 1e-7, 1 - 1e-7), labels=[0, 1])
                       if is_proba else np.nan)
    return out


def top1_predictions(frame: pd.DataFrame, scores) -> pd.DataFrame:
    """Per observation, the candidate with the highest score (ties -> nearest)."""
    df = frame[["trip_id", "point_pos", "arc_id", "canon_arc", "dist_rank"]].copy()
    df["score"] = np.asarray(scores, dtype=float)
    if "label" in frame:
        df["label"] = frame["label"].to_numpy()
    df = df.sort_values(["trip_id", "point_pos", "score", "dist_rank"],
                        ascending=[True, True, False, True], kind="stable")
    return df.groupby(["trip_id", "point_pos"], sort=False).head(1).reset_index(drop=True)


def top1_accuracy(frame: pd.DataFrame, scores) -> float:
    """Fraction of observations whose top-scored candidate is the reference arc.
    Observations whose true arc is not among the candidates count as wrong."""
    return float(top1_predictions(frame, scores)["label"].mean())


def candidate_miss_rate(frame: pd.DataFrame) -> float:
    """Fraction of observations whose reference arc is absent from the candidate set."""
    return float(1 - frame.groupby(["trip_id", "point_pos"])["label"].max().mean())


def route_continuity(pred: pd.DataFrame, u: dict, v: dict) -> float:
    """Share of consecutive predicted-arc changes that are physically connected
    (same arc, or previous arc's end node == next arc's start node).

    ``pred``: output of ``top1_predictions`` (or HMM output) with trip_id,
    point_pos, arc_id. ``u``/``v``: {trip_id: array of arc start/end nodes}.
    """
    ok = total = 0
    for tid, g in pred.sort_values(["trip_id", "point_pos"]).groupby("trip_id"):
        a = g["arc_id"].to_numpy()
        prev, nxt = a[:-1], a[1:]
        change = prev != nxt
        total += int(change.sum())
        ok += int((v[tid][prev[change]] == u[tid][nxt[change]]).sum())
    return ok / total if total else 1.0


def network_endpoints(trip_ids, partition: str) -> tuple[dict, dict]:
    """Load arc start/end node arrays for route_continuity()."""
    from ..data.io import load_trip
    u, v = {}, {}
    for tid in trip_ids:
        arcs = load_trip(partition, tid, with_route=False).arcs
        u[tid], v[tid] = arcs["u"].to_numpy(), arcs["v"].to_numpy()
    return u, v


def evaluate(frame: pd.DataFrame, scores, partition: str | None = None,
             is_proba: bool = True, endpoints: tuple[dict, dict] | None = None) -> dict:
    """All metrics required by the report's comparison table in one call."""
    res = candidate_metrics(frame["label"], scores, is_proba=is_proba)
    res["top1_accuracy"] = top1_accuracy(frame, scores)
    res["candidate_miss_rate"] = candidate_miss_rate(frame)
    if endpoints is None and partition is not None:
        endpoints = network_endpoints(frame["trip_id"].unique(), partition)
    if endpoints is not None:
        res["route_continuity"] = route_continuity(top1_predictions(frame, scores), *endpoints)
    return res
