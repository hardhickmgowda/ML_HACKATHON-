"""Evaluation metrics: classifier-level, point-level and route-level (Debasditya / Hardhick)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score, roc_auc_score


def top1_table(df: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    """Per labelled GPS point: predicted arc (argmax score), true arc present among candidates?, correct?"""
    d = df[["trip_id", "pt", "arc", "y"]].copy()
    d["s"] = score
    d = d.sort_values(["trip_id", "pt", "s"], ascending=[True, True, False], kind="stable")
    top = d.groupby(["trip_id", "pt"], sort=False).head(1).set_index(["trip_id", "pt"])
    has_true = d.groupby(["trip_id", "pt"], sort=False)["y"].max()
    top["has_true"] = has_true.reindex(top.index).to_numpy()
    top["correct"] = top["y"] == 1
    return top


def classifier_metrics(df: pd.DataFrame, p: np.ndarray, nearest_baseline: bool = False) -> dict:
    """df must contain only labelled points. p = P(candidate is the true arc)."""
    y = df["y"].to_numpy()
    p = np.clip(p, 1e-6, 1 - 1e-6)
    top = top1_table(df, p)
    pred_top = np.zeros(len(df), bool)
    key = pd.MultiIndex.from_frame(df[["trip_id", "pt", "arc"]])
    pred_top = key.isin(pd.MultiIndex.from_frame(top.reset_index()[["trip_id", "pt", "arc"]])).astype(int)
    thr = (p >= 0.5).astype(int)
    out = dict(
        precision=precision_score(y, thr, zero_division=0), recall=recall_score(y, thr, zero_division=0),
        f1=f1_score(y, thr, zero_division=0), macro_f1=f1_score(y, pred_top, average="macro"),
        roc_auc=roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan,
        log_loss=log_loss(y, p, labels=[0, 1]), top1_acc=float(top["correct"].mean()),
        miss_rate=float(1 - top["has_true"].mean()), n_points=int(len(top)), n_rows=int(len(df)))
    return out


def continuity(seq, net) -> float:
    """Fraction of consecutive distinct-arc pairs that are physically connected (end(a) == start(b))."""
    seq = np.asarray(seq)
    if len(seq) < 2:
        return 1.0
    return float((net.arc_to[seq[:-1]] == net.arc_from[seq[1:]]).mean())


def dedupe(seq) -> np.ndarray:
    seq = np.asarray(seq)
    if len(seq) == 0:
        return seq
    return seq[np.r_[True, seq[1:] != seq[:-1]]]


def route_prf(pred_arcs, ref_arcs, net=None) -> dict:
    """Arc-set precision / recall / F1 of a reconstructed route vs the reference (length-weighted if net given)."""
    P, R = set(map(int, pred_arcs)), set(map(int, ref_arcs))
    if not P or not R:
        return dict(route_precision=0.0, route_recall=0.0, route_f1=0.0)
    inter = P & R
    w = (lambda s: float(sum(net.arc_len[a] for a in s))) if net is not None else (lambda s: float(len(s)))
    pr, rc = w(inter) / max(w(P), 1e-9), w(inter) / max(w(R), 1e-9)
    return dict(route_precision=pr, route_recall=rc, route_f1=2 * pr * rc / max(pr + rc, 1e-9))
