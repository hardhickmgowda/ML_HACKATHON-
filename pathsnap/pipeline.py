"""End-to-end inference pipeline: raw trip folder -> candidates -> classifier -> HMM/Viterbi -> route (Pranav/Hardhick).

The reference .route is NEVER used to produce a prediction. If present it is only read afterwards to score it.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .config import load_config, resolve
from .data_io import load_trip, Trip
from .dataset import prepare_trip, PreparedTrip
from .hmm import HMMParams, viterbi_decode, connected_route, snapped_xy
from .labels import align_to_route
from .metrics import continuity, dedupe, route_prf


def score_trip(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    return bundle["model"].predict_proba(df[bundle["features"]].to_numpy(np.float32))[:, 1]


def classifier_arcs(df: pd.DataFrame, prob: np.ndarray, T: int) -> np.ndarray:
    """argmax-score arc for every GPS point (-1 if no candidates)."""
    d = pd.DataFrame({"pt": df["pt"].to_numpy(), "arc": df["arc"].to_numpy(), "p": prob})
    top = d.sort_values(["pt", "p"], ascending=[True, False], kind="stable").drop_duplicates("pt")
    out = np.full(T, -1, np.int64)
    out[top["pt"].to_numpy()] = top["arc"].to_numpy()
    return out


def evaluate_against_reference(p: PreparedTrip, clf_arc, hmm_arc, clf_route_raw, hmm_route_raw, hmm_route, clf_route_filled, cfg):
    """Score predictions against the reference route (evaluation only)."""
    ref = p.trip.route
    if ref is None or len(ref) == 0:
        return None
    _, true_arc = align_to_route(p.net, p.ct.xy, ref, cfg)
    lab = true_arc >= 0
    n = int(lab.sum())
    m = dict(n_labeled_points=n, labeled_frac=float(lab.mean()))
    if n:
        m["top1_clf"] = float((clf_arc[lab] == true_arc[lab]).mean())
        m["top1_hmm"] = float((hmm_arc[lab] == true_arc[lab]).mean())
    ref_u = np.unique(ref)
    m["continuity_clf"] = continuity(clf_route_raw, p.net)
    m["continuity_hmm_raw"] = continuity(hmm_route_raw, p.net)
    m["continuity_hmm_filled"] = continuity(hmm_route, p.net)
    for tag, r in (("clf", clf_route_filled), ("hmm", hmm_route)):
        for k, v in route_prf(r, ref_u, p.net).items():
            m[f"{k}_{tag}"] = v
    for k, v in route_prf(clf_route_raw, ref_u, p.net).items():
        m[f"{k}_clf_raw"] = v
    return m


def run_decoding(p: PreparedTrip, prob: np.ndarray, params: HMMParams, cfg: dict) -> dict:
    T = p.ct.n
    clf_arc = classifier_arcs(p.df, prob, T)
    t0 = time.time()
    res = viterbi_decode(p.net, p.ct.xy, p.ct.ts, p.df, prob, params)
    t_hmm = time.time() - t0
    hmm_route, breaks = connected_route(p.net, res.arc, params.fill_limit_m)
    clf_route_filled, _ = connected_route(p.net, clf_arc, params.fill_limit_m)
    clf_route_raw = dedupe(clf_arc[clf_arc >= 0])
    hmm_route_raw = dedupe(res.arc[res.arc >= 0])
    return dict(clf_arc=clf_arc, hmm=res, hmm_route=hmm_route, hmm_route_raw=hmm_route_raw, clf_route_raw=clf_route_raw,
                clf_route_filled=clf_route_filled, unbridged=breaks, t_hmm=t_hmm)


class PathSnapPipeline:
    def __init__(self, cfg: dict | None = None, model_path: str | None = None, hmm_path: str | None = None):
        self.cfg = cfg or load_config()
        art = resolve(self.cfg["paths"]["artifacts"])
        mp = Path(model_path) if model_path else (art / "model.joblib")
        if not mp.exists():
            raise FileNotFoundError(f"model not found: {mp}. Run scripts/03_tune_best.py first.")
        self.bundle = joblib.load(mp)
        hp = Path(hmm_path) if hmm_path else (art / "hmm_params.json")
        self.params = HMMParams.from_dict({**self.cfg["hmm"], **(json.load(open(hp)) if hp.exists() else {})})

    def run(self, folder: str | Path, trip_id: str | None = None) -> dict:
        t_all = time.time()
        trip: Trip = load_trip(folder, trip_id)
        t0 = time.time(); p = prepare_trip(trip, self.cfg, use_route=False); t_prep = time.time() - t0
        t0 = time.time(); prob = score_trip(self.bundle, p.df); t_clf = time.time() - t0
        d = run_decoding(p, prob, self.params, self.cfg)
        metrics = evaluate_against_reference(p, d["clf_arc"], d["hmm"].arc, d["clf_route_raw"], d["hmm_route_raw"],
                                             d["hmm_route"], d["clf_route_filled"], self.cfg)
        net, ct = p.net, p.ct
        lon, lat = net.proj.inv(ct.xy_raw[:, 0], ct.xy_raw[:, 1])
        sp = snapped_xy(net, d["hmm"].arc, d["hmm"].frac)
        slon, slat = net.proj.inv(np.nan_to_num(sp[:, 0]), np.nan_to_num(sp[:, 1]))
        ok = d["hmm"].arc >= 0
        return dict(
            trip_id=trip.trip_id, n_points_raw=int(ct.n_raw), n_points=int(ct.n), n_arcs=int(net.n_arcs),
            track=np.c_[lat, lon].round(6).tolist(),
            matched=[dict(i=int(i), arc=int(a), lat=round(float(la), 6), lon=round(float(lo), 6))
                     for i, (a, la, lo, k) in enumerate(zip(d["hmm"].arc, slat, slon, ok)) if k],
            route_arcs=d["hmm_route"].tolist(), route=net.arc_polyline_ll(d["hmm_route"]),
            classifier_route=net.arc_polyline_ll(d["clf_route_filled"]),
            reference_route=net.arc_polyline_ll(trip.route) if trip.route is not None else None,
            hmm=dict(restarts=d["hmm"].restarts, teleports=d["hmm"].teleports, unbridged_breaks=int(d["unbridged"])),
            metrics=metrics,
            timing=dict(prepare_s=round(t_prep, 3), classify_s=round(t_clf, 3), hmm_s=round(d["t_hmm"], 3),
                        total_s=round(time.time() - t_all, 3)),
            _prepared=p, _decoded=d, _prob=prob,
        )
