"""Data-validation checks.

Two layers:

* ``validate_raw_trip``      - checks on the raw files of one trip
* ``validate_processed``     - checks on the processed tables before they are
                               handed to model training

Each check yields an ``Issue`` with a severity:
    ERROR   - data cannot be used as-is (pipeline should stop)
    WARNING - usable, but needs special handling / should be reported
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .. import config
from .features import FEATURE_COLUMNS
from .geo import LocalProjection
from .io import Trip


@dataclass
class Issue:
    trip_id: str
    severity: str
    check: str
    detail: str


def _route_points(trip: Trip, proj: LocalProjection, step: float = 5.0):
    nodes = trip.nodes
    u = trip.arcs["u"].to_numpy()[trip.route]
    v = trip.arcs["v"].to_numpy()[trip.route]
    ax, ay = proj.to_xy(nodes["lon"].to_numpy()[u], nodes["lat"].to_numpy()[u])
    bx, by = proj.to_xy(nodes["lon"].to_numpy()[v], nodes["lat"].to_numpy()[v])
    L = np.hypot(bx - ax, by - ay)
    k = np.maximum(np.ceil(L / step).astype(int), 1) + 1
    starts = np.repeat(np.cumsum(k) - k, k)
    s = (np.arange(k.sum()) - starts) / np.repeat(k - 1, k)
    rep = lambda arr: np.repeat(arr, k)
    return np.c_[rep(ax) + s * rep(bx - ax), rep(ay) + s * rep(by - ay)], L.sum()


def validate_raw_trip(trip: Trip) -> tuple[dict, list[Issue]]:
    tid = trip.trip_id
    issues: list[Issue] = []
    add = lambda sev, chk, det: issues.append(Issue(tid, sev, chk, det))
    tr, nd, ar = trip.track, trip.nodes, trip.arcs
    m: dict = {"trip_id": tid, "partition": trip.partition,
               "n_points": len(tr), "n_nodes": len(nd), "n_arcs": len(ar),
               "route_len": None if trip.route is None else len(trip.route)}

    # --- structural / type checks
    for name, df in (("track", tr), ("nodes", nd), ("arcs", ar)):
        n_nan = int(df.isna().sum().sum())
        m[f"nan_{name}"] = n_nan
        if n_nan:
            add("ERROR", f"{name}_nan", f"{n_nan} missing / non-numeric values")
    if len(tr) < 2:
        add("ERROR", "track_too_short", f"{len(tr)} points")
        return m, issues

    bad_ll = (~tr["lon"].between(-180, 180) | ~tr["lat"].between(-90, 90)).sum()
    if bad_ll:
        add("ERROR", "track_lonlat_range", f"{bad_ll} points outside valid lon/lat")
    # coordinate-order sanity: GPS must overlap the road network's extent
    if not (nd["lon"].min() <= tr["lon"].median() <= nd["lon"].max()
            and nd["lat"].min() <= tr["lat"].median() <= nd["lat"].max()):
        add("ERROR", "track_outside_network", "GPS median not within node bbox (swapped lon/lat?)")

    # --- time
    dt = np.diff(tr["t"].to_numpy())
    m.update({"dt_min": dt.min(), "dt_max": dt.max(), "frac_dt_1s": float(np.mean(dt == 1)),
              "duration_s": float(tr["t"].iloc[-1] - tr["t"].iloc[0])})
    if (dt <= 0).any():
        add("ERROR", "time_not_increasing", f"{int((dt <= 0).sum())} non-increasing steps")
    n_gaps = int((dt > config.GAP_SECONDS).sum())
    m["n_time_gaps"] = n_gaps
    if n_gaps:
        add("WARNING", "time_gaps", f"{n_gaps} gaps > {config.GAP_SECONDS:.0f}s (max {dt.max():.0f}s)")

    # --- kinematics
    proj = LocalProjection.from_points(tr["lon"].to_numpy(), tr["lat"].to_numpy())
    x, y = proj.to_xy(tr["lon"].to_numpy(), tr["lat"].to_numpy())
    step = np.hypot(np.diff(x), np.diff(y))
    v = step / np.where(dt > 0, dt, np.nan)
    m["track_length_km"] = float(step.sum() / 1000)
    m["speed_p99_mps"] = float(np.nanpercentile(v, 99))
    n_fast = int(np.nansum(v > config.MAX_SPEED_MPS))
    m["n_speed_violations"] = n_fast
    if n_fast:
        add("WARNING", "speed_spikes", f"{n_fast} steps faster than {config.MAX_SPEED_MPS} m/s")
    n_dup = int(tr.duplicated(["lon", "lat"]).sum())
    m["n_duplicate_positions"] = n_dup
    m["extent_km"] = float(max(np.ptp(x), np.ptp(y)) / 1000)

    # --- network
    u, w = ar["u"].to_numpy(), ar["v"].to_numpy()
    bad_ids = int(((u < 0) | (u >= len(nd)) | (w < 0) | (w >= len(nd))).sum())
    if bad_ids:
        add("ERROR", "arc_node_ids", f"{bad_ids} arcs reference missing nodes")
        return m, issues
    m["n_self_loops"] = int((u == w).sum())
    m["n_duplicate_arcs"] = int(ar.duplicated().sum())
    if m["n_duplicate_arcs"]:
        add("WARNING", "duplicate_arcs", f"{m['n_duplicate_arcs']} duplicate (u,v) arcs")
    used = np.zeros(len(nd), bool)
    used[u] = used[w] = True
    m["n_isolated_nodes"] = int((~used).sum())
    g = coo_matrix((np.ones(len(u)), (u, w)), shape=(len(nd), len(nd)))
    n_wcc, lab = connected_components(g, directed=True, connection="weak")
    n_scc, slab = connected_components(g, directed=True, connection="strong")
    m["n_weak_components"] = int(n_wcc)
    m["largest_wcc_frac"] = float(np.bincount(lab).max() / len(nd))
    m["largest_scc_frac"] = float(np.bincount(slab).max() / len(nd))

    # --- reference route
    if trip.route is not None:
        r = trip.route
        if ((r < 0) | (r >= len(ar))).any():
            add("ERROR", "route_arc_ids", "route references arcs that do not exist")
            return m, issues
        cont = w[r[:-1]] == u[r[1:]]
        m["route_continuity"] = float(cont.mean()) if len(cont) else 1.0
        if not cont.all():
            add("WARNING", "route_discontinuous", f"{int((~cont).sum())} breaks between consecutive arcs")
        rpts, rlen = _route_points(trip, proj)
        d, _ = cKDTree(rpts).query(np.c_[x, y])
        m["route_length_km"] = float(rlen / 1000)
        m["gps_to_route_median_m"] = float(np.median(d))
        m["gps_to_route_p95_m"] = float(np.percentile(d, 95))
        m["frac_points_far_from_route"] = float(np.mean(d > config.LABEL_MAX_DIST_M))
        m["track_route_length_ratio"] = m["track_length_km"] / max(m["route_length_km"], 1e-9)
        if m["frac_points_far_from_route"] > 0.05:
            add("WARNING", "route_partial_coverage",
                f"{m['frac_points_far_from_route']:.0%} of GPS points >{config.LABEL_MAX_DIST_M:.0f} m "
                f"from the reference route (track {m['track_length_km']:.1f} km vs route "
                f"{m['route_length_km']:.1f} km)")
    return m, issues


def validate_processed(points: pd.DataFrame, cands: pd.DataFrame) -> list[Issue]:
    """Checks that the model-ready tables are consistent."""
    issues: list[Issue] = []
    add = lambda sev, chk, det, tid="ALL": issues.append(Issue(tid, sev, chk, det))

    missing_cols = [c for c in FEATURE_COLUMNS if c not in cands.columns]
    if missing_cols:
        add("ERROR", "feature_columns", f"missing {missing_cols}")
        return issues
    n_nan = cands[FEATURE_COLUMNS].isna().sum()
    if n_nan.any():
        add("ERROR", "feature_nan", n_nan[n_nan > 0].to_dict().__repr__())
    if not np.isfinite(cands[FEATURE_COLUMNS].to_numpy(dtype=float)).all():
        add("ERROR", "feature_inf", "non-finite feature values")

    per_trip_pts = points.groupby("trip_id").size()
    per_trip_cpts = cands.groupby("trip_id")["point_pos"].nunique()
    no_cand = (per_trip_pts - per_trip_cpts.reindex(per_trip_pts.index).fillna(0)).astype(int)
    for tid, k in no_cand[no_cand > 0].items():
        add("ERROR", "point_without_candidates", f"{k} points", tid)

    if cands.duplicated(["trip_id", "point_pos", "canon_arc"]).any():
        add("ERROR", "duplicate_candidates", "same canonical arc twice for a point")

    if "label" in cands.columns:
        pos_per_point = cands.groupby(["trip_id", "point_pos"])["label"].sum()
        if (pos_per_point > 1).any():
            add("ERROR", "multiple_positives", f"{int((pos_per_point > 1).sum())} points")
        if "label_valid" in points.columns:
            vf = points.groupby("trip_id")["label_valid"].mean()
            for tid, f in vf[vf < 0.9].items():
                add("WARNING", "low_label_validity", f"only {f:.0%} of points have a reliable label", tid)
    return issues


def issues_frame(issues: list[Issue]) -> pd.DataFrame:
    return pd.DataFrame([asdict(i) for i in issues], columns=["trip_id", "severity", "check", "detail"])
