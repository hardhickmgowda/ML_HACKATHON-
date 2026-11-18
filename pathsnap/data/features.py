"""Feature engineering for (GPS point, candidate arc) pairs.

The ML task is framed as binary classification: for each GPS observation and
each of its candidate road segments, predict whether that segment is the one
the vehicle was on. The classifier's probability later becomes the HMM
emission score.

Features use ONLY the GPS track and the road network - never the route.

Feature groups
--------------
Geometric fit     dist_m, dist_rank, dist_minus_min, dist_ratio, proj_t, n_cands
Direction         heading_diff_deg, heading_valid, is_two_way
Road structure    arc_len_m, out_deg_end, in_deg_start
Kinematics        speed_mps, dt_prev, is_segment_start, is_stationary
Local context     same_as_prev_nearest, connects_from_prev_nearest,
                  connects_to_next_nearest, fallback
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import angle_diff_deg
from .network import RoadNetwork

FEATURE_COLUMNS = [
    "dist_m", "dist_rank", "dist_minus_min", "dist_ratio", "proj_t", "n_cands",
    "heading_diff_deg", "heading_valid", "is_two_way",
    "arc_len_m", "out_deg_end", "in_deg_start",
    "speed_mps", "dt_prev", "is_segment_start", "is_stationary",
    "same_as_prev_nearest", "connects_from_prev_nearest", "connects_to_next_nearest",
    "fallback",
]


def build_features(points: pd.DataFrame, cands: pd.DataFrame, net: RoadNetwork) -> pd.DataFrame:
    df = cands.copy()
    p = df["point_pos"].to_numpy()
    a = df["arc_id"].to_numpy()

    # geometric fit relative to the other candidates of the same point
    grp = df.groupby("point_pos")["dist_m"]
    dmin = grp.transform("min")
    df["dist_minus_min"] = df["dist_m"] - dmin
    df["dist_ratio"] = df["dist_m"] / (dmin + 1.0)
    df["n_cands"] = grp.transform("size")

    # direction agreement between GPS motion and the directed arc
    heading = points["heading_deg"].to_numpy()[p]
    hvalid = ~np.isnan(heading)
    df["heading_diff_deg"] = np.where(hvalid, angle_diff_deg(heading, net.bearing[a]), 90.0)
    df["heading_valid"] = hvalid.astype(np.int8)
    df["is_two_way"] = net.is_two_way[a].astype(np.int8)

    # road structure
    df["arc_len_m"] = net.length[a]
    df["out_deg_end"] = net.out_deg[net.v[a]]
    df["in_deg_start"] = net.in_deg[net.u[a]]

    # kinematics of the observation
    df["speed_mps"] = points["speed_mps"].to_numpy()[p]
    df["dt_prev"] = points["dt_prev"].fillna(0.0).to_numpy()[p]
    df["is_segment_start"] = points["is_segment_start"].to_numpy()[p].astype(np.int8)
    df["is_stationary"] = points["is_stationary"].to_numpy()[p].astype(np.int8)

    # local context: relation to the neighbours' closest candidate (same segment only)
    n = len(points)
    nearest = np.full(n, -1, dtype=np.int64)
    top = df[df["dist_rank"] == 0]
    nearest[top["point_pos"].to_numpy()] = top["arc_id"].to_numpy()
    seg = points["segment_id"].to_numpy()

    prev_p = p - 1
    has_prev = (prev_p >= 0)
    has_prev[has_prev] &= seg[prev_p[has_prev]] == seg[p[has_prev]]
    prev_arc = np.where(has_prev, nearest[np.clip(prev_p, 0, n - 1)], -1)
    ok = prev_arc >= 0
    df["same_as_prev_nearest"] = (ok & (net.canon[np.where(ok, prev_arc, 0)] == net.canon[a])).astype(np.int8)
    df["connects_from_prev_nearest"] = (ok & (net.v[np.where(ok, prev_arc, 0)] == net.u[a])).astype(np.int8)

    next_p = p + 1
    has_next = next_p < n
    has_next[has_next] &= seg[next_p[has_next]] == seg[p[has_next]]
    next_arc = np.where(has_next, nearest[np.clip(next_p, 0, n - 1)], -1)
    ok = next_arc >= 0
    df["connects_to_next_nearest"] = (ok & (net.v[a] == net.u[np.where(ok, next_arc, 0)])).astype(np.int8)

    df["fallback"] = df["fallback"].astype(np.int8)
    df["canon_arc"] = net.canon[a]
    return df
