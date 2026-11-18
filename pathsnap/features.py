"""Per GPS-candidate feature extraction (Harsh)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .geo import point_segment, wrap_angle
from .network import RoadNetwork

FEATURES = [
    "dist", "dist_rel", "dist_rank", "n_cands", "frac", "arc_len",
    "hdiff", "cos_h", "hdiff_long", "cos_h_long", "heading_valid", "speed",
    "lat_off", "d_prev", "d_next", "along_move", "out_deg_end", "in_deg_start", "has_reverse",
]


def _heading(xy, w, min_disp):
    T = len(xy)
    i = np.arange(T)
    j0, j1 = np.maximum(i - w, 0), np.minimum(i + w, T - 1)
    v = xy[j1] - xy[j0]
    n = np.hypot(v[:, 0], v[:, 1])
    return np.arctan2(v[:, 1], v[:, 0]), n >= min_disp


def build_features(net: RoadNetwork, xy: np.ndarray, ts: np.ndarray, cand, cfg: dict) -> pd.DataFrame:
    pt, arc, dist, frac = cand
    T = len(xy)
    md = cfg["preprocess"]["heading_min_disp_m"]
    h1, v1 = _heading(xy, 1, md)
    h3, v3 = _heading(xy, 3, md)
    i = np.arange(T)
    j0, j1 = np.maximum(i - 1, 0), np.minimum(i + 1, T - 1)
    dtm = np.maximum(ts[j1] - ts[j0], 1e-3)
    speed = np.hypot(*(xy[j1] - xy[j0]).T) / dtm
    bear = net.arc_bearing[arc]
    hd = np.abs(wrap_angle(h1[pt] - bear))
    hdl = np.abs(wrap_angle(h3[pt] - bear))
    hd = np.where(v1[pt], hd, np.pi / 2)
    hdl = np.where(v3[pt], hdl, np.pi / 2)
    # signed lateral offset (left of arc direction positive)
    ax, ay, bx, by = net.arc_a[arc, 0], net.arc_a[arc, 1], net.arc_b[arc, 0], net.arc_b[arc, 1]
    L = np.maximum(net.arc_len[arc], 1e-6)
    dx, dy = (bx - ax) / L, (by - ay) / L
    qx, qy = ax + frac * (bx - ax), ay + frac * (by - ay)
    lat_off = dx * (xy[pt, 1] - qy) - dy * (xy[pt, 0] - qx)
    # neighbour consistency: distance & along-arc position of points i-2 / i+2 w.r.t. same arc
    ip, inx = np.maximum(pt - 2, 0), np.minimum(pt + 2, T - 1)
    d_prev, t_prev, _, _ = point_segment(xy[ip, 0], xy[ip, 1], ax, ay, bx, by)
    d_next, t_next, _, _ = point_segment(xy[inx, 0], xy[inx, 1], ax, ay, bx, by)
    df = pd.DataFrame({
        "pt": pt.astype(np.int32), "arc": arc.astype(np.int64),
        "dist": dist, "frac": frac, "arc_len": net.arc_len[arc],
        "hdiff": hd, "cos_h": np.cos(hd), "hdiff_long": hdl, "cos_h_long": np.cos(hdl),
        "heading_valid": v3[pt].astype(np.float32), "speed": speed[pt],
        "lat_off": lat_off, "d_prev": d_prev, "d_next": d_next,
        "along_move": (t_next - t_prev) * net.arc_len[arc],
        "out_deg_end": net.out_deg[net.arc_to[arc]], "in_deg_start": net.in_deg[net.arc_from[arc]],
        "has_reverse": (net.reverse[arc] >= 0).astype(np.float32),
    })
    g = df.groupby("pt", sort=False)["dist"]
    df["dist_rank"] = g.cumcount().astype(np.float32)
    df["dist_rel"] = df["dist"] - g.transform("min")
    df["n_cands"] = g.transform("size").astype(np.float32)
    return df
