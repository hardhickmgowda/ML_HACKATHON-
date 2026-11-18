"""Map reference .route arcs onto GPS points -> training labels (Harsh).

Monotone dynamic-programming alignment: each GPS point is assigned to one position of the reference route;
positions may only stay or move forward (routes are traversed in order). Points farther than labels.radius_m
from every route arc are left unlabeled (-1) instead of being forced onto a wrong arc.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from .geo import point_segment
from .network import RoadNetwork


def align_to_route(net: RoadNetwork, xy: np.ndarray, route: np.ndarray, cfg: dict):
    lc = cfg["labels"]
    R, T = lc["radius_m"], len(xy)
    true_pos = np.full(T, -1, np.int64)
    if route is None or len(route) == 0:
        return true_pos, np.full(T, -1, np.int64)
    route = np.asarray(route, np.int64)
    route = route[(route >= 0) & (route < net.n_arcs)]
    L = net.arc_len[route]
    n_s = np.maximum(np.ceil(L / 5.0).astype(np.int64) + 1, 2)
    pos = np.repeat(np.arange(len(route)), n_s)
    off = np.cumsum(n_s) - n_s
    frac = (np.arange(len(pos)) - np.repeat(off, n_s)) / (np.repeat(n_s, n_s) - 1)
    a, b = net.arc_a[route[pos]], net.arc_b[route[pos]]
    tree = cKDTree(a + frac[:, None] * (b - a))
    hits = tree.query_ball_point(xy, r=R, return_sorted=False)
    P, J = [], []
    for i, h in enumerate(hits):
        if h:
            u = np.unique(pos[np.asarray(h)])
            P.append(np.full(len(u), i)); J.append(u)
    if not P:
        return true_pos, np.full(T, -1, np.int64)
    P, J = np.concatenate(P), np.concatenate(J)
    arcs = route[J]
    d, _, _, _ = point_segment(xy[P, 0], xy[P, 1], net.arc_a[arcs, 0], net.arc_a[arcs, 1], net.arc_b[arcs, 0], net.arc_b[arcs, 1])
    ok = d <= R
    P, J, d = P[ok], J[ok], d[ok]
    if len(P) == 0:
        return true_pos, np.full(T, -1, np.int64)
    bounds = np.searchsorted(P, np.arange(T + 1))
    cost_all = 0.5 * (d / 10.0) ** 2
    steps = [i for i in range(T) if bounds[i + 1] > bounds[i]]
    back, prevJ, prev_cost = [], None, None
    for i in steps:
        sl = slice(bounds[i], bounds[i + 1])
        Jc, cc = J[sl], cost_all[sl]
        if prevJ is None:
            tot, bp = cc.copy(), np.full(len(Jc), -1)
        else:
            delta = Jc[None, :] - prevJ[:, None]
            M = np.where(delta >= 0, lc["skip_penalty"] * delta, np.inf) + prev_cost[:, None]
            bp = np.argmin(M, axis=0)
            best = M[bp, np.arange(len(Jc))]
            bad = ~np.isfinite(best)                       # no monotone predecessor: restart the chain here
            best = np.where(bad, prev_cost.min(), best)
            bp = np.where(bad, np.argmin(prev_cost), bp)
            tot = best + cc
        back.append((Jc, bp))
        prevJ, prev_cost = Jc, tot
    k = int(np.argmin(prev_cost))
    for s in range(len(steps) - 1, -1, -1):
        Jc, bp = back[s]
        true_pos[steps[s]] = Jc[k]
        k = int(bp[k]) if bp[k] >= 0 else 0
    true_arc = np.where(true_pos >= 0, route[np.clip(true_pos, 0, len(route) - 1)], -1)
    return true_pos, true_arc
