"""HMM + Viterbi route reconstruction (Hardhick).

Hidden state at GPS point t  = one of its candidate arcs.
Emission   log e(t,a) = alpha*log P_clf(a|x_t)  -  (1-alpha) * d(t,a)^2 / (2 sigma^2)
Transition log T(a->b)= -| d_net(a->b) - d_gps | / beta(dt)     (Newson & Krumm 2009, with a learned emission)
   d_net  = along-network distance between the two projected positions (bounded Dijkstra on the directed graph)
   unreachable           -> -invalid_penalty   (a "teleport": keeps Viterbi defined, flags a route break)
   U-turn (a -> reverse(a)) -> extra -uturn_penalty
   time gap > gap_seconds or jump > gap_metres -> hard restart (uniform transitions)
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, fields
import math
import numpy as np
import pandas as pd
from .network import RoadNetwork


@dataclass
class HMMParams:
    alpha: float = 0.7
    sigma_m: float = 15.0
    beta0_m: float = 8.0
    beta_per_s: float = 1.0
    invalid_penalty: float = 25.0
    uturn_penalty: float = 6.0
    gap_seconds: float = 180.0
    gap_metres: float = 3000.0
    max_sp_limit_m: float = 3000.0
    fill_limit_m: float = 4000.0
    normalize_per_point: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "HMMParams":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HMMResult:
    arc: np.ndarray          # (T,) decoded arc per cleaned GPS point (-1 if the point had no candidates)
    frac: np.ndarray         # (T,) position along that arc
    restarts: int            # hard restarts (GPS gaps)
    teleports: int           # transitions chosen despite being unreachable (route breaks)


def emission_logprob(prob: np.ndarray, dist: np.ndarray, starts: np.ndarray, counts: np.ndarray, p: HMMParams) -> np.ndarray:
    prob = np.asarray(prob, np.float64)
    if p.normalize_per_point:
        prob = prob / np.repeat(np.maximum(np.add.reduceat(prob, starts), 1e-12), counts)
    return p.alpha * np.log(prob + 1e-4) - (1 - p.alpha) * 0.5 * (np.asarray(dist, np.float64) / p.sigma_m) ** 2


def transition_logprob(net: RoadNetwork, a_arcs, a_frac, b_arcs, b_frac, dgc: float, dt: float, p: HMMParams):
    """(ka,kb) log-transition matrix between two consecutive GPS points. Returns (matrix, n_invalid_mask)."""
    beta = p.beta0_m + p.beta_per_s * min(dt, 60.0)
    limit = min(p.max_sp_limit_m, math.ceil((3.0 * dgc + 150.0) / 300.0) * 300.0)
    ka, kb = len(a_arcs), len(b_arcs)
    T = np.empty((ka, kb))
    invalid = np.zeros((ka, kb), bool)
    arc_len, arc_from, arc_to, rev = net.arc_len, net.arc_from, net.arc_to, net.reverse
    b_start = [int(arc_from[b]) for b in b_arcs]
    b_len = [float(arc_len[b]) for b in b_arcs]
    for i in range(ka):
        a = int(a_arcs[i]); la = float(arc_len[a]); fa = float(a_frac[i])
        reach = net.reach(int(arc_to[a]), limit)
        ra = int(rev[a])
        for j in range(kb):
            b = int(b_arcs[j]); fb = float(b_frac[j])
            if a == b:
                dnet = abs(fb - fa) * la
            else:
                sp = reach.get(b_start[j])
                if sp is None:
                    T[i, j] = -p.invalid_penalty
                    invalid[i, j] = True
                    continue
                dnet = (1.0 - fa) * la + sp + fb * b_len[j]
            v = -abs(dnet - dgc) / beta
            if b == ra:
                v -= p.uturn_penalty
            T[i, j] = v
    return T, invalid


def viterbi_decode(net: RoadNetwork, xy: np.ndarray, ts: np.ndarray, df: pd.DataFrame, prob: np.ndarray,
                   params: HMMParams) -> HMMResult:
    """df: candidate rows sorted by (pt, dist) with columns pt, arc, dist, frac. prob aligned with df rows."""
    T = len(xy)
    pt = df["pt"].to_numpy()
    arc = df["arc"].to_numpy(np.int64)
    frac = df["frac"].to_numpy(np.float64)
    dist = df["dist"].to_numpy(np.float64)
    uniq, starts, counts = np.unique(pt, return_index=True, return_counts=True)
    em = emission_logprob(prob, dist, starts, counts, params)
    n = len(uniq)
    out_arc, out_frac = np.full(T, -1, np.int64), np.zeros(T)
    if n == 0:
        return HMMResult(out_arc, out_frac, 0, 0)
    sl = [slice(s, s + c) for s, c in zip(starts, counts)]
    delta = em[sl[0]].copy()
    back: list[np.ndarray] = []
    restarts = 0
    tele_flags: list[np.ndarray] = []
    for k in range(1, n):
        u, v = uniq[k - 1], uniq[k]
        dt = float(ts[v] - ts[u])
        dgc = float(np.hypot(*(xy[v] - xy[u])))
        if dt > params.gap_seconds or dgc > params.gap_metres:
            Tm = np.zeros((len(delta), counts[k])); inv = np.zeros_like(Tm, bool); restarts += 1
        else:
            Tm, inv = transition_logprob(net, arc[sl[k - 1]], frac[sl[k - 1]], arc[sl[k]], frac[sl[k]], dgc, dt, params)
        sc = delta[:, None] + Tm
        bp = np.argmax(sc, axis=0)
        delta = sc[bp, np.arange(sc.shape[1])] + em[sl[k]]
        back.append(bp)
        tele_flags.append(inv[bp, np.arange(inv.shape[1])])
    state = np.empty(n, np.int64)
    state[-1] = int(np.argmax(delta))
    teleports = 0
    for k in range(n - 1, 0, -1):
        teleports += int(tele_flags[k - 1][state[k]])
        state[k - 1] = back[k - 1][state[k]]
    for k in range(n):
        out_arc[uniq[k]] = arc[sl[k]][state[k]]
        out_frac[uniq[k]] = frac[sl[k]][state[k]]
    return HMMResult(out_arc, out_frac, restarts, teleports)


def connected_route(net: RoadNetwork, point_arcs: np.ndarray, fill_limit: float = 4000.0):
    """Collapse per-point arcs into a route and bridge every gap with a shortest path.
    Returns (route_arc_ids, n_unbridged_breaks)."""
    seq = np.asarray([a for a in point_arcs if a >= 0], np.int64)
    if len(seq) == 0:
        return np.empty(0, np.int64), 0
    seq = seq[np.r_[True, seq[1:] != seq[:-1]]]
    route, breaks = [int(seq[0])], 0
    for b in seq[1:]:
        a = route[-1]
        if net.arc_to[a] != net.arc_from[b]:
            path = net.shortest_path_arcs(int(net.arc_to[a]), int(net.arc_from[b]), fill_limit)
            if path is None:
                breaks += 1
            else:
                route.extend(path)
        route.append(int(b))
    return np.asarray(route, np.int64), breaks


def snapped_xy(net: RoadNetwork, arcs: np.ndarray, frac: np.ndarray) -> np.ndarray:
    """Metric coordinates of the projected position on each decoded arc (NaN where arc < 0)."""
    ok = arcs >= 0
    a = np.where(ok, arcs, 0)
    p = net.arc_a[a] + frac[:, None] * (net.arc_b[a] - net.arc_a[a])
    p[~ok] = np.nan
    return p
