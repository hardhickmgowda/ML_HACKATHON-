"""cKDTree spatial index + candidate generation (Harsh)."""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from .geo import point_segment
from .network import RoadNetwork


class ArcIndex:
    """KD-tree over points sampled along every arc (spacing <= sample_spacing_m)."""

    def __init__(self, net: RoadNetwork, spacing: float = 10.0):
        self.net = net
        n_s = np.maximum(np.ceil(net.arc_len / spacing).astype(np.int64) + 1, 2)
        arc_id = np.repeat(np.arange(net.n_arcs), n_s)
        off = np.cumsum(n_s) - n_s
        local = np.arange(len(arc_id)) - np.repeat(off, n_s)
        frac = local / (np.repeat(n_s, n_s) - 1)
        a, b = net.arc_a[arc_id], net.arc_b[arc_id]
        self.samples = a + frac[:, None] * (b - a)
        self.sample_arc = arc_id
        self.tree = cKDTree(self.samples)

    def _pairs(self, xy: np.ndarray, radius: float, pts: np.ndarray):
        hits = self.tree.query_ball_point(xy[pts], r=radius, return_sorted=False)
        lens = np.fromiter((len(h) for h in hits), dtype=np.int64, count=len(hits))
        if lens.sum() == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        flat = np.concatenate([np.asarray(h, dtype=np.int64) for h in hits if len(h)])
        p = np.repeat(pts, lens)
        key = np.unique(p * self.net.n_arcs + self.sample_arc[flat])
        return key // self.net.n_arcs, key % self.net.n_arcs

    def query(self, xy: np.ndarray, radius: float, max_cands: int, fallback_radius: float):
        """Return (pt, arc, dist, frac) sorted by (pt, dist), at most max_cands per point, >=1 per point."""
        T = len(xy)
        pt, arc = self._pairs(xy, radius, np.arange(T))
        have = np.zeros(T, bool)
        have[pt] = True
        miss = np.flatnonzero(~have)
        if len(miss):
            p2, a2 = self._pairs(xy, fallback_radius, miss)
            pt, arc = np.concatenate([pt, p2]), np.concatenate([arc, a2])
        net = self.net
        d, t, _, _ = point_segment(xy[pt, 0], xy[pt, 1], net.arc_a[arc, 0], net.arc_a[arc, 1],
                                   net.arc_b[arc, 0], net.arc_b[arc, 1])
        lim = np.where(have[pt], radius, fallback_radius)
        keep = d <= lim
        pt, arc, d, t = pt[keep], arc[keep], d[keep], t[keep]
        o = np.lexsort((d, pt))
        pt, arc, d, t = pt[o], arc[o], d[o], t[o]
        first = np.searchsorted(pt, pt, side="left")
        rank = np.arange(len(pt)) - first
        k = rank < max_cands
        return pt[k], arc[k], d[k], t[k]
