"""Directed road network + bounded shortest paths (Harsh/Hardhick)."""
from __future__ import annotations
import heapq
import numpy as np
from .geo import Projector


class RoadNetwork:
    def __init__(self, nodes_ll: np.ndarray, arcs: np.ndarray, proj: Projector):
        self.proj = proj
        nx, ny = proj.fwd(nodes_ll[:, 0], nodes_ll[:, 1])
        self.node_xy = np.c_[nx, ny]
        self.n_nodes = len(nodes_ll)
        self.arc_from = arcs[:, 0].astype(np.int64)
        self.arc_to = arcs[:, 1].astype(np.int64)
        self.n_arcs = len(arcs)
        a, b = self.node_xy[self.arc_from], self.node_xy[self.arc_to]
        self.arc_a, self.arc_b = a, b
        d = b - a
        self.arc_len = np.hypot(d[:, 0], d[:, 1])
        self.arc_bearing = np.arctan2(d[:, 1], d[:, 0])
        self.out_deg = np.bincount(self.arc_from, minlength=self.n_nodes)
        self.in_deg = np.bincount(self.arc_to, minlength=self.n_nodes)
        # CSR of outgoing arcs per node
        order = np.argsort(self.arc_from, kind="stable")
        self._out_arcs = order
        self._out_ptr = np.searchsorted(self.arc_from[order], np.arange(self.n_nodes + 1))
        # reverse arc (to,from) lookup
        key = self.arc_from * self.n_nodes + self.arc_to
        rkey = self.arc_to * self.n_nodes + self.arc_from
        so = np.argsort(key)
        pos = np.clip(np.searchsorted(key[so], rkey), 0, self.n_arcs - 1)
        hit = key[so][pos] == rkey
        self.reverse = np.where(hit, so[pos], -1)
        self._py = None
        self._reach_cache: dict[int, tuple[float, dict]] = {}

    # ---- python-list mirrors: scalar indexing of numpy is slow inside Dijkstra loops
    def _pylists(self):
        if self._py is None:
            self._py = (self._out_ptr.tolist(), self._out_arcs.tolist(), self.arc_to.tolist(), self.arc_len.tolist())
        return self._py

    def reach(self, src: int, limit: float) -> dict[int, float]:
        """All nodes within `limit` metres of `src` along directed arcs (cached)."""
        c = self._reach_cache.get(src)
        if c is not None and c[0] >= limit:
            return c[1]
        ptr, oa, to, ln = self._pylists()
        dist = {src: 0.0}
        pq = [(0.0, src)]
        done = set()
        while pq:
            d, u = heapq.heappop(pq)
            if u in done:
                continue
            done.add(u)
            for k in range(ptr[u], ptr[u + 1]):
                a = oa[k]
                v, nd = to[a], d + ln[a]
                if nd <= limit and nd < dist.get(v, 1e18):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        if len(self._reach_cache) > 200_000:
            self._reach_cache.clear()
        self._reach_cache[src] = (limit, dist)
        return dist

    def shortest_path_arcs(self, src: int, dst: int, limit: float) -> list[int] | None:
        """Arc-id path from node src to node dst (None if farther than `limit`). [] if src==dst."""
        if src == dst:
            return []
        ptr, oa, to, ln = self._pylists()
        dist, par = {src: 0.0}, {}
        pq, done = [(0.0, src)], set()
        while pq:
            d, u = heapq.heappop(pq)
            if u in done:
                continue
            if u == dst:
                break
            done.add(u)
            for k in range(ptr[u], ptr[u + 1]):
                a = oa[k]
                v, nd = to[a], d + ln[a]
                if nd <= limit and nd < dist.get(v, 1e18):
                    dist[v], par[v] = nd, a
                    heapq.heappush(pq, (nd, v))
        if dst not in par:
            return None
        path, v = [], dst
        while v != src:
            a = par[v]
            path.append(a)
            v = int(self.arc_from[a])
        return path[::-1]

    def arc_polyline_ll(self, arc_ids) -> list[list[float]]:
        """Connected [lat, lon] polyline for an arc sequence (for maps)."""
        arc_ids = np.asarray(arc_ids, dtype=np.int64)
        if len(arc_ids) == 0:
            return []
        pts = [self.node_xy[self.arc_from[arc_ids[0]]]]
        for a in arc_ids:
            pts.append(self.node_xy[self.arc_to[a]])
        pts = np.asarray(pts)
        lon, lat = self.proj.inv(pts[:, 0], pts[:, 1])
        return np.c_[lat, lon].round(6).tolist()
