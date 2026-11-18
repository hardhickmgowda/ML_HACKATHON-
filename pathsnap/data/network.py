"""Road-network representation and candidate generation.

Arcs are *directed* (a two-way street appears as u->v and v->u). Some trips
contain exact duplicate arcs (same u, v) and zero-length self-loops; these are
collapsed onto a *canonical* arc id so that a GPS point never sees two
identical candidates. ``canon[arc_id]`` maps any arc to its canonical id and
evaluation should compare canonical ids.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .. import config
from .geo import LocalProjection, bearing_deg, point_segment_distance


class RoadNetwork:
    def __init__(self, nodes: pd.DataFrame, arcs: pd.DataFrame, proj: LocalProjection,
                 densify_step: float = config.DENSIFY_STEP_M):
        self.proj = proj
        self.n_nodes = len(nodes)
        self.n_arcs = len(arcs)
        self.node_x, self.node_y = proj.to_xy(nodes["lon"].to_numpy(), nodes["lat"].to_numpy())

        self.u = arcs["u"].to_numpy()
        self.v = arcs["v"].to_numpy()
        self.ax, self.ay = self.node_x[self.u], self.node_y[self.u]
        self.bx, self.by = self.node_x[self.v], self.node_y[self.v]
        self.length = np.hypot(self.bx - self.ax, self.by - self.ay)
        self.bearing = bearing_deg(self.bx - self.ax, self.by - self.ay)

        # canonical ids: first occurrence of every (u, v) pair
        uv = pd.DataFrame({"u": self.u, "v": self.v, "i": np.arange(self.n_arcs)})
        self.canon = uv.groupby(["u", "v"])["i"].transform("min").to_numpy(np.int64)
        self.is_self_loop = self.u == self.v
        self.is_duplicate = self.canon != np.arange(self.n_arcs)

        # degree information (used as features and later by the HMM)
        self.out_deg = np.bincount(self.u, minlength=self.n_nodes)
        self.in_deg = np.bincount(self.v, minlength=self.n_nodes)
        rev = pd.MultiIndex.from_arrays([self.v, self.u])
        fwd = pd.MultiIndex.from_arrays([self.u, self.v])
        self.is_two_way = rev.isin(fwd)

        self._build_index(densify_step)

    # ------------------------------------------------------------------ index
    def _build_index(self, step: float) -> None:
        usable = np.flatnonzero(~self.is_self_loop & ~self.is_duplicate)
        n_samples = np.maximum(np.ceil(self.length[usable] / step).astype(int), 1) + 1
        arc_of_sample = np.repeat(usable, n_samples)
        # position 0..1 along each arc
        starts = np.repeat(np.cumsum(n_samples) - n_samples, n_samples)
        offsets = (np.arange(n_samples.sum()) - starts) / np.repeat(n_samples - 1, n_samples)
        sx = self.ax[arc_of_sample] + offsets * (self.bx - self.ax)[arc_of_sample]
        sy = self.ay[arc_of_sample] + offsets * (self.by - self.ay)[arc_of_sample]
        self._sample_arc = arc_of_sample
        self._tree = cKDTree(np.c_[sx, sy])
        self._step = step

    # ------------------------------------------------------------- candidates
    def candidates(self, px: np.ndarray, py: np.ndarray,
                   radius: float = config.CANDIDATE_RADIUS_M,
                   k: int = config.MAX_CANDIDATES) -> pd.DataFrame:
        """Candidate arcs for each point.

        Returns one row per (point_pos, arc_id) with the exact perpendicular
        distance ``dist_m``, projection position ``proj_t`` and ``dist_rank``.
        A point with nothing inside ``radius`` falls back to its nearest arcs
        and is marked ``fallback=True`` (so the model / HMM still has options).
        """
        pts = np.c_[px, py]
        search_r = radius + self._step / 2
        hits = self._tree.query_ball_point(pts, r=search_r)
        lens = np.fromiter((len(h) for h in hits), dtype=int, count=len(hits))
        p_idx = np.repeat(np.arange(len(pts)), lens)
        s_idx = np.concatenate([np.asarray(h, dtype=int) for h in hits]) if lens.sum() else np.array([], int)
        df = pd.DataFrame({"point_pos": p_idx, "arc_id": self._sample_arc[s_idx]})
        df = df.drop_duplicates()
        df = self._exact(df, px, py)
        df = df[df["dist_m"] <= radius]
        df["fallback"] = False

        missing = np.setdiff1d(np.arange(len(pts)), df["point_pos"].unique())
        if len(missing):
            kk = min(len(self._sample_arc), k * 20)
            _, nn = self._tree.query(pts[missing], k=kk)
            nn = np.atleast_2d(nn)
            fb = pd.DataFrame({"point_pos": np.repeat(missing, nn.shape[1]),
                               "arc_id": self._sample_arc[nn.ravel()]}).drop_duplicates()
            fb = self._exact(fb, px, py)
            fb["fallback"] = True
            df = pd.concat([df, fb], ignore_index=True)

        df = df.sort_values(["point_pos", "dist_m", "arc_id"], kind="stable")
        df = df.groupby("point_pos", sort=False).head(k).reset_index(drop=True)
        df["dist_rank"] = df.groupby("point_pos").cumcount()
        return df

    def _exact(self, df: pd.DataFrame, px, py) -> pd.DataFrame:
        a = df["arc_id"].to_numpy()
        p = df["point_pos"].to_numpy()
        d, t = point_segment_distance(px[p], py[p], self.ax[a], self.ay[a], self.bx[a], self.by[a])
        df = df.copy()
        df["dist_m"] = d
        df["proj_t"] = t
        return df

    # ----------------------------------------------------------- utilities
    def arc_polyline_lonlat(self, arc_ids) -> list[list[tuple[float, float]]]:
        """[(lat, lon), (lat, lon)] per arc - handy for folium."""
        arc_ids = np.asarray(arc_ids)
        alon, alat = self.proj.to_lonlat(self.ax[arc_ids], self.ay[arc_ids])
        blon, blat = self.proj.to_lonlat(self.bx[arc_ids], self.by[arc_ids])
        return [[(la, lo), (lb, lob)] for la, lo, lb, lob in zip(alat, alon, blat, blon)]
