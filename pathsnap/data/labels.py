"""Per-observation labels derived from the reference route.

The ``.route`` file is an ordered list of arc ids, *not* one label per GPS
point (a trip with 1 315 points may have a 525-arc route). To get a label per
observation we solve a monotonic alignment:

    choose route positions j_0 <= j_1 <= ... <= j_{n-1}
    minimising  sum_i  dist(point_i, route_arc[j_i])

i.e. the vehicle moves forward along the route and never jumps back. This is
solved exactly with dynamic programming:

    D[i, j] = d[i, j] + min_{k <= j} D[i-1, k]

(the inner min is a running prefix-minimum, so each row costs O(m)). Ties
are broken towards earlier route positions.

The aligned arc is the point's reference arc. If it is still farther than
``LABEL_MAX_DIST_M`` the reference route probably does not cover that part of
the drive (seen in a few training trips), so the point gets
``label_valid = False`` and should be excluded from training / scoring.

This module needs the reference route and must NEVER be imported by the
inference path (API prediction code).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from .geo import point_segment_distance
from .network import RoadNetwork


def _dist_matrix_rows(px, py, net: RoadNetwork, route, rows):
    ax, ay = net.ax[route][None, :], net.ay[route][None, :]
    bx, by = net.bx[route][None, :], net.by[route][None, :]
    d, _ = point_segment_distance(px[rows, None], py[rows, None], ax, ay, bx, by)
    return d.astype(np.float32)


def align_to_route(px: np.ndarray, py: np.ndarray, net: RoadNetwork,
                   route: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Return the aligned route position (index into ``route``) for each point."""
    n, m = len(px), len(route)
    D = np.empty((n, m), dtype=np.float32)
    prev = None
    for s in range(0, n, chunk):
        rows = np.arange(s, min(s + chunk, n))
        d = _dist_matrix_rows(px, py, net, route, rows)
        for r, i in enumerate(rows):
            if prev is None:
                D[i] = d[r]
            else:
                D[i] = d[r] + np.minimum.accumulate(prev)
            prev = D[i]
    pos = np.empty(n, dtype=np.int64)
    j = int(np.argmin(D[-1]))
    pos[-1] = j
    for i in range(n - 1, 0, -1):
        j = int(np.argmin(D[i - 1, : j + 1]))
        pos[i - 1] = j
    return pos


def point_labels(points: pd.DataFrame, net: RoadNetwork, route: np.ndarray) -> pd.DataFrame:
    """Per-point reference arc + distance + validity flag."""
    px, py = points["x"].to_numpy(), points["y"].to_numpy()
    pos = align_to_route(px, py, net, route)
    arc = route[pos]
    dist, _ = point_segment_distance(px, py, net.ax[arc], net.ay[arc], net.bx[arc], net.by[arc])
    return pd.DataFrame({
        "point_pos": np.arange(len(points)),
        "route_pos": pos,
        "true_arc": arc,
        "true_canon": net.canon[arc],
        "true_dist_m": dist,
        "label_valid": dist <= config.LABEL_MAX_DIST_M,
    })
