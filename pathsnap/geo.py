"""Local metric projection + geometric helpers (Harsh)."""
from __future__ import annotations
import numpy as np

M_PER_DEG_LAT = 110_540.0
M_PER_DEG_LON_EQ = 111_320.0


class Projector:
    """Equirectangular projection around (lon0, lat0). Accurate to <0.1% over a ~10 km city extent."""

    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = float(lon0), float(lat0)
        self.kx = M_PER_DEG_LON_EQ * np.cos(np.radians(lat0))
        self.ky = M_PER_DEG_LAT

    def fwd(self, lon, lat):
        return (np.asarray(lon) - self.lon0) * self.kx, (np.asarray(lat) - self.lat0) * self.ky

    def inv(self, x, y):
        return np.asarray(x) / self.kx + self.lon0, np.asarray(y) / self.ky + self.lat0


def point_segment(px, py, ax, ay, bx, by):
    """Vectorised point->segment distance. Returns (dist, t in [0,1], qx, qy)."""
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    t = ((px - ax) * dx + (py - ay) * dy) / np.where(l2 > 0, l2, 1.0)
    t = np.where(l2 > 0, np.clip(t, 0.0, 1.0), 0.0)
    qx, qy = ax + t * dx, ay + t * dy
    return np.hypot(px - qx, py - qy), t, qx, qy


def wrap_angle(a):
    """Wrap to (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi
