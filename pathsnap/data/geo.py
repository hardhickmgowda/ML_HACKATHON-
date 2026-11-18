"""Geometry helpers.

Coordinates are (lon, lat) in degrees. Distances in degrees are meaningless
(1 deg of longitude is ~111 km at the equator but ~55 km at 60 N), so every
trip is projected to a *local equirectangular plane* centred on the trip's GPS
centroid:

    x = R * cos(lat0) * (lon - lon0)   [m]
    y = R * (lat - lat0)               [m]

Trips span at most a few tens of km, where this projection's distance error is
well below 1 % (far below the ~5 m GPS noise). It is cheap, needs no external
CRS library and works anywhere on the globe (the dataset covers 6 continents,
so a single UTM zone would not). ``haversine_m`` is provided for checks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import EARTH_RADIUS_M


@dataclass(frozen=True)
class LocalProjection:
    lon0: float
    lat0: float

    @classmethod
    def from_points(cls, lon: np.ndarray, lat: np.ndarray) -> "LocalProjection":
        return cls(float(np.mean(lon)), float(np.mean(lat)))

    def to_xy(self, lon, lat) -> tuple[np.ndarray, np.ndarray]:
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        x = np.radians(lon - self.lon0) * EARTH_RADIUS_M * np.cos(np.radians(self.lat0))
        y = np.radians(lat - self.lat0) * EARTH_RADIUS_M
        return x, y

    def to_lonlat(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        lon = self.lon0 + np.degrees(np.asarray(x) / (EARTH_RADIUS_M * np.cos(np.radians(self.lat0))))
        lat = self.lat0 + np.degrees(np.asarray(y) / EARTH_RADIUS_M)
        return lon, lat


def haversine_m(lon1, lat1, lon2, lat2) -> np.ndarray:
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def bearing_deg(dx, dy) -> np.ndarray:
    """Compass bearing (0 = north, 90 = east) of a planar vector, in [0, 360)."""
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def angle_diff_deg(a, b) -> np.ndarray:
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = np.abs((np.asarray(a) - np.asarray(b)) % 360.0)
    return np.minimum(d, 360.0 - d)


def point_segment_distance(px, py, ax, ay, bx, by):
    """Distance from points P to segments AB (broadcasting).

    Returns (distance_m, t) where t in [0, 1] is the position of the
    projection of P along AB (0 = at A, 1 = at B).
    """
    dx, dy = bx - ax, by - ay
    seg2 = dx * dx + dy * dy
    seg2 = np.where(seg2 == 0, 1e-12, seg2)
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / seg2, 0.0, 1.0)
    qx, qy = ax + t * dx, ay + t * dy
    return np.hypot(px - qx, py - qy), t
