"""Robust parsers for .track / .nodes / .arcs / .route files (Harsh).

Formats (verified on the provided data):
  .nodes  one node per line:  lon  lat          (node id = 0-based line number)
  .arcs   one arc per line:   from_node to_node (arc id  = 0-based line number, directed)
  .route  one arc id per line (reference route, optional at inference)
  .track  lon  lat  timestamp_seconds           (may use CRLF line endings)
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

_SPLIT = re.compile(r"[\s,;]+")


def _read_table(path: Path, ncols: int, dtype) -> np.ndarray:
    """Tolerant numeric table reader: handles CRLF, blank lines, comma/space/tab separators, empty files."""
    rows = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p for p in _SPLIT.split(line) if p]
            if len(parts) < ncols:
                continue                                    # malformed line -> skip
            try:
                rows.append([float(p) for p in parts[:ncols]])
            except ValueError:
                continue
    if not rows:
        return np.empty((0, ncols), dtype=dtype)
    return np.asarray(rows, dtype=np.float64).astype(dtype, copy=False)


@dataclass
class Trip:
    trip_id: str
    folder: Path
    nodes_ll: np.ndarray            # (N,2) lon, lat
    arcs: np.ndarray                # (M,2) from, to  (int64)
    track: np.ndarray               # (T,3) lon, lat, ts
    route: np.ndarray | None = None  # (R,) arc ids or None
    meta: dict = field(default_factory=dict)


def load_trip(folder: str | Path, trip_id: str | None = None, load_route: bool = True) -> Trip:
    folder = Path(folder)
    trip_id = trip_id or folder.name
    base = folder / trip_id
    nodes = _read_table(base.with_suffix(".nodes"), 2, np.float64)
    arcs = _read_table(base.with_suffix(".arcs"), 2, np.int64)
    track = _read_table(base.with_suffix(".track"), 3, np.float64)
    route = None
    rp = base.with_suffix(".route")
    if load_route and rp.exists():
        r = _read_table(rp, 1, np.int64)
        route = r[:, 0] if len(r) else np.empty(0, np.int64)
    if len(nodes) == 0 or len(arcs) == 0 or len(track) == 0:
        raise ValueError(f"Trip {trip_id}: empty nodes/arcs/track file(s) in {folder}")
    if arcs.min() < 0 or arcs.max() >= len(nodes):
        raise ValueError(f"Trip {trip_id}: arc references node outside .nodes range")
    return Trip(trip_id, folder, nodes, arcs, track, route)


def discover_trips(roots: list[str | Path]) -> dict[str, Path]:
    """Find every folder <root>/**/<id>/<id>.track  ->  {trip_id: folder}."""
    found: dict[str, Path] = {}
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for tf in sorted(root.rglob("*.track")):
            if tf.parent.name == tf.stem:
                found[tf.stem] = tf.parent
    return found
