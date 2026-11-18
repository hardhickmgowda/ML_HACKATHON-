"""Readers for the raw PathSnap trip files.

Each trip lives in ``<root>/<partition>/<trip_id>/`` and contains:

* ``<trip_id>.track``  lon, lat, t (seconds)       - GPS observations
* ``<trip_id>.nodes``  lon, lat                    - row index = node_id
* ``<trip_id>.arcs``   from_node_id, to_node_id    - row index = arc_id (directed)
* ``<trip_id>.route``  arc_id per line, in order   - reference route (labels)

All files are whitespace separated with no header. The road network is
*per trip* (each trip comes with its own local extract), so node/arc ids are
only meaningful inside their own trip.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from .. import config

EXTENSIONS = ("track", "nodes", "arcs", "route")


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: pd.DataFrame          # columns: lon, lat, t
    nodes: pd.DataFrame          # columns: lon, lat ; index = node_id
    arcs: pd.DataFrame           # columns: u, v     ; index = arc_id
    route: np.ndarray | None = None  # arc_ids in travel order (None = hidden)
    meta: dict = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return len(self.track)

    def without_route(self) -> "Trip":
        """Copy with the reference route removed (what the predictor may see)."""
        return Trip(self.trip_id, self.partition, self.track, self.nodes,
                    self.arcs, None, dict(self.meta))


def trip_dir(root: Path, partition: str, trip_id: str) -> Path:
    return Path(root) / partition / trip_id


def list_trips(partition: str, root: Path | None = None) -> list[str]:
    """Sorted trip ids (sub-folder names) in a partition, ignoring junk files."""
    base = Path(root or config.DATA_ROOT) / partition
    if not base.is_dir():
        raise FileNotFoundError(f"Partition folder not found: {base}")
    return sorted(p.name for p in base.iterdir()
                  if p.is_dir() and not p.name.startswith("."))


def _read(path: Path, names: list[str], dtypes: dict) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=names,
                     engine="c", comment=None)
    for col, dt in dtypes.items():
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if np.issubdtype(np.dtype(dt), np.integer) and df[col].notna().all():
            df[col] = df[col].astype(dt)
    return df


def read_track(path: Path) -> pd.DataFrame:
    return _read(path, ["lon", "lat", "t"],
                 {"lon": "float64", "lat": "float64", "t": "float64"})


def read_nodes(path: Path) -> pd.DataFrame:
    df = _read(path, ["lon", "lat"], {"lon": "float64", "lat": "float64"})
    df.index.name = "node_id"
    return df


def read_arcs(path: Path) -> pd.DataFrame:
    df = _read(path, ["u", "v"], {"u": "int64", "v": "int64"})
    df.index.name = "arc_id"
    return df


def read_route(path: Path) -> np.ndarray:
    df = _read(path, ["arc_id"], {"arc_id": "int64"})
    return df["arc_id"].to_numpy()


def load_trip(partition: str, trip_id: str, root: Path | None = None,
              with_route: bool = True) -> Trip:
    """Load one trip. Set ``with_route=False`` for the prediction path so the
    reference labels never enter it."""
    d = trip_dir(root or config.DATA_ROOT, partition, trip_id)
    files = {ext: d / f"{trip_id}.{ext}" for ext in EXTENSIONS}
    missing = [ext for ext in ("track", "nodes", "arcs") if not files[ext].exists()]
    if missing:
        raise FileNotFoundError(f"Trip {partition}/{trip_id} is missing {missing}")

    route = None
    if with_route and files["route"].exists():
        route = read_route(files["route"])

    meta = {ext: files[ext].stat().st_size for ext in EXTENSIONS if files[ext].exists()}
    return Trip(trip_id=trip_id, partition=partition,
                track=read_track(files["track"]),
                nodes=read_nodes(files["nodes"]),
                arcs=read_arcs(files["arcs"]),
                route=route,
                meta={"file_sizes": meta, "dir": str(d)})


def iter_trips(partition: str, root: Path | None = None, with_route: bool = True,
               trip_ids: list[str] | None = None) -> Iterator[Trip]:
    for tid in trip_ids or list_trips(partition, root):
        yield load_trip(partition, tid, root, with_route)
