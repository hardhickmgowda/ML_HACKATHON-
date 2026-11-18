"""Turn a raw trip into model-ready tables (shared by training AND the API -> no train/serve skew)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .data_io import Trip
from .geo import Projector
from .network import RoadNetwork
from .preprocess import clean_track, CleanTrack
from .candidates import ArcIndex
from .features import build_features
from .labels import align_to_route


@dataclass
class PreparedTrip:
    trip: Trip
    net: RoadNetwork
    ct: CleanTrack
    df: pd.DataFrame            # one row per (GPS point, candidate arc)
    true_arc: np.ndarray        # (T,) reference arc per point, -1 = unlabeled / no reference


def prepare_trip(trip: Trip, cfg: dict, use_route: bool = True) -> PreparedTrip:
    proj = Projector(trip.nodes_ll[:, 0].mean(), trip.nodes_ll[:, 1].mean())   # per-trip geometry only
    net = RoadNetwork(trip.nodes_ll, trip.arcs, proj)
    ct = clean_track(trip.track, proj, cfg)
    c = cfg["candidates"]
    idx = ArcIndex(net, c["sample_spacing_m"])
    cand = idx.query(ct.xy, c["radius_m"], c["max_candidates"], c["fallback_radius_m"])
    df = build_features(net, ct.xy, ct.ts, cand, cfg)
    true_arc = np.full(ct.n, -1, np.int64)
    if use_route and trip.route is not None and len(trip.route):
        _, true_arc = align_to_route(net, ct.xy, trip.route, cfg)
    df["y"] = (df["arc"].to_numpy() == true_arc[df["pt"].to_numpy()]).astype(np.int8)
    df["labeled"] = (true_arc[df["pt"].to_numpy()] >= 0)
    df.insert(0, "trip_id", trip.trip_id)
    return PreparedTrip(trip, net, ct, df, true_arc)
