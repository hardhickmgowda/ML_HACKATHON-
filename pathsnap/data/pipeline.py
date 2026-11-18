"""End-to-end preprocessing of one trip / one partition.

    raw trip --clean--> points --network index--> candidates --> features
                                                       |
                         (only if a route is given) labels

``process_trip(trip.without_route())`` is exactly what the API will run at
inference time, so training and serving share the same code path.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import config
from .clean import clean_track
from .features import build_features
from .io import Trip, iter_trips
from .network import RoadNetwork


@dataclass
class ProcessedTrip:
    trip_id: str
    points: pd.DataFrame
    candidates: pd.DataFrame
    network: RoadNetwork
    stats: dict = field(default_factory=dict)


def process_trip(trip: Trip) -> ProcessedTrip:
    t0 = time.perf_counter()
    points, proj, report = clean_track(trip.track)
    net = RoadNetwork(trip.nodes, trip.arcs, proj)
    cands = net.candidates(points["x"].to_numpy(), points["y"].to_numpy())
    cands = build_features(points, cands, net)

    stats = {"trip_id": trip.trip_id, "partition": trip.partition, **report,
             "n_candidates": len(cands),
             "mean_cands_per_point": len(cands) / max(len(points), 1),
             "frac_fallback_points": float(cands.groupby("point_pos")["fallback"].max().mean())}

    if trip.route is not None:
        from .labels import point_labels  # imported lazily: never needed for inference
        lab = point_labels(points, net, trip.route)
        points = points.merge(lab, on="point_pos", how="left")
        cands = cands.merge(lab[["point_pos", "true_canon", "label_valid"]], on="point_pos", how="left")
        cands["label"] = (cands["canon_arc"] == cands["true_canon"]).astype(np.int8)
        cands = cands.drop(columns="true_canon")

        valid = points["label_valid"].to_numpy()
        hit = cands[cands["label"] == 1]["point_pos"].unique()
        covered = np.zeros(len(points), bool)
        covered[hit] = True
        nearest_ok = np.zeros(len(points), bool)
        top = cands[(cands["dist_rank"] == 0) & (cands["label"] == 1)]["point_pos"].to_numpy()
        nearest_ok[top] = True
        stats.update({
            "frac_label_valid": float(valid.mean()),
            "candidate_miss_rate": float(1 - covered[valid].mean()) if valid.any() else np.nan,
            "nearest_arc_top1_acc": float(nearest_ok[valid].mean()) if valid.any() else np.nan,
            "median_true_dist_m": float(points["true_dist_m"].median()),
        })

    points.insert(0, "trip_id", trip.trip_id)
    cands.insert(0, "trip_id", trip.trip_id)
    stats["seconds"] = round(time.perf_counter() - t0, 3)
    return ProcessedTrip(trip.trip_id, points, cands, net, stats)


def build_partition(partition: str, root=None, trip_ids=None, verbose=True):
    """Process every trip in a partition and write parquet files to
    ``data_processed/<partition>/``."""
    out = config.PROCESSED_DIR / partition
    out.mkdir(parents=True, exist_ok=True)
    pts, cds, sts = [], [], []
    for trip in iter_trips(partition, root=root, trip_ids=trip_ids):
        pt = process_trip(trip)
        pts.append(pt.points)
        cds.append(pt.candidates)
        sts.append(pt.stats)
        if verbose:
            s = pt.stats
            print(f"[{partition}] {trip.trip_id}: {s['n_clean']:>6} pts, "
                  f"{s['mean_cands_per_point']:.1f} cands/pt, "
                  f"miss={s.get('candidate_miss_rate', float('nan')):.3f}, "
                  f"valid={s.get('frac_label_valid', float('nan')):.3f}, {s['seconds']}s")
    points = pd.concat(pts, ignore_index=True)
    cands = pd.concat(cds, ignore_index=True)
    stats = pd.DataFrame(sts)
    points.to_parquet(out / "points.parquet", index=False)
    cands.to_parquet(out / "candidates.parquet", index=False)
    stats.to_csv(out / "trip_stats.csv", index=False)
    return points, cands, stats
