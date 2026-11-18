"""Tests for the data pipeline. Run with:  python -m pytest -q"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pathsnap import config
from pathsnap.data.clean import clean_track, find_speed_spikes
from pathsnap.data.geo import LocalProjection, angle_diff_deg, haversine_m, point_segment_distance
from pathsnap.data.io import Trip
from pathsnap.data.labels import align_to_route
from pathsnap.data.network import RoadNetwork
from pathsnap.data.pipeline import process_trip
from pathsnap.data.validate import validate_processed, validate_raw_trip

# 1 metre in degrees of latitude
M = 1 / 111_195.0


def toy_trip(with_route=True) -> Trip:
    """An L-shaped two-way road: (0,0) -> (0,200m) -> (200m,200m), plus a
    parallel street 40 m east. Vehicle drives north then east at ~10 m/s."""
    lat0, lon0 = 50.0, 8.0
    mlon = M / np.cos(np.radians(lat0))
    nodes = pd.DataFrame({"lon": [lon0, lon0, lon0 + 200 * mlon, lon0 + 40 * mlon, lon0 + 40 * mlon],
                          "lat": [lat0, lat0 + 200 * M, lat0 + 200 * M, lat0, lat0 + 160 * M]})
    arcs = pd.DataFrame({"u": [0, 1, 1, 2, 3, 4, 0],
                         "v": [1, 0, 2, 1, 4, 3, 1]})  # arc 6 duplicates arc 0
    t = np.arange(40, dtype=float)
    rng = np.random.default_rng(0)
    north = np.c_[np.zeros(20), np.arange(20) * 10.0]
    east = np.c_[np.arange(1, 21) * 10.0, np.full(20, 200.0)]
    xy = np.r_[north, east] + rng.normal(0, 3, (40, 2))
    xy[25] += [0, 900]  # one GPS spike
    track = pd.DataFrame({"lon": lon0 + xy[:, 0] * mlon, "lat": lat0 + xy[:, 1] * M, "t": t})
    route = np.array([0, 2]) if with_route else None
    return Trip("toy", "train", track, nodes, arcs, route)


def test_geometry_basics():
    assert haversine_m(0, 0, 0, 1) == pytest.approx(111_195, rel=1e-3)
    proj = LocalProjection(8.0, 50.0)
    x, y = proj.to_xy(8.01, 50.01)
    assert np.hypot(x, y) == pytest.approx(haversine_m(8, 50, 8.01, 50.01), rel=2e-3)
    lon, lat = proj.to_lonlat(x, y)
    assert lon == pytest.approx(8.01) and lat == pytest.approx(50.01)
    d, t = point_segment_distance(np.array([5.0]), np.array([3.0]), 0, 0, 10, 0)
    assert d[0] == pytest.approx(3.0) and t[0] == pytest.approx(0.5)
    assert angle_diff_deg(350, 10) == pytest.approx(20)


def test_spike_detection():
    x = np.arange(10) * 10.0
    y = np.zeros(10)
    y[4] = 500
    out = find_speed_spikes(x, y, np.arange(10.0))
    assert out.tolist() == [i == 4 for i in range(10)]


def test_clean_drops_spike_and_keeps_mapping():
    pts, _, rep = clean_track(toy_trip().track)
    assert rep["n_speed_spikes"] == 1
    assert 25 not in pts["orig_idx"].tolist()
    assert len(pts) == 39 and pts["point_pos"].tolist() == list(range(39))


def test_candidates_and_duplicates():
    trip = toy_trip()
    pts, proj, _ = clean_track(trip.track)
    net = RoadNetwork(trip.nodes, trip.arcs, proj)
    assert net.canon[6] == 0 and net.is_duplicate[6]
    c = net.candidates(pts["x"].to_numpy(), pts["y"].to_numpy())
    assert 6 not in c["arc_id"].values           # duplicate never offered
    assert c.groupby("point_pos").size().max() <= config.MAX_CANDIDATES
    assert set(c["point_pos"]) == set(range(len(pts)))


def test_route_alignment_is_monotonic_and_correct():
    trip = toy_trip()
    pts, proj, _ = clean_track(trip.track)
    net = RoadNetwork(trip.nodes, trip.arcs, proj)
    pos = align_to_route(pts["x"].to_numpy(), pts["y"].to_numpy(), net, trip.route)
    assert np.all(np.diff(pos) >= 0)
    assert pos[0] == 0 and pos[-1] == 1


def test_process_trip_labels_and_heading():
    p = process_trip(toy_trip())
    c = p.candidates
    assert c.groupby("point_pos")["label"].sum().max() == 1
    # direction matters: the true arc should mostly be the one aligned with motion
    best = c.sort_values(["point_pos", "dist_minus_min", "heading_diff_deg"]) \
            .assign(close=lambda d: d.dist_minus_min < 1).query("close") \
            .groupby("point_pos").head(1)
    assert best["label"].mean() > 0.8
    assert not [i for i in validate_processed(p.points, c) if i.severity == "ERROR"]


def test_features_do_not_depend_on_route():
    """Leakage guard: features must be identical with and without the route."""
    a = process_trip(toy_trip(with_route=True)).candidates
    b = process_trip(toy_trip(with_route=False)).candidates
    from pathsnap.data.features import FEATURE_COLUMNS
    pd.testing.assert_frame_equal(a[FEATURE_COLUMNS].reset_index(drop=True),
                                  b[FEATURE_COLUMNS].reset_index(drop=True))
    assert "label" not in b.columns


def test_raw_validation_toy():
    m, issues = validate_raw_trip(toy_trip())
    assert not [i for i in issues if i.severity == "ERROR"]
    assert m["n_duplicate_arcs"] == 1


@pytest.mark.skipif(not (config.DATA_ROOT / "student_test").exists(), reason="dataset not present")
def test_real_trip_smoke():
    from pathsnap.data import list_trips, load_trip
    tid = list_trips("student_test")[0]
    p = process_trip(load_trip("student_test", tid))
    assert p.stats["candidate_miss_rate"] < 0.05
    assert not [i for i in validate_processed(p.points, p.candidates) if i.severity == "ERROR"]
