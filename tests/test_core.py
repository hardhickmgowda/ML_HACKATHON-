"""Smoke tests: python -m pytest -q   (or python tests/test_core.py)"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathsnap.geo import Projector, point_segment
from pathsnap.network import RoadNetwork
from pathsnap.hmm import HMMParams, viterbi_decode, connected_route
import pandas as pd


def _line_net():
    # 0 -> 1 -> 2 -> 3 along x (100 m arcs), plus reverse arcs
    p = Projector(0, 0); lon = np.array([0, 100, 200, 300]) / p.kx
    nodes = np.c_[lon, np.zeros(4)]
    arcs = np.array([[0, 1], [1, 2], [2, 3], [1, 0], [2, 1], [3, 2]])
    return RoadNetwork(nodes, arcs, p)


def test_point_segment():
    d, t, _, _ = point_segment(np.array([50.0]), np.array([10.0]), 0.0, 0.0, 100.0, 0.0)
    assert abs(d[0] - 10) < 1e-9 and abs(t[0] - 0.5) < 1e-9


def test_reverse_and_reach():
    n = _line_net()
    assert n.reverse[0] == 3 and n.reverse[1] == 4
    assert abs(n.reach(0, 250)[2] - 200) < 1e-6
    assert n.shortest_path_arcs(0, 3, 500) == [0, 1, 2]


def test_viterbi_prefers_forward_connected_path():
    n = _line_net()
    xy = np.array([[50.0, 3], [150, 3], [250, 3]]); ts = np.array([0.0, 10, 20])
    rows = [(0, 0, 3.0, .5), (0, 3, 3.0, .5), (1, 1, 3.0, .5), (1, 4, 3.0, .5), (2, 2, 3.0, .5), (2, 5, 3.0, .5)]
    df = pd.DataFrame(rows, columns=["pt", "arc", "dist", "frac"])
    res = viterbi_decode(n, xy, ts, df, np.full(6, 0.5), HMMParams())
    assert res.arc.tolist() == [0, 1, 2]
    route, br = connected_route(n, np.array([0, 2]), 500)
    assert route.tolist() == [0, 1, 2] and br == 0


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
