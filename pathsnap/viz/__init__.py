"""Folium visualisation helpers.

trip_map.py: per-trip result maps used by scripts/05_evaluate.py (raw GPS, classifier, HMM, reference).
maps.py:     EDA maps with the surrounding road network used by scripts/eda.py.
"""
from .trip_map import make_map, save_trip_map  # noqa: F401
