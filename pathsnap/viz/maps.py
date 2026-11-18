"""Folium map builder shared by EDA, the report and the UI.

Layers (each toggleable via LayerControl):
    Road network (nearby)   grey
    Reference route         green   (only when a route is supplied)
    Predicted route         orange  (only when predicted arcs are supplied)
    GPS observations        blue dots
    Segment starts / gaps   red markers
"""
from __future__ import annotations

import folium
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ..data.network import RoadNetwork

COLORS = {"network": "#9e9e9e", "route": "#1b9e3a", "pred": "#ff7f0e",
          "gps": "#1f4fd1", "gap": "#d62728"}

_LEGEND = """
<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;
 padding:8px 12px;border:1px solid #888;border-radius:6px;font:13px sans-serif;">
<b>{title}</b><br>{rows}</div>"""


def _legend_row(color, label, dot=False):
    shape = ("border-radius:50%;width:10px;height:10px;" if dot
             else "width:22px;height:4px;")
    return (f'<span style="display:inline-block;{shape}background:{color};'
            f'margin-right:6px;vertical-align:middle"></span>{label}<br>')


def nearby_arcs(net: RoadNetwork, px, py, buffer_m: float = 150.0) -> np.ndarray:
    """Arcs whose midpoint is within ``buffer_m`` of any GPS point."""
    mx, my = (net.ax + net.bx) / 2, (net.ay + net.by) / 2
    tree = cKDTree(np.c_[px, py])
    d, _ = tree.query(np.c_[mx, my], distance_upper_bound=buffer_m + float(net.length.max()) / 2)
    keep = (d <= buffer_m + net.length / 2) & ~net.is_duplicate & ~net.is_self_loop
    return np.flatnonzero(keep)


def trip_map(points: pd.DataFrame, net: RoadNetwork, title: str,
             route: np.ndarray | None = None, predicted_arcs: np.ndarray | None = None,
             network_buffer_m: float = 150.0, max_gps_markers: int = 4000) -> folium.Map:
    lat, lon = points["lat"].to_numpy(), points["lon"].to_numpy()
    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]], padding=(20, 20))

    rows = []
    arcs = nearby_arcs(net, points["x"].to_numpy(), points["y"].to_numpy(), network_buffer_m)
    fg = folium.FeatureGroup(name=f"Road network ({len(arcs)} arcs nearby)")
    folium.PolyLine(net.arc_polyline_lonlat(arcs), color=COLORS["network"], weight=2,
                    opacity=0.6).add_to(fg)
    fg.add_to(m)
    rows.append(_legend_row(COLORS["network"], "Road network"))

    def _path(arc_ids):
        lines = net.arc_polyline_lonlat(arc_ids)
        return [pt for seg in lines for pt in seg]

    if route is not None and len(route):
        fg = folium.FeatureGroup(name="Reference route")
        folium.PolyLine(_path(route), color=COLORS["route"], weight=6, opacity=0.8,
                        tooltip="Reference route").add_to(fg)
        fg.add_to(m)
        rows.append(_legend_row(COLORS["route"], "Reference route"))

    if predicted_arcs is not None and len(predicted_arcs):
        fg = folium.FeatureGroup(name="Predicted route")
        uniq = pd.unique(np.asarray(predicted_arcs))
        folium.PolyLine(net.arc_polyline_lonlat(uniq), color=COLORS["pred"], weight=4,
                        opacity=0.9, dash_array="6 4", tooltip="Predicted").add_to(fg)
        fg.add_to(m)
        rows.append(_legend_row(COLORS["pred"], "Predicted route"))

    fg = folium.FeatureGroup(name="GPS observations")
    stride = max(1, len(points) // max_gps_markers)
    for _, r in points.iloc[::stride].iterrows():
        folium.CircleMarker((r["lat"], r["lon"]), radius=2, color=COLORS["gps"],
                            fill=True, fill_opacity=0.8, weight=0,
                            tooltip=f"t={r['t']:.0f}s").add_to(fg)
    fg.add_to(m)
    lbl = "GPS observations" + (f" (every {stride}th)" if stride > 1 else "")
    rows.append(_legend_row(COLORS["gps"], lbl, dot=True))

    if "is_segment_start" in points:
        starts = points[points["is_segment_start"]]
        if len(starts) > 1:
            fg = folium.FeatureGroup(name="Segment starts (after GPS gap)")
            for _, r in starts.iloc[1:].iterrows():
                folium.CircleMarker((r["lat"], r["lon"]), radius=6, color=COLORS["gap"],
                                    fill=True, tooltip=f"gap of {r['dt_prev']:.0f}s").add_to(fg)
            fg.add_to(m)
            rows.append(_legend_row(COLORS["gap"], "Start after GPS gap", dot=True))

    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(_LEGEND.format(title=title, rows="".join(rows))))
    return m
