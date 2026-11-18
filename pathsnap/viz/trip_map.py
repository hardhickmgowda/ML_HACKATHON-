"""Folium visualisation of one matched trip: raw GPS, nearby road network, reference, classifier-only and HMM routes (Harsh)."""
from __future__ import annotations
from pathlib import Path
import folium
import numpy as np
from scipy.spatial import cKDTree

LEGEND = """<div style="position:fixed;bottom:28px;left:12px;z-index:9999;background:#fff;padding:8px 12px;
 border:1px solid #888;border-radius:6px;font:13px sans-serif"><b>{title}</b><br>{rows}</div>"""


def _row(colour, label, dot=False, dash=False):
    shape = "border-radius:50%;width:9px;height:9px;" if dot else "width:24px;height:0;border-top:4px {} {};".format(
        "dashed" if dash else "solid", colour)
    bg = f"background:{colour};" if dot else ""
    return f'<span style="display:inline-block;{shape}{bg}margin-right:6px;vertical-align:middle"></span>{label}<br>'


def _network_segments(net, xy, buffer_m=120.0, max_arcs=15000):
    """Arcs whose midpoint lies within buffer_m of the GPS track, as [[lat,lon],[lat,lon]] segments."""
    mid = (net.arc_a + net.arc_b) / 2
    d, _ = cKDTree(xy).query(mid, distance_upper_bound=buffer_m + 200)
    keep = np.flatnonzero(d <= buffer_m + np.minimum(net.arc_len, 400) / 2)[:max_arcs]
    lo_a, la_a = net.proj.inv(net.arc_a[keep, 0], net.arc_a[keep, 1])
    lo_b, la_b = net.proj.inv(net.arc_b[keep, 0], net.arc_b[keep, 1])
    return [[[a, b], [c, e]] for a, b, c, e in zip(la_a, lo_a, la_b, lo_b)]


def make_map(track_latlon, layers: dict, out_html: str | Path | None = None, title: str = "", network=None) -> folium.Map:
    """layers: {name: (polyline [[lat,lon],...], colour, dash_array_or_None, weight)}; network: list of segments."""
    tr = np.asarray(track_latlon)
    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([tr.min(0).tolist(), tr.max(0).tolist()], padding=(20, 20))
    rows = []
    if network:
        fg = folium.FeatureGroup(name=f"Road network ({len(network)} arcs near the track)", show=True)
        folium.PolyLine(network, color="#8c8c8c", weight=1.5, opacity=0.7).add_to(fg)
        fg.add_to(m); rows.append(_row("#8c8c8c", "Road network (nearby arcs)"))
    for name, (line, colour, dash, weight) in layers.items():
        if line:
            fg = folium.FeatureGroup(name=name, show=True)
            folium.PolyLine(line, color=colour, weight=weight, opacity=0.85, dash_array=dash, tooltip=name).add_to(fg)
            fg.add_to(m); rows.append(_row(colour, name, dash=bool(dash)))
    raw = folium.FeatureGroup(name="Raw GPS observations", show=True)
    step = max(1, len(tr) // 2500)
    for p in tr[::step]:
        folium.CircleMarker(p.tolist(), radius=2, color="#d62728", weight=0, fill=True, fill_opacity=0.8).add_to(raw)
    raw.add_to(m); rows.append(_row("#d62728", "Raw GPS observations", dot=True))
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(LEGEND.format(title=title, rows="".join(rows))))
    if out_html:
        Path(out_html).parent.mkdir(parents=True, exist_ok=True)
        m.save(str(out_html))
    return m


def save_trip_map(result: dict, out_html: str | Path) -> None:
    """result: output of PathSnapPipeline.run (uses the private _prepared entry for the road network)."""
    p = result.get("_prepared")
    network = _network_segments(p.net, p.ct.xy) if p is not None else None
    layers = {"Reference route": (result.get("reference_route"), "#1a9850", None, 8),
              "Classifier-only route": (result["classifier_route"], "#f39c12", "6,6", 4),
              "Classifier + HMM route": (result["route"], "#2166ac", None, 4)}
    make_map(result["track"], layers, out_html, title=f"Trip {result['trip_id']}", network=network)
