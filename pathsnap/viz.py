"""Folium visualisation: raw GPS vs ground-truth / predicted paths (Harsh)."""
from __future__ import annotations
from pathlib import Path
import folium


def make_map(track_latlon, layers: dict, out_html: str | Path | None = None, title: str = "") -> folium.Map:
    """layers: {name: (polyline [[lat,lon],...], colour, dash_array_or_None)}."""
    c = track_latlon[len(track_latlon) // 2]
    m = folium.Map(location=c, zoom_start=14, tiles="OpenStreetMap", control_scale=True)
    raw = folium.FeatureGroup(name="Raw GPS", show=True)
    step = max(1, len(track_latlon) // 2500)
    for p in track_latlon[::step]:
        folium.CircleMarker(p, radius=1.5, color="#555", weight=1, fill=True, fill_opacity=0.6).add_to(raw)
    raw.add_to(m)
    for name, (line, colour, dash) in layers.items():
        if line:
            fg = folium.FeatureGroup(name=name, show=True)
            folium.PolyLine(line, color=colour, weight=4, opacity=0.85, dash_array=dash, tooltip=name).add_to(fg)
            fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    if title:
        m.get_root().html.add_child(folium.Element(f"<h4 style='position:fixed;top:8px;left:60px;z-index:9999;background:#fff;padding:4px 8px'>{title}</h4>"))
    if out_html:
        Path(out_html).parent.mkdir(parents=True, exist_ok=True)
        m.save(str(out_html))
    return m


def save_trip_map(result: dict, out_html: str | Path) -> None:
    layers = {"Reference route": (result.get("reference_route"), "#1a9850", None),
              "Classifier-only": (result["classifier_route"], "#f39c12", "6,6"),
              "Classifier + HMM": (result["route"], "#2166ac", None)}
    make_map(result["track"], layers, out_html, title=f"Trip {result['trip_id']}")
