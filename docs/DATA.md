# PathSnap — Dataset documentation & data pipeline

Owner: Data Engineering & Preprocessing. All numbers below are produced by
`scripts/validate_data.py`, `scripts/build_dataset.py` and `scripts/eda.py`
(see `results/data_validation/` and `results/eda/`).

## 1. How the raw data is organised

```
Student folder/
  train/<trip_id>/<trip_id>.{track,nodes,arcs,route}         80 trips
  student_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}  10 trips
```
(Evaluator-testing trips: 10, not distributed to us.)

| File | Columns (no header, whitespace) | Meaning |
|---|---|---|
| `.track` | lon, lat, t (s) | GPS observations, ~1 Hz |
| `.nodes` | lon, lat | road-network nodes; row index = node_id |
| `.arcs`  | from_node_id, to_node_id | **directed** road segments; row index = arc_id |
| `.route` | arc_id | reference route, arcs in travel order |

Relationships: `route -> arc_id -> (u, v) -> node lon/lat`. **There is no
row-to-row correspondence between `.track` and `.route`**: a trip with 1 315
GPS points can have a 525-arc route. Each trip ships its *own* local road
network, so ids are only meaningful within a trip. Trips are spread across six
continents (lon −100…137, lat −34…61).

## 2. Inventory (`results/eda/partition_summary.csv`)

| | train | student_test |
|---|---|---|
| trips | 80 | 10 |
| GPS points total | 208 417 | 19 160 |
| points / trip (median, min–max) | 1 529 (361–19 313) | 1 420 (262–3 958) |
| network nodes / trip (median, max) | 9 608 (155 003) | 6 597 (26 635) |
| route arcs / trip (median) | 280 | 250 |
| driven distance | 2 979 km | 297 km |

## 3. Data quality findings

| Check | Result | Handling |
|---|---|---|
| Missing values / unparsable rows | **0** in every file of every trip (`missing_values.csv`) | loader still coerces & the cleaner drops invalid rows defensively |
| Files present | all 4 files in all 90 trips | loader raises if track/nodes/arcs missing |
| Timestamp order | strictly increasing in all trips | cleaner sorts + de-duplicates anyway |
| Sampling | 96.5 % of steps exactly 1 s; 42 trips have gaps > 10 s, up to 42 h (trip 90) | trajectory split into **segments** at gaps > 10 s or jumps > 300 m (`segment_id`, `is_segment_start`) |
| Speed spikes | 11 trips have steps > 70 m/s (252 km/h) | isolated spikes removed; see "timestamp bursts" below |
| Timestamp bursts | e.g. trip 13: after a short dropout, ~4 buffered fixes arrive with compressed timestamps → apparent 80–140 m/s | positions are fine, so points are kept; speed feature uses a rolling median |
| Stationary points | duplicated consecutive positions (e.g. 402 in trip 54) | flagged `is_stationary`; heading carried forward through stops |
| Duplicate arcs | 60 / 90 trips contain identical (u,v) arcs (up to 322) | collapsed onto a canonical arc id (`canon_arc`); compare canonical ids when scoring |
| Self-loops | a few zero-length u==v arcs | excluded from candidate index |
| Route continuity | 11 trips have a few breaks (to-node ≠ next from-node); overall 99.9 % continuous | noted; alignment does not assume continuity |
| **Route only partly matches GPS** | trips **54, 33, 2, 43** have > 5 % of points > 50 m from the route | points get `label_valid=False` and are excluded from training/scoring |

The partial-match trips come in two kinds (diagnosed in EDA):
* **GPS bias** — trip 54: from ~10 % into the trip the track is shifted ~138 m
  north and only 1 % of those points lie within 15 m of *any* road. The
  reference route itself looks right; the GPS is offset. Trip 2 is similar at
  its start and end.
* **Incomplete route** — trip 33: 89 % of the far points lie on real roads
  that the route simply does not contain (track 47.8 km vs route 6.4 km).

So the reference route is **not error-free ground truth**; we treat it as
reference labels and report how many points we trust (88.8 % of train points,
99.9 % of student-test points).

## 4. GPS noise (`gps_noise_stats.csv`, plot `02_gps_noise_radius.png`)

Distance from GPS point to its reference arc (points with reliable labels):

| | median | 90 % | 95 % | 99 % |
|---|---|---|---|---|
| train | 4.9 m | 13.2 m | 17.5 m | 34.6 m |
| student_test | 5.2 m | 13.4 m | 16.7 m | 27.0 m |

Train and student-test noise distributions match closely. This sets the
**candidate radius (60 m)**: 99 % of reliably-labelled train points are within
35 m of their arc, and 60 m leaves headroom while keeping ~6 candidates per
point. (Points > 50 m are by definition `label_valid=False`, so the miss-rate
curve in the plot is shown up to that cap.) The median/σ is also the natural
starting point for the HMM emission noise.

## 5. Coordinate handling

Lon/lat are angular, so each trip is projected to a **local equirectangular
plane** centred on its GPS centroid (`pathsnap/data/geo.py`). Over a trip's
extent (≤ ~60 km) the distance error is < 1 %, far below GPS noise, and unlike
UTM it works for every trip without zone handling. Verified against haversine
in `tests/`. Coordinate order (lon, lat) is checked: the GPS median must lie
inside the network's bounding box.

## 6. Pipeline

```
raw trip ─► clean_track ─► RoadNetwork index ─► candidates ─► features ─► (labels)
```

1. **Cleaning** (`clean.py`): drop invalid rows, sort, de-duplicate timestamps,
   remove isolated speed spikes, split into segments at gaps, compute speed and a
   windowed heading. `orig_idx` maps every point back to its raw row.
2. **Candidates** (`network.py`): arcs are densified every 10 m into a KD-tree;
   every arc within 60 m of a point is a candidate (max 8, nearest first). If
   none is within 60 m, the nearest arcs are used and marked `fallback=1`.
3. **Labels** (`labels.py`, training only): the route is aligned to the GPS
   points with a monotonic dynamic programme (vehicle never moves backwards along
   the route), giving one reference arc per point. A candidate's `label` = 1 if
   its canonical arc equals that point's reference arc.
4. **Features** (`features.py`, `FEATURE_COLUMNS`): distance (absolute, rank,
   relative to nearest), position along arc, heading difference, two-way flag,
   arc length, node degrees, speed, gap/stationary flags, and whether the
   candidate continues the neighbouring points' nearest arc.

Features never read the route (enforced by `test_features_do_not_depend_on_route`).
The API runs exactly `process_trip(trip.without_route())`.

## 7. Key EDA insight for modelling (`baselines.csv`, `04_feature_separation.png`)

| Baseline Top-1 (reliable points) | train | student_test |
|---|---|---|
| nearest arc only | 55.6 % | 53.9 % |
| nearest arc, direction resolved by heading | 80.6 % | 81.3 % |
| candidate miss rate (upper bound on Top-1 = 1 − miss) | 0.85 % | 0.28 % |
| route continuity of nearest-arc predictions | – | 42.8 % |

Most two-way streets are stored as two arcs (u→v, v→u) with *identical*
geometry, so distance alone is a coin-flip on direction. Heading difference
fixes this, which is why `heading_diff_deg` is one of the strongest features
(corr −0.41 with the label, next to `dist_rank` −0.47). The remaining gap
(~81 % → ~99 %) is what the classifier + HMM must close: junctions, parallel
roads and GPS bias. Point-wise choices are also disconnected: only 42.8 % of
arc changes in the nearest-arc baseline are between connected arcs, which is
exactly what HMM/Viterbi decoding is meant to fix.

## 8. Output tables for the modelling team

`data_processed/<partition>/candidates.csv` — one row per (point, candidate):
`trip_id, point_pos, arc_id, canon_arc, <FEATURE_COLUMNS>, label, label_valid`

`data_processed/<partition>/points.csv` — one row per cleaned observation:
`trip_id, point_pos, orig_idx, lon, lat, t, x, y, segment_id, speed_mps,
heading_deg, …, true_arc, true_dist_m, label_valid`

Rules for using them:
* Train only on rows with `label_valid == True`.
* Cross-validate with **GroupKFold on `trip_id`** (never split a trip).
* Class balance: ~16 % positives (≈ 6 candidates per point).
* Use `pathsnap.data.dataset.load_split / cv_folds` and `pathsnap.eval.metrics.evaluate`
  so every model is scored identically (Top-1 = per point, max-score candidate vs label).
