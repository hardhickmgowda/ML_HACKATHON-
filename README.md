# PathSnap – distributed map-matching & route reconstruction

GPS track + road network -> candidate arcs -> ML classifier -> HMM/Viterbi -> connected route.
Two-laptop deployment: **Laptop 1** runs the FastAPI backend (model + HMM), **Laptop 2** opens the browser UI.

## Layout
```
config.yaml            every tunable number (seeds, radii, training sizes, HMM defaults)
main.py                FastAPI backend (Laptop 1)                         [Pranav]
ui/index.html          Leaflet UI (Laptop 2, or served by the API at /)   [Pranav]
pathsnap/data_io.py    .track/.nodes/.arcs/.route parsers                 [Harsh]
pathsnap/network.py    directed graph, reverse arcs, bounded Dijkstra
pathsnap/preprocess.py outlier filter + median smoothing                  [Harsh]
pathsnap/candidates.py cKDTree candidate generation (50 m)                [Harsh]
pathsnap/features.py   19 GPS-candidate features                          [Harsh]
pathsnap/labels.py     monotone DP aligning .route to GPS -> 0/1 labels   [Harsh]
pathsnap/viz.py        Folium maps                                        [Harsh]
pathsnap/models.py     5 model families                                   [Debasditya]
pathsnap/metrics.py    classifier / point / route metrics
pathsnap/hmm.py        emission, transition, Viterbi, gap handling        [Hardhick]
pathsnap/pipeline.py   raw trip -> route, used by training AND the API
scripts/01..05_*.py    reproducible experiment steps
artifacts/             trained models (*.joblib), hmm_params.json, feature cache
results/               all CSV/JSON result tables + per-trip Folium maps
```

## Install
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Put the dataset in `data/` so it contains `train/` and `student_test/` (each `<id>/<id>.track|.nodes|.arcs|.route`).
On evaluation day put the 10 evaluation trips in `data/evaluation/<id>/` (the `.route` file is optional).
Override locations with `PATHSNAP_DATA_DIRS="/path/a:/path/b"` (`;` on Windows).

## Run the API (Laptop 1)
Trained artifacts are already in `artifacts/`, so no retraining is needed:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
Find Laptop 1's IP (`ipconfig` / `ip a`) and allow port 8000 through the firewall.
Endpoints: `GET /health`, `GET /trips`, `POST /match {"trip_id": "00000052"}`, `GET /match/{trip_id}`, `DELETE /cache`.
Each call runs parse -> clean -> candidates -> features -> classifier -> HMM live (no per-trip shortcuts; the reference
`.route` is never used for the prediction, only to score it when present).

## Run the UI (Laptop 2)
Open `http://<laptop1-ip>:8000/` in a browser (served by the API), **or** open `ui/index.html` locally and type the API
address. Pick a trip, press *Match route*. Layers: raw GPS, classifier-only, classifier+HMM, reference route.
The UI loads Leaflet and map tiles from the internet.

## Reproduce everything
```bash
./run_all.sh          # ~6 min on one CPU; seeds fixed (config.yaml: seed 42)
python tests/test_core.py
```
Student-test data is read only by `scripts/05_evaluate.py`; training, model selection and tuning use `train/` only.
