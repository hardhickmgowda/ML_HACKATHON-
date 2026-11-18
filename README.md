# PathSnap — Distributed Map-Matching & Route Reconstruction

UE24CS352A Machine Learning Hackathon, PES University.

## Repository layout

```
pathsnap/            Python package
  config.py          all thresholds / paths (single source of truth)
  data/              loading, validation, cleaning, candidates, labels, features
  eval/metrics.py    shared metrics (Top-1, macro F1, log loss, miss rate, continuity)
  viz/maps.py        Folium map builder (EDA, report, UI)
scripts/             runnable entry points
tests/               pytest suite
docs/DATA.md         dataset documentation + EDA findings
results/             generated tables, plots, maps (committed)
data_processed/      generated CSV tables (git-ignored, rebuild with scripts)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place the organiser data at `./Student folder/` (or set `PATHSNAP_DATA=/path/to/data`).

## Data pipeline

```bash
python scripts/validate_data.py   # raw-data checks   -> results/data_validation/
python scripts/build_dataset.py   # model-ready data  -> data_processed/<partition>/
python scripts/eda.py             # EDA tables/plots/maps -> results/eda/
python -m pytest -q               # tests
```

Loading data in your own code:

```python
from pathsnap.data.dataset import load_split, cv_folds
from pathsnap.eval.metrics import evaluate

train = load_split("train")              # reliable labels only
test = load_split("student_test")
for fit_idx, val_idx in cv_folds(train): # 5 folds, grouped by trip, seed 42
    model.fit(train.X.iloc[fit_idx], train.y.iloc[fit_idx])
    ...
model.fit(train.X, train.y)
proba = model.predict_proba(test.X)[:, 1]
print(evaluate(test.frame, proba, partition="student_test"))
# -> accuracy, precision, recall, macro_f1, log_loss, confusion_matrix,
#    top1_accuracy, candidate_miss_rate, route_continuity

# inference path (no labels, identical preprocessing) - for the API
from pathsnap.data import load_trip
from pathsnap.data.pipeline import process_trip
p = process_trip(load_trip("student_test", "00000000").without_route())
```

See `docs/DATA.md` for the dataset description, quality issues and design decisions.
