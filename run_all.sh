#!/usr/bin/env bash
# Full reproducible pipeline (about 6 minutes on 1 CPU). Run from the repo root.
set -euo pipefail
python scripts/01_build_dataset.py   # parse + candidates + features + labels (train AND student_test cached; test never used for fitting)
python scripts/02_train_compare.py   # 5 models, GroupKFold-by-trip CV, saves artifacts/models/*.joblib, picks winner by CV only
python scripts/03_tune_best.py       # Optuna TPE on the winner, saves artifacts/model.joblib + OOF scores
python scripts/04_tune_hmm.py        # Optuna TPE on HMM params using OOF scores, saves artifacts/hmm_params.json
python scripts/05_evaluate.py        # FIRST and only use of student_test: tables + folium maps in results/
python scripts/06_report_figures.py  # extra experiments + every figure/table used in the technical report -> results/report/
