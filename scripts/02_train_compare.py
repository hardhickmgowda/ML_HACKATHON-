"""Step 2 (Debasditya): five models, GroupKFold-by-trip CV on the TRAIN partition only, save fitted models.
student_test is never loaded here (no leakage); selection uses CV only."""
import json, time, glob, joblib, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from _common import load_config, resolve, set_seed, get_logger
from pathsnap.features import FEATURES
from pathsnap.models import MODEL_NAMES, PRETTY, FAMILY, make_model, sample_points
from pathsnap.metrics import classifier_metrics

cfg = load_config(); set_seed(cfg["seed"]); log = get_logger(); tc = cfg["training"]; seed = cfg["seed"]
cache = resolve(cfg["paths"]["artifacts"], "cache"); res = resolve(cfg["paths"]["results"]); res.mkdir(exist_ok=True)
mdir = resolve(cfg["paths"]["artifacts"], "models"); mdir.mkdir(parents=True, exist_ok=True)

summ = pd.read_csv(res / "dataset_summary.csv", dtype={"trip_id": str})
tr = summ[summ.split == "train"]
good = tr[tr.labeled_frac >= cfg["labels"]["min_labeled_frac"]].trip_id.tolist()
bad = tr[tr.labeled_frac < cfg["labels"]["min_labeled_frac"]].trip_id.tolist()
log.info("train trips: %d usable, %d excluded as noisy-label (labelled frac < %.2f): %s", len(good), len(bad),
         cfg["labels"]["min_labeled_frac"], bad)
df = pd.concat([joblib.load(cache / f"train_{t}.pkl") for t in good], ignore_index=True)
df = df[df.labeled]                                        # labelled points only

cvdf = sample_points(df, tc["cv_points_per_trip"], seed)
log.info("CV sample: %d points / %d rows / %d trips", cvdf.groupby(['trip_id','pt']).ngroups, len(cvdf), cvdf.trip_id.nunique())
gkf = GroupKFold(n_splits=tc["cv_folds"])
folds = list(gkf.split(cvdf, groups=cvdf["trip_id"]))

# ---- nearest-arc baseline (no learning): score = -distance
base = classifier_metrics(cvdf, np.clip(1 / (1 + cvdf["dist"].to_numpy()), 1e-6, 1 - 1e-6))
log.info("nearest-arc baseline CV-sample top1 = %.4f", base["top1_acc"])

rows, fold_rows = [], []
for name in MODEL_NAMES:
    ms = []
    t0 = time.time()
    for k, (a, b) in enumerate(folds):
        trn, val = cvdf.iloc[a], cvdf.iloc[b]
        if name == "knn":
            trn = sample_points(trn, 10**9, seed, max_rows=tc["knn_max_rows"])
        m = make_model(name, seed).fit(trn[FEATURES].to_numpy(np.float32), trn["y"])
        p = m.predict_proba(val[FEATURES].to_numpy(np.float32))[:, 1]
        r = classifier_metrics(val, p); r.update(model=name, fold=k); fold_rows.append(r); ms.append(r)
    agg = {k: float(np.mean([m[k] for m in ms])) for k in ["precision", "recall", "f1", "macro_f1", "roc_auc", "log_loss", "top1_acc", "miss_rate"]}
    agg.update({k + "_std": float(np.std([m[k] for m in ms])) for k in ["top1_acc", "macro_f1"]})
    agg.update(model=name, name=PRETTY[name], family=FAMILY[name], cv_seconds=round(time.time() - t0, 1))
    rows.append(agg)
    log.info("%-22s CV top1=%.4f macroF1=%.4f AUC=%.4f logloss=%.4f (%.0fs)", PRETTY[name], agg["top1_acc"], agg["macro_f1"], agg["roc_auc"], agg["log_loss"], agg["cv_seconds"])
cv = pd.DataFrame(rows)
cv.insert(0, "nearest_arc_baseline_top1", base["top1_acc"])
cv.to_csv(res / "cv_results.csv", index=False); pd.DataFrame(fold_rows).to_csv(res / "cv_folds.csv", index=False)

# ---- winner chosen on CV only (top-1 accuracy, then log-loss)
win = cv.sort_values(["top1_acc", "log_loss"], ascending=[False, True]).iloc[0].model
json.dump({"winner": win, "criterion": "CV top-1 accuracy (GroupKFold by trip_id), tie-break log-loss",
           "baseline_top1": base["top1_acc"], "train_trips": good, "excluded_trips": bad}, open(res / "selected.json", "w"), indent=2)
log.info("WINNER = %s", PRETTY[win])

# ---- refit every model on the full train sample and save .joblib
fit_df = sample_points(df, tc["train_points_per_trip"], seed)
for name in MODEL_NAMES:
    d = sample_points(fit_df, 10**9, seed, max_rows=tc["knn_max_rows"]) if name == "knn" else fit_df
    t0 = time.time()
    m = make_model(name, seed).fit(d[FEATURES].to_numpy(np.float32), d["y"])
    joblib.dump({"name": name, "model": m, "features": FEATURES, "rows": len(d)}, mdir / f"{name}.joblib", compress=3)
    log.info("saved %s.joblib (%d rows, %.0fs)", name, len(d), time.time() - t0)
