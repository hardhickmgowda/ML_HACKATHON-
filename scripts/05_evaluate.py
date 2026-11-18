"""Step 5 (Hardhick + Debasditya): FINAL evaluation on student_test -- the first time it is used.
(a) five frozen models (fitted on train only) + the tuned winner, scored on classifier metrics;
(b) classifier-only vs classifier+HMM route reconstruction through the REAL inference pipeline (route file unused)."""
import json, time, joblib, numpy as np, pandas as pd
from _common import load_config, resolve, set_seed, get_logger
from pathsnap.data_io import load_trip, discover_trips
from pathsnap.dataset import prepare_trip
from pathsnap.models import MODEL_NAMES, PRETTY, FAMILY
from pathsnap.metrics import classifier_metrics
from pathsnap.pipeline import PathSnapPipeline, score_trip
from pathsnap.viz import save_trip_map

cfg = load_config(); set_seed(cfg["seed"]); log = get_logger()
art = resolve(cfg["paths"]["artifacts"]); res = resolve(cfg["paths"]["results"]); (res / "maps").mkdir(parents=True, exist_ok=True)
folders = discover_trips([resolve(cfg["paths"]["data_root"]) / cfg["paths"]["test_dir"]])
log.info("student_test trips: %s", list(folders))
prep = {t: prepare_trip(load_trip(f), cfg) for t, f in folders.items()}
df = pd.concat([p.df for p in prep.values()], ignore_index=True); dfl = df[df.labeled].reset_index(drop=True)
n_pts = dfl.groupby(["trip_id", "pt"]).ngroups
cv = pd.read_csv(res / "cv_results.csv").set_index("model")

rows = []
bundles = {n: joblib.load(art / "models" / f"{n}.joblib") for n in MODEL_NAMES}
bundles["tuned"] = joblib.load(art / "model.joblib")
for n, b in bundles.items():
    t0 = time.time(); p = score_trip(b, dfl); dt = time.time() - t0
    m = classifier_metrics(dfl, p)
    base = n if n != "tuned" else b["name"]
    rows.append(dict(model=PRETTY[base] + (" (Bayes-tuned)" if n == "tuned" else ""), family=FAMILY[base],
                     cv_macro_f1=cv.loc[base, "macro_f1"] if n != "tuned" else json.load(open(res / "tuning_best.json"))["best_cv_macro_f1"],
                     test_top1=m["top1_acc"], test_macro_f1=m["macro_f1"], test_precision=m["precision"], test_recall=m["recall"],
                     test_f1=m["f1"], test_roc_auc=m["roc_auc"], test_log_loss=m["log_loss"], candidate_miss_rate=m["miss_rate"],
                     infer_s_per_1k_points=1000 * dt / n_pts))
base_m = classifier_metrics(dfl, np.clip(1 / (1 + dfl["dist"].to_numpy()), 1e-6, 1 - 1e-6))
rows.insert(0, dict(model="Nearest-arc baseline (no ML)", family="Rule", test_top1=base_m["top1_acc"], test_macro_f1=base_m["macro_f1"],
                    candidate_miss_rate=base_m["miss_rate"]))
tab = pd.DataFrame(rows); tab.to_csv(res / "student_test_models.csv", index=False)
log.info("\n%s", tab[["model", "test_top1", "test_macro_f1", "test_roc_auc", "test_log_loss"]].round(4).to_string(index=False))

# ---- route reconstruction through the production pipeline
pipe = PathSnapPipeline(cfg)
per = []
for tid, folder in folders.items():
    r = pipe.run(folder)
    m = r["metrics"]; per.append(dict(trip_id=tid, n_points=r["n_points"], **m, restarts=r["hmm"]["restarts"], teleports=r["hmm"]["teleports"],
                                      seconds=r["timing"]["total_s"], hmm_seconds=r["timing"]["hmm_s"]))
    save_trip_map(r, res / "maps" / f"{tid}.html")
    log.info("trip %s top1 clf=%.3f hmm=%.3f | routeF1 clf=%.3f hmm=%.3f | %.1fs", tid, m["top1_clf"], m["top1_hmm"], m["route_f1_clf"], m["route_f1_hmm"], r["timing"]["total_s"])
pt = pd.DataFrame(per); pt.to_csv(res / "student_test_routes_per_trip.csv", index=False)
w = pt.n_labeled_points
agg = {c: float(np.average(pt[c], weights=w)) for c in ["top1_clf", "top1_hmm"]}
agg.update({c: float(pt[c].mean()) for c in pt.columns if c.startswith(("continuity", "route_"))})
json.dump(agg, open(res / "student_test_routes_summary.json", "w"), indent=2)
log.info("SUMMARY %s", json.dumps({k: round(v, 4) for k, v in agg.items()}))
