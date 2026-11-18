"""Step 3 (Debasditya): Bayesian hyper-parameter optimisation (Optuna TPE) of the CV winner with group-aware CV
on TRAIN trips only; refit on the train sample; save artifacts/model.joblib; write out-of-fold scores for HMM tuning."""
import json, time, joblib, numpy as np, pandas as pd, optuna
from sklearn.model_selection import GroupKFold
from _common import load_config, resolve, set_seed, get_logger
from pathsnap.features import FEATURES
from pathsnap.models import PRETTY, make_model, sample_points
from pathsnap.metrics import classifier_metrics

cfg = load_config(); set_seed(cfg["seed"]); log = get_logger(); tc = cfg["training"]; seed = cfg["seed"]
cache = resolve(cfg["paths"]["artifacts"], "cache"); res = resolve(cfg["paths"]["results"]); art = resolve(cfg["paths"]["artifacts"])
sel = json.load(open(res / "selected.json")); name = sel["winner"]; good = sel["train_trips"]
log.info("tuning winner: %s on %d train trips", PRETTY[name], len(good))
full = {t: joblib.load(cache / f"train_{t}.pkl") for t in good}
df = pd.concat(full.values(), ignore_index=True); df = df[df.labeled]
tune_df = sample_points(df, tc["optuna_points_per_trip"], seed + 1)
folds = list(GroupKFold(n_splits=tc["optuna_folds"]).split(tune_df, groups=tune_df["trip_id"]))

def space(tr):
    if name == "hgb":
        return dict(max_iter=tr.suggest_int("max_iter", 100, 500, step=50), learning_rate=tr.suggest_float("learning_rate", 0.02, 0.3, log=True),
                    max_leaf_nodes=tr.suggest_int("max_leaf_nodes", 8, 96), min_samples_leaf=tr.suggest_int("min_samples_leaf", 10, 200),
                    l2_regularization=tr.suggest_float("l2_regularization", 1e-3, 30, log=True), max_depth=tr.suggest_int("max_depth", 3, 14))
    if name == "rf":
        return dict(n_estimators=tr.suggest_int("n_estimators", 60, 200, step=20), max_depth=tr.suggest_int("max_depth", 6, 24),
                    min_samples_leaf=tr.suggest_int("min_samples_leaf", 1, 30), max_features=tr.suggest_float("max_features", 0.2, 1.0))
    if name == "mlp":
        return dict(hidden=(tr.suggest_int("h1", 32, 128, step=32), tr.suggest_int("h2", 16, 64, step=16)),
                    alpha=tr.suggest_float("alpha", 1e-6, 1e-2, log=True), lr=tr.suggest_float("lr", 1e-4, 1e-2, log=True))
    if name == "knn":
        return dict(n_neighbors=tr.suggest_int("n_neighbors", 5, 100))
    return dict(C=tr.suggest_float("C", 1e-3, 100, log=True))

def objective(tr):
    params = space(tr); s = []
    for a, b in folds:
        m = make_model(name, seed, params).fit(tune_df.iloc[a][FEATURES].to_numpy(np.float32), tune_df.iloc[a]["y"])
        p = m.predict_proba(tune_df.iloc[b][FEATURES].to_numpy(np.float32))[:, 1]
        s.append(classifier_metrics(tune_df.iloc[b], p)["macro_f1"])
    return float(np.mean(s))

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=6))
t0 = time.time()
for i in range(tc["optuna_trials"]):
    study.optimize(objective, n_trials=1)
    log.info("trial %2d  macroF1=%.4f  best=%.4f  (%.0fs)", i + 1, study.trials[-1].value, study.best_value, time.time() - t0)
best = study.best_params
if name == "mlp":
    best = dict(hidden=(best.pop("h1"), best.pop("h2")), **best)
log.info("best params: %s", best)
default_cv = float(pd.read_csv(res / "cv_results.csv").set_index("model").loc[name, "macro_f1"])
json.dump({"model": name, "best_params": best, "best_cv_macro_f1": study.best_value, "default_cv_macro_f1_other_sample": default_cv,
           "n_trials": len(study.trials)}, open(res / "tuning_best.json", "w"), indent=2, default=list)
pd.DataFrame([dict(number=t.number, value=t.value, **t.params) for t in study.trials]).to_csv(res / "tuning_trials.csv", index=False)

# ---- refit on the full train sample, save as the deployed model
fit_df = sample_points(df, tc["train_points_per_trip"], seed)
final = make_model(name, seed, best).fit(fit_df[FEATURES].to_numpy(np.float32), fit_df["y"])
joblib.dump({"name": name, "model": final, "features": FEATURES, "rows": len(fit_df), "params": best}, art / "model.joblib", compress=3)
log.info("saved artifacts/model.joblib")

# ---- out-of-fold scores (all rows of 30 smaller trips) for HMM tuning: model never sees the trip it scores
summ = pd.read_csv(res / "dataset_summary.csv", dtype={"trip_id": str}); summ = summ[summ.trip_id.isin(good) & (summ.split == "train")]
cand = summ[(summ.n_points <= 3000) & (summ.n_arcs <= 80000)].sort_values("trip_id")
rng = np.random.default_rng(seed); hmm_trips = sorted(rng.choice(cand.trip_id.values, size=min(30, len(cand)), replace=False).tolist())
trips_arr = np.array(good); oof = {}
for k, (a, b) in enumerate(GroupKFold(n_splits=4).split(trips_arr, groups=trips_arr)):
    trn_trips = set(trips_arr[a]); held = [t for t in trips_arr[b] if t in hmm_trips]
    d = sample_points(df[df.trip_id.isin(trn_trips)], tc["train_points_per_trip"], seed)
    m = make_model(name, seed, best).fit(d[FEATURES].to_numpy(np.float32), d["y"])
    for t in held:
        oof[t] = m.predict_proba(full[t][FEATURES].to_numpy(np.float32))[:, 1]
joblib.dump(oof, art / "oof_hmm.pkl", compress=3); json.dump(hmm_trips, open(res / "hmm_tune_trips.json", "w"))
log.info("OOF scores for %d trips saved", len(oof))
