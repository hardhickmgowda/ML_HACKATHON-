"""Step 4 (Hardhick): Bayesian (Optuna TPE) tuning of the HMM emission/transition parameters on TRAIN trips using
out-of-fold classifier scores. Objective: mean of point top-1 accuracy and route F1 vs the reference route."""
import json, time, joblib, numpy as np, optuna
from _common import load_config, resolve, set_seed, get_logger
from pathsnap.data_io import load_trip, discover_trips
from pathsnap.dataset import prepare_trip
from pathsnap.hmm import HMMParams
from pathsnap.pipeline import run_decoding, evaluate_against_reference

cfg = load_config(); set_seed(cfg["seed"]); log = get_logger()
art = resolve(cfg["paths"]["artifacts"]); res = resolve(cfg["paths"]["results"])
oof = joblib.load(art / "oof_hmm.pkl"); folders = discover_trips([resolve(cfg["paths"]["data_root"]) / cfg["paths"]["train_dir"]])
prep = {t: prepare_trip(load_trip(folders[t]), cfg) for t in oof}
log.info("HMM tuning on %d train trips (%d points)", len(prep), sum(p.ct.n for p in prep.values()))

def score(params: HMMParams):
    t1, rf = [], []
    for t, p in prep.items():
        d = run_decoding(p, oof[t], params, cfg)
        m = evaluate_against_reference(p, d["clf_arc"], d["hmm"].arc, d["clf_route_raw"], d["hmm_route_raw"], d["hmm_route"], d["clf_route_filled"], cfg)
        if m and "top1_hmm" in m:
            t1.append(m["top1_hmm"]); rf.append(m["route_f1_hmm"])
    return float(np.mean(t1)), float(np.mean(rf))

# reference: classifier only (OOF) + default HMM
base_t1 = np.mean([evaluate_against_reference(p, (c := run_decoding(p, oof[t], HMMParams.from_dict(cfg["hmm"]), cfg))["clf_arc"], c["hmm"].arc,
                   c["clf_route_raw"], c["hmm_route_raw"], c["hmm_route"], c["clf_route_filled"], cfg)["top1_clf"] for t, p in prep.items()])
d_t1, d_rf = score(HMMParams.from_dict(cfg["hmm"]))
log.info("OOF classifier-only top1=%.4f | default HMM top1=%.4f routeF1=%.4f", base_t1, d_t1, d_rf)

def objective(tr):
    p = HMMParams.from_dict({**cfg["hmm"], "alpha": tr.suggest_float("alpha", 0.2, 1.0), "sigma_m": tr.suggest_float("sigma_m", 5, 40),
        "beta0_m": tr.suggest_float("beta0_m", 2, 30), "beta_per_s": tr.suggest_float("beta_per_s", 0.0, 3.0),
        "invalid_penalty": tr.suggest_float("invalid_penalty", 5, 60), "uturn_penalty": tr.suggest_float("uturn_penalty", 0, 20)})
    a, b = score(p)
    return 0.5 * (a + b)

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg["seed"], n_startup_trials=8))
t0 = time.time()
for i in range(30):
    study.optimize(objective, n_trials=1)
    log.info("trial %2d value=%.4f best=%.4f (%.0fs)", i + 1, study.trials[-1].value, study.best_value, time.time() - t0)
best = HMMParams.from_dict({**cfg["hmm"], **study.best_params})
b_t1, b_rf = score(best)
json.dump(best.to_dict(), open(art / "hmm_params.json", "w"), indent=2)
json.dump({"oof_classifier_top1": base_t1, "default_hmm": {"top1": d_t1, "route_f1": d_rf}, "tuned_hmm": {"top1": b_t1, "route_f1": b_rf},
           "best_params": study.best_params}, open(res / "hmm_tuning.json", "w"), indent=2)
log.info("tuned HMM top1=%.4f routeF1=%.4f  params=%s", b_t1, b_rf, study.best_params)
