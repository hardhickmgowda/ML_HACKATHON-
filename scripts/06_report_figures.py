"""Step 6: extra experiments + every figure used in the technical report.

Everything here is computed from the saved artifacts and the cached train/student_test tables; nothing is typed in.
Outputs -> results/report/ (CSV + PNG)

  (a) per-model student-test route reconstruction: classifier-only vs classifier+HMM, route continuity
  (b) per-model confusion matrices on student_test (candidate rows, threshold 0.5)
  (c) like-for-like tuned vs untuned HGB: both configurations on the SAME CV folds (comparison folds and Optuna folds)
  (d) real training histories: MLP (per-epoch train/val log loss, trip-grouped hold-out) and HGB (per boosting iteration)
  (e) comparison plots: CV macro-F1, student-test top-1 / macro-F1, log loss, inference time, Optuna history, HMM per trip
"""
import json, time, joblib, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from _common import load_config, resolve, set_seed, get_logger
from pathsnap.data_io import load_trip, discover_trips
from pathsnap.dataset import prepare_trip
from pathsnap.features import FEATURES
from pathsnap.hmm import HMMParams
from pathsnap.metrics import classifier_metrics
from pathsnap.models import MODEL_NAMES, PRETTY, make_model, sample_points
from pathsnap.pipeline import run_decoding, evaluate_against_reference, score_trip

cfg = load_config(); seed = cfg["seed"]; set_seed(seed); log = get_logger(); tc = cfg["training"]
art = resolve(cfg["paths"]["artifacts"]); res = resolve(cfg["paths"]["results"]); cache = art / "cache"
out = res / "report"; out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
SHORT = {"logreg": "LogReg", "knn": "KNN", "rf": "RF", "hgb": "HGB", "mlp": "MLP", "tuned": "HGB-tuned"}
C_MAIN, C_ALT, C_GREY = "#2a6fdb", "#e07a1f", "#9aa0a6"

bundles = {n: joblib.load(art / "models" / f"{n}.joblib") for n in MODEL_NAMES}
bundles["tuned"] = joblib.load(art / "model.joblib")
hmm_params = HMMParams.from_dict({**cfg["hmm"], **json.load(open(art / "hmm_params.json"))})

# ---------------------------------------------------------------- (a) + (b) student test, every model
folders = discover_trips([resolve(cfg["paths"]["data_root"]) / cfg["paths"]["test_dir"]])
prep = {t: prepare_trip(load_trip(f), cfg) for t, f in folders.items()}
route_rows, cms, flip_rows = [], {}, []
for n, b in bundles.items():
    ys, ps = [], []
    for tid, p in prep.items():
        prob = score_trip(b, p.df)
        d = run_decoding(p, prob, hmm_params, cfg)
        m = evaluate_against_reference(p, d["clf_arc"], d["hmm"].arc, d["clf_route_raw"], d["hmm_route_raw"],
                                       d["hmm_route"], d["clf_route_filled"], cfg)
        route_rows.append(dict(model=n, trip_id=tid, n_labeled_points=m["n_labeled_points"], **{k: m[k] for k in
                          ["top1_clf", "top1_hmm", "continuity_clf", "continuity_hmm_raw", "continuity_hmm_filled",
                           "route_f1_clf", "route_f1_hmm"]}, teleports=d["hmm"].teleports))
        lab = p.df["labeled"].to_numpy()
        ys.append(p.df["y"].to_numpy()[lab]); ps.append(prob[lab])
        if n == "tuned":                       # where did the HMM help / hurt at point level (deployed model)?
            ok = p.true_arc >= 0
            c_ok, h_ok = d["clf_arc"] == p.true_arc, d["hmm"].arc == p.true_arc
            flip_rows.append(dict(trip_id=tid, labeled=int(ok.sum()), both_right=int((ok & c_ok & h_ok).sum()),
                                  fixed_by_hmm=int((ok & ~c_ok & h_ok).sum()), broken_by_hmm=int((ok & c_ok & ~h_ok).sum()),
                                  both_wrong=int((ok & ~c_ok & ~h_ok).sum()), teleports=d["hmm"].teleports,
                                  unbridged_breaks=int(d["unbridged"]), restarts=d["hmm"].restarts))
    y, pr = np.concatenate(ys), np.concatenate(ps)
    cms[n] = confusion_matrix(y, (pr >= 0.5).astype(int), labels=[0, 1])
    log.info("%-6s done", n)
pd.DataFrame(flip_rows).to_csv(out / "hmm_point_flips_deployed.csv", index=False)
rt = pd.DataFrame(route_rows); rt.to_csv(out / "student_test_routes_per_model_per_trip.csv", index=False)
w = rt.n_labeled_points
agg = rt.groupby("model", sort=False).apply(lambda g: pd.Series({
    "top1_clf": np.average(g.top1_clf, weights=g.n_labeled_points), "top1_hmm": np.average(g.top1_hmm, weights=g.n_labeled_points),
    "continuity_clf": g.continuity_clf.mean(), "continuity_hmm_raw": g.continuity_hmm_raw.mean(),
    "continuity_hmm_filled": g.continuity_hmm_filled.mean(), "route_f1_clf": g.route_f1_clf.mean(),
    "route_f1_hmm": g.route_f1_hmm.mean(), "teleports": g.teleports.sum()}), include_groups=False)
agg.to_csv(out / "student_test_routes_per_model.csv")
log.info("\n%s", agg.round(4).to_string())
pd.DataFrame([dict(model=n, tn=c[0, 0], fp=c[0, 1], fn=c[1, 0], tp=c[1, 1]) for n, c in cms.items()]).to_csv(
    out / "confusion_matrices_student_test.csv", index=False)

fig, axs = plt.subplots(2, 3, figsize=(9, 5.6))
for ax, (n, c) in zip(axs.ravel(), cms.items()):
    cn = c / c.sum(axis=1, keepdims=True)
    ax.imshow(cn, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{c[i, j]:,}\n({cn[i, j]:.1%})", ha="center", va="center", fontsize=7.5,
                    color="white" if cn[i, j] > 0.6 else "black")
    ax.set_xticks([0, 1], ["pred: wrong arc", "pred: true arc"]); ax.set_yticks([0, 1], ["wrong arc", "true arc"])
    ax.set_title(PRETTY.get(n, "HGB (Bayes-tuned)") if n != "tuned" else "HGB (Bayes-tuned)", fontsize=9)
fig.suptitle("Confusion matrices on student_test candidate rows (threshold 0.5, row-normalised)")
fig.tight_layout(); fig.savefig(out / "fig_confusion_matrices.png"); plt.close(fig)

# ---------------------------------------------------------------- (c) like-for-like tuned vs untuned
sel = json.load(open(res / "selected.json")); good = sel["train_trips"]
df = pd.concat([joblib.load(cache / f"train_{t}.pkl") for t in good], ignore_index=True); df = df[df.labeled]
best = json.load(open(res / "tuning_best.json"))["best_params"]

def cv_eval(sample, n_splits, params):
    rows, t_fit, t_pred, n_pts = [], 0.0, 0.0, 0
    for a, b in GroupKFold(n_splits=n_splits).split(sample, groups=sample["trip_id"]):
        t0 = time.time(); m = make_model("hgb", seed, params).fit(sample.iloc[a][FEATURES].to_numpy(np.float32), sample.iloc[a]["y"]); t_fit += time.time() - t0
        t0 = time.time(); p = m.predict_proba(sample.iloc[b][FEATURES].to_numpy(np.float32))[:, 1]; t_pred += time.time() - t0
        r = classifier_metrics(sample.iloc[b], p); rows.append(r); n_pts += r["n_points"]
    return dict(cv_macro_f1=np.mean([r["macro_f1"] for r in rows]), cv_macro_f1_std=np.std([r["macro_f1"] for r in rows]),
                cv_top1=np.mean([r["top1_acc"] for r in rows]), cv_log_loss=np.mean([r["log_loss"] for r in rows]),
                fit_seconds=t_fit, infer_s_per_1k_points=1000 * t_pred / n_pts)

samples = {"comparison folds (5x, 120 pts/trip, seed 42)": (sample_points(df, tc["cv_points_per_trip"], seed), tc["cv_folds"]),
           "Optuna folds (4x, 100 pts/trip, seed 43)": (sample_points(df, tc["optuna_points_per_trip"], seed + 1), tc["optuna_folds"])}
tv = []
for sname, (s, k) in samples.items():
    for cfgname, params in (("untuned (default)", {}), ("Bayes-tuned", best)):
        tv.append(dict(cv_sample=sname, config=cfgname, **cv_eval(s, k, params)))
        log.info("%s | %s | %s", sname, cfgname, {k2: round(v, 4) for k2, v in tv[-1].items() if isinstance(v, float)})
pd.DataFrame(tv).to_csv(out / "tuned_vs_untuned_cv.csv", index=False)

# ---------------------------------------------------------------- (d0) in-sample vs CV vs test (over/underfitting evidence)
fit_df = sample_points(df, tc["train_points_per_trip"], seed)
cvr = pd.read_csv(res / "cv_results.csv").set_index("model")
stm = pd.read_csv(res / "student_test_models.csv"); stm = stm[stm.family != "Rule"].reset_index(drop=True); stm.index = MODEL_NAMES + ["tuned"]
fit_rows = []
for n, b in bundles.items():
    m_in = classifier_metrics(fit_df, score_trip(b, fit_df))
    fit_rows.append(dict(model=n, train_rows=b["rows"], train_top1=m_in["top1_acc"], train_macro_f1=m_in["macro_f1"], train_log_loss=m_in["log_loss"],
                         cv_top1=cvr.loc[n, "top1_acc"] if n in cvr.index else np.nan, cv_macro_f1=cvr.loc[n, "macro_f1"] if n in cvr.index else np.nan,
                         cv_log_loss=cvr.loc[n, "log_loss"] if n in cvr.index else np.nan, cv_seconds=cvr.loc[n, "cv_seconds"] if n in cvr.index else np.nan,
                         test_top1=stm.loc[n, "test_top1"], test_macro_f1=stm.loc[n, "test_macro_f1"], test_log_loss=stm.loc[n, "test_log_loss"]))
pd.DataFrame(fit_rows).to_csv(out / "train_cv_test_gap.csv", index=False)
log.info("\n%s", pd.DataFrame(fit_rows).round(4).to_string(index=False))

# ---------------------------------------------------------------- (d) real training histories
gi_tr, gi_va = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(fit_df, groups=fit_df["trip_id"]))
Xtr, ytr = fit_df.iloc[gi_tr][FEATURES].to_numpy(np.float32), fit_df.iloc[gi_tr]["y"].to_numpy()
Xva, yva = fit_df.iloc[gi_va][FEATURES].to_numpy(np.float32), fit_df.iloc[gi_va]["y"].to_numpy()
sc = StandardScaler().fit(Xtr); Xtr_s, Xva_s = sc.transform(Xtr), sc.transform(Xva)
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, learning_rate_init=1e-3, random_state=seed)
hist = []
rng = np.random.default_rng(seed)
for ep in range(1, 61):                       # same architecture/optimiser as models.py; one pass = one epoch
    idx = rng.permutation(len(Xtr_s))
    for s in range(0, len(idx), 200):
        mlp.partial_fit(Xtr_s[idx[s:s + 200]], ytr[idx[s:s + 200]], classes=[0, 1])
    hist.append(dict(epoch=ep, train_log_loss=log_loss(ytr, mlp.predict_proba(Xtr_s)[:, 1]),
                     val_log_loss=log_loss(yva, mlp.predict_proba(Xva_s)[:, 1])))
mh = pd.DataFrame(hist); mh.to_csv(out / "mlp_training_history.csv", index=False)
deployed = bundles["mlp"]["model"].named_steps["clf"]
pd.DataFrame({"epoch": np.arange(1, len(deployed.loss_curve_) + 1), "train_loss": deployed.loss_curve_,
              "val_accuracy": deployed.validation_scores_}).to_csv(out / "mlp_deployed_curve.csv", index=False)

hgb_hist = {}
for cfgname, params in (("untuned", {}), ("tuned", best)):
    m = make_model("hgb", seed, params).fit(Xtr, ytr)
    tr_l = [log_loss(ytr, p[:, 1]) for p in m.staged_predict_proba(Xtr)]
    va_l = [log_loss(yva, p[:, 1]) for p in m.staged_predict_proba(Xva)]
    hgb_hist[cfgname] = (tr_l, va_l)
pd.DataFrame({f"{c}_{s}": pd.Series(v[i]) for c, v in hgb_hist.items() for i, s in enumerate(["train", "val"])}).to_csv(
    out / "hgb_training_history.csv", index_label="iteration")

fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.2))
axs[0].plot(mh.epoch, mh.train_log_loss, color=C_MAIN, label="train"); axs[0].plot(mh.epoch, mh.val_log_loss, color=C_ALT, label="validation (held-out trips)")
axs[0].axvline(mh.val_log_loss.idxmin() + 1, color=C_GREY, ls=":", lw=1)
axs[0].set(title="MLP (64,32): log loss per epoch", xlabel="epoch", ylabel="log loss"); axs[0].legend(frameon=False)
ax2 = axs[1]; ax2.plot(deployed.loss_curve_, color=C_MAIN, label="train loss"); ax2.set(title="Deployed MLP (early stopping)", xlabel="epoch", ylabel="train loss")
ax3 = ax2.twinx(); ax3.plot(deployed.validation_scores_, color=C_ALT, label="val accuracy"); ax3.set_ylabel("val accuracy (10% split)")
ax2.legend(handles=ax2.get_lines() + ax3.get_lines(), frameon=False, loc="center right")
for c, col in (("untuned", C_GREY), ("tuned", C_MAIN)):
    axs[2].plot(hgb_hist[c][0], color=col, ls="--", lw=1, label=f"{c} train"); axs[2].plot(hgb_hist[c][1], color=col, label=f"{c} val")
axs[2].set(title="HGB: log loss per iteration", xlabel="iteration", ylabel="log loss", ylim=(0.03, 0.15)); axs[2].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(out / "fig_training_curves.png"); plt.close(fig)

# ---------------------------------------------------------------- (e) comparison plots
cv = pd.read_csv(res / "cv_results.csv").set_index("model").loc[MODEL_NAMES]
st = pd.read_csv(res / "student_test_models.csv")
st = st[st.family != "Rule"].reset_index(drop=True); st["key"] = MODEL_NAMES + ["tuned"]
lab = [SHORT[k] for k in st.key]

fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
x = np.arange(5)
axs[0].bar(x, cv.macro_f1, yerr=cv.macro_f1_std, color=[C_MAIN if k == "hgb" else C_GREY for k in cv.index], capsize=3)
axs[0].set(xticks=x, xticklabels=[SHORT[k] for k in cv.index], ylim=(0.9, 0.96), title="5-fold GroupKFold CV macro-F1 (train)", ylabel="macro-F1")
for i, v in enumerate(cv.macro_f1): axs[0].text(i, v + 0.009, f"{v:.4f}", ha="center", fontsize=7)
cf = pd.read_csv(res / "cv_folds.csv")
for i, k in enumerate(cv.index):
    axs[1].scatter(np.full(5, i) + np.linspace(-0.12, 0.12, 5), cf[cf.model == k].top1_acc, color=C_MAIN if k == "hgb" else C_GREY, s=14)
axs[1].set(xticks=x, xticklabels=[SHORT[k] for k in cv.index], title="Per-fold CV top-1 accuracy (stability)", ylabel="top-1 accuracy")
fig.tight_layout(); fig.savefig(out / "fig_cv_comparison.png"); plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.2))
x = np.arange(len(st)); wd = 0.38
axs[0].bar(x - wd / 2, st.test_top1, wd, color=C_MAIN, label="top-1 accuracy"); axs[0].bar(x + wd / 2, st.test_macro_f1, wd, color=C_ALT, label="macro-F1")
axs[0].set(xticks=x, xticklabels=lab, ylim=(0.85, 0.96), title="student_test: top-1 and macro-F1"); axs[0].legend(frameon=False, fontsize=7)
axs[0].tick_params(axis="x", rotation=30)
axs[1].bar(x, st.test_log_loss, color=C_MAIN); axs[1].set(xticks=x, xticklabels=lab, title="student_test: log loss (lower is better)")
for i, v in enumerate(st.test_log_loss): axs[1].text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=7)
axs[1].tick_params(axis="x", rotation=30)
axs[2].bar(x, st.infer_s_per_1k_points * 1000, color=C_MAIN); axs[2].set_yscale("log")
axs[2].set(xticks=x, xticklabels=lab, title="Inference time (ms per 1k GPS points, log)"); axs[2].tick_params(axis="x", rotation=30)
fig.tight_layout(); fig.savefig(out / "fig_student_test_comparison.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 3.2))
order = list(agg.index); x = np.arange(len(order))
ax.bar(x - 0.3, agg.top1_clf, 0.2, color=C_GREY, label="top-1, classifier only")
ax.bar(x - 0.1, agg.top1_hmm, 0.2, color=C_MAIN, label="top-1, classifier + HMM")
ax.bar(x + 0.1, agg.continuity_clf, 0.2, color="#c9c9c9", hatch="//", label="continuity, classifier only")
ax.bar(x + 0.3, agg.continuity_hmm_raw, 0.2, color=C_ALT, label="continuity, HMM (raw arcs)")
ax.set(xticks=x, xticklabels=[SHORT[k] for k in order], ylim=(0.6, 1.0), title="student_test: effect of HMM/Viterbi decoding for every model")
ax.legend(frameon=False, ncol=4, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.32))
fig.tight_layout(); fig.savefig(out / "fig_hmm_per_model.png"); plt.close(fig)

pt = pd.read_csv(res / "student_test_routes_per_trip.csv", dtype={"trip_id": str})
fig, axs = plt.subplots(1, 2, figsize=(10, 3.2)); x = np.arange(len(pt))
axs[0].bar(x - 0.2, pt.top1_clf, 0.4, color=C_GREY, label="classifier only"); axs[0].bar(x + 0.2, pt.top1_hmm, 0.4, color=C_MAIN, label="classifier + HMM")
axs[0].set(xticks=x, xticklabels=pt.trip_id.str[-2:], ylim=(0.7, 1.01), title="Point top-1 per student_test trip (deployed model)", xlabel="trip id (last 2 digits)")
axs[0].legend(frameon=False, fontsize=7)
axs[1].bar(x - 0.2, pt.route_f1_clf, 0.4, color=C_GREY, label="classifier only"); axs[1].bar(x + 0.2, pt.route_f1_hmm, 0.4, color=C_MAIN, label="classifier + HMM")
axs[1].set(xticks=x, xticklabels=pt.trip_id.str[-2:], ylim=(0.5, 1.01), title="Route F1 per student_test trip", xlabel="trip id (last 2 digits)")
fig.tight_layout(); fig.savefig(out / "fig_hmm_per_trip.png"); plt.close(fig)

tr = pd.read_csv(res / "tuning_trials.csv")
fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
axs[0].scatter(tr.number + 1, tr.value, color=C_GREY, s=14, label="trial"); axs[0].plot(tr.number + 1, tr.value.cummax(), color=C_MAIN, label="best so far")
axs[0].axvline(6.5, color=C_GREY, ls=":", lw=1); axs[0].text(6.7, tr.value.min(), "TPE starts", fontsize=7)
axs[0].set(title="Optuna TPE history (HGB)", xlabel="trial", ylabel="CV macro-F1 (4 folds)"); axs[0].legend(frameon=False, fontsize=7)
axs[1].scatter(tr.learning_rate, tr.value, c=tr.min_samples_leaf, cmap="viridis", s=18); axs[1].set_xscale("log")
axs[1].set(title="Trial score vs learning rate (colour = min_samples_leaf)", xlabel="learning_rate", ylabel="CV macro-F1")
fig.tight_layout(); fig.savefig(out / "fig_optuna.png"); plt.close(fig)
log.info("report figures written to %s", out)
