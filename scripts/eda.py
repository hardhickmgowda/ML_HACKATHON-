"""Exploratory data analysis: tables, plots and Folium maps.

Prerequisites:  python scripts/validate_data.py && python scripts/build_dataset.py
Usage:          python scripts/eda.py
Writes:         results/eda/*.csv, results/eda/plots/*.png, results/eda/maps/*.html

Every output names the partition it was computed on. Choices that feed into the
pipeline (thresholds) are based on the TRAIN partition only.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap import config  # noqa: E402
from pathsnap.data import load_trip  # noqa: E402
from pathsnap.data.dataset import read_processed  # noqa: E402
from pathsnap.data.pipeline import process_trip  # noqa: E402
from pathsnap.viz.maps import trip_map  # noqa: E402

OUT = config.RESULTS_DIR / "eda"
PLOTS = OUT / "plots"
MAPS = OUT / "maps"
PART_COLORS = {"train": "#2a6fdb", "student_test": "#e8710a"}

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(PLOTS / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def load_processed(part):
    return (read_processed(part, "points"), read_processed(part, "candidates"),
            read_processed(part, "trip_stats"))


def main():
    for p in (PLOTS, MAPS):
        p.mkdir(parents=True, exist_ok=True)
    checks = pd.read_csv(config.RESULTS_DIR / "data_validation" / "trip_checks.csv", dtype={"trip_id": str})
    issues = pd.read_csv(config.RESULTS_DIR / "data_validation" / "issues.csv", dtype={"trip_id": str})
    data = {p: load_processed(p) for p in config.PARTITIONS}

    # ------------------------------------------------------------ 1. inventory
    sizes = checks["file_sizes"].apply(ast.literal_eval).apply(pd.Series).add_suffix("_bytes")
    inv = pd.concat([checks[["partition", "trip_id", "n_points", "duration_s", "track_length_km",
                             "extent_km", "n_nodes", "n_arcs", "route_len", "route_length_km"]], sizes], axis=1)
    inv.to_csv(OUT / "inventory.csv", index=False)
    part_summary = inv.groupby("partition").agg(
        trips=("trip_id", "size"), total_points=("n_points", "sum"),
        median_points=("n_points", "median"), min_points=("n_points", "min"), max_points=("n_points", "max"),
        median_nodes=("n_nodes", "median"), max_nodes=("n_nodes", "max"),
        median_route_arcs=("route_len", "median"), total_track_km=("track_length_km", "sum"))
    part_summary.to_csv(OUT / "partition_summary.csv")
    print(part_summary.T.to_string(), "\n")

    # ------------------------------------------------------- 2. missing values
    miss = checks.groupby("partition")[["nan_track", "nan_nodes", "nan_arcs"]].sum()
    miss["files_missing"] = 0  # every trip folder has all 4 files (checked by the loader)
    miss.to_csv(OUT / "missing_values.csv")
    issues.groupby(["severity", "check"]).agg(n_trips=("trip_id", "nunique"),
                                              trips=("trip_id", lambda s: " ".join(sorted(set(s)))))\
          .to_csv(OUT / "data_quality_issues.csv")

    # --------------------------------------------- 3. points / sampling / speed
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for part, g in checks.groupby("partition"):
        ax[0].hist(g["n_points"], bins=np.logspace(2, 4.5, 25), alpha=0.7, label=part, color=PART_COLORS[part])
    ax[0].set_xscale("log"); ax[0].set_xlabel("GPS points per trip (log)"); ax[0].set_ylabel("trips")
    ax[0].set_title("Trip length (train vs student_test)"); ax[0].legend()

    pts_tr = data["train"][0]
    dt = pts_tr["dt_prev"].dropna()
    bins = [0.5, 1.5, 2.5, 5.5, 10.5, 30.5, 120.5, 600.5, dt.max() + 1]
    cnt = pd.cut(dt, bins).value_counts(sort=False)
    labels = ["1s", "2s", "3-5s", "6-10s", "11-30s", "31-120s", "2-10min", ">10min"]
    ax[1].bar(labels, cnt.values, color=PART_COLORS["train"])
    ax[1].set_yscale("log"); ax[1].set_ylabel("steps (log)"); ax[1].tick_params(axis="x", rotation=30)
    ax[1].set_title(f"Sampling interval (train): {np.mean(dt == 1):.1%} exactly 1 s")

    sp = pts_tr.loc[~pts_tr["is_segment_start"], "speed_prev_mps"].dropna()
    ax[2].hist(sp.clip(upper=80) * 3.6, bins=80, color=PART_COLORS["train"])
    ax[2].axvline(config.MAX_SPEED_MPS * 3.6, color="red", ls="--", label=f"spike threshold {config.MAX_SPEED_MPS*3.6:.0f} km/h")
    ax[2].set_xlabel("step speed (km/h, clipped at 288)"); ax[2].set_yscale("log")
    ax[2].set_title("Raw step speed (train)"); ax[2].legend()
    save(fig, "01_points_sampling_speed")

    # ----------------------------------------------- 4. GPS noise / radius choice
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for part in config.PARTITIONS:
        p = data[part][0]
        d = p.loc[p["label_valid"], "true_dist_m"]
        ax[0].hist(d.clip(upper=50), bins=100, density=True, histtype="step", lw=2,
                   label=f"{part} (median {d.median():.1f} m)", color=PART_COLORS[part])
    ax[0].set_xlabel("distance GPS point -> reference arc (m)"); ax[0].set_ylabel("density")
    ax[0].set_title("GPS noise (points with reliable labels)"); ax[0].legend()

    d_all = pts_tr.loc[pts_tr["label_valid"], "true_dist_m"].to_numpy()
    r = np.arange(5, 101, 1)
    ax[1].plot(r, [np.mean(d_all > x) * 100 for x in r], color=PART_COLORS["train"])
    ax[1].axvline(config.CANDIDATE_RADIUS_M, color="red", ls="--", label=f"chosen radius {config.CANDIDATE_RADIUS_M:.0f} m")
    ax[1].set_yscale("log"); ax[1].set_xlabel("candidate search radius (m)")
    ax[1].set_ylabel("% points whose true arc is outside radius")
    ax[1].set_title("Candidate miss rate vs radius (train)"); ax[1].legend()
    save(fig, "02_gps_noise_radius")
    noise = pd.DataFrame({part: data[part][0].loc[data[part][0]["label_valid"], "true_dist_m"].describe(
        percentiles=[.5, .9, .95, .99]) for part in config.PARTITIONS})
    noise.to_csv(OUT / "gps_noise_stats.csv")

    # ------------------------------------------- 5. network size & connectivity
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for part, g in checks.groupby("partition"):
        ax[0].scatter(g["n_nodes"], g["n_arcs"], s=18, alpha=0.8, label=part, color=PART_COLORS[part])
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("nodes"); ax[0].set_ylabel("arcs (directed)")
    ax[0].set_title(f"Road network size per trip (arcs/node ≈ {np.median(checks.n_arcs/checks.n_nodes):.2f})")
    ax[0].legend()
    ax[1].hist(checks["largest_scc_frac"] * 100, bins=30, color=PART_COLORS["train"], alpha=0.7, label="largest strongly-conn.")
    ax[1].hist(checks["largest_wcc_frac"] * 100, bins=30, color="#666", alpha=0.5, label="largest weakly-conn.")
    ax[1].set_xlabel("% of nodes in largest component"); ax[1].set_ylabel("trips")
    ax[1].set_title("Network connectivity (all 90 trips)"); ax[1].legend()
    save(fig, "03_network_size_connectivity")
    checks[["partition", "trip_id", "n_nodes", "n_arcs", "n_isolated_nodes", "n_self_loops",
            "n_duplicate_arcs", "n_weak_components", "largest_wcc_frac", "largest_scc_frac",
            "route_continuity"]].to_csv(OUT / "network_connectivity.csv", index=False)

    # ------------------------------ 6. feature behaviour: positives vs negatives
    c = data["train"][1]
    c = c[c["label_valid"]]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for lab, col, name in ((0, "#999", "other candidates"), (1, "#1b9e3a", "true arc")):
        s = c[c["label"] == lab]
        ax[0].hist(s["dist_m"], bins=60, density=True, histtype="step", lw=2, color=col, label=name)
        ax[1].hist(s.loc[s["heading_valid"] == 1, "heading_diff_deg"], bins=36, density=True,
                   histtype="step", lw=2, color=col, label=name)
        ax[2].hist(s["dist_rank"], bins=np.arange(-0.5, config.MAX_CANDIDATES), density=True,
                   histtype="step", lw=2, color=col, label=name)
    ax[0].set_xlabel("dist_m"); ax[0].set_title("Distance to candidate (train)")
    ax[1].set_xlabel("heading_diff_deg"); ax[1].set_title("GPS heading vs arc direction (train)")
    ax[2].set_xlabel("dist_rank"); ax[2].set_title("Rank by distance (train)")
    for a in ax:
        a.legend()
    save(fig, "04_feature_separation")

    # baseline: nearest arc vs nearest arc with direction resolved by heading
    base = []
    for part in config.PARTITIONS:
        cc = data[part][1]
        cc = cc[cc["label_valid"]]
        nearest = cc[cc["dist_rank"] == 0]
        # among candidates within 1 m of the nearest one, choose the best heading
        tie = cc[cc["dist_minus_min"] < 1.0].sort_values(["trip_id", "point_pos", "heading_diff_deg"])
        dir_aware = tie.groupby(["trip_id", "point_pos"]).head(1)
        base.append({"partition": part,
                     "nearest_arc_top1": nearest["label"].mean(),
                     "nearest_arc_plus_heading_top1": dir_aware["label"].mean(),
                     "candidate_miss_rate": 1 - cc.groupby(["trip_id", "point_pos"])["label"].max().mean(),
                     "positive_rate": cc["label"].mean(),
                     "mean_candidates": cc.groupby(["trip_id", "point_pos"]).size().mean()})
    base = pd.DataFrame(base)
    base.to_csv(OUT / "baselines.csv", index=False)
    print(base.round(4).to_string(index=False), "\n")

    feats = [col for col in c.columns if col not in ("trip_id", "point_pos", "arc_id", "canon_arc",
                                                     "label", "label_valid")]
    c[feats + ["label"]].corr(numeric_only=True)["label"].drop("label").sort_values()\
        .to_csv(OUT / "feature_label_correlation_train.csv")

    # -------------------------------------------- 7. trajectory plots (metres)
    sample = checks[checks["partition"] == "train"].sort_values("n_points")["trip_id"].to_numpy()
    sample = sample[np.linspace(0, len(sample) - 1, 8).astype(int)]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, tid in zip(axes.ravel(), sample):
        trip = load_trip("train", tid)
        pt = process_trip(trip)
        net, pts = pt.network, pt.points
        for a in trip.route:
            ax.plot([net.ax[a], net.bx[a]], [net.ay[a], net.by[a]], color="#1b9e3a", lw=3, alpha=0.6)
        ax.scatter(pts["x"], pts["y"], s=1, color="#1f4fd1")
        ax.set_aspect("equal"); ax.set_title(f"train/{tid}: {len(pts)} pts", fontsize=9)
        ax.tick_params(labelsize=7)
    fig.suptitle("GPS trajectories (blue) vs reference route (green), local metres — train partition")
    save(fig, "05_trajectories_train")

    # --------------------------------------------------------- 8. folium maps
    picks = [("train", "00000001", "typical trip"),
             ("train", "00000054", "GPS biased ~138 m off the road network"),
             ("train", "00000033", "reference route misses part of the drive"),
             ("train", "00000013", "timestamp bursts after gaps"),
             ("student_test", "00000056", "longest student-test trip")]
    for part, tid, why in picks:
        trip = load_trip(part, tid)
        pt = process_trip(trip)
        m = trip_map(pt.points, pt.network, f"{part}/{tid} — {why}", route=trip.route)
        m.save(str(MAPS / f"{part}_{tid}.html"))
    print(f"EDA written to {OUT}")


if __name__ == "__main__":
    main()
