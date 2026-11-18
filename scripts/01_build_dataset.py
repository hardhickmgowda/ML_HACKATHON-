"""Step 1 (Harsh): parse every trip, generate candidates + features + labels, cache to artifacts/cache/."""
import time, joblib, numpy as np, pandas as pd
from _common import load_config, resolve, set_seed, get_logger
from pathsnap.data_io import load_trip, discover_trips
from pathsnap.dataset import prepare_trip

cfg = load_config(); set_seed(cfg["seed"]); log = get_logger()
data = resolve(cfg["paths"]["data_root"])
out = resolve(cfg["paths"]["artifacts"], "cache"); out.mkdir(parents=True, exist_ok=True)
summary = []
for split in ("train_dir", "test_dir"):
    trips = discover_trips([data / cfg["paths"][split]])
    for tid, folder in trips.items():
        t0 = time.time()
        p = prepare_trip(load_trip(folder), cfg)
        df = p.df
        labeled_frac = float((p.true_arc >= 0).mean())
        miss = float(1 - df[df.labeled].groupby("pt")["y"].max().mean()) if labeled_frac > 0 else np.nan
        first = df[df.dist_rank == 0]
        near = float(first[first.labeled].set_index("pt")["y"].mean()) if labeled_frac > 0 else np.nan
        for c in df.select_dtypes("float64").columns:
            df[c] = df[c].astype("float32")
        joblib.dump(df, out / f"{split[:-4]}_{tid}.pkl", compress=3)
        summary.append(dict(split=split[:-4], trip_id=tid, n_points=p.ct.n, n_raw=p.ct.n_raw, n_arcs=p.net.n_arcs,
                            rows=len(df), labeled_frac=round(labeled_frac, 3), candidate_miss=round(miss, 4),
                            nearest_arc_top1=round(near, 4), secs=round(time.time() - t0, 1)))
        log.info("%s %s pts=%d rows=%d labeled=%.2f miss=%.4f nearest=%.3f (%.1fs)", split[:-4], tid, p.ct.n, len(df),
                 labeled_frac, miss, near, time.time() - t0)
res = resolve(cfg["paths"]["results"]); res.mkdir(exist_ok=True)
pd.DataFrame(summary).to_csv(res / "dataset_summary.csv", index=False)
log.info("done: %d trips", len(summary))
