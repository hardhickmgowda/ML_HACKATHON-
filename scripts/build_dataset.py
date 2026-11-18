"""Build model-ready candidate/feature tables.

Usage:  python scripts/build_dataset.py [--partitions train student_test]

Writes, per partition, to data_processed/<partition>/:
    points.parquet       one row per cleaned GPS observation (+ labels)
    candidates.parquet   one row per (observation, candidate arc) with
                         FEATURE_COLUMNS, `label` and `label_valid`
    trip_stats.csv       cleaning / candidate / labelling stats per trip

For model training, use rows with label_valid == True and group by trip_id
for cross-validation (GroupKFold), never by row.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap import config  # noqa: E402
from pathsnap.data.pipeline import build_partition  # noqa: E402
from pathsnap.data.validate import issues_frame, validate_processed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", nargs="+", default=list(config.PARTITIONS))
    args = ap.parse_args()

    for part in args.partitions:
        points, cands, stats = build_partition(part)
        idf = issues_frame(validate_processed(points, cands))
        idf.to_csv(config.PROCESSED_DIR / part / "processed_checks.csv", index=False)
        n_valid = int(points["label_valid"].sum()) if "label_valid" in points else len(points)
        print(f"\n== {part}: {points['trip_id'].nunique()} trips, {len(points):,} points "
              f"({n_valid:,} with reliable labels), {len(cands):,} candidate rows")
        if "candidate_miss_rate" in stats:
            print(f"   mean candidate miss rate: {stats['candidate_miss_rate'].mean():.4f}")
            print(f"   nearest-arc Top-1 (baseline): {stats['nearest_arc_top1_acc'].mean():.4f}")
        print(f"   processed-table checks: "
              f"{'OK' if idf.empty else idf.groupby(['severity','check']).size().to_dict()}")
        if (idf["severity"] == "ERROR").any():
            print(idf[idf["severity"] == "ERROR"].to_string(index=False))
            sys.exit(1)


if __name__ == "__main__":
    main()
