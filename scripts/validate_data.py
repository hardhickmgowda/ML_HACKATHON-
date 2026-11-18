"""Run raw-data validation checks on every trip.

Usage:  python scripts/validate_data.py [--partitions train student_test]
Writes: results/data_validation/trip_checks.csv, issues.csv
Exit code 1 if any ERROR-level issue is found.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap import config  # noqa: E402
from pathsnap.data import iter_trips  # noqa: E402
from pathsnap.data.validate import issues_frame, validate_raw_trip  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", nargs="+", default=list(config.PARTITIONS))
    args = ap.parse_args()

    out = config.RESULTS_DIR / "data_validation"
    out.mkdir(parents=True, exist_ok=True)
    metrics, issues = [], []
    for part in args.partitions:
        for trip in iter_trips(part):
            m, iss = validate_raw_trip(trip)
            m["file_sizes"] = trip.meta["file_sizes"]
            metrics.append(m)
            issues.extend(iss)

    mdf = pd.DataFrame(metrics)
    idf = issues_frame(issues)
    mdf.to_csv(out / "trip_checks.csv", index=False)
    idf.to_csv(out / "issues.csv", index=False)

    print(f"Validated {len(mdf)} trips -> {out}")
    print(idf.groupby(["severity", "check"]).size().rename("n_trips").to_string())
    n_err = int((idf["severity"] == "ERROR").sum()) if len(idf) else 0
    if n_err:
        print(f"\n{n_err} ERROR(s):")
        print(idf[idf["severity"] == "ERROR"].to_string(index=False))
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
