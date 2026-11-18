"""GPS trajectory cleaning.

The cleaner never invents or moves observations. It:

1. drops rows that cannot be used at all (NaN / impossible lon-lat /
   duplicated timestamps) and sorts by time;
2. flags isolated *speed spikes* (a point that is reached and left at an
   impossible speed, while skipping it gives a plausible speed) as outliers;
3. splits the trajectory into *segments* at GPS dropouts (large time or
   distance jumps) so the HMM can treat each segment's transitions carefully;
4. derives per-point kinematics (step length, speed, smoothed heading).

``orig_idx`` always points back to the row in the raw .track file, so model
outputs can be mapped back to every original observation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from .geo import LocalProjection, bearing_deg


def _speeds(x, y, t):
    step = np.hypot(np.diff(x), np.diff(y))
    dt = np.diff(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(dt > 0, step / dt, np.inf)
    return step, dt, v


def find_speed_spikes(x, y, t, max_speed=config.MAX_SPEED_MPS, max_iter=5):
    """Boolean mask of spike outliers.

    Point i is a spike when v(i-1 -> i) > max_speed AND v(i -> i+1) > max_speed
    but v(i-1 -> i+1) <= max_speed (the trajectory is fine without it).
    End points are spikes if their only neighbour is reached too fast and the
    next pair is plausible. Repeats until no more spikes are found.
    """
    n = len(x)
    out = np.zeros(n, dtype=bool)
    for _ in range(max_iter):
        keep = np.flatnonzero(~out)
        if len(keep) < 3:
            break
        kx, ky, kt = x[keep], y[keep], t[keep]
        _, _, v = _speeds(kx, ky, kt)
        v_in = np.r_[np.nan, v]
        v_out = np.r_[v, np.nan]
        skip_dt = kt[2:] - kt[:-2]
        skip_v = np.hypot(kx[2:] - kx[:-2], ky[2:] - ky[:-2]) / np.where(skip_dt > 0, skip_dt, np.nan)
        skip_v = np.r_[np.nan, skip_v, np.nan]
        spike = (v_in > max_speed) & (v_out > max_speed) & (skip_v <= max_speed)
        # end points
        if v_out[0] > max_speed and len(v) > 1 and v[1] <= max_speed:
            spike[0] = True
        if v_in[-1] > max_speed and len(v) > 1 and v[-2] <= max_speed:
            spike[-1] = True
        if not spike.any():
            break
        out[keep[spike]] = True
    return out


def _segment_heading(x, y, seg, half_window):
    """Heading from point i-w to i+w, clipped to the point's own segment.

    Using a window instead of consecutive points makes the heading robust to
    the ~5 m GPS jitter, which otherwise dominates at low speed.
    """
    n = len(x)
    idx = np.arange(n)
    # segment start/end index for every point
    starts = pd.Series(idx).groupby(seg).transform("min").to_numpy()
    ends = pd.Series(idx).groupby(seg).transform("max").to_numpy()
    a = np.clip(idx - half_window, starts, ends)
    b = np.clip(idx + half_window, starts, ends)
    dx, dy = x[b] - x[a], y[b] - y[a]
    disp = np.hypot(dx, dy)
    heading = bearing_deg(dx, dy)
    valid = disp >= 2.0  # need a couple of metres of movement for a direction
    heading = np.where(valid, heading, np.nan)
    # carry the last reliable heading forward through stops (within a segment)
    h = pd.Series(heading).groupby(seg).ffill().groupby(seg).bfill().to_numpy()
    return h, valid


def clean_track(track: pd.DataFrame, proj: LocalProjection | None = None,
                drop_outliers: bool = True) -> tuple[pd.DataFrame, LocalProjection, dict]:
    """Return (cleaned points, projection, cleaning report)."""
    report: dict = {"n_raw": len(track)}
    df = track.reset_index(drop=True).copy()
    df["orig_idx"] = np.arange(len(df))

    bad = (df[["lon", "lat", "t"]].isna().any(axis=1)
           | ~df["lon"].between(-180, 180) | ~df["lat"].between(-90, 90))
    report["n_invalid_rows"] = int(bad.sum())
    df = df[~bad]

    df = df.sort_values("t", kind="stable")
    dup_t = df["t"].duplicated(keep="first")
    report["n_duplicate_timestamps"] = int(dup_t.sum())
    df = df[~dup_t].reset_index(drop=True)

    if proj is None:
        proj = LocalProjection.from_points(df["lon"].to_numpy(), df["lat"].to_numpy())
    x, y = proj.to_xy(df["lon"].to_numpy(), df["lat"].to_numpy())
    df["x"], df["y"] = x, y

    spikes = find_speed_spikes(x, y, df["t"].to_numpy())
    df["is_outlier"] = spikes
    report["n_speed_spikes"] = int(spikes.sum())
    if drop_outliers:
        df = df[~df["is_outlier"]].reset_index(drop=True)

    x, y, t = df["x"].to_numpy(), df["y"].to_numpy(), df["t"].to_numpy()
    step, dt, v = _speeds(x, y, t)
    df["dt_prev"] = np.r_[np.nan, dt]
    df["step_prev_m"] = np.r_[np.nan, step]
    df["speed_prev_mps"] = np.r_[np.nan, v]
    df["dt_next"] = np.r_[dt, np.nan]
    df["step_next_m"] = np.r_[step, np.nan]

    new_seg = (df["dt_prev"] > config.GAP_SECONDS) | (df["step_prev_m"] > config.GAP_METRES)
    df["segment_id"] = new_seg.cumsum().astype(int)
    df["is_segment_start"] = new_seg | (df.index == 0)
    report["n_segments"] = int(df["segment_id"].nunique())
    report["n_gaps"] = int(new_seg.sum())

    df["is_stationary"] = df["step_prev_m"].fillna(np.inf) < config.STATIONARY_METRES
    report["n_stationary"] = int(df["is_stationary"].sum())

    # Robust speed: rolling median of per-step speed inside each segment
    sp = df["speed_prev_mps"].where(~df["is_segment_start"])
    df["speed_mps"] = (sp.groupby(df["segment_id"])
                         .transform(lambda s: s.rolling(5, center=True, min_periods=1).median())
                         .fillna(0.0))

    heading, hvalid = _segment_heading(x, y, df["segment_id"].to_numpy(),
                                       config.HEADING_HALF_WINDOW)
    df["heading_deg"] = heading
    df["heading_valid"] = hvalid
    df["point_pos"] = np.arange(len(df))
    report["n_clean"] = len(df)
    return df, proj, report
