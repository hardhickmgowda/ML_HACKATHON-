"""Track cleaning: outlier removal + smoothing (Harsh)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import median_filter
from .geo import Projector


@dataclass
class CleanTrack:
    keep_idx: np.ndarray      # indices into the raw track that survived cleaning
    xy: np.ndarray            # (T,2) smoothed metric coordinates
    xy_raw: np.ndarray        # (T,2) unsmoothed metric coordinates
    ts: np.ndarray            # (T,)
    n_raw: int

    @property
    def n(self) -> int:
        return len(self.ts)


def clean_track(track: np.ndarray, proj: Projector, cfg: dict) -> CleanTrack:
    """1) drop NaN / non-monotonic timestamps  2) drop physically impossible jumps  3) light median smoothing."""
    pc = cfg["preprocess"]
    lon, lat, ts = track[:, 0], track[:, 1], track[:, 2]
    x, y = proj.fwd(lon, lat)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(ts)
    idx = np.flatnonzero(ok)
    keep: list[int] = []
    consecutive_drop = 0
    for i in idx:
        if not keep:
            keep.append(i)
            continue
        j = keep[-1]
        dt = ts[i] - ts[j]
        if dt <= 0:
            continue
        d = np.hypot(x[i] - x[j], y[i] - y[j])
        if d / dt > pc["max_speed_mps"] and d > 30.0 and consecutive_drop < 5:
            consecutive_drop += 1          # outlier; but after 5 in a row accept (the baseline was wrong)
            continue
        consecutive_drop = 0
        keep.append(i)
    keep = np.asarray(keep, dtype=np.int64)
    xr, yr = x[keep], y[keep]
    w = int(pc["smooth_window"])
    if w > 1 and len(keep) >= w:
        xs, ys = median_filter(xr, size=w, mode="nearest"), median_filter(yr, size=w, mode="nearest")
    else:
        xs, ys = xr, yr
    return CleanTrack(keep, np.c_[xs, ys], np.c_[xr, yr], ts[keep], len(track))
