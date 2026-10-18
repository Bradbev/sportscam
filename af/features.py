"""Per-sample features from on-ice player positions, with offline temporal context.

Detections are sampled at HZ.  Each sample gets quantiles and a histogram of
player camera-x, the player count, motion summaries (af.motion), and the same
values a few seconds before and after, so the model can see where play is going.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d

HZ = 5.0
QS = (0.1, 0.25, 0.5, 0.75, 0.9)
NBINS = 12
OFFSETS_S = (-8, -4, -2, -1, 0, 1, 2, 4, 8)
EXTRA_OFFSETS_S = (-4, -2, -1, 0, 1, 2, 4)


def fill_nan(a):
    idx = np.where(~np.isnan(a))[0]
    if len(idx) == 0:
        return np.zeros_like(a)
    return np.interp(np.arange(len(a)), idx, a[idx])


def robust_centre(xs_list, q=0.5):
    out = np.full(len(xs_list), np.nan)
    for i, xs in enumerate(xs_list):
        if len(xs):
            out[i] = np.quantile(xs, q)
    return out


def frame_features(xs_list, max_x):
    """Per-sample quantiles, normalised histogram and count of on-ice player x."""
    lo, hi = -960.0, max_x + 960.0
    n = len(xs_list)
    q = np.full((n, len(QS)), np.nan)
    hist = np.zeros((n, NBINS))
    cnt = np.zeros(n)
    for i, xs in enumerate(xs_list):
        cnt[i] = len(xs)
        if len(xs):
            q[i] = np.quantile(xs, QS)
            h, _ = np.histogram(xs, bins=NBINS, range=(lo, hi))
            hist[i] = h / len(xs)
    for j in range(q.shape[1]):
        q[:, j] = fill_nan(q[:, j])
    return q, hist, cnt


def build_features(xs_list, max_x, extra=None, extra_names=()):
    q, hist, cnt = frame_features(xs_list, max_x)
    # light per-sample smoothing (1s) before stacking, detections are noisy
    k = int(HZ)
    qs = uniform_filter1d(q, k, axis=0, mode="nearest")
    hs = uniform_filter1d(hist, k, axis=0, mode="nearest")
    cs = uniform_filter1d(cnt, k, mode="nearest")
    n = len(xs_list)
    cols, names = [], []
    idx = np.arange(n)
    for off in OFFSETS_S:
        j = np.clip(idx + int(off * HZ), 0, n - 1)
        cols.append(qs[j]); names += [f"q{int(p*100)}@{off}" for p in QS]
        cols.append(cs[j][:, None]); names += [f"cnt@{off}"]
        if off in (-2, 0, 2):
            cols.append(hs[j]); names += [f"h{b}@{off}" for b in range(NBINS)]
    # motion of the distribution: median velocity over a few horizons
    for d in (1, 2, 4):
        j0 = np.clip(idx - int(d * HZ), 0, n - 1)
        j1 = np.clip(idx + int(d * HZ), 0, n - 1)
        cols.append(((qs[j1, 2] - qs[j0, 2]) / (2 * d))[:, None]); names.append(f"vmed{d}")
    # longer-term context: where has play been over the last/next 20s
    for w in (10, 20):
        cols.append(uniform_filter1d(qs[:, 2], int(w * HZ), mode="nearest")[:, None])
        names.append(f"med_avg{w}")
    if extra is not None:
        es = uniform_filter1d(extra, k, axis=0, mode="nearest")
        for off in EXTRA_OFFSETS_S:
            j = np.clip(idx + int(off * HZ), 0, n - 1)
            cols.append(es[j]); names += [f"{nm}@{off}" for nm in extra_names]
        # cumulative motion over longer windows: net drift of play
        for w in (2, 4):
            cols.append(uniform_filter1d(extra[:, :1], int(w * HZ), axis=0, mode="nearest"))
            names.append(f"v_mean_avg{w}")
    return np.hstack(cols), names


def segment_ids(human, times):
    """Index of the human's framed segment containing each time, -1 outside."""
    sid = np.full(len(times), -1)
    for k, (s, e) in enumerate(human.segments()):
        sid[(times >= s) & (times <= e)] = k
    return sid
