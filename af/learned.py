"""Learned framing: player-distribution features (with offline temporal context)
-> human camera x, via gradient boosting.

Honest scoring: GroupKFold over the human's framed segments, so every reported
prediction comes from a model that never saw that stretch of play.

Run:  python -m af.learned [--folds 6]
"""

import argparse

import numpy as np
from scipy.ndimage import uniform_filter1d

from .human import HumanPath
from .metrics import evaluate, fmt
from .players import Detections
from .simple_methods import DETS, EDGES, HZ, PKL, fill_nan

QS = (0.1, 0.25, 0.5, 0.75, 0.9)
NBINS = 12
OFFSETS_S = (-8, -4, -2, -1, 0, 1, 2, 4, 8)


def frame_features(xs_list, max_x):
    """Per-sample features from the set of on-ice player x."""
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


def build_features(xs_list, max_x):
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
    return np.hstack(cols), names


def segment_ids(human, times):
    segs = human.segments()
    sid = np.full(len(times), -1)
    for k, (s, e) in enumerate(segs):
        sid[(times >= s) & (times <= e)] = k
    return sid


def cross_val_predict(X, y, groups, folds, params):
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    pred = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=folds).split(X, y, groups):
        m = lgb.LGBMRegressor(**params, verbose=-1)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=6)
    args = ap.parse_args()

    human = HumanPath(PKL)
    human.set_frame_size(3840, 2880)
    det = Detections(DETS, EDGES, human.max_x)
    t = det.times
    xs = det.per_sample(det.on_ice())
    X, names = build_features(xs, human.max_x)

    scored = human.scored_mask(t)
    sid = segment_ids(human, t)
    use = scored & (sid >= 0)
    y = human.x_at(t[use])
    print(f"{use.sum()} labelled samples in {len(np.unique(sid[use]))} segments, {X.shape[1]} features")

    params = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0)
    p = cross_val_predict(X[use], y, sid[use], args.folds, params)
    p = np.clip(p, 0, human.max_x)
    full = np.full(len(t), np.nan)
    full[use] = p
    full = fill_nan(full)
    print(fmt("lgbm cv raw", evaluate(t, full, human)))
    for sm in (1, 2, 3):
        v = uniform_filter1d(full, int(sm * HZ), mode="nearest")
        print(fmt(f"lgbm cv sm{sm}", evaluate(t, v, human)))
    np.savez(r"D:/temp/af/learned_cv.npz", t=t, pred=full)


if __name__ == "__main__":
    main()
