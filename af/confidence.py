"""How sure is the auto camera?  Expected error, live-play probability, review segments.

Signals for each sample:
  quantile spread   LightGBM q10 / q90 of camera x around the camera model's x
  disagreement      std of a small ensemble (different seeds / subsamples)
  path speed        fast camera moves are where transitions get missed
  evidence          player count and its stability, distance from the median player
Two models sit on top of the signals (plus the usual features):
  error model       expected |auto x - human x|
  live model        probability the operator would be framing (not a stoppage or
                    intermission), so review time isn't spent where nothing is shown
Review score = expected error x P(live); high-score stretches become segments.

Training is cross-game: each game's signals and errors come from models trained
on the other games, so the error and live models learn what a *new* game looks like.

  python -m af.confidence DIR[=CACHE] DIR[=CACHE] ...    evaluate (review curve per game)
Production models are trained by `autotrack.py train` and used by `autotrack.py DIR`.
"""

import json
import os
import sys

import lightgbm as lgb
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.stats import spearmanr

from .features import HZ
from .model import PARAMS, finish

ENSEMBLE = 5
SEG_SMOOTH_S = 3.0   # the score is smoothed before thresholding
SEG_PCT = 80         # stretches above this percentile of the smoothed score are flagged
SEG_MERGE_S = 5.0    # flagged stretches closer than this are merged
SEG_PAD_S = 2.0      # context added either side of a segment
SEG_MIN_S = 3.0
MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def _fit(X, y, **over):
    return lgb.LGBMRegressor(**dict(PARAMS, **over), verbose=-1).fit(X, y)


def _boosters(models):
    return [m.booster_ if hasattr(m, "booster_") else m for m in models]


class SignalModels:
    """Quantile and ensemble camera models; their spread says how sure the camera is."""

    def __init__(self, q10, q90, ens):
        self.q10, self.q90, self.ens = q10, q90, ens

    @classmethod
    def fit(cls, X, y):
        return cls(_fit(X, y, objective="quantile", alpha=0.1), _fit(X, y, objective="quantile", alpha=0.9),
                   [_fit(X, y, subsample=0.6, colsample_bytree=0.5, random_state=s) for s in range(ENSEMBLE)])

    def signals(self, g, pred):
        q10, q90 = (finish(m.predict(g.X), g.max_x) for m in (self.q10, self.q90))
        ens = np.std([m.predict(g.X) for m in self.ens], axis=0)
        k = int(2 * HZ)
        speed = np.abs(np.gradient(pred)) * HZ
        cnt = g.X[:, g.names.index("cnt@0")]
        cols = {
            "spread": q90 - q10,
            "above": q90 - pred,
            "below": pred - q10,
            "ens_std": uniform_filter1d(ens, k, mode="nearest"),
            "speed": uniform_filter1d(speed, k, mode="nearest"),
            "speed_max4": uniform_filter1d(speed, int(4 * HZ), mode="nearest"),
            "from_median": np.abs(pred - g.med),
            "count": cnt,
            "count_std": np.sqrt(np.maximum(uniform_filter1d(cnt ** 2, k) - uniform_filter1d(cnt, k) ** 2, 0)),
            "pred": pred,
        }
        return np.hstack([np.column_stack(list(cols.values())), g.X]), list(cols) + list(g.names)


class ReviewModels:
    def __init__(self, signal_models, err_model, live_model, names):
        self.sig, self.err, self.live, self.names = signal_models, err_model, live_model, names

    def score(self, g, pred):
        """(review score, expected error px, P(live), signal matrix) per sample."""
        S, names = self.sig.signals(g, pred)
        if names != self.names:
            raise SystemExit("review models were trained on a different feature layout; retrain them")
        err = np.maximum(self.err.predict(S), 0)
        live = self.live.predict(S)
        live = live[:, 1] if live.ndim == 2 else live
        live = uniform_filter1d(live, int(2 * HZ), mode="nearest")
        return err * live, err, live, S

    def save(self, models_dir=MODELS):
        os.makedirs(models_dir, exist_ok=True)
        files = {"q10": self.sig.q10, "q90": self.sig.q90, "err": self.err, "live": self.live}
        files.update({f"ens{k}": m for k, m in enumerate(self.sig.ens)})
        for name, m in files.items():
            _boosters([m])[0].save_model(os.path.join(models_dir, f"review_{name}.txt"))
        with open(os.path.join(models_dir, "review_models.json"), "w") as f:
            json.dump({"features": self.names, "ensemble": len(self.sig.ens)}, f, indent=1)

    @classmethod
    def load(cls, models_dir=MODELS):
        meta_path = os.path.join(models_dir, "review_models.json")
        if not os.path.exists(meta_path):
            return None
        meta = json.load(open(meta_path))
        b = lambda n: lgb.Booster(model_file=os.path.join(models_dir, f"review_{n}.txt"))
        sig = SignalModels(b("q10"), b("q90"), [b(f"ens{k}") for k in range(meta["ensemble"])])
        return cls(sig, b("err"), b("live"), meta["features"])


def game_span(g):
    """Between the operator's first and last camera points (the start and end they set)."""
    return (g.t >= g.human.t[0]) & (g.t <= g.human.t[-1])


def abs_error(g, pred):
    err = np.full(len(g.t), np.nan)
    err[g.use] = np.abs(pred[g.use] - g.y[g.use])
    return err


def cross_game_signals(games):
    """Per game: (camera x, signals, |error|), every model trained on the other games."""
    out = []
    for g in games:
        others = [o for o in games if o is not g]
        X = np.vstack([o.X[o.use] for o in others])
        y = np.concatenate([o.y[o.use] for o in others])
        pred = finish(_fit(X, y).predict(g.X), g.max_x)
        S, names = SignalModels.fit(X, y).signals(g, pred)
        out.append((pred, S, abs_error(g, pred)))
    return out, names


def fit_err_live(games, rows):
    """Error and live models from cross-game rows of the given games."""
    S_err = np.vstack([S[~np.isnan(e)] for _, S, e in rows])
    e_err = np.concatenate([uniform_filter1d(np.nan_to_num(e), int(2 * HZ))[~np.isnan(e)] for _, _, e in rows])
    err_model = _fit(S_err, e_err, objective="l2")
    S_live = np.vstack([S[game_span(g)] for g, (_, S, _) in zip(games, rows)])
    y_live = np.concatenate([g.use[game_span(g)] for g in games])
    live_model = lgb.LGBMClassifier(**dict(PARAMS, objective="binary"), verbose=-1).fit(S_live, y_live)
    return err_model, live_model


def train(games):
    """Production review models from every hand-tracked game (at least two)."""
    games = [g for g in games if g.use.any()]
    if len(games) < 2:
        raise SystemExit("review models need at least two hand-tracked games")
    rows, names = cross_game_signals(games)
    err_model, live_model = fit_err_live(games, rows)
    X = np.vstack([g.X[g.use] for g in games])
    y = np.concatenate([g.y[g.use] for g in games])
    return ReviewModels(SignalModels.fit(X, y), err_model, live_model, names)


def segments(t, score, start_ms=None, end_ms=None):
    """All review segments, best first: [(t0_ms, t1_ms, expected px*s)]."""
    s = uniform_filter1d(score, int(SEG_SMOOTH_S * HZ), mode="nearest")
    inside = np.ones(len(t), bool)
    if start_ms is not None:
        inside &= t >= start_ms
    if end_ms is not None:
        inside &= t <= end_ms
    hot = inside & (s > np.percentile(s[inside], SEG_PCT))
    d = np.diff(np.concatenate([[0], hot.astype(int), [0]]))
    merged = []
    for a, b in zip(np.where(d > 0)[0], np.where(d < 0)[0]):
        if merged and (t[a] - t[merged[-1][1] - 1]) < SEG_MERGE_S * 1000:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    out = []
    pad = int(SEG_PAD_S * HZ)
    for a, b in merged:
        a, b = max(0, a - pad), min(len(t), b + pad)
        if (t[b - 1] - t[a]) >= SEG_MIN_S * 1000:
            out.append((float(t[a]), float(t[b - 1]), float(s[a:b].sum() / HZ)))
    return sorted(out, key=lambda r: -r[2])


def take(segs, budget_s):
    keep, used = [], 0.0
    for r in segs:
        if used >= budget_s:
            break
        keep.append(r)
        used += (r[1] - r[0]) / 1000
    return keep


REASONS = (("speed", "fast transition"), ("spread", "unsure where play is"),
           ("ens_std", "models disagree"), ("count_std", "player detections unstable"))


def describe(t, S, names, segs):
    """A short reason per segment: the signal most above its game-wide median."""
    out = []
    z = {}
    for col, _ in REASONS:
        v = S[:, names.index(col)]
        med, iqr = np.median(v), np.subtract(*np.percentile(v, [75, 25])) or 1.0
        z[col] = (v - med) / iqr
    for t0, t1, total in segs:
        m = (t >= t0) & (t <= t1)
        col = max(REASONS, key=lambda r: z[r[0]][m].mean())
        out.append(col[1])
    return out


def reviewed(g, pred, segs):
    """The path after a perfect review of segs: human x inside them."""
    out = pred.copy()
    for t0, t1, _ in segs:
        m = (g.t >= t0) & (g.t <= t1) & g.use
        out[m] = g.y[m]
    return out


def main():
    from sklearn.metrics import roc_auc_score
    from .game import open_game
    from .metrics import evaluate
    games = [open_game(s) for s in sys.argv[1:]]
    assert len(games) >= 2, "needs at least two hand-tracked games"
    rows, _ = cross_game_signals(games)
    minutes = (0, 1, 2, 3, 5, 7.5, 10, 15, 20)
    for i, g in enumerate(games):
        others = [j for j in range(len(games)) if j != i]
        err_model, live_model = fit_err_live([games[j] for j in others], [rows[j] for j in others])
        pred, S, err = rows[i]
        rm = ReviewModels(None, err_model, live_model, None)
        e = np.maximum(err_model.predict(S), 0)
        live = uniform_filter1d(live_model.predict_proba(S)[:, 1], int(2 * HZ), mode="nearest")
        score = e * live
        span = game_span(g)
        ok = ~np.isnan(err)
        print(f"\n== {g.video_dir}: camera MAE {evaluate(g.t, pred, g.human)['mae']:.0f} (trained on the other games)")
        print(f"  Spearman(expected, actual error) on framed time {spearmanr(e[ok], err[ok]).correlation:.2f}; "
              f"live-play AUC {roc_auc_score(g.use[span], live[span]):.2f}")
        oracle = np.where(ok, uniform_filter1d(np.nan_to_num(err), int(2 * HZ)), 0)
        cands = {"predicted": segments(g.t, score, g.human.t[0], g.human.t[-1]),
                 "oracle": segments(g.t, oracle, g.human.t[0], g.human.t[-1])}
        print("  review min | MAE predicted / oracle | within 300px predicted | segments")
        for mins in minutes:
            r = {k: evaluate(g.t, reviewed(g, pred, take(v, mins * 60)), g.human) for k, v in cands.items()}
            print(f"  {mins:9.1f}  | {r['predicted']['mae']:5.0f} / {r['oracle']['mae']:5.0f}     "
                  f"| {r['predicted']['within300']:6.1%}                | {len(take(cands['predicted'], mins * 60))}")


if __name__ == "__main__":
    main()
