"""Cross-game evaluation of the learned framer.

  python -m af.transfer g1 g2

Reports, for each game: the median baseline, the human path delayed 1 s / 2 s,
within-game grouped CV, a model trained only on each other game, and a model
trained on every other game plus out-of-fold segments of this one.
"""

import sys

import lightgbm as lgb
import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.model_selection import GroupKFold

from .games import Game
from .metrics import evaluate, fmt
from .simple_methods import HZ

PARAMS = dict(objective="l1", n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0)
FOLDS = 6


def fit(Xs, ys):
    return lgb.LGBMRegressor(**PARAMS, verbose=-1).fit(np.vstack(Xs), np.concatenate(ys))


def finish(g, p):
    return uniform_filter1d(np.clip(p, 0, g.max_x), int(2 * HZ), mode="nearest")


def oof(g, others=()):
    """Out-of-fold predictions on g's labelled samples, optionally with other games' labels in every fold."""
    p = np.full(len(g.t), np.nan)
    ui = np.where(g.use)[0]
    for tr, te in GroupKFold(FOLDS).split(ui, groups=g.sid[ui]):
        m = fit([g.X[ui[tr]]] + [o.X[o.use] for o in others], [g.y[ui[tr]]] + [o.y[o.use] for o in others])
        p[ui[te]] = m.predict(g.X[ui[te]])
    idx = np.where(~np.isnan(p))[0]
    return np.interp(np.arange(len(p)), idx, p[idx])


def main():
    names = sys.argv[1:] or ["g1", "g2"]
    games = [Game(n) for n in names]
    for g in games:
        h = g.human
        print(f"== {g.name}: {g.use.sum() / HZ / 60:.1f} labelled min")
        print(fmt("  median sm2", evaluate(g.t, finish(g, g.med), h)))
        for d in (1, 2):
            print(fmt(f"  human +{d}s", evaluate(g.t, np.interp(g.t - d * 1000, h.t, h.x), h)))
        cv = finish(g, oof(g))
        print(fmt("  within-game CV", evaluate(g.t, cv, h)))
        np.savez(f"D:/temp/af/feats/pred_{g.name}_cv.npz", t=g.t, pred=cv)
        others = [o for o in games if o is not g]
        for o in others:
            p = finish(g, fit([o.X[o.use]], [o.y[o.use]]).predict(g.X))
            print(fmt(f"  trained on {o.name} only", evaluate(g.t, p, h)))
            np.savez(f"D:/temp/af/feats/pred_{g.name}_from_{o.name}.npz", t=g.t, pred=p)
        if others:
            p = finish(g, oof(g, others))
            print(fmt(f"  CV + all of {'+'.join(o.name for o in others)}", evaluate(g.t, p, h)))
            np.savez(f"D:/temp/af/feats/pred_{g.name}_cv_plus.npz", t=g.t, pred=p)


if __name__ == "__main__":
    main()
