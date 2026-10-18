"""Cross-game evaluation of the camera-x model on hand-tracked folders.

  python -m af.transfer DIR[=CACHE] DIR[=CACHE] ...

Reports, for each game: the median baseline, the human path delayed 1 s / 2 s,
within-game grouped CV, a model trained only on each other game, and a model
trained on every other game plus out-of-fold segments of this one.
Predictions are saved to CACHE/pred_*.npz for af.compare.
"""

import os
import sys

import numpy as np
from sklearn.model_selection import GroupKFold

from .features import fill_nan
from .game import open_game
from .metrics import evaluate, fmt
from .model import finish, fit

FOLDS = 6


def oof(g, others=()):
    """Out-of-fold predictions on g's labelled samples, optionally with other games' labels in every fold."""
    p = np.full(len(g.t), np.nan)
    ui = np.where(g.use)[0]
    for tr, te in GroupKFold(FOLDS).split(ui, groups=g.sid[ui]):
        m = fit([g.X[ui[tr]]] + [o.X[o.use] for o in others], [g.y[ui[tr]]] + [o.y[o.use] for o in others])
        p[ui[te]] = m.predict(g.X[ui[te]])
    return fill_nan(p)


def save(g, tag, p):
    np.savez(os.path.join(g.cache, f"pred_{tag}.npz"), t=g.t, pred=p)


def main():
    games = [open_game(s) for s in sys.argv[1:]]
    for i, g in enumerate(games):
        h = g.human
        if h is None:
            print(f"== {g.video_dir}: no hand-tracked path, skipped")
            continue
        print(f"== game {i}: {g.video_dir}, {g.use.sum() / 5 / 60:.1f} labelled min")
        print(fmt("  median sm2", evaluate(g.t, finish(g.med, g.max_x), h)))
        for d in (1, 2):
            print(fmt(f"  human +{d}s", evaluate(g.t, np.interp(g.t - d * 1000, h.t, h.x), h)))
        cv = finish(oof(g), g.max_x)
        print(fmt("  within-game CV", evaluate(g.t, cv, h)))
        save(g, "cv", cv)
        others = [o for o in games if o is not g and o.human is not None]
        for j, o in enumerate(others):
            p = finish(fit([o.X[o.use]], [o.y[o.use]]).predict(g.X), g.max_x)
            print(fmt(f"  trained on {os.path.basename(o.video_dir.rstrip('/'))} only", evaluate(g.t, p, h)))
            save(g, f"from_{j}", p)
        if others:
            p = finish(oof(g, others), g.max_x)
            print(fmt("  CV + all other games", evaluate(g.t, p, h)))
            save(g, "cv_plus", p)


if __name__ == "__main__":
    main()
