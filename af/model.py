"""The camera-x model: LightGBM (L1) on af.features, then 2 s smoothing.

Saved as LightGBM's text format plus a small json with the feature names, so
a model trained on one feature layout is never applied to another.
"""

import json
import os

import lightgbm as lgb
import numpy as np
from scipy.ndimage import uniform_filter1d

from .features import HZ

PARAMS = dict(objective="l1", n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0)
SMOOTH_S = 2.0
DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "models", "autotrack_lgbm.txt")


def fit(Xs, ys):
    return lgb.LGBMRegressor(**PARAMS, verbose=-1).fit(np.vstack(Xs), np.concatenate(ys))


def finish(pred, max_x):
    """Raw per-sample predictions -> the camera path that gets written."""
    return uniform_filter1d(np.clip(pred, 0, max_x), int(SMOOTH_S * HZ), mode="nearest")


def train(games):
    used = [g for g in games if g.use.any()]
    if not used:
        raise SystemExit("none of the games has a hand-tracked camera path")
    return fit([g.X[g.use] for g in used], [g.y[g.use] for g in used]), used


def save(m, names, path, trained_on):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    m.booster_.save_model(path)
    with open(path + ".json", "w") as f:
        json.dump({"features": names, "trained_on": trained_on, "params": PARAMS, "smooth_s": SMOOTH_S}, f, indent=1)


class Model:
    def __init__(self, path=DEFAULT_MODEL):
        if not os.path.exists(path):
            raise SystemExit(f"no model at {path}; train one with: python autotrack.py train DIR [DIR ...]")
        self.path = path
        self.booster = lgb.Booster(model_file=path)
        self.meta = json.load(open(path + ".json"))

    def predict(self, game):
        if list(game.names) != self.meta["features"]:
            raise SystemExit(f"{self.path} was trained on a different feature layout; retrain it")
        return finish(self.booster.predict(game.X), game.max_x)
