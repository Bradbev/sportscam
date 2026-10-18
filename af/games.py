"""The hand-tracked games, and their per-sample features and labels.

Each game is a human pkl plus its detection cache and ice edges.  Features are
cached in D:/temp/af/feats/<name>.npz because the motion matching is slow.

  python -m af.games g1 g2      # build / refresh the feature caches
"""

import os
import sys

import numpy as np

from .geometry import ColToX
from .human import HumanPath
from .learned import build_features, segment_ids
from .players import Detections

GAMES = {
    "g1": dict(pkl=r"D:/RawVideo/autoframe/DJI_20260822110551_0015_D.pkl",
               dets=r"D:/temp/af/dets", edges=r"D:/temp/af/icemask/ice_edges.npz"),
    "g2": dict(pkl=r"D:/temp/af/g2/DJI_20260815095044_0011_D.pkl",
               dets=r"D:/temp/af/g2/dets", edges=r"D:/temp/af/g2/icemask/ice_edges.npz"),
}
FEATS = r"D:/temp/af/feats"
W, H = 3840, 2880


class Game:
    def __init__(self, name, exact=True, rebuild=False):
        cfg = GAMES[name]
        self.name = name
        self.human = HumanPath(cfg["pkl"])
        self.human.set_frame_size(W, H)
        self.max_x = self.human.max_x
        cache = os.path.join(FEATS, f"{name}{'' if exact else '_lin'}.npz")
        if os.path.exists(cache) and not rebuild:
            d = np.load(cache, allow_pickle=True)
            self.t, self.X, self.y, self.use, self.sid = d["t"], d["X"], d["y"], d["use"], d["sid"]
            self.names = list(d["names"])
            self.med = d["med"]
            return
        e = np.load(cfg["edges"])
        col_map = ColToX(self.human.pkl, W, H, int(e["band0"])) if exact else None
        det = Detections(cfg["dets"], cfg["edges"], self.max_x, col_map)
        from .motion import summarise, velocities
        from .simple_methods import fill_nan, robust_centre
        keep = det.on_ice()
        xs = det.per_sample(keep)
        extra, enames = summarise(velocities(det, keep, with_x=True))
        self.X, self.names = build_features(xs, self.max_x, extra, enames)
        self.t = det.times
        self.sid = segment_ids(self.human, self.t)
        self.use = self.human.scored_mask(self.t) & (self.sid >= 0)
        self.y = np.full(len(self.t), np.nan)
        self.y[self.use] = self.human.x_at(self.t[self.use])
        self.med = np.clip(fill_nan(robust_centre(xs)), 0, self.max_x)
        os.makedirs(FEATS, exist_ok=True)
        np.savez(cache, t=self.t, X=self.X, y=self.y, use=self.use, sid=self.sid,
                 names=np.array(self.names), med=self.med)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        g = Game(n, rebuild=True)
        print(n, len(g.t), "samples,", g.use.sum(), "labelled,", g.X.shape[1], "features")
