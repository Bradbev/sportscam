"""One game: its videos, calibration, detection cache, features and (if hand-tracked) labels.

Cache layout (default DIR/autotrack_cache):
  dets/       per-video detection npz + half-res band proxy, meta.json
  icemask/    ice_edges.npz and preview images
  features.npz
Features are rebuilt when the calibration changes (it moves every camera x).
"""

import json
import os

import numpy as np

from .calibration import calib_key, load_calibration
from .features import build_features, fill_nan, robust_centre, segment_ids
from .geometry import ColToX, max_x
from .human import HumanPath
from .motion import summarise, velocities
from .players import Detections

CACHE_NAME = "autotrack_cache"


class Game:
    def __init__(self, video_dir, cache_dir=None, rebuild=False):
        self.video_dir = video_dir
        self.cache = cache_dir or os.path.join(video_dir, CACHE_NAME)
        self.dets = os.path.join(self.cache, "dets")
        self.edges = os.path.join(self.cache, "icemask", "ice_edges.npz")
        self.calib, pkl = load_calibration(video_dir)
        meta = json.load(open(os.path.join(self.dets, "meta.json")))
        self.files = meta["files"]
        self.w, self.h = self.files[0]["w"], self.files[0]["h"]
        self.max_x = max_x(self.calib, self.w, self.h)
        has_path = pkl is not None and len(pkl["camera_path"].camera_targets) > 1
        self.human = HumanPath(pkl) if has_path else None
        if self.human is not None:
            self.human.set_frame_size(self.w, self.h)

        path = os.path.join(self.cache, "features.npz")
        key = calib_key(self.calib)
        if os.path.exists(path) and not rebuild:
            d = np.load(path, allow_pickle=True)
            if str(d["calib"]) == key:
                self.t, self.X, self.names, self.med = d["t"], d["X"], list(d["names"]), d["med"]
                self._labels()
                return
        e = np.load(self.edges)
        det = Detections(self.dets, self.edges, self.max_x, ColToX(self.calib, self.w, self.h, int(e["band0"])))
        keep = det.on_ice()
        xs = det.per_sample(keep)
        extra, enames = summarise(velocities(det, keep, with_x=True))
        self.X, self.names = build_features(xs, self.max_x, extra, enames)
        self.t = det.times
        self.med = np.clip(fill_nan(robust_centre(xs)), 0, self.max_x)
        np.savez(path, t=self.t, X=self.X, names=np.array(self.names), med=self.med, calib=key)
        self._labels()

    def _labels(self):
        """Human camera x on the samples the human actually framed (use), else NaN."""
        self.y = np.full(len(self.t), np.nan)
        self.use = np.zeros(len(self.t), bool)
        self.sid = np.full(len(self.t), -1)
        if self.human is None:
            return
        self.sid = segment_ids(self.human, self.t)
        self.use = self.human.scored_mask(self.t) & (self.sid >= 0)
        self.y[self.use] = self.human.x_at(self.t[self.use])


def open_game(spec):
    """'DIR' or 'DIR=CACHE' (for games whose cache is not DIR/autotrack_cache)."""
    d, _, c = spec.partition("=")
    return Game(d, c or None)
