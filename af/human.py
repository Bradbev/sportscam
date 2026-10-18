"""Load the human reference path and turn it into a scoreable time series.

The app samples the operator's camera every ~1.04s (auto record).  Large gaps
between targets are stretches the operator cut out of the final video (camera
cuts, stoppages); they are excluded from scoring because the human never framed
them.
"""

import os
import pickle
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

GAP_MS = 5000  # a gap longer than this between human targets is not scored


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


class HumanPath:
    def __init__(self, pkl_path):
        self.pkl = load_pkl(pkl_path)
        self.path = self.pkl["camera_path"]
        ct = self.path.camera_targets
        self.t = np.array([c.time for c in ct], dtype=np.float64)
        self.x = np.array([c.x for c in ct], dtype=np.float64)
        self.cut = np.array([c.cut_to for c in ct])
        self.max_x = None  # filled by set_frame_size

    def set_frame_size(self, w, h, out_w=1920):
        import math
        a = math.radians(self.pkl["rotation"] + self.pkl["angle_right"])
        self.max_x = int(math.sin(a) * h + math.cos(a) * w - out_w)

    def x_at(self, times_ms):
        """Exactly what the app renders (includes its direction-change easing)."""
        get = self.path.get_camera_at_time
        return np.array([get(float(t))[0].x for t in times_ms])

    def scored_mask(self, times_ms):
        """True where the human was actively framing (inside the path, not in a gap/cut)."""
        times_ms = np.asarray(times_ms)
        idx = np.searchsorted(self.t, times_ms, side="right") - 1
        ok = (idx >= 0) & (idx < len(self.t) - 1)
        idx_c = np.clip(idx, 0, len(self.t) - 2)
        gap = self.t[idx_c + 1] - self.t[idx_c]
        ok &= gap <= GAP_MS
        ok &= ~self.cut[idx_c]
        return ok

    def segments(self):
        """[(start_ms, end_ms)] continuous framed stretches."""
        segs = []
        start = self.t[0]
        for i in range(len(self.t) - 1):
            if self.t[i + 1] - self.t[i] > GAP_MS or self.cut[i]:
                segs.append((start, self.t[i]))
                start = self.t[i + 1]
        segs.append((start, self.t[-1]))
        return segs
