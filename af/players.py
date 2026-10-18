"""Turn the raw detection cache into per-sample on-ice player positions.

Coordinates:
  col   column in the base-rotation full-res frame (width fixed_w, ~4799)
  x     app camera x (left edge of the 1920 crop) that would centre on col,
        using the linear mapping x = col / fixed_w * pan_span - 960.  The
        app's small per-x extra rotation is ignored; learned stages absorb it.
"""

import glob
import json
import os

import numpy as np

OUT_W = 1920


class Detections:
    def __init__(self, dets_dir, edges_path, max_x):
        self.meta = json.load(open(os.path.join(dets_dir, "meta.json"))) \
            if os.path.exists(os.path.join(dets_dir, "meta.json")) else None
        files = sorted(glob.glob(os.path.join(dets_dir, "*.npz")))
        times, boxes, confs, sid = [], [], [], []
        offset = 0
        for f in files:
            d = np.load(f)
            n = len(d["times_ms"])
            times.append(d["times_ms"])
            boxes.append(d["boxes"])
            confs.append(d["confs"])
            sid.append(np.repeat(np.arange(offset, offset + n), d["counts"]))
            offset += n
        self.times = np.concatenate(times)
        self.boxes = np.concatenate(boxes)
        self.confs = np.concatenate(confs)
        self.sample = np.concatenate(sid)
        e = np.load(edges_path)
        self.top, self.bot = e["top"], e["bot"]
        self.band0 = int(e["band0"])
        self.fixed_w = len(self.top)
        self.max_x = max_x
        self.pan_span = max_x + OUT_W

    def col_to_x(self, col):
        return col / self.fixed_w * self.pan_span - OUT_W / 2

    def on_ice(self, conf_min=0.25, top_tol=15, bot_tol=30, min_h=20):
        b = self.boxes
        cx = 0.5 * (b[:, 0] + b[:, 2])
        feet = b[:, 3] - self.band0
        h = b[:, 3] - b[:, 1]
        c = np.clip(cx.astype(int), 0, self.fixed_w - 1)
        keep = (self.confs >= conf_min) & (h >= min_h)
        keep &= feet >= self.top[c] - top_tol
        keep &= feet <= self.bot[c] + bot_tol
        return keep

    def per_sample(self, keep):
        """List over samples of arrays of player camera-centre x (x + 960 space -> x)."""
        b = self.boxes[keep]
        s = self.sample[keep]
        x = self.col_to_x(0.5 * (b[:, 0] + b[:, 2]))
        order = np.argsort(s, kind="stable")
        s, x = s[order], x[order]
        bounds = np.searchsorted(s, np.arange(len(self.times) + 1))
        return [x[bounds[i]:bounds[i + 1]] for i in range(len(self.times))]

    def rows(self, keep):
        """(sample_index, x, box) arrays for kept detections, for feature building."""
        b = self.boxes[keep]
        return self.sample[keep], self.col_to_x(0.5 * (b[:, 0] + b[:, 2])), b, self.confs[keep]
