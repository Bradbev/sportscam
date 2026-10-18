"""Exact mapping between band columns and app camera x, per game calibration.

For every camera x the app shows a 1920x720 crop of the frame rotated by
rotation + angle(x).  Its centre, mapped back through that rotation and forward
through the base-rotation band transform, lands on one band column.  Inverting
that (monotonic) curve gives the camera x that centres a given column, using
the game's own calibration instead of a linear approximation.
"""

import math

import cv2
import numpy as np

from .extract_dets import band_matrix
from util import rotate_point

OUT_W, OUT_H = 1920, 720


def max_x(pkl, w, h):
    a = math.radians(pkl["rotation"] + pkl["angle_right"])
    return int(math.sin(a) * h + math.cos(a) * w - OUT_W)


def crop_matrix(pkl, w, h, x):
    """Raw frame -> crop pixels affine, as the app renders camera x (zoom 1)."""
    mx = max_x(pkl, w, h)
    xp = [0, mx // 2 + pkl["center_x_offset"], mx]
    x = int(np.clip(x, 0, mx))
    angle = float(np.interp(x, xp, [pkl["angle_left"], 0, pkl["angle_right"]]))
    y_live = -pkl["top_of_roi"] + float(np.interp(x, xp, [pkl["top_of_roi_left"], 0, pkl["top_of_roi_right"]]))
    size = np.array([w, h], dtype=np.float64)
    M = cv2.getRotationMatrix2D(tuple(size / 2.0), pkl["rotation"] + angle, 1.0)
    new = np.abs(M[:, :2]) @ size
    M[:, -1] += (new - size) / 2.0
    rot_w = int(new[0])
    y = int(rotate_point((x + OUT_W // 2 - rot_w // 2, y_live), -angle)[1])
    x = int(min(x, rot_w - OUT_W))
    M[0, 2] -= x
    M[1, 2] -= y
    return M


def centre_cols(pkl, w, h, band_y0, step=8):
    """(camera x values, band column of each crop's centre)."""
    mx = max_x(pkl, w, h)
    B, _ = band_matrix(w, h, pkl["rotation"], band_y0)
    xs = np.arange(0, mx + 1, step, dtype=np.float64)
    cols = np.empty_like(xs)
    for i, x in enumerate(xs):
        inv = cv2.invertAffineTransform(crop_matrix(pkl, w, h, x))
        raw = inv @ np.array([OUT_W / 2, OUT_H / 2, 1.0])
        cols[i] = (B @ np.array([raw[0], raw[1], 1.0]))[0]
    return xs, cols


class ColToX:
    """Band column -> camera x that centres it; linear extrapolation outside."""

    def __init__(self, pkl, w, h, band_y0):
        self.xs, self.cols = centre_cols(pkl, w, h, band_y0)
        assert np.all(np.diff(self.cols) > 0), "centre column not monotonic in x"
        n = len(self.xs) // 10
        self.sl = np.polyfit(self.cols[:n], self.xs[:n], 1)[0]
        self.sr = np.polyfit(self.cols[-n:], self.xs[-n:], 1)[0]

    def __call__(self, col):
        col = np.asarray(col, dtype=np.float64)
        x = np.interp(col, self.cols, self.xs)
        x = np.where(col < self.cols[0], self.xs[0] + (col - self.cols[0]) * self.sl, x)
        return np.where(col > self.cols[-1], self.xs[-1] + (col - self.cols[-1]) * self.sr, x)
