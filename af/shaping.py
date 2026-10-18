"""Post-processing that makes a camera target behave like an operator:
hold still through small wobble, then pan decisively.

Operates on a 5 Hz target series.  Offline, so it may look ahead.
"""

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

HZ = 5.0


def hold_and_pan(target, deadband=150.0, vmax=900.0, settle=40.0):
    """Hold the camera until the target is more than `deadband` px away, then
    move toward it at up to `vmax` px/s until within `settle` px."""
    out = np.empty_like(target)
    cam = target[0]
    moving = False
    step = vmax / HZ
    for i, tgt in enumerate(target):
        d = tgt - cam
        if not moving and abs(d) > deadband:
            moving = True
        if moving:
            cam += np.clip(d, -step, step)
            if abs(tgt - cam) <= settle:
                moving = False
        out[i] = cam
    return out


def lookahead(x, seconds):
    """Shift a series earlier in time (use the future value now)."""
    n = int(round(seconds * HZ))
    if n == 0:
        return x
    return np.concatenate([x[n:], np.repeat(x[-1], n)]) if n > 0 else np.concatenate([np.repeat(x[0], -n), x[:n]])


def shape(x, med_s=0.0, smooth_s=0.0, deadband=0.0, vmax=900.0, settle=40.0, lead_s=0.0):
    y = x.copy()
    if med_s:
        y = median_filter(y, size=max(1, int(med_s * HZ)), mode="nearest")
    if smooth_s:
        y = uniform_filter1d(y, max(1, int(smooth_s * HZ)), mode="nearest")
    if lead_s:
        y = lookahead(y, lead_s)
    if deadband:
        y = hold_and_pan(y, deadband, vmax, settle)
    return y
