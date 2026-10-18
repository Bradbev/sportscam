"""Automatic ice mask in the base-rotation band, from a temporal median.

The camera is fixed, so the per-pixel median over frames spread across the game
is an (almost) empty rink.  Ice is the large bright, low-saturation region; the
mask is its largest connected component, hole-filled.  A detection is 'on ice'
when its feet point (box bottom centre) falls inside the dilated mask, which
rejects spectators in the far stands (feet above the far boards) and people on
the camera side of the near boards (box extends below the near boards).
"""

import json
import os
import subprocess

import cv2
import numpy as np

from .extract_dets import ROT_BAND, band_matrix, probe


def sample_band_frames(path, rotation, times_s):
    w, h, _, _ = probe(path)
    M, rot_w = band_matrix(w, h, rotation, ROT_BAND[0])
    out = []
    for t in times_s:
        buf = subprocess.check_output([
            "ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", path, "-frames:v", "1",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-"])
        if len(buf) < w * h * 3:
            continue
        frame = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        out.append(cv2.warpAffine(frame, M, (rot_w, ROT_BAND[1] - ROT_BAND[0])))
    return out


def _envelope(xs, ys, upper, deg=6, iters=30):
    """Smooth polynomial envelope of an edge that occluders can only shrink.

    upper=True: the true edge is the minimum row (far boards); points below the
    fit are occluded and get pulled up to it.  upper=False: mirror image for the
    near boards.
    """
    u = (xs - xs.mean()) / xs.std()
    y = ys.copy()
    for _ in range(iters):
        fit = np.polyval(np.polyfit(u, y, deg), u)
        y = np.where(upper, np.minimum(ys, fit), np.maximum(ys, fit))
    return np.polyval(np.polyfit(u, y, deg), u)


def build(frames):
    med = np.median(np.stack(frames), axis=0).astype(np.uint8)
    hsv = cv2.cvtColor(med, cv2.COLOR_BGR2HSV)
    bright = (hsv[..., 2] > 150) & (hsv[..., 1] < 60)
    m = bright.astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ice = (lab == biggest).astype(np.uint8) * 255
    # Rebuild from per-column top/bottom edges.  Occluders (bench crowd in front
    # of the near boards, players against the far boards) can only make the ice
    # look shorter, so a rolling max of the bottom edge and a rolling min of the
    # top edge recover the boards lines.
    cols = np.where(ice.any(axis=0))[0]
    top = np.full(ice.shape[1], -1)
    bot = np.full(ice.shape[1], -1)
    for c in cols:
        rows = np.where(ice[:, c])[0]
        top[c], bot[c] = rows[0], rows[-1]
    xs = cols.astype(np.float64)
    top_s = np.full(ice.shape[1], 0.0)
    bot_s = np.full(ice.shape[1], 0.0)
    top_s[cols] = _envelope(xs, top[cols].astype(np.float64), upper=True)
    bot_s[cols] = _envelope(xs, bot[cols].astype(np.float64), upper=False)
    # Extend both curves past the detected ice linearly (the ice behind the nets is
    # often too shadowed to segment), so the on-ice test covers the full width.
    w = ice.shape[1]
    lo, hi = cols[0], cols[-1]
    # Slopes are taken a little way in: polynomial fits misbehave at their ends.
    a, b = 300, 800
    for arr in (top_s, bot_s):
        arr[lo:lo + a] = arr[lo + a] + (arr[lo + b] - arr[lo + a]) / (b - a) * (np.arange(lo, lo + a) - lo - a)
        arr[hi - a + 1:hi + 1] = arr[hi - a] + (arr[hi - a] - arr[hi - b]) / (b - a) * (np.arange(hi - a + 1, hi + 1) - hi + a)
        sl = (arr[lo + b] - arr[lo + a]) / (b - a)
        sr = (arr[hi - a] - arr[hi - b]) / (b - a)
        arr[:lo] = arr[lo] + sl * (np.arange(lo) - lo)
        arr[hi + 1:] = arr[hi] + sr * (np.arange(hi + 1, w) - hi)
    filled = np.zeros_like(ice)
    for c in range(w):
        a, b = int(max(0, top_s[c])), int(min(ice.shape[0] - 1, bot_s[c]))
        if b > a:
            filled[a:b + 1, c] = 255
    return med, filled, top_s, bot_s


def build_cache(path, out_dir, rotation):
    """Ice edges from 31 frames spread across one source video, plus preview images."""
    dur = float(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path]))
    frames = sample_band_frames(path, rotation, np.linspace(30, dur - 30, 31))
    med, mask, top, bot = build(frames)
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, "median.png"), med)
    cv2.imwrite(os.path.join(out_dir, "ice_mask.png"), mask)
    vis = med.copy()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0, 0, 255), 3)
    cv2.imwrite(os.path.join(out_dir, "ice_mask_vis.jpg"), cv2.resize(vis, None, fx=0.3, fy=0.3))
    np.savez(os.path.join(out_dir, "ice_edges.npz"), top=top, bot=bot, band0=ROT_BAND[0])
    return int((mask > 0).sum())


if __name__ == "__main__":
    import sys
    path, out_dir = sys.argv[1], sys.argv[2]
    rotation = float(sys.argv[3]) if len(sys.argv) > 3 else 37.0
    print("ice px", build_cache(path, out_dir, rotation))
