"""The rink scoreboard, read from the half-res band proxy: clock running, score changes.

1. Locate the clock: in a search window at the top right of the band, the seconds
   digits are the pixels that change with a strong 1 Hz rhythm.
2. Score digits sit at a fixed offset below the clock (same board, same mount):
   home (blue) to the left, away (pink) to the right.
3. A score change is a lasting step in a digit's appearance: the 7 s median image
   10 s after differs from 10 s before, while each side is itself steady.  Without
   reading the digits the board's score is counted from 0-0, so every change is
   also offered for review; the board may be wrong, late, or off.

  python -m af.scoreboard DIR[=CACHE] ...
"""

import glob
import json
import os
import subprocess
import sys

import numpy as np
from scipy.ndimage import binary_dilation, label, median_filter, uniform_filter1d

SEARCH = (1900, 40, 420, 240)  # x0, y0, w, h in proxy px
# offsets from the seconds digits' top-left, (dx, dy, w, h), measured on two games
HOME_DIGIT = (-34, 38, 18, 24)
AWAY_DIGIT = (6, 38, 22, 24)
CLOCK_BOX = (-14, -2, 30, 16)   # minutes and seconds
STEP_S = 10
STEADY_S = 10
STEADY_RATIO = 3   # the change must be this much bigger than the drift either side
MIN_STEP = 4       # x the median change: score digits change at 6-13, board on/off at 15+
BOARD_RESET = 15   # bigger steps are the board being switched on/off or reset


def _proxies(cache):
    meta = json.load(open(os.path.join(cache, "dets", "meta.json")))
    out = []
    for f in meta["files"]:
        stem = os.path.splitext(os.path.basename(f["path"]))[0]
        n = len(np.load(os.path.join(cache, "dets", stem + ".npz"))["times_ms"])
        out.append((os.path.join(cache, "dets", stem + "_band.mp4"), n, f))
    return out


def read_crop(path, box, start_s=None, dur_s=None, n=None):
    x0, y0, w, h = box
    cmd = ["ffmpeg", "-v", "error"]
    if start_s is not None:
        cmd += ["-ss", f"{start_s:.2f}", "-t", f"{dur_s:.2f}"]
    cmd += ["-i", path, "-vf", f"crop={w}:{h}:{x0}:{y0}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    rgb = np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.uint8).reshape(-1, h, w, 3)
    fr = rgb.max(axis=3)  # brightness that keeps blue digits bright (grey makes them dim)
    if n is not None:
        fr = fr[:n]
        if len(fr) < n:
            fr = np.concatenate([fr, np.repeat(fr[-1:], n - len(fr), 0)])
    return fr


def locate_clock(cache, hz=5):
    """Top-left (x, y) of the clock's seconds digits in proxy px, or None."""
    path, n, f = max(_proxies(cache), key=lambda r: r[1])
    mid = n / hz / 2
    fr = read_crop(path, SEARCH, mid - 90, 180).astype(np.float32)
    d = np.abs(np.diff(fr, axis=0))  # frame-to-frame change
    k = np.arange(len(d))
    one_hz = np.abs((d * np.exp(-2j * np.pi * k / hz)[:, None, None]).mean(0))
    score = one_hz / (d.mean(0) + 1.0)
    mask = score > max(0.25, 0.5 * score.max())
    lab, nlab = label(binary_dilation(mask, iterations=2))
    if nlab == 0:
        return None
    sizes = [(lab == i).sum() for i in range(1, nlab + 1)]
    ys, xs = np.where(lab == 1 + int(np.argmax(sizes)))
    return SEARCH[0] + int(xs.min()), SEARCH[1] + int(ys.min())


def _box(origin, rel):
    return origin[0] + rel[0], origin[1] + rel[1], rel[2], rel[3]


def read_game(cache, origin):
    """Per sample: home digit, away digit and clock crops (uint8 stacks)."""
    out = {"home": [], "away": [], "clock": []}
    for path, n, _ in _proxies(cache):
        for key, rel in (("home", HOME_DIGIT), ("away", AWAY_DIGIT), ("clock", CLOCK_BOX)):
            out[key].append(read_crop(path, _box(origin, rel), n=n))
    return {k: np.concatenate(v) for k, v in out.items()}


def per_second(frames, hz=5):
    n = len(frames) // hz
    return np.median(frames[:n * hz].reshape(n, hz, *frames.shape[1:]).astype(np.float32), axis=1)


def steps(frames, hz=5):
    """[(second index, strength)] of lasting changes in a digit crop."""
    s = per_second(frames, hz)
    m = median_filter(s, size=(7, 1, 1), mode="nearest")
    n = len(m)
    def diff(a, b):  # mean of the most-changed quarter of pixels: one glyph is a small part of the crop
        d = np.abs(m[a] - m[b]).reshape(n, -1)
        return np.sort(d, axis=1)[:, -d.shape[1] // 4:].mean(axis=1)
    idx = np.arange(n)
    a, b = np.clip(idx - STEP_S, 0, n - 1), np.clip(idx + STEP_S, 0, n - 1)
    a2, b2 = np.clip(idx - STEP_S - STEADY_S, 0, n - 1), np.clip(idx + STEP_S + STEADY_S, 0, n - 1)
    change = diff(b, a)
    steady = np.maximum(diff(a, a2), diff(b, b2))
    noise = np.median(change) + 1e-3
    score = np.where(change > STEADY_RATIO * steady, change / noise, 0)
    out = []
    for i in np.argsort(-score):
        if score[i] < MIN_STEP:
            break
        if all(abs(i - j) > 30 for j, _ in out):
            out.append((int(i), float(score[i])))
    return sorted(out)


def clock_running(clock_frames, hz=5):
    """Per sample: True while the clock's seconds are ticking (1 s lag change)."""
    f = clock_frames.astype(np.float32)
    d = np.zeros(len(f))
    d[hz:] = np.abs(f[hz:] - f[:-hz]).mean(axis=(1, 2))
    s = uniform_filter1d(d, 2 * hz)
    # half the running level: the clock runs most of a game (70-90% of samples), so the
    # 80th percentile is a running level; a stopped clock changes by a third of that or less
    return s > 0.5 * np.percentile(s, 80)


def main():
    from .game import open_game
    for spec in sys.argv[1:]:
        g = open_game(spec)
        origin = locate_clock(g.cache)
        print(f"== {g.video_dir}: clock seconds at proxy {origin}")
        if origin is None:
            continue
        crops = read_game(g.cache, origin)
        np.savez_compressed(os.path.join(g.cache, "scoreboard.npz"), origin=origin, **crops)
        for side in ("home", "away"):
            ev = steps(crops[side])
            print(f"  {side}: {[(s, round(v, 1)) for s, v in ev]}")
        run = clock_running(crops["clock"])
        print(f"  clock running {run.mean():.0%} of samples")


if __name__ == "__main__":
    main()
