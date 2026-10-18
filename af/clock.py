"""Is the scoreboard game clock running?  A play / stoppage signal.

The scoreboard sits inside the detection band, so the half-res band proxies
already contain it.  The seconds digits change once a second while the clock
runs and freeze at whistles, so the per-sample difference between the clock
region now and 1 s earlier is high during play and near zero in stoppages.

  python -m af.clock DETS_DIR X0 Y0 W H OUT.npz     # region in proxy pixels
"""

import glob
import os
import subprocess
import sys

import numpy as np


def read_region(path, x0, y0, w, h):
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"crop={w}:{h}:{x0}:{y0},format=gray",
                        "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, h, w)


def change_signal(frames, lag=5):
    f = frames.astype(np.float32)
    d = np.zeros(len(f))
    d[lag:] = np.abs(f[lag:] - f[:-lag]).mean(axis=(1, 2))
    return d


def main():
    dets, x0, y0, w, h, out = sys.argv[1], *map(int, sys.argv[2:6]), sys.argv[6]
    sig, frames_all = [], []
    for p in sorted(glob.glob(os.path.join(dets, "*_band.mp4"))):
        n = len(np.load(p.replace("_band.mp4", ".npz"))["times_ms"])
        fr = read_region(p, x0, y0, w, h)[:n]
        if len(fr) < n:  # pad if the proxy is a frame short
            fr = np.concatenate([fr, np.repeat(fr[-1:], n - len(fr), 0)])
        frames_all.append(fr)
        sig.append(change_signal(fr))
    np.savez(out, diff=np.concatenate(sig), frames=np.concatenate(frames_all))


if __name__ == "__main__":
    main()
