"""Hand-built framing methods from on-ice player positions.

Run:  python -m af.simple_methods
"""

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

from .human import HumanPath
from .metrics import evaluate, fmt
from .players import Detections

PKL = r"D:/RawVideo/autoframe/DJI_20260822110551_0015_D.pkl"
DETS = r"D:/temp/af/dets"
EDGES = r"D:/temp/af/icemask/ice_edges.npz"
HZ = 5.0


def robust_centre(xs_list, q=0.5, fill=np.nan):
    out = np.full(len(xs_list), fill)
    for i, xs in enumerate(xs_list):
        if len(xs):
            out[i] = np.quantile(xs, q)
    return out


def fill_nan(a):
    idx = np.where(~np.isnan(a))[0]
    if len(idx) == 0:
        return np.zeros_like(a)
    return np.interp(np.arange(len(a)), idx, a[idx])


def smooth(a, seconds):
    n = max(1, int(round(seconds * HZ)))
    return uniform_filter1d(a, n, mode="nearest")


def stretch(x, max_x, centre, gain):
    """Push away from centre: the operator commits to end zones."""
    return np.clip(centre + (x - centre) * gain, 0, max_x)


def main():
    human = HumanPath(PKL)
    human.set_frame_size(3840, 2880)
    det = Detections(DETS, EDGES, human.max_x)
    t = det.times
    print(f"{len(t)} samples, {t[0]/1000:.0f}-{t[-1]/1000:.0f}s; boxes {len(det.boxes)}")
    keep = det.on_ice()
    print(f"kept {keep.sum()} / {len(keep)} boxes; mean on-ice per sample {keep.sum()/len(t):.1f}")
    xs = det.per_sample(keep)
    mx = human.max_x

    raw_med = np.clip(fill_nan(robust_centre(xs)), 0, mx)
    raw_mean = np.clip(fill_nan(np.array([x.mean() if len(x) else np.nan for x in xs])), 0, mx)
    for name, base in (("median", raw_med), ("mean", raw_mean)):
        for sm in (0, 2, 4):
            v = smooth(base, sm) if sm else base
            print(fmt(f"{name} sm{sm}", evaluate(t, v, human)))
    base = smooth(raw_med, 2)
    for gain in (1.3, 1.6, 2.0, 2.5):
        print(fmt(f"median sm2 stretch{gain}", evaluate(t, stretch(base, mx, mx / 2, gain), human)))


if __name__ == "__main__":
    main()
