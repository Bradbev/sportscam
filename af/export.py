"""Write an automatic camera path as the sidecar the app copies in: DIR/auto_path.pkl.

The sidecar holds the calibration the path was computed with, the camera path
(a target every `step_ms`, y derived from x exactly as the app does at zoom 1),
and an `autotrack` dict describing how it was made.  It never touches the
operator's own `<first video>.pkl`.

  python -m af.export PRED.npz VIDEO_DIR [--max-x N]    (normally run via autotrack.py)
"""

import argparse
import os
import pickle
import time

import numpy as np

from .calibration import CALIB_KEYS, load_calibration
from camerapath import CameraPath
from cameratarget import CameraTarget

SIDECAR = "auto_path.pkl"


def make_sidecar(calib, times_ms, xs, max_x, info=None, step_ms=1000):
    xp = [0, max_x // 2 + calib["center_x_offset"], max_x]
    fp = [calib["top_of_roi_left"], 0, calib["top_of_roi_right"]]
    path = CameraPath()
    # built directly: add_camera_target rescans the path on every call
    for t in np.arange(times_ms[0], times_ms[-1], step_ms):
        x = int(np.clip(np.interp(t, times_ms, xs), 0, max_x))
        y = -calib["top_of_roi"] + float(np.interp(x, xp, fp))
        path.camera_targets.append(CameraTarget(float(t), x=x, y=y, zoom=1.0, source="auto"))
    out = {k: calib[k] for k in CALIB_KEYS}
    out["camera_path"] = path
    out["autotrack"] = dict(info or {}, created=time.strftime("%Y-%m-%d %H:%M:%S"))
    return out


def write_sidecar(video_dir, sidecar):
    dst = os.path.join(video_dir, SIDECAR)
    tmp = dst + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(sidecar, f)
    os.replace(tmp, dst)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred", help="npz with t (ms) and pred (camera x)")
    ap.add_argument("video_dir")
    ap.add_argument("--max-x", type=int, default=2819)
    args = ap.parse_args()
    p = np.load(args.pred)
    calib, _ = load_calibration(args.video_dir)
    dst = write_sidecar(args.video_dir, make_sidecar(calib, p["t"], p["pred"], args.max_x, {"from": args.pred}))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
