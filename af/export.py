"""Write an automatic camera path as a sportscam.py pickle.

Copies the calibration from a source pkl and replaces the camera path with
targets every `step_ms`, y derived from x exactly as the app does at zoom 1.0:
    y = -top_of_roi + interp(x, [0, max_x//2 + center_x_offset, max_x],
                                [top_of_roi_left, 0, top_of_roi_right])

Never writes next to the source video unless told to; default output dir is
D:/temp/af/out.

  python -m af.export PRED.npz SOURCE.pkl OUT_DIR [--start S --end S]
"""

import argparse
import copy
import os
import pickle

import numpy as np

from .human import load_pkl  # also puts the repo on sys.path
from camerapath import CameraPath
from cameratarget import CameraTarget
from highlight import Highlights


def make_pkl(src, times_ms, xs, max_x, step_ms=1000, start_ms=None, end_ms=None):
    out = {k: copy.deepcopy(v) for k, v in src.items() if k not in ("camera_path", "highlights")}
    xp = [0, max_x // 2 + src["center_x_offset"], max_x]
    fp = [src["top_of_roi_left"], 0, src["top_of_roi_right"]]
    s = times_ms[0] if start_ms is None else start_ms
    e = times_ms[-1] if end_ms is None else end_ms
    path = CameraPath()
    # built directly: add_camera_target rescans the path each call (quadratic)
    for t in np.arange(s, e, step_ms):
        x = int(np.clip(np.interp(t, times_ms, xs), 0, max_x))
        y = -src["top_of_roi"] + float(np.interp(x, xp, fp))
        path.camera_targets.append(CameraTarget(float(t), x=x, y=y, zoom=1.0))
    out["camera_path"] = path
    out["highlights"] = Highlights()
    out["bookmark_times_ms"] = []
    out["rendered"] = False
    out["finished"] = True
    out["auto_record"] = False
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred")
    ap.add_argument("source_pkl")
    ap.add_argument("out_dir")
    ap.add_argument("--start", type=float)
    ap.add_argument("--end", type=float)
    ap.add_argument("--max-x", type=int, default=2819)
    args = ap.parse_args()
    p = np.load(args.pred)
    src = load_pkl(args.source_pkl)
    out = make_pkl(src, p["t"], p["pred"], args.max_x,
                   start_ms=args.start * 1000 if args.start else None,
                   end_ms=args.end * 1000 if args.end else None)
    os.makedirs(args.out_dir, exist_ok=True)
    dst = os.path.join(args.out_dir, os.path.basename(args.source_pkl))
    if os.path.normcase(os.path.abspath(dst)) == os.path.normcase(os.path.abspath(args.source_pkl)):
        raise SystemExit("refusing to overwrite the source pkl")
    with open(dst, "wb") as f:
        pickle.dump(out, f)
    print(f"wrote {dst}: {len(out['camera_path'].camera_targets)} targets")


if __name__ == "__main__":
    main()
