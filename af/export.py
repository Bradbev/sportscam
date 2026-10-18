"""Write an automatic camera path as the sidecar the app copies in: DIR/auto_path.pkl.

The sidecar holds the calibration the path was computed with, the camera path
(a target every `step_ms`, y derived from x exactly as the app does at zoom 1),
and an `autotrack` dict describing how it was made.  It never touches the
operator's own `<first video>.pkl`.

  python -m af.export PRED.npz VIDEO_DIR [--max-x N]    (normally run via autotrack.py)
"""

import argparse
import json
import os
import pickle
import time

import numpy as np

from .calibration import CALIB_KEYS, load_calibration
from camerapath import CameraPath
from cameratarget import CameraTarget
from gamestate import GameState

SIDECAR = "auto_path.pkl"
REVIEW = "review.json"


def make_sidecar(calib, times_ms, xs, max_x, info=None, step_ms=1000, game=None):
    """`game` (af.gameinfo.analyse) sets period and score on every target and adds
    period / goal bookmarks, the same way recording in the app would."""
    xp = [0, max_x // 2 + calib["center_x_offset"], max_x]
    fp = [calib["top_of_roi_left"], 0, calib["top_of_roi_right"]]
    path = CameraPath()
    bookmarks = []
    goals = sorted(game["scored"], key=lambda gl: gl["t"]) if game else []
    periods = game["periods"] if game else []
    # built directly: add_camera_target truncates forward on every call
    for t in np.arange(times_ms[0], times_ms[-1], step_ms):
        x = int(np.clip(np.interp(t, times_ms, xs), 0, max_x))
        y = -calib["top_of_roi"] + float(np.interp(x, xp, fp))
        gs = GameState()
        if game:
            gs.period = 1 + sum(1 for a, _ in periods[1:] if t >= a - 60000)
            score = [0, 0]
            for gl in goals:
                if gl["t"] <= t:
                    score[gl["team"]] += 1
            gs.score = tuple(score)
        path.camera_targets.append(CameraTarget(float(t), x=x, y=y, zoom=1.0, game_state=gs, source="auto"))
    if game:
        score = [0, 0]
        for gl in goals:
            score[gl["team"]] += 1
            bookmarks.append((float(gl["t"]), f"goal {score[0]}-{score[1]}"))
        bookmarks += [(float(a - 60000), f"period - {k + 2}") for k, (a, _) in enumerate(periods[1:])]
    out = {k: calib[k] for k in CALIB_KEYS}
    out["camera_path"] = path
    out["bookmark_times_ms"] = sorted(bookmarks)
    out["autotrack"] = dict(info or {}, created=time.strftime("%Y-%m-%d %H:%M:%S"))
    return out


def write_sidecar(video_dir, sidecar):
    dst = os.path.join(video_dir, SIDECAR)
    tmp = dst + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(sidecar, f)
    os.replace(tmp, dst)
    return dst


def write_review(video_dir, entries):
    """DIR/review.json: what the app's review mode steps through, best first.

    Each entry has kind ("camera", later "goal"/"faceoff"/...), t0/t1 in source ms,
    and kind-specific fields.  Tiers are worked out by the app, after the start
    point is known, so warm-up segments never use up review time."""
    dst = os.path.join(video_dir, REVIEW)
    tmp = dst + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"version": 1, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "entries": entries}, f, indent=1)
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
