"""Which videos make up a game, and the calibration to use for them.

The app keeps calibration in `<first video>.pkl`.  When that exists it is used
as is; otherwise the defaults below (the average of the two hand-calibrated
games at this rink) stand in until the operator adjusts them.
"""

import glob
import os
import pickle
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:  # pickles reference the app's CameraPath / CameraTarget
    sys.path.insert(0, REPO)

CALIB_KEYS = ("top_of_roi", "top_of_roi_left", "top_of_roi_right", "rotation", "mini_rotation",
              "angle_left", "angle_right", "mini_view_x", "mini_view_y", "center_x_offset",
              "zone_left", "zone_right")

# Average of 2026-08-15 and 2026-08-22 as calibrated by hand.
DEFAULTS = dict(top_of_roi=-1755, top_of_roi_left=-75, top_of_roi_right=152, rotation=37,
                mini_rotation=6.3, angle_left=-9, angle_right=9, mini_view_x=3376, mini_view_y=2022,
                center_x_offset=62, zone_left=793, zone_right=2129)


def video_files(video_dir):
    """Source videos in the order the app plays them (processed renders excluded)."""
    files = [f for f in glob.glob(os.path.join(video_dir, "*.mp4")) if "_processed" not in f]
    return sorted({os.path.normcase(f): f for f in files}.values(), key=os.path.normcase)


def main_pkl_path(video_dir):
    files = video_files(video_dir)
    if not files:
        raise SystemExit(f"no .mp4 files in {video_dir}")
    return os.path.splitext(files[0])[0] + ".pkl"


def load_pkl(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_calibration(video_dir):
    """(calibration dict, the loaded main pkl or None)."""
    pkl = load_pkl(main_pkl_path(video_dir))
    if pkl is None:
        return dict(DEFAULTS), None
    return {k: pkl[k] for k in CALIB_KEYS}, pkl


def calib_key(calib):
    return ",".join(f"{k}={calib[k]}" for k in CALIB_KEYS)
