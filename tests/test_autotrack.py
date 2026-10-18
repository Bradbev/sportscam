"""autotrack: geometry, sidecar format and video discovery (no GPU or video needed)."""
import pickle

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("scipy")

from af.calibration import CALIB_KEYS, DEFAULTS, video_files
from af.export import make_sidecar
from af.geometry import ColToX, centre_cols, max_x

W, H = 3840, 2880


def test_col_to_x_inverts_the_app_crop_geometry():
    xs, cols = centre_cols(DEFAULTS, W, H, band_y0=1550)
    back = ColToX(DEFAULTS, W, H, 1550)(cols)
    assert np.abs(back - xs).max() < 1.0


def test_max_x_matches_the_known_frame():
    assert max_x(DEFAULTS, W, H) == 2819


def test_sidecar_holds_calibration_and_an_auto_path_the_app_can_play():
    mx = max_x(DEFAULTS, W, H)
    t = np.arange(0, 60000, 200.0)
    x = np.linspace(-100, mx + 100, len(t))  # deliberately out of range at both ends
    side = pickle.loads(pickle.dumps(make_sidecar(DEFAULTS, t, x, mx)))
    for k in CALIB_KEYS:
        assert side[k] == DEFAULTS[k]
    ct = side["camera_path"].camera_targets
    assert len(ct) == 60
    assert all(c.source == "auto" for c in ct)
    assert all(0 <= c.x <= mx for c in ct)
    # y follows x exactly as the app derives it at zoom 1
    mid = ct[30]
    xp = [0, mx // 2 + DEFAULTS["center_x_offset"], mx]
    expect = -DEFAULTS["top_of_roi"] + np.interp(mid.x, xp, [DEFAULTS["top_of_roi_left"], 0, DEFAULTS["top_of_roi_right"]])
    assert mid.y == pytest.approx(expect)
    cam, _, _ = side["camera_path"].get_camera_at_time(30500.0)
    assert ct[30].x <= cam.x <= ct[31].x


def test_video_files_are_ordered_and_skip_renders(tmp_path):
    for n in ("B_0002_D.MP4", "A_0001_D.MP4", "A_0001_D_processed.mp4", "notes.txt"):
        (tmp_path / n).write_bytes(b"")
    names = [p.split("\\")[-1].split("/")[-1] for p in video_files(str(tmp_path))]
    assert names == ["A_0001_D.MP4", "B_0002_D.MP4"]
