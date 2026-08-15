"""Fused ROI warp must match legacy full-frame rotate + crop (+ resize)."""
import os
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from util import fused_roi_from_raw, get_rotate_image_matrix_and_size, rotate_image

# Default path on the creator machine; override with SPORTSCAM_TEST_VIDEO for other layouts.
_DEFAULT_VIDEO = Path(r"D:\RawVideo\2026-05-09\DJI_20260509095217_0009_D.MP4")
TEST_VIDEO_PATH = Path(os.environ.get("SPORTSCAM_TEST_VIDEO", str(_DEFAULT_VIDEO)))

ROI_BASE = (1920, 720)
OUT_WH = (1920, 720)


def reference_roi_from_raw(raw_frame, angle_deg, frame_x, frame_y, zoom, roi_base_wh, out_wh):
    """
    Legacy pipeline from sportscam interactive encode preview: full rotate, crop ROI,
    optional resize to out when zoom > 1 (same ops as util.fused docstring).
    """
    M_rot, Wr, Hr = get_rotate_image_matrix_and_size(raw_frame, angle_deg)
    w_crop = int(roi_base_wh[0] / zoom)
    h_crop = int(roi_base_wh[1] / zoom)
    full = rotate_image(raw_frame, angle_deg)
    assert full.shape[0] == Hr and full.shape[1] == Wr

    x_pos = int(min(frame_x, Wr - w_crop))
    y_pos = int(frame_y)
    roi = full[y_pos : y_pos + h_crop, x_pos : x_pos + w_crop]
    if zoom > 1.0:
        return cv2.resize(roi, out_wh, interpolation=cv2.INTER_LINEAR)
    return roi


def assert_pixels_close(a, b, max_absdiff=5):
    assert a.shape == b.shape, (a.shape, b.shape)
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    mx = int(d.max())
    assert mx <= max_absdiff, f"max absdiff={mx} (allowed {max_absdiff})"


@pytest.fixture(scope="module")
def video_first_frame():
    if not TEST_VIDEO_PATH.is_file():
        pytest.skip(f"missing test video (set SPORTSCAM_TEST_VIDEO): {TEST_VIDEO_PATH}")
    cap = cv2.VideoCapture(str(TEST_VIDEO_PATH))
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        pytest.skip(f"could not decode first frame from {TEST_VIDEO_PATH}")
    assert frame.ndim == 3 and frame.shape[2] == 3
    return frame


@pytest.mark.parametrize("angle_deg", [0.0, 15.5, 37.2])
@pytest.mark.parametrize("zoom", [1.0, 1.25])
def test_fused_matches_legacy_video(video_first_frame, angle_deg, zoom):
    raw = video_first_frame
    _, Wr, Hr = get_rotate_image_matrix_and_size(raw, angle_deg)
    w_crop = int(ROI_BASE[0] / zoom)
    h_crop = int(ROI_BASE[1] / zoom)
    frame_x = min(Wr // 3, max(0, Wr - w_crop - 1))
    frame_y = min(Hr // 3, max(0, Hr - h_crop - 1))

    fused = fused_roi_from_raw(raw, angle_deg, frame_x, frame_y, zoom, ROI_BASE, OUT_WH)
    leg = reference_roi_from_raw(raw, angle_deg, frame_x, frame_y, zoom, ROI_BASE, OUT_WH)
    assert_pixels_close(fused, leg)


def test_fused_matches_legacy_synthetic_random():
    """Runs without video; compares pixels on random data (not shape-only)."""
    rng = np.random.default_rng(2026)
    raw = rng.integers(0, 255, size=(2160, 3840, 3), dtype=np.uint8)
    angle_deg = 37.2
    zoom = 1.25
    _, Wr, Hr = get_rotate_image_matrix_and_size(raw, angle_deg)
    w_crop = int(ROI_BASE[0] / zoom)
    h_crop = int(ROI_BASE[1] / zoom)
    frame_x = min(400, max(0, Wr - w_crop - 1))
    frame_y = min(120, max(0, Hr - h_crop - 1))

    fused = fused_roi_from_raw(raw, angle_deg, frame_x, frame_y, zoom, ROI_BASE, OUT_WH)
    leg = reference_roi_from_raw(raw, angle_deg, frame_x, frame_y, zoom, ROI_BASE, OUT_WH)
    assert_pixels_close(fused, leg)


def test_fused_mean_nonzero_real_video(video_first_frame):
    """Sanity: ROI from real pixels should not be a flat black buffer."""
    raw = video_first_frame
    fused = fused_roi_from_raw(raw, 37.0, 400, 200, 1.0, ROI_BASE, OUT_WH)
    assert fused.mean() > 5.0
