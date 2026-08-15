# Sportscam optimization context (handoff)

## Goal

Speed up rendering on Windows with an RTX 4080; establish profiling baselines; optimize the `-r` encode path without breaking interactive editing.

## What was implemented

### Phase 0 — profiling ([sportscam.py](sportscam.py))

- `--profile` / `SPORTSCAM_PROFILE=1`: stage timers (`decode`, `mini_view`, `rotate_full`, `output_roi`, `annotate_logo`, `encode`, `preview_ui`, `editor_ui`, `frame_total`).
- `--max-frames N`: stop after N completed frames for short benchmarks.
- Baseline example folder discussed: `D:\RawVideo\2026-05-09`.

### Tier 1 — ROI pipeline ([util.py](util.py))

- **`get_rotate_image_matrix_and_size`**: shared rotation matrix + `(Wr, Hr)` matching `rotate_image`.
- **`fused_roi_from_raw`**: for **`-r` encode only**, avoids allocating a full **Wr×Hr** rotated buffer. Uses **`warpAffine(raw, M_rot @ T, (w_crop, h_crop))`** (warp only the crop-sized destination), then **`cv2.resize`** when `zoom > 1`.
- **Not done**: a single affine that also folds in `resize`; OpenCV’s resize grid does not match a uniform affine scale for all angles, so pixel parity would break.
- **`cv2.setUseOptimized(True)`** after imports in `sportscam.py`.

### Behavior split

- **`writeOutputFile` (`-r`)**: `fused_roi_from_raw` → `_finalize_roi_output`; **`rotate_full`** not used for geometry (still decodes full frames).
- **Interactive (no `-r`)**: full **`rotate_image`** for `frame` (HUD / rectangles / main window).

### Tests

- [tests/test_fused_roi.py](tests/test_fused_roi.py): smoke tests on output shape for zoom 1 and zoom > 1.

## Bugs fixed along the way

1. **Blank encode/preview when fused affine was wrong**: replaced broken single-matrix warp with explicit crop-sized warp + resize (correct output).
2. **Blank main HUD (`frame` window)**: vertical slice was hardcoded `frame[1200:2800]`. If rotated height **< 1200**, that slice is empty → blank. **Fix**: clamp band — if start row ≥ height use full frame; else if end exceeds height slice `[1200:h_rot]`; else `[1200:2800]`.

## Likely next steps (not implemented)

- Decode threading / prefetch (~19 ms/frame in baseline).
- NVENC (`ffmpeg` pipe) if encode becomes the bottleneck after ROI optimization.
- `--no-preview` during `-r` to skip `imshow` overhead.
- Golden-frame regression test (`tests/fixtures`, absdiff vs legacy).
- [widescreen.py](widescreen.py): same patterns for 4K output.

## Files touched most

- [sportscam.py](sportscam.py) — profiling, `-r` vs interactive paths, HUD slice fix.
- [util.py](util.py) — `get_rotate_image_matrix_and_size`, `fused_roi_from_raw`, `rotate_image` refactor.

## Reference baseline (from user’s `baseline.txt`)

Rough averages over 300 frames with `-r`: **decode ~19 ms**, **rotate_full ~12.5 ms** (before skipping full rotate on encode path), **output_roi ~1.4 ms**, **encode ~0.7 ms** (also saw OpenCV `acv1` codec warning — verify writer/codec).
