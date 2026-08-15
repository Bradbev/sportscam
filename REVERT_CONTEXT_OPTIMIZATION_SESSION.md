# Optimization / UI session — summary for revert & next attempts

Use this file as context after reverting the repo. It records what changed, measured results, and subjective outcome.

---

## Subjective outcome (author)

- **Does not feel like a net win** in daily use.
- **Encode/render output appears to “burst”** (speed uneven), which makes the app harder to use; likely causes to investigate separately include **decode variance**, **single-threaded pipeline**, **preview `imshow` + encode competing**, or **Windows/OpenCV scheduling** — not proven here.

---

## Goals that motivated the work

1. Speed up **`-r` / encode** on Windows (RTX 4080), especially vs full-frame rotate.
2. Fix **blank main preview / HUD** while miniview still showed content.
3. Replace **weak tests** (shape-only on random arrays) with **real-video pixel checks** vs legacy rotate+crop.

---

## Code areas touched

### `util.py`

- **`fused_roi_from_raw`** (encode path only):
  - **Original broken approach** (from earlier iteration): composed `M_rot @ T` and passed it to `warpAffine` in a way that did **not** match OpenCV’s actual convention for the rotation matrix. Effect: ROI could be **all black** while legacy `rotate_image` → crop looked correct.
  - **Current approach**: build the warp matrix via  
    `inv_rot = invertAffineTransform(M_rot)`, compose with translation `T` (crop origin in full rotated destination), then **`M_crop_warp = invertAffineTransform(inv_crop)`** so `warpAffine` samples the same pixels as **cropping** the full rotated canvas.
  - **Docstring** in `fused_roi_from_raw` may still mention an older `M_rot @ T` story; the implementation uses the invert chain above.

- **`get_rotate_image_matrix_and_size`**, **`rotate_image`**: shared helpers (existing refactor direction).

### `sportscam.py`

- **Profiling** (already present in project direction): `--profile`, `SPORTSCAM_PROFILE`, `--max-frames`, `ProfCollector`, spans for `decode`, `mini_view`, `rotate_full`, `output_roi`, etc.
- **`cv2.setUseOptimized(True)`** after imports (from earlier handoff).
- **Interactive HUD / blank band**: vertical slice for the large `frame` window was clamped so `[1200:2800]` never yields an empty slice when rotated height is short.
- **`frame_y` helper**: `rotate_point(..., 0)` instead of `rotate_point(..., -angle)` so vertical ROI isn’t skewed by pan-angle rotation around the origin (matches `widescreen.py`-style behavior).
- **Mouse → pan mapping**:
  - Globals **`mouse_win_x` / `mouse_win_y`** (raw pixel coords in the **`frame`** window, size **`view_size`** = 3340×1050).
  - Each frame: **`clamped_mouse_x = clip(mouse_win_x / view_size[0] * max_x, 0, max_x)`** so horizontal pan is **[0, max_x]** aligned with window width; **`mouse_y_scaled`** uses **`view_size`** for vertical centering (not **`out_size`**).
  - **`mouse_x`** updated each frame from **`clamped_mouse_x`** for keyboard shortcuts that depend on pan position.
- **Encode path**: still calls **`fused_roi_from_raw`** when `writeOutputFile`; interactive path uses **`rotate_image`** + **`_create_output_frame`**.

### `tests/test_fused_roi.py`

- Loads **first frame** from  
  `D:\RawVideo\2026-05-09\DJI_20260509095217_0009_D.MP4`  
  unless **`SPORTSCAM_TEST_VIDEO`** overrides.
- **`reference_roi_from_raw`**: legacy **`rotate_image` → crop → resize** (matches interactive geometry).
- Asserts **pixel-wise** closeness (max abs channel diff **≤ 5**) between fused and legacy on parametrized angles/zooms + synthetic random frame.
- **`test_fused_mean_nonzero_real_video`**: fused ROI mean **> 5** on real frame (guards flat black).

---

## Profile snapshots (same workload concept: ~300 frames, `-r`)

Captured in-repo as **`baseline.txt`** (pre–fused-path profile includes **`rotate_full`**) vs **`postopts.txt`** (post-change encode path: **`rotate_full`** absent).

| Stage (avg ms) | `baseline.txt` | `postopts.txt` |
|----------------|------------------|----------------|
| decode | ~19.06 | ~25.08 |
| rotate_full | ~12.50 | *(not run on encode path)* |
| output_roi | ~1.44 | ~2.94 |
| encode | ~0.73 | ~0.78 |
| frame_total | ~42.64 | ~37.95 |

**Interpretation:**

- **`rotate_full` removed** from encode path as intended; **`output_roi`** rises vs slicing a pre-rotated buffer (expected).
- **`frame_total`** drops ~**4.7 ms** on average in these two snapshots — smaller than “save ~12.5 ms from dropping rotate” because **`decode` regressed ~+6 ms** between runs (likely noise, load, or environment — treat as **not a controlled A/B**).
- **Encode** stayed ~**0.7–0.8 ms** in both.

---

## What actually failed before fixes (for regression hunting after revert)

1. **Wrong fused affine**: encode ROI could be **black**; tests comparing to legacy **rotate+crop** exposed it.
2. **Blank editor vs miniview**: mix of **ROI at x=0** (warp padding), **wrong mouse→pan scaling** (`out_size` vs **`view_size`**), and **hardcoded vertical band** for HUD.

---

## If you revert everything

- Restore **behavior + tests** from your chosen baseline commit.
- Keep **this file** (or copy it elsewhere) so the next attempt doesn’t repeat the same blind spots:
  - fused warp must be **parity-tested** against **`rotate_image` + crop** on **real frames**, not only shape checks;
  - interactive mouse mapping must match **`view_size`** for the window that owns the callback.

---

## Ideas for “other methods” (not implemented here)

- **`--no-preview`** during `-r** to remove **`preview_ui`** contention.
- **Decode prefetch / second thread** if decode remains noisy or bursty.
- **Hardware encode** (NVENC / ffmpeg pipe) if CPU encode or mux stalls dominate.
- **Stable benchmarking**: same machine, close apps, multiple runs, median of **`frame_total`** / per-stage histograms — averages alone hide burstiness.

---

*Generated as handoff context before reverting the working tree.*
