# af-harness: scoring harness and detection cache

Branch `af-harness`, created from `autoframe`. Worktree: `C:\dev\sportscam-wt\af-harness`.
New code is in `af/`. The older `autoframe/` package is left as it was.

## Plan (written 2026-10-18, before starting)

1. **Reference.** Load the human camera path from the pkl. It has 2182 targets, sampled
   about every 1.04 s, and covers all four files as one timeline (the same times FastCap
   uses). Score only the stretches the human actually framed. A gap of more than 5 s
   between targets, or a `cut_to`, ends a stretch. That leaves 36 segments, 36.9 minutes.
2. **Metric.** The primary metric is MAE of camera x in pixels (crop width 1920, x range
   0–2819). Supporting numbers:
   - median and p90 error
   - % of time within 300 px
   - containment: whether the human's crop centre is inside the auto crop
   - best time shift and the MAE at that shift, which separates "wrong place" from
     "right place, late"
   - travel ratio, as a measure of jitter
3. **Detection cache, one pass over the full game.**
   - ffmpeg decodes every file on NVDEC. Every 5th frame (5 Hz) is kept.
   - Each kept frame is rotated by the calibrated 37°. Only rows 1550–2600 of the rotated
     frame (the ice band) are kept, at full 4K resolution.
   - YOLO11x person detection runs on that band, keeping every box with conf ≥ 0.10 so
     filtering can be tuned offline.
   - A half-resolution band proxy video is written alongside, for debug overlays and pixel
     features.
   - Output goes to `D:\temp\af\dets`.
4. **Then branch out into approaches.** Each gets its own branch and note:
   - **players**: robust centre of on-ice players → x, plus zone dwell (a cleaned-up
     version of the earlier v2 plan).
   - **learned**: a regression from per-time detection features (histogram of player
     positions along the rink, counts, motion, with a centred temporal window) to human x.
     Trained on 37 minutes of labels with blocked cross-validation by segment, so it is
     always scored on stretches it did not train on.
   - **motion**: direction of play from optical flow and player velocities, to lead
     transitions instead of lagging them.
   - **path shaping**: post-processing that makes the camera hold and pan the way the
     human does (the human sits deep in an end zone 43% of the time).

## Findings so far

- Human x is strongly bimodal. The 5th–95th percentile is 10–2819 px. 22% of the time
  x < 280 and 22% x > 2540: the operator parks deep in an end zone and commits to it.
- Lag matters a lot. A copy of the human path delayed by just 1 s already has MAE 155 px.
  Delayed by 2 s it is 302 px; by 4 s, 557 px. So an MAE of about 250–300 would be
  roughly "human quality with a 2 s reaction delay".
- Trivial baselines: a constant centre camera scores MAE 921, and a constant median 917.
- Detector check on one frame (t = 1000 s), on the full-res 4K band:
  - YOLO11x finds the far-end skaters at about 55 px tall. At half resolution most of
    those are missed, so detection runs at full resolution.
  - It also detects spectators in the far stands (high in the band) and people on the
    near benches and boards (big boxes at the bottom). Both need geometric filters.
  - A scoreboard with the game clock is visible at the far right. It could later be used
    to detect stoppages.

## Progress log

- 2026-10-18 15:05: full-game detection pass started (about 75 min at 5 samples/s). It
  runs at below-normal priority.
- **Ice mask** (`af/icemask.py`), automatic:
  - It takes the median of 31 frames, finds the bright low-saturation ice, then fits
    smooth envelopes to the far-boards and near-boards lines. Occluders such as the bench
    crowd can only shrink the visible ice, so taking the envelope recovers the true lines.
  - A box counts as on ice if its feet point lies between those two lines, within a
    tolerance. This rejects far-stand spectators and most bench and near-side people.
  - It keeps about 10 boxes per sample (out of about 50 raw).
  - Leftover problem: bench players whose legs are hidden by someone in front of them get
    boxed from the waist up, so they look like near-board skaters. It's occasional, not
    systematic: a long-term occupancy heatmap shows only the two goalie spots as
    persistently occupied.
- **File 1 results** (only 5.4 framed minutes, since the human path starts at 940 s):

  | Method | MAE | Best lag | Contain |
  |---|---|---|---|
  | Median of on-ice players, smoothed 2 s | 266 | +0.8 s | 97% |
  | Mean of players | 394 | | |
  | Median, stretched ×1.3 away from centre | 255 | | |
  | LightGBM on player-distribution features, 4-fold grouped CV | **198** | 0.0 s | 98.6% |

  - The +0.8 s lag on the median means **the human moves about 0.8 s ahead of the player
    centroid**. The operator anticipates, presumably by following the puck.
  - The learned model has no lag, because it can use ±8 s of future context. That's fine
    here: processing is offline.
- **Full game (16:20).** The detection pass finished: 21,510 samples. Decode plus YOLO11x
  ran at about 6.5 samples/s, roughly 52 min in total. On average 11.7 boxes per sample
  are kept as on-ice. Scores over all 36.9 framed minutes:

  | Method | MAE | Median | p90 | <300 px | Contain |
  |---|---|---|---|---|---|
  | Constant centre | 921 | | | 14% | 44% |
  | Median of on-ice players | 266 | 174 | 607 | 69% | 96% |
  | **LightGBM, 6-fold grouped CV by segment** | **196** | 126 | 419 | 82% | 98% |
  | LightGBM trained on P1–P2, tested on P3 only | 240 | 154 | 513 | 76% | 97% |

  On the P3 span the cross-validated predictions also score 243, so P3 itself is harder.
  The time-split model is not overfitting.
- The plot (`D:\temp\af\paths_learned.png`) shows the learned path following the human's
  end-to-end swings closely. The remaining error is:
  - small wobble while the human holds still,
  - transitions about 0.5–1 s early or late,
  - the occasional fast reversal that the model only follows halfway.
- **Path shaping** (`af/shaping.py`) means hold-and-pan, median and smoothing filters.
  They barely change MAE: the best was 197. A deadband of 100 px with a 600 px/s pan cap
  costs 2 px of MAE but brings travel down to 0.89× the human's, which is closer to how
  an operator moves. So the error is in the decisions, not in jitter.
- `af/export.py` writes any prediction as a sportscam pkl, with calibration copied from
  the source and y computed exactly as the app does (verified against the human pkl: max
  error 0). Output goes to `D:\temp\af\out` only.
- Separate venv for experiments: `D:\temp\af\venv`. It reads the main venv's packages
  through a `.pth` path file and adds scipy, scikit-learn and lightgbm. The main venv is
  untouched.
