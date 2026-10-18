# autotrack: bringing the tracker to main

Branch `autotrack`. It holds the `af-*` commits rebased onto `main` (after the bookmark
fix), without the older `autoframe` branch commits. Worktree:
`C:\dev\sportscam-wt\autotrack`.

## Plan (2026-10-18, agreed with you)

- **One command:** `python autotrack.py DIR` runs detection, then the ice mask, then
  features, then the saved model, and writes `DIR\auto_path.pkl`. Everything is cached in
  `DIR\autotrack_cache\`, so a re-run resumes and skips finished files.
- **Always a sidecar.** The tracker never writes `DIR\<first video>.pkl`.
- **App copies the sidecar in once.** If the main pkl has no camera targets and
  `auto_path.pkl` exists, the app loads that camera path. If there was no main pkl at
  all, it also takes the sidecar's calibration. Your next save makes the main pkl the
  working copy, and the sidecar stays as the original auto result.
- **Calibration:** taken from the main pkl if there is one, otherwise the average of the
  two hand-tracked games. Re-running after a calibration change takes seconds, because
  detection is cached.
- **Saved model:** a LightGBM model trained on G1 and G2, stored in `models/`, with
  `python autotrack.py train` to retrain as more hand-tracked games arrive.
- **Trim:** remove the TCN, path shaping and simple baselines. Keep the evaluation
  tools (human path, metrics, transfer, compare) and the clock and cut planner.
- **Dependencies:** `requirements-autotrack.txt`.
- **Tests:** geometry round trip, sidecar format, and an end-to-end smoke run.
- Clean up the `af-*` and `fix-goal-bookmark` branches and worktrees, because
  `autotrack` contains all their commits. Your other branches, including `autoframe`,
  are left alone.

## Progress log

- Rebased onto `main`, keeping 9 tracker commits.
- **Packaging done** (commit 332ae29):
  - `autotrack.py` with `run` and `train`.
  - `af/` now holds `calibration`, `game`, `features`, `model`, `pipeline` and
    `export` (the sidecar), plus the existing detection, ice mask, geometry, players and
    motion modules.
  - Dev tools take `DIR[=CACHE]`: `transfer`, `compare`, `debugvid`, `clock`, `phases`.
  - Removed: the TCN, path shaping, simple baselines, `learned.py` and `games.py`.
  - Detection now refuses to cache a file it couldn't decode, instead of caching zero
    samples.
- **Saved model:** `models/autotrack_lgbm.txt` (600 KB), trained on both games (74
  hand-tracked minutes). The cross-game scores after the restructure are identical to
  before: 179/180 transfer, 177/166 with both games.
- **App:** `sportscam.py` copies `auto_path.pkl` in once when the game has no camera
  targets. It takes the sidecar's calibration only when there was no main pkl at all.
  `CameraTarget` has a new `source` field ("human" by default, "auto" from the tracker).
- **Tests:** `tests/test_autotrack.py`, 4 tests, all passing:
  - the geometry round trip is under 1 px;
  - the sidecar format and playback work;
  - video ordering is right.

  `tests/test_fused_roi.py` already fails on `main`, because `util.fused_roi_from_raw`
  doesn't exist. That's not related to this branch.
- **End-to-end check:** a 2-minute game 2 clip in `D:\temp\af\smoke`, with no pkl, so
  it used the default calibration.
  - `autotrack.py` took 2 min 48 s: detection, then the ice mask, then the sidecar.
  - Then `sportscam.py D:\temp\af\smoke -r` printed "Loaded 120 auto camera targets
    from auto_path.pkl" and rendered at 22.7 fps. The saved main pkl has the auto
    targets and the default calibration.
  - The frames look right.
- **Cleanup:**
  - Removed the `af-harness`, `af-learned`, `af-game2` and `fix-goal-bookmark` branches
    and worktrees. `autotrack` has all their content, and the bookmark fix is in
    `main`. The old tips were 8b8acca, d4529ae, ac23e83 and a40d599, recoverable from
    the reflog for a while.
  - My scratch scripts were archived to `D:\temp\af\scratch_af_archive`.
  - `autoframe` and your other branches were left alone.

- **Your decision:** no low-priority mode, so that code is removed. The README also
  had two `\a` control characters from my edit; those are fixed.

### Merging into main

`main`'s working folder has untracked `docs\` files, and this branch tracks
`docs/progress/*.md`. Git refuses to merge over untracked files. The branch's copies are
the same notes, so move or delete the untracked `docs\progress` first, then
`git merge autotrack`.

After merging:
1. Run `pip install -r requirements-autotrack.txt` in your venv.
2. Copy `D:\temp\af\yolo11x.pt` to `models\`, or let the first run download it.
