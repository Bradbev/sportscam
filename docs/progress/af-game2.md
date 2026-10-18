# af-game2: does it generalise to a second game?

Branch `af-game2`, created from `af-learned`. Worktree: `C:\dev\sportscam-wt\af-game2`.
Game 2 data: `D:\RawVideo\2026-08-15` (read only). Outputs: `D:\temp\af\g2\`.

## Plan (written 2026-10-18, before starting)

Every result so far was trained and tested on one game (`autoframe`, 2026-08-22, called
**G1** below). The 2026-08-15 game (**G2**) is hand-tracked too: 2194 targets, 29 framed
segments, 37.2 framed minutes. It's the same rink with the same camera rotation (37°),
but the calibration differs slightly (top_of_roi -1790 vs -1720, center_x_offset 80 vs
43). So this tests a new game, new teams and a new day, but not a new rink.

1. Run detection on G2 with the same settings as G1 (5 Hz, YOLO11x on the full-res ice
   band), at low priority. Build G2's ice mask automatically, the same way as G1's.
2. Let the code handle more than one game. Each game is a (pkl, detections, ice edges)
   triple.
3. Map player positions to camera x with each game's own calibration, using the exact
   app geometry instead of the current linear approximation. Otherwise a model trained
   on G1 would carry G1's calibration offsets into G2.
4. The tests that matter:
   - **G1 → G2**: train on all of G1, test on all of G2. This is the honest "new game"
     number.
   - **G2 → G1**: the reverse.
   - **G1+G2 combined**, with grouped CV and leave-one-game-out. Does a second game's
     worth of labels help?
   - The simple median baseline on G2, for scale. Also the human delayed by 1 s and 2 s
     on G2, because G2's operator style may differ.
5. If transfer is much worse than within-game CV, find out why: calibration mapping, a
   different ice mask, operator style, or overfitting to G1.

## Progress log

- Detection on G2 started (about 50 min). The band check (`D:\temp\af\g2\bandcheck_g2.jpg`)
  shows the ice sits about 60 px lower than G1's in the full-res band, still well inside
  the detection band.
- G2's ice mask was built automatically from the 0012 file and looks clean
  (`D:\temp\af\g2\icemask\ice_mask_vis.jpg`).
- **Exact column → camera-x mapping** (`af/geometry.py`). It inverts the app's crop
  geometry for each game's calibration. The two games' mappings differ by less than
  10 px, so calibration is not a transfer risk. The old linear mapping differs from the
  exact one by up to 85 px at the ends, but it's consistent across games. Re-running G1
  with the exact mapping gives MAE 180 (was 177; that's noise).
- Multi-game code: `af/games.py` (per-game features cached in `D:\temp\af\feats`) and
  `af/transfer.py` (median baseline, delayed human, within-game CV, train-on-other-game,
  and CV with the other game added).

### Result: it transfers (2026-10-18 evening)

G2 detection finished (52 min). Same settings, same automatic ice mask, exact
calibration mapping. `python -m af.transfer g1 g2`:

| Scored on | Median baseline | Human +1 s | Human +2 s | Within-game CV | **Trained on the other game only** | CV + all of the other game |
|---|---|---|---|---|---|---|
| G1 (36.9 min) | 272 | 154 | 302 | 180 | **179** | 177 |
| G2 (37.2 min) | 257 | 145 | 283 | 171 | **180** | **166** |

- **A model trained on one game does as well on the other as within-game CV.**
  G2 → G1: 179 vs 180. G1 → G2: 180 vs 171. Within 300 px about 82–83% of the time,
  and the human's crop centre is inside the auto crop about 99% of the time. Nothing
  here is overfitted to G1.
- Adding the other game's labels helps a little (G2: 171 → 166). More games should keep
  helping slowly.
- The two operator sessions look alike: the delayed-human and median baselines are
  within about 15 px of each other.
- So the earlier P3 holdout (207) was pessimistic: one period is a small, hard test set.
  The honest new-game number is **about 180 px**, between "human 1 s late" and "human
  2 s late".

### Side finding: the scoreboard clock (for cuts)

How the app treats gaps: only `cut_to` skips footage when rendering. A gap in the human
path without a cut is still rendered; the camera just moves between the targets on
either side. So the only real edit decisions in G1 are the start, 3 cuts, and the end.
The cuts are the 2 intermissions plus one near the start.

The scoreboard is inside the detection band. Its seconds digits change once a second
while the clock runs, so a 34×18 px crop of the band proxy gives a clean
"clock running" signal (`af/clock.py`, 10 s for the whole game).

- This league runs a **running clock**: it does not stop at whistles. So it can't detect
  the human's 10–30 s pauses at stoppages, but those pauses aren't cut in the render
  anyway.
- It marks the periods exactly in G1:

  | Period | Clock runs (s) |
  |---|---|
  | P1 | 993–1895 |
  | Intermission countdown | 1903–2017 |
  | P2 | 2059–2962 |
  | Intermission countdown | 2970–3084 |
  | P3 | 3132–3694 |
  | 80 s stoppage | 3694–3774 |
  | P3 resumes | 3774–4116 |

- Your intermission cuts line up within ±3 s of the clock stopping and restarting. Your
  last target (4141 s) is 25 s after the final clock stop (4116 s).
- An intermission and an in-period stoppage look similar on the clock. On-ice player
  count separates them: about 6 during intermissions, about 13 during stoppages.
- **Draft cut planner** (`af/phases.py`):
  1. Live = clock running AND the 20 s median on-ice count is at least 9.
  2. Clock runs shorter than 20 s are ignored.
  3. Interruptions shorter than 120 s are bridged.

  On G1, compared with your spans:

  | Your span (s) | Planner start | Planner end |
  |---|---|---|
  | 940–1897 | +53 s | −3 s |
  | 2057–2963 | +2 s | −1 s |
  | 3129–4141 | +3 s | −25 s |

  The +53 s is the opening faceoff: you started at 940 s, but the clock flickers until
  993 s. The −25 s is the post-game: you kept filming about 25 s after the final clock
  stop. **Your decision (2026-10-18):** game start and end are quick for a human to set,
  so no further tuning there. The planner only needs to find intermissions, and it
  already does that within 3 s.
