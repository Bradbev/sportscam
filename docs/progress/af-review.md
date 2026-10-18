# af-review: human verification before the final render (design only)

Status: **in progress** on branch `af-review` (from `main`), worktree
`C:\dev\sportscam-wt\af-review`. The plan and log are at the end of this file.

Goal: the auto pass produces a camera path, a score history and cut points. The human
then spends a few minutes checking only the parts the system is unsure of, and renders.
You said start and end of game don't matter: a human sets those in seconds.

Everything below was checked against the code on `main` and the two hand-tracked games:
G1 is `D:\RawVideo\autoframe`, 2026-08-22; G2 is `D:\RawVideo\2026-08-15`. Both were
read only.

## 1. Camera and scoreboard calibration: defaults or guesses?

**What exists today.** Calibration is a set of pkl fields:

- **Rotation and band:** `rotation`, `angle_left`/`angle_right`, `top_of_roi`,
  `top_of_roi_left`/`top_of_roi_right`, `center_x_offset`.
- **Zone lines** (used for the possession display): `zone_left`/`zone_right`.
- **Scoreboard mini-view** pasted at the top left: `mini_view_x`/`mini_view_y`,
  `mini_rotation`.

New files start from the defaults in `sportscam.py:140-160`, and the operator then
adjusts by hand with the keys. Those defaults are: rotation 37, angles ±9, top_of_roi
-1600 (left and right offsets 0), center_x_offset 0, zones 720/2048, mini-view
3550/2000, mini_rotation 8.

**Calibrated values in the pkls:**

| Field | Default | G1 | G2 | G2 `calibrated.pkl` |
|---|---|---|---|---|
| rotation | 37 | 37 | 37.0 | 36.4 |
| angle_left / right | -9 / 9 | -9 / 9 | -9 / 9 | -9 / 9 |
| top_of_roi | -1600 | -1720 | -1790 | -1790 |
| top_of_roi_left / right | 0 / 0 | -59 / 163 | -90 / 142 | -90 / 142 |
| center_x_offset | 0 | 43 | 80 | 82 |
| zone_left / right | 720 / 2048 | 782 / 2100 | 804 / 2158 | 804 / 2158 |
| mini_view_x / y | 3550 / 2000 | 3377 / 1999 | 3376 / 2046 | 3376 / 2046 |
| mini_rotation | 8 | 7.4 | 5.2 | 5.2 |

The two games differ by small amounts. The vertical ones are about 70 px (top_of_roi,
mini_view_y), probably because the camera sits a little lower on its mount. The ice in
G2 sits about 60 px lower in the detection band, which matches.

**Proposed approach.**

1. **Better defaults first.** Use G1 and G2's averages as the app's defaults, which
   costs almost nothing. With the same rink and mount, that is probably within tens of
   px everywhere.
2. **Then a cheap automatic correction, only for the vertical shift.** The automatic ice
   mask (`af/icemask.py`) already finds the far-boards and near-boards lines.
   - top_of_roi follows from the near-boards line at mid-ice.
   - top_of_roi_left and _right follow from the line at each end.
   - Lock the offsets between G1 and G2 so they match the human's choices.
   - The same shift (the median frame's offset against a stored reference median, via
     phase correlation) moves the scoreboard crop (`mini_view_y`) and the clock crop used
     by `af/clock.py`.
3. **Rotation and angle_left/right** stay at their defaults: they haven't changed.
4. **center_x_offset** (43 vs 80) and **zone lines** could come from the centre red line
   and blue lines in the median frame. They're low value: center_x_offset only shifts the
   per-x angle knot. Leave it to the human, or use the average.

**How to measure.** Run the guess for G1 and G2 and compare with the human's values. More
usefully, render a few crops with guessed versus human calibration and check the pixel
offset of the boards line. A tolerance of about 20 px vertical is invisible in the
1920×720 output.

**Open questions.**
- Does the camera always go on the same mount in this rink, or are other rinks coming?
  A different rink needs the automatic route, or a per-rink default set.
> Answer: One rink only, usually the same or similar mount - but it will vary slightly every time (not a totally fixed mount)
- Is the scoreboard mini-view always shown? It assumes the same scoreboard position.
> Answer: Often.  Always show the mini-view.  But sometimes the in-rink scoreboard breaks, so anything relying on the board needs a fallback.

**Effort.** Defaults: an hour. Automatic vertical correction from the ice mask: about
half a day plus checking.

## 2. Confidence scores, so the human only reviews the doubtful parts

**What exists today.** Nothing; the model outputs a single x. We do know where it fails:
- About 30% of error is in 47 episodes of more than 500 px lasting at least 1 s.
- Those are mostly breakouts, where the human jumps to the lead skaters, and the
  post-game.
- The model also stops about 100 px short when the human parks deep in a zone.

**Candidate signals** (per 200 ms sample, then smoothed into segments):

| Signal | Why it might flag errors | Cost |
|---|---|---|
| **Quantile spread**: LightGBM at q10 and q90 alongside the L1 median | Directly says "the human could be anywhere in this range"; wide during transitions | 2 extra models |
| **Model disagreement**: spread of the 6 fold models, or a TCN versus LightGBM gap | Disagreement means it's unsure | Free; the fold models already exist |
| **Speed of the predicted path, and the rush-front versus median gap** | Breakouts: fast skaters far ahead of the pack | Free (features exist) |
| **Player evidence**: on-ice count far from about 13, low detection confidence, many rejected boxes near the boards | Bad input means a bad output (bench crowd, occlusion) | Free |
| **Clock state**: stopped or flickering | Stoppages and intermissions, where the human behaves differently (pauses, faceoff jumps) | Free (`af/clock.py`) |
| **Bimodal player distribution** (two clusters, at both ends) | The human must pick one; the model averages them | Free (histogram features) |

Combine them with a small **error model**: a second LightGBM trained out-of-fold to
predict the main model's |error| from these signals. Its output is the confidence score.
Train it on G1 and test on G2, and vice versa, so it isn't graded on its own training
data.

**Segments to review.** Threshold the predicted error, merge flagged samples less than
5 s apart, pad each segment by about 2 s, and rank the segments by total predicted error
(px × s).

**How to validate.** Use both games, scoring each with a model that never saw it.
- **Ranking quality:** the Spearman correlation between predicted and actual error, and
  the precision and recall for the more-than-500-px episodes.
- **Error caught per minute reviewed:** sort the segments by predicted error. For N =
  1, 2, 5 and 10 minutes reviewed, measure the share of total error (px × s) inside the
  reviewed segments. Compare that with an oracle (the actual worst segments) and with
  random segments.
- **After-review MAE:** replace the reviewed segments with the human path, as a
  simulated perfect review, and recompute MAE. This gives a curve to put in front of you
  ("5 min of review takes MAE from X to Y"); it's the number that matters.

**Open questions.**
- How much review time is acceptable per game? 5 minutes? 10?
> Answer: 5-10 minutes would be ideal, but if possible we could have tiers of review for second and third passes?
- Is "close enough" the right goal, or does any visibly wrong framing (the puck carrier
  out of frame) matter more than average px? Containment-based error (is the human crop
  centre outside the auto crop?) may be a better target for the error model than MAE.
> Answer: close enough is the goal - I don't think we can get perfect.

**Effort.** Quantile models plus the error model plus the review curve: about a day.
That's offline work and needs no app changes.

## 3. Changing "auto" from truncate to replace-in-segment

**What exists today.**
- `CameraPath.add_camera_target` (`camerapath.py`) calls `truncate_path_to_time(t)`. That
  **drops every target after t** before appending.
- Auto record (`a` key) adds a target every 1000 ms of video at the mouse x
  (`sportscam.py:667-684`). It also calls `truncate_bookmarks_to_time`, which deletes
  later bookmarks.
- Ending a highlight truncates too (`h` key).

This is right for "record from here on", but it destroys an auto path that already exists
everywhere after the review point. The search is also a linear scan, quadratic over a
whole game (seen when exporting).
>Question: will quadratic matter for this datasize?
>
>Answer (Claude): not for live use. `camera_index` scans about 2–4k targets (around
>0.3 ms in Python), once per frame and once per recorded target, which is nothing next to
>decoding a 4K frame. It only adds up in bulk operations: writing a whole auto path one
>target at a time is n²/2, about 9M steps, a few seconds. That's why the exporter appends
>directly. `replace_range` will use bisect anyway because it's no harder, and
>`camera_index` could switch to bisect for free. Low priority.

**Proposed data model.**

- **Replace in a range.** Add `CameraPath.replace_range(t0, t1, new_targets)`, which
  removes the targets with t0 ≤ time ≤ t1, inserts the new ones, and keeps the list
  sorted. Use bisect on a parallel times list, not a scan.
- **Boundaries.** At t1, keep the next existing target as is. The camera then glides from
  the human's last point to the auto path over one interval (about 1 s), which is the
  same easing `get_camera_at_time` already does. Optionally insert a blend target at
  t1 + 1 s if the jump is over about 400 px.
- **Provenance.** Add `source` to `CameraTarget` (`"auto"`, `"human"`, `"auto-reviewed"`),
  plus an optional `conf`. That gives review UI colours, lets later auto runs replace
  only `auto` targets, and tells training which labels are real human ones.
  - Old pickles have no `source`, so `CameraTarget.__init__` needs a default.
  - `pickle` restores `__dict__` without calling `__init__`, so loading old objects needs
    a `getattr(c, "source", "human")` fallback, or a `__setstate__`.
- **Bookmarks and score.** `truncate_bookmarks_to_time` must not wipe later bookmarks in
  review mode. Only bookmarks inside [t0, t1] should be replaced.

**Review UI flow** (sketch):

1. `sportscam.py --review DIR` loads the auto pkl plus a sidecar `review.json` with the
   ranked low-confidence segments (t0, t1, predicted error, reason).
>Note: when run with only `sportscam.py DIR`, we should provide a small & simple TUI for the various ways to run the script
2. `n` / `N` jump to the next or previous segment. Playback starts 3 s before t0, with a
   timeline strip showing the segments in red and the auto path drawn as now.
3. In a segment, the operator either:
   - accepts (`Enter`, which marks it `auto-reviewed` and moves on); or
   - holds the record key and tracks with the mouse as today. While recording, new
     targets replace the targets in [t_start, t_now]. On release, the path after t_now
     is untouched.
4. A summary line shows how many segments are reviewed out of the total, and the review
   time used.
5. Then render as normal.

**How to measure.** Simulate on G1 and G2: apply the human path inside the selected
segments with `replace_range` and score with the existing harness. That's the same
"after-review MAE" as in item 2. Separately, check that the app renders a merged path
with no visible jumps at segment edges, by eye on 2–3 clips with `af.compare`.

**Open questions.**
- Should normal (non-review) recording keep truncate-forward? Suggest yes. Make replace
  the mode only when the path has `auto` targets after the current time, or add an
  explicit toggle.
>Answer : yes
- Is a 1 s glide at segment ends acceptable, or should the human's last point snap?
>Answer: unsure, assume 1s glide is OK

**Effort.** `CameraPath` changes plus tests: half a day. Review mode in `sportscam.py`:
1–2 days, since `process()` is one long loop with key handling spread through it.

## 4. Score changes from the scoreboard

**What exists today.**
- `GameState` (period, score, shots) is copied onto **every `CameraTarget`**, so each pkl
  already holds the score history.
- `[`/`]` and the middle click (goal) change the score. Goals also append a bookmark.
- `annotate_ouput_frame` draws "Period N / Score a-b" on the output.
- The mini-view shows the scoreboard itself.
- Bookmarks are saved to the pkl in source time and written to
  `<file>_bookmarks.txt` in output time, as YouTube links.
- `GameState._get_lr` swaps left and right in period 2 (teams change ends).

**Existing bug:** the goal bookmark text at `sportscam.py:674` is missing its `f`
prefix. Every goal bookmark in both games reads
`goal {self.game_state.score[0]}-{self.game_state.score[1]`, and the closing brace is
missing too.

**Ground truth we already have:** score changes, from the targets' game_state.

| Game | Goals (source time, s) | Final |
|---|---|---|
| G1 | 1538, 1583, 1705, 2069, 2293, 3339, 3683 | 3-4 |
| G2 | 2429, 2429 (double entry, 0.6 s apart; looks like a double click), 2473, 2525, 2789, 2824 | 2-4 |

These are the operator's click times, typically a second or two after the puck went in.

**Proposed approach.**

1. **Locate the digits.** The scoreboard sits at a fixed place in the band proxy. The clock
   is at about proxy (2060, 114) in G1, and the two score digits are below it. Find the
   digit regions from frame differences at the known goal times: the pixels that change
   exactly once there. Store them as per-rink offsets from the clock, shifted per game by
   item 1's vertical correction.
2. **Read the digits.** Recognising 7-segment-style LED digits at about 13×11 px in the
   half-res proxy is marginal. Better to sample the full-res frame at 1 Hz, where the
   digits are about 25 px tall. Then classify 0–9 with a template match against digits
   collected from the two games, or a tiny CNN. The OCR is easy because the font is fixed.
3. **Make it robust.** Accept a change only when the new value holds for 10 s or more,
   increases by exactly 1, and isn't undone. Flag anything else (a scoreboard reset, a
   wrong increment, an occluded board) as low confidence and put it in the review list
   from item 2.
4. **Map to sides.** The scoreboard shows home/away; the app wants left/right with the
   period-2 swap. Read each team's attacking direction from where the camera was just
   before the goal (item 5), or have the human confirm once per game.
5. **Output.** Set `game_state.score` on the targets from the goal time onwards, and add
   bookmarks with fixed text.

**How to measure.** Compare detected goal times and scores with the table above:
precision and recall of goals within ±15 s, and final score correct. That's 13 goals
across 2 games: enough to find failures, not enough to prove reliability.

**Open questions.**
- How often is the scoreboard wrong or not updated in your experience: late updates,
  corrections, off for a game?
>Answer: usually right, but wrong often enough that failure needs graceful fallback (or feature is not critical)
- Should the period come from the clock too? Period end is when the clock hits 0, which
  `af/clock.py` already sees.
>Answer: yes
- Are shots wanted automatically? Probably not feasible.
>Answer:  no

**Effort.** Digit location plus template OCR plus the change logic: about 1 day. Wiring
into `game_state` and bookmarks: half a day, plus the bookmark bug fix.

## 5. Goal detection from camera behaviour

**The signature,** checked in the pkls:

- **Before the goal:** the camera is parked in an end zone (x < ~400 or > ~2400).
- **Then a pause:** the human stops auto record. In the current pkls the next target
  comes 15–25 s later.
- **Then a jump to the centre faceoff,** x ≈ 1450 (1444–1478 across both games).

G1 examples:

| Goal (s) | Camera before | Next target |
|---|---|---|
| 1538 | x 2794 | x 1444 at 1557 |
| 1583 | x 2736 | x 2738 (the human kept recording; centre faceoff later) |

Period starts also go to centre (x 1450–1478), so the clock state is needed to tell them
apart. Faceoffs elsewhere go to the zone dots (x ≈ 460 / 2420 in G1).

**Two sources in the automatic pipeline:**

1. **Behaviour from detections, not the human path.** At review time there's no human
   path, so the signature has to come from the players:
   - Play is concentrated in an end zone.
   - Then a stoppage. The clock doesn't stop at whistles in this league, so use the
     player motion collapsing, which the motion features already show.
   - Then the players line up at **centre ice** for a faceoff: a tight cluster at the
     centre with near-zero velocity, both teams split either side of the red line.
   - Other stoppages end at a zone dot. Centre-ice faceoffs happen only after goals and
     at period starts, which the clock rules out. So "centre-ice faceoff not at a period
     start" is close to a goal detector by itself.
   - Bonus: the celebration (a group of 4–6 players clustering near the boards or net,
     then skating to the bench) is another cue.
2. **Scoreboard** (item 4).

**Cross-checking:**

| Faceoff | Scoreboard | Result |
|---|---|---|
| Centre ice | +1 within ±60 s | Confirmed goal |
| Centre ice | No change | Possible missed scoreboard update; put in review |
| None | Change | Possible scoreboard correction or mistake; put in review |

Side attribution comes from which end zone play was in before the stoppage, plus the
period (via `_get_lr`).

**How to measure.**
- Labels: the 13 goals in the table above.
- Negatives: every other stoppage, about 30 per game.
- Report precision and recall for each method alone and combined.
- Also check the faceoff detector on all stoppages against the human's resume x. Centre
  (about 1450) versus zone dots tells whether "where is the faceoff" works, independent
  of goals.

**Open questions.**
- Are there other centre-ice faceoffs in this league (after a penalty shot or similar)?
>Answer: Yes.  Rare.  But perhaps every center ice face off should be flagged for human review
- Do you want goal highlights generated automatically (the `h` mechanism) from the
  detected goal minus about 15 s?
>Answer: yes, but in a later phase

**Effort.** Faceoff-location detector plus goal logic, offline against both games: about
1 day. Combining with item 4: half a day.

## Suggested order

| Step | Item | Effort | Why this order |
|---|---|---|---|
| 1 | Calibration defaults | ~1 h | Cheap |
| 2 | Confidence and review curve | Offline | Tells whether "review a few minutes" is realistic before building UI |
| 3 | Replace-in-range data model, then review mode in the app | | |
| 4 | Faceoff and goal detection, then scoreboard digits, then cross-check | | Can run in parallel with step 3 |

Branch: one `af-review` branch for the offline analysis (items 2, 4, 5). The app changes
(item 3, the bookmark bug, defaults) are better as a separate small branch from `main`,
because they touch `sportscam.py`, which you use day to day.

## Implementation plan (2026-10-18, before starting)

You asked for this whole design, plus one extra rule: **the first manually recorded
segment marks the render start.** Auto points before your first recorded point are
skipped when rendering, so to trim the warm-up you record a moment at the first faceoff.
It's non-destructive: the auto points stay in the pkl.

Order, with a commit and a log entry after each stage:

| Stage | What | Where |
|---|---|---|
| A | `CameraPath`: bisect lookup, `replace_range`, provenance (`source`); render starts at the first human point; app calibration defaults set to the two-game average | app |
| B | Confidence: an error model trained across games, review segments ranked by expected error, and the "minutes reviewed → MAE after review" curve with tiers; `autotrack.py` writes `review.json` | offline + autotrack |
| C | Review mode: a small start-up menu when `sportscam.py DIR` runs; jump to the next or previous segment; record replaces only the segment; accept; status line and timeline strip | app |
| D | Scoreboard and goals: find the clock automatically, read the score digits, take the period from the clock, a centre-ice faceoff detector, cross-check, then write bookmarks, score on the targets, review entries and intermission cut suggestions | offline + autotrack + app |

Your answers that shape it:
- Normal recording keeps truncate-forward; replace-in-range applies only in review mode.
- A 1 s glide at segment edges is fine.
- Review in tiers: 5–10 minutes first, then optional second and third passes.
- Close enough is the goal.
- The scoreboard can break, so anything using it has a fallback and goes to review when
  unsure.
- Every centre-ice faceoff is flagged for review.
- No automatic shots. Goal highlights come later.

## Implementation log

### Stage A: path data model, render start, defaults (done)

- `CameraPath.camera_index` now uses bisect instead of a linear scan. A test checks it
  matches the old scan exactly.
- New `CameraPath.replace_range(t0, t1, targets)`. It's not used by normal recording yet,
  which still truncates forward.
- New `CameraPath.get_start_time()`. On a path that mixes auto and recorded points, the
  first recorded point is the start; otherwise it's the first point. The app's opening
  "jump to the first camera" uses it, so playback and rendering start at your first
  recorded point.
  - **Caveat:** if the first thing you record is a fix in the middle of the game, that
    becomes the start. Review mode (stage C) will show the start time and ask you to set
    it first.
- App calibration defaults, and `af/calibration.py`, are now the average of the
  hand-calibrated games: top_of_roi -1732 (left -75, right 161), center_x_offset 54,
  zones 793/2129, mini-view 3384/2020, rotation 6.3. The zones and mini-view rotation
  come only from the two games where you had set them.
- Tests: `tests/test_camerapath.py` (3) and `tests/test_autotrack.py` (4) all pass.

### Stage B: confidence and review segments (done)

`af/confidence.py`. Each sample gets a **review score = expected error × P(live play)**.

- **Expected error:**
  - The inputs are how sure the camera is (the q10–q90 spread of quantile models, and
    disagreement within a 5-model ensemble), how fast the path moves, the player count
    and its stability, distance from the median player, and the usual features.
  - A LightGBM error model maps those inputs to the expected error.
  - It's trained across games: each game's inputs and errors come from models trained on
    the *other* game.
- **P(live):** a classifier for "would the operator be framing now?". It matters a lot:
  without it the top segments landed in stoppages and intermissions, and 5 minutes of
  review did no better than random. Cross-game AUC is 0.94 on G1 and 0.97 on G2.
- **Segments:**
  - Flag where the 3 s smoothed score is in the top 20%.
  - Merge flagged stretches less than 5 s apart, pad each by 2 s, and drop any shorter
    than 3 s.
  - Rank by total expected error.
  - Each segment gets a reason: fast transition, unsure where play is, models disagree,
    or player detections unstable.

**Review curve** (each game scored by models that never saw it; reviewing = replacing
the auto path with yours inside the chosen segments):

| Minutes reviewed | G1 MAE | G2 MAE | Best possible (G1 / G2) | Within 300 px (G2) |
|---|---|---|---|---|
| 0 | 179 | 180 | | 82% |
| 3 | 162 | 158 | 146 / 143 | 85% |
| 5 | 148 | 147 | 127 / 128 | 86% |
| 10 | 122 | 119 | 89 / 88 | 90% |
| 15 | 96 | 90 | 59 / 58 | 93% |
| 20 | 76 | 84 | 40 / 48 | 93% |

**Suggested tiers:** first pass 5 min (about 180 → 148), second pass +5 min (→ about
120), third pass +10 min (→ about 80).

**In autotrack:**
- `autotrack.py train` also saves the review models: `models/review_*.txt` plus
  `review_models.json`.
- `autotrack.py DIR` writes `DIR\review.json`, listing every segment best first with
  t0, t1, score, expected px, P(live) and reason.
- The app splits segments into tiers after you set the start point, so warm-up segments
  don't use up review time.
- On G2's videos with no pkl (`D:\temp\af\render_g2`), it found 87 segments in 12 s.
  The reasons: 71 fast transition, 13 unsure, 2 models disagree, 1 detections unstable.

### Stage C: review mode in the app (done)

**Start-up menu.** `python sportscam.py DIR` with no other flags first prints what's in
the folder: number of videos, game pkl camera points (auto and recorded), and whether
autotrack has run and how many review segments it found. Then it asks:
1. Review the auto path.
2. Edit / record (the old default; `--edit` skips the menu).
3. Render.
4. Run autotrack.

`--review` opens review mode directly.

**Review mode** (`review.py` holds `ReviewSession`; the app wiring is in `sportscam.py`):

| Key | Action |
|---|---|
| `=` / `-` | Next / previous segment. Playback starts 3 s before and pauses 1 s after. |
| Enter | Segment OK (saved in the game pkl as `review_done`). |
| `a` | Re-record. While recording, new points replace the path; the rest is untouched, with about a 1 s glide back into the auto path. A single click replaces one point. |
| `T` | Start: a recorded point here. Rendering starts at the first *recorded* point (your rule). Any earlier recorded points stop counting. |
| `Y` | End: the path stops here, and rendering ends at the last point. |
| `c` … `c` | Cut from the first press to the second (for example over an intermission). It deletes the points between and flags the first, so the render jumps. `C` cancels. |

- **Tiers** are worked out live after the start point: tier 1 is about the first 5 min
  of segments by rank, tier 2 the next 5 min, tier 3 the next 10. Within a tier,
  segments play in time order. Status lines show per-tier progress, the start and end,
  and the current segment's reason and expected error.
- **Timeline strip** along the bottom: segments in red, orange and yellow by tier, green
  once reviewed. A white box marks the current segment and a white line the playhead.
- Points re-recorded in review mode are tagged `review`, not `human`, so a mid-game fix
  can never become the render start; only `T`, or recording outside review mode, sets it.
- In review mode the old `c` (cut at the *last* point) and the cut that `p` makes are
  disabled, because on an auto path the last point is the end of the video.
- Tests: `tests/test_review.py` (3), plus the camera path and autotrack tests: 10 pass.
- **Headless drive** of `--review` on G2's videos, with the windows stubbed and keys
  scripted: `T`, `=`, Enter, `=`, `a`…`a`, Enter, `c`…`c`, `Y`, `q`. The saved pkl has
  the start at 0, two review points from the re-record, a cut point and its restart
  point, the end at 1266 s, `review_done` with both segments, and a still-sorted path.
- **Not yet checked by a person:** how it feels to use. Key choices, pause timing and
  timeline readability need you to try it.

### Stage D: scoreboard, goals, periods, cuts (done)

**Scoreboard** (`af/scoreboard.py`), read from the half-res band proxy:
- **Clock:** found automatically. The seconds digits are the pixels that change with a
  strong 1 Hz rhythm. Found at proxy (2080, 117) in G1 and (2100, 133) in G2, both
  correct.
- **Score digits:** the same offsets below the clock in both games. Brightness is the
  per-pixel max channel, because greyscale makes the blue home digits nearly vanish.
- **Score change:** a lasting step. The 7 s median image 10 s after differs from 10 s
  before (measured on the most-changed quarter of pixels), while both sides are steady.
  Very large steps are the board being switched on, off or reset.
- **Clock running** gives the periods: running with players on the ice, at least 8 min,
  gaps under 2 min bridged. The intermission countdown also runs, but on empty ice.

**Play** (`af/events.py`):
- **Faceoffs:** a few seconds with almost nobody moving fast and a tight group, then a
  restart. The place is the group's median x.
- **Goals:** a centre-ice faceoff that isn't a period opener, after end-zone play. The
  end decides the team, the way the app does (`attribute_to_left_team`, period-2 swap).
- **Intermissions:** a minute or more of nearly empty ice. This is the fallback for
  periods when there's no board.

**Combined** (`af/gameinfo.py`):
- Board goals matched to play goals within 25 s are "confirmed". Others are "board only"
  or "play only".
- Home/away is mapped onto the app's score[0]/score[1] by vote from confirmed goals.
- When the board works, only board goals count. When it doesn't, play goals count, and
  all of them go to review.

| | G1 | G2 |
|---|---|---|
| Board, final | 4-3 (Tigers–Warriors) | **6-2** (Huskies–Tigers) |
| Operator's log | 7 goals, score history identical to ours | 6 entries, 2-4: **missed 2 goals** (about 2205 s and 3717 s) and double-clicked one (2429 s) |
| Ours (board + play) | 7 of 7 at the right times, sides right, history = yours | 8 of 8, final 2-6 in the app's team order |
| Play alone | 5 of 7, 1 extra | found 2 goals the operator missed; 1 extra (P3 start), missed 1 |
| Periods / cuts | clock periods 993–1894, 2060–2962, 3132–4116; cuts within 3–7 s of yours | P1 includes the warm-up countdown (gap < 2 min); cuts close to yours |

**In autotrack and the app:**
- `auto_path.pkl` now carries period and score on every point, plus `period - N` and
  `goal a-b` bookmarks. The app copies the bookmarks in, along with the path, when the
  game has none.
- `review.json` gets event entries ahead of the camera segments:
  - **goal:** board only, or play only. Enter = it's a goal (adds it if it wasn't
    counted); Backspace = not a goal (takes it out). The score on every later point and
    all goal bookmarks are rebuilt.
  - **faceoff:** a centre-ice faceoff nothing explains. Enter = fine.
  - **cut:** an intermission. The review jumps to its start and pauses; Enter applies
    the cut, Backspace keeps it.
- On G2 as a new game, autotrack reports "3 periods, 8 goals ((2, 6)), scoreboard used"
  and writes 4 event entries (2 doubtful goals, 2 cuts) plus 87 camera segments.
- **Headless drives** of the app:
  - both cuts applied (no points left inside them);
  - adding a play-only goal gave a new `goal 2-5` bookmark and shifted the later one;
  - rejecting a board-only goal took it back out; the final score stayed consistent.
- Tests: 12 pass (event ordering and sidecar game info added).

**Known limits:**
- The digits aren't read, only counted from 0-0. A board correction (a goal taken back)
  looks like another goal. It goes to review if play doesn't confirm it.
- Two-digit scores are untested.
- If the scoreboard moves position (another rink), the clock is still found
  automatically, but the digit offsets are fixed for this board.

## Checkpoint, 2026-10-18 night: what to try

All on branch `af-review` (not merged). To try it on game 2's videos as if it were a
new game:

```
cd C:\dev\sportscam-wt\af-review
python sportscam.py D:\temp\af\render_g2
```
Choose **1** in the menu. Then:
1. Press `T` at the first faceoff.
2. Press `=` to step: intermission cuts and goal questions first, then camera segments
   by tier.
3. For each one, Enter, Backspace, or `a` to re-record.
4. Press `q` to save.

A render (`-r`) then starts at your `T` point and skips the cut intermissions.

`D:\temp\af\render_g2` holds hardlinks to the 2026-08-15 videos plus the autotrack
outputs; your own pkl there isn't touched. To run it on your new game, run
`autotrack.py D:\RawVideo\2026-10-18` again from this worktree (seconds, since detection
is cached) and open it the same way.

### Fixes after the first real use (2026-10-19, game 2026-10-18)

What happened:
- You chose menu option 2 (edit), which never shows review.
- Your pkl kept the camera path from the first autotrack run, because the sidecar is only
  copied in once. So it had no scores or bookmarks.
- A goal you middle-clicked in review only changed that one point.
- The scoreboard wasn't used for this game.

Changes:
- **Menu:** review is item 1 and the default (Enter). The menu shows the start, end,
  review progress, and whether a newer auto path is waiting. Option 2 says
  "R switches to review".
- **`R`** switches between review and edit inside the app. Edit mode shows a hint when
  review items exist.
- **Timeline** moved to just under the status lines, because the window can be taller
  than the screen.
- **Refresh:** when autotrack has been re-run (the sidecar's `created` differs from the
  game pkl's `auto_path_created`) and the game's path is an auto path, the app takes the
  new path and keeps the game's own points:
  - start, review fixes, cut and restart markers, and the end;
  - removed stretches inside cuts, and nothing after the end;
  - goal decisions, replayed.

  It makes a backup first: `<game>.pkl.before-refresh`.
- **Goal decisions are saved** in `review_goal_changes`. Goals typed into the old app
  (one point scored, the rest not) are recognised and converted. Any within 60 s of a
  goal the new path already has are dropped, so they don't count twice: your 30:06 goal
  matched the board's 30:45 goal.
- **Goals entered in review mode** (middle click, or `[ { ] }`) change the score from
  that moment on, and neither click moves the camera.
- **Cut items** already covered by a cut in the path count as reviewed.
- **Scoreboard:** the clock-running threshold assumed a stopped clock at the 20th
  percentile. This game's clock runs about 90% of the time, so running stretches dropped
  out and no periods were found. The threshold is now half the running level, giving
  periods 197–1159, 1321–2283 and 2449–3411 s. G1 and G2 are unchanged.
- **Faceoff check:** a centre faceoff within 30 s *before* a board goal counts as
  explained, because the board updates late.
- **Accuracy line:** "vs the hand-tracked path" is skipped when the folder's path is
  mostly auto.

**2026-10-18 result:** the board gives 6 goals at the right times, checked frame by
frame against the board: 4-2 to the Bombshells. Your copy refreshed to 2-4 in the app's
team order, with all goal and period bookmarks, and kept your start 3:09, both cuts, the
30:06 point and the end 57:21. Tests: 15 pass.
- **Menu option 5, convert for sharing:** does what `convert.bat` does
  (`ffmpeg -i <first video>_processed.mp4 -c:v hevc_nvenc -preset fast -vtag hvc1`) and
  writes next to your other finished games. It prompts `name: 2026-10-18-` and you type
  the rest, e.g. `Bombshells_v_Supernovas`, giving
  `D:\RawVideo\2026-10-18-Bombshells_v_Supernovas.mp4`. It asks before overwriting. A
  2-minute test render went from 771 MB to 31 MB.
