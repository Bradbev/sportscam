# af-learned: better features and models from the detection cache

Branch `af-learned`, created from `af-harness`. Worktree: `C:\dev\sportscam-wt\af-learned`.

## Plan (written 2026-10-18, before starting)

Starting point: LightGBM on per-sample quantiles and histograms of player x, with ±8 s
context. MAE 196 (CV) and 240 (P3 holdout). The errors are mostly transition decisions,
so this branch tries giving the model more information about **where play is going**:

1. **Player motion features.** Match each player to the nearest player in the next sample
   (200 ms apart) to get per-player x velocity. Summarise it per sample: mean signed
   velocity, the fraction of players moving right versus left, and the velocity of the
   fastest players (a rush shows up as a few fast skaters heading the same way).
2. **Spatial weighting.** Down-weight the goalies (whoever sits nearest each crease) and
   people persistently in the bench areas.
3. **A sequence model** (a small temporal CNN or GRU over the per-sample features, about
   ±10 s), trained with the same grouped CV, as an alternative to LightGBM on
   hand-stacked offsets.
4. Report every result two ways: grouped CV and the P3 time holdout. The holdout is the
   better guide to how it will do on a new game.

## Progress log

Scores are MAE in px: 6-fold grouped CV, then the P3 time holdout. Baseline 196 / 240.

- **Per-player velocities.** Each player is matched to the next sample by Hungarian
  assignment on feet position, with a 160 px cutoff. Features: mean and median velocity,
  the fraction moving fast in each direction, and the 3 fastest in each direction.
  → **189 / 235.**
- **Error analysis.** About 30% of total error comes from 47 episodes (each ≥1 s at
  >500 px off, 135 s in total).
  - The biggest is the last 23 s of the game (4118 s). That's probably handshakes or
    players leaving, and needs a "game over" or stoppage rule, not better tracking.
  - The typical miss (contact sheet `D:\temp\af\sheet_2213.jpg`) is a breakout. A few
    skaters sprint toward one end and the human follows the leaders at once. The model
    waits for the median player to catch up.
- **Rush-front features.** The x of the most advanced fast-moving skater in each
  direction, the speed-weighted centre, and the x of the fastest skater.
  → **182 / 216.** The P3 holdout improved by 19 px, the best gain so far.
- **Temporal CNN** (`af/seqmodel.py`): dilated 1-D convolutions with ±12 s of context, on
  the same features. → 206 / 228. That's worse than LightGBM. Blending 25% of it into
  LightGBM gives no gain. Parked: with only 37 minutes of strongly correlated samples,
  boosted trees on hand-made context features fit better.
- **LightGBM sweep.** Finer time offsets and leaf or child settings each change MAE by
  only ±3 px. The **L1 objective** is better: → **177 / 207**, now the default.
- **Track-smoothed velocities.** Frame matches are chained into tracks and velocity is
  fitted over ±3 samples. → 179 / 205, no real change. Velocity noise is not the
  bottleneck.
- **Isotonic recalibration** of predictions (fitted out-of-fold) → 186. Worse, because
  the L1 model is already calibrated.
- **Where the error is now** (L1 model):

  | Human speed (px/s) | Share of time | MAE | Share of error |
  |---|---|---|---|
  | Holding still (<50) | 49% | 146 | 40% |
  | 50–300 | 30% | 186 | 32% |
  | 300–600 | 16% | 220 | 20% |
  | >600 | 5% | 293 | 8% |

  The model also stops short when the human parks deep: by about 93 px at the left end
  and 125 px at the right.

## Checkpoint, 2026-10-18 evening

**Best model:** LightGBM (L1 objective) on on-ice player positions plus motion and
rush-front features, with ±8 s of context, then smoothed over 2 s. `python -m af.learned
--motion`

| | MAE | Median | p90 | Within 300 px | Contain |
|---|---|---|---|---|---|
| 6-fold grouped CV, whole game | **177** | 118 | 399 | 83% | 98.5% |
| Trained on P1–P2, tested on P3 | **207** | 128 | 471 | 80% | 97% |

For scale: your own path delayed by 1 s scores 155, and delayed by 2 s scores 302.

**Things you can look at:**
- `D:\temp\af\compare_P3_3130.mp4`: 2 minutes from period 3, your camera on top and the
  auto camera below. It uses the holdout model, which never saw period 3.
- `D:\temp\af\paths_learned.png`: your x against auto x over several segments.
- `D:\temp\af\render\`: a folder of hardlinks to your four raw MP4s (same files, no extra
  disk used) plus an **auto-generated pkl for the whole game**. Run
  `python sportscam.py D:\temp\af\render` to scrub through it with the auto path, or add
  `-r` to render it. Everything stays inside `D:\temp\af\render`.

**What it doesn't do yet:**
- Detect stoppages, warm-up or the end of the game. It frames the whole video, and your
  cuts aren't reproduced.
- The last 23 s of the game is the biggest single error.
