# Autoframe progress index

Newest first. One line per approach. Details are in the linked note.

Baseline numbers are on the 0015 game: 36.9 framed minutes scored against the human path.
For scale: a constant centre camera scores MAE 921 px. The human's own path, delayed by
1 s, scores MAE 155 px; delayed by 2 s, MAE 302 px.

| Branch | Approach | Status | Best MAE (px) | Note |
|---|---|---|---|---|
| `autotrack` | Packaging the tracker for main: one command, sidecar, saved model. Contains all `af-*` commits | **ready to merge** | trained on both games; ~180 on a new game | [autotrack.md](autotrack.md) |
| TBD (`af-review`?) | Human review before render: calibration defaults, confidence scores, replace-in-range recording, scoreboard and goal detection | design only, not started | - | [af-review.md](af-review.md) |
| `af-game2` (now in `autotrack`) | Generalisation test on a second hand-tracked game (2026-08-15) | **transfers**: trained on one game, tested on the other | **179 (G2→G1) / 180 (G1→G2)**; 166 on G2 with both games | [af-game2.md](af-game2.md) |
| `af-learned` (now in `autotrack`) | LightGBM + player motion/rush-front features, L1 | **current best**; checkpoint reached | **177 (CV) / 207 (P3 holdout)** | [af-learned.md](af-learned.md) |
| `af-harness` (now in `autotrack`) | Harness, detection cache, median and first LightGBM baselines | done (base for the others) | 196 (CV) / 240 (P3 holdout) | [af-harness.md](af-harness.md) |
