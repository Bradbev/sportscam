"""autotrack steps: detect -> ice mask -> features -> model -> DIR/auto_path.pkl."""

import os
import time

from .calibration import load_calibration, video_files
from .extract_dets import detect, probe
from . import confidence as C
from . import gameinfo
from .export import make_sidecar, write_review, write_sidecar
from .game import CACHE_NAME, Game
from .icemask import build_cache
from . import model as M
from .features import HZ

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(REPO, "models", "yolo11x.pt")


def ensure_weights(path=WEIGHTS):
    if not os.path.exists(path):
        from ultralytics.utils.downloads import attempt_download_asset
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"downloading detector weights to {path}")
        attempt_download_asset(path)
    return path


def prepare(video_dir, cache_dir=None, rebuild_features=False):
    """Run (or resume) detection and the ice mask, then load the game's features."""
    cache = cache_dir or os.path.join(video_dir, CACHE_NAME)
    files = video_files(video_dir)
    if not files:
        raise SystemExit(f"no .mp4 files in {video_dir}")
    calib, _ = load_calibration(video_dir)
    t0 = time.time()
    detect(files, os.path.join(cache, "dets"), calib["rotation"], ensure_weights)  # weights fetched only if needed
    edges = os.path.join(cache, "icemask", "ice_edges.npz")
    if not os.path.exists(edges):
        longest = max(files, key=lambda f: probe(f)[3])
        print(f"building ice mask from {os.path.basename(longest)}")
        build_cache(longest, os.path.dirname(edges), calib["rotation"])
    g = Game(video_dir, cache, rebuild=rebuild_features)
    print(f"{len(g.t)} samples ready ({time.time() - t0:.0f}s)")
    return g


def game_entries(game):
    """Review entries for everything about the game that isn't certain."""
    out = []
    for gl in game["goals"]:
        if gl["status"] == "confirmed" and game["board_ok"]:
            continue
        scored = gl in game["scored"]
        why = {"board only": "scoreboard changed, no centre faceoff seen after",
               "play only": "centre faceoff after end-zone play, scoreboard didn't change",
               "confirmed": "goal from play (no scoreboard)"}[gl["status"]]
        out.append({"kind": "goal", "t": gl["t"], "t0": gl["t"] - 12000, "t1": gl["t"] + 5000, "team": gl["team"],
                    "scored": scored, "reason": f"{'goal' if scored else 'possible goal'}: {why}"})
    for f in game["centre_faceoffs"]:
        out.append({"kind": "faceoff", "t0": f["t"] - 25000, "t1": f["t"] + 3000,
                    "reason": "centre-ice faceoff with no goal or period start: a missed goal?"})
    for a, b in game["cuts"]:
        out.append({"kind": "cut", "t0": a, "t1": b, "reason": "intermission: Enter cuts it out"})
    return out


def run(video_dir, cache_dir=None, model_path=M.DEFAULT_MODEL):
    model = M.Model(model_path)  # fail before an hour of detection, not after
    g = prepare(video_dir, cache_dir)
    pred = model.predict(g)
    info = {"model": os.path.basename(model_path), "trained_on": model.meta["trained_on"],
            "calibration": "main pkl" if os.path.exists(os.path.splitext(video_files(video_dir)[0])[0] + ".pkl")
            else "defaults"}
    try:
        game = gameinfo.analyse(g, pred)
    except Exception as e:  # periods and goals are a bonus; never lose the camera path over them
        print(f"periods/goals skipped: {e!r}")
        game = None
    if game:
        info["scoreboard"] = "used" if game["board_ok"] else "not found (goals from play only)"
        final = [0, 0]
        for gl in game["scored"]:
            final[gl["team"]] += 1
        print(f"{len(game['periods'])} periods, {len(game['scored'])} goals ({tuple(final)}), "
              f"scoreboard {info['scoreboard']}")
    dst = write_sidecar(video_dir, make_sidecar(g.calib, g.t, pred, g.max_x, info, game=game))
    print(f"wrote {dst} ({len(g.t) / HZ / 60:.0f} min of video)")
    entries = game_entries(game) if game else []
    review = C.ReviewModels.load()
    if review is None:
        print("no review models in models/; review.json has no camera segments (autotrack.py train makes them)")
    else:
        score, err, live, S = review.score(g, pred)
        segs = C.segments(g.t, score)
        reasons = C.describe(g.t, S, review.names, segs)
        for (t0, t1, total), why in zip(segs, reasons):
            m = (g.t >= t0) & (g.t <= t1)
            entries.append({"kind": "camera", "t0": t0, "t1": t1, "score": round(total),
                            "expected_px": round(float(err[m].mean())), "live": round(float(live[m].mean()), 2),
                            "reason": why})
    print(f"wrote {write_review(video_dir, entries)} ({len(entries)} entries)")
    if g.human is not None:
        from .metrics import evaluate, fmt
        print(fmt("  vs the hand-tracked path in this folder", evaluate(g.t, pred, g.human)))
    return dst


def train(specs, model_path=M.DEFAULT_MODEL):
    """specs: video dirs, each optionally DIR=CACHE when its cache lives elsewhere."""
    games = []
    for s in specs:
        d, _, c = s.partition("=")
        games.append(prepare(d, c or None))
    m, used = M.train(games)
    M.save(m, games[0].names, model_path, [g.video_dir for g in used])
    n = sum(g.use.sum() for g in used) / HZ / 60
    print(f"trained on {len(used)} games ({n:.0f} hand-tracked min) -> {model_path}")
    C.train(used).save()
    print(f"review models -> {C.MODELS}")
