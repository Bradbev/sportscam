"""autotrack steps: detect -> ice mask -> features -> model -> DIR/auto_path.pkl."""

import os
import time

from .calibration import load_calibration, video_files
from .extract_dets import detect, probe
from .export import make_sidecar, write_sidecar
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


def run(video_dir, cache_dir=None, model_path=M.DEFAULT_MODEL):
    model = M.Model(model_path)  # fail before an hour of detection, not after
    g = prepare(video_dir, cache_dir)
    pred = model.predict(g)
    info = {"model": os.path.basename(model_path), "trained_on": model.meta["trained_on"],
            "calibration": "main pkl" if os.path.exists(os.path.splitext(video_files(video_dir)[0])[0] + ".pkl")
            else "defaults"}
    dst = write_sidecar(video_dir, make_sidecar(g.calib, g.t, pred, g.max_x, info))
    print(f"wrote {dst} ({len(g.t) / HZ / 60:.0f} min of video)")
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
