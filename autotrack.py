"""Automatic camera path for a folder of game video.

  python autotrack.py DIR                  detect players, predict the camera path, write DIR/auto_path.pkl
  python autotrack.py train DIR [DIR ...]  retrain the model from hand-tracked folders (DIR=CACHE to use
                                           a cache that is not DIR/autotrack_cache)

Detection takes about 50 min per game on an RTX 4080 and is cached in
DIR/autotrack_cache, so re-runs (e.g. after changing the calibration in the
app) take seconds.

sportscam.py copies auto_path.pkl into the game once, when the game's own pkl
has no camera targets yet.
"""

import argparse
import sys

from af import model as M
from af import pipeline


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        ap = argparse.ArgumentParser(prog="autotrack.py train")
        ap.add_argument("dirs", nargs="+")
        ap.add_argument("--model", default=M.DEFAULT_MODEL)
        args = ap.parse_args(sys.argv[2:])
        pipeline.train(args.dirs, args.model)
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--cache", help="default: DIR/autotrack_cache")
    ap.add_argument("--model", default=M.DEFAULT_MODEL)
    args = ap.parse_args()
    pipeline.run(args.dir, args.cache, args.model)


if __name__ == "__main__":
    main()
