# sportscam

Create a python venv with `python -m venv venv`

Activate with `venv\scripts\activate`
Install with `pip install -r requirements.txt`

# Automatic camera path

`python autotrack.py DIR` detects the players in every video in DIR, predicts the
camera path a human operator would have recorded and writes it to `DIR\auto_path.pkl`.
Opening DIR in sportscam.py then starts from that path (it is copied in once, while the
game has no camera targets of its own).

* Install with `pip install -r requirements-autotrack.txt` (CUDA torch, see the file).
* Detection takes about 50 min per game on an RTX 4080.
  It is cached in `DIR\autotrack_cache` (about 650 MB), so re-running after
  changing the calibration in the app takes seconds.
* Calibration comes from the game's pkl if there is one, otherwise the defaults in
  `af/calibration.py`.
* The model is `models/autotrack_lgbm.txt`. Retrain it from hand-tracked games with
  `python autotrack.py train DIR [DIR ...]`.
* Dev tools: `python -m af.transfer DIR DIR` (cross-game scores) and
  `python -m af.compare` (side-by-side clip of human and auto camera).

# BUGS

# Future features
* Refactor game state into its own class.  CameraTargets will store that
* Add shot count / blocked shots?