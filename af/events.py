"""Game events from the players: faceoffs, goals (a centre-ice faceoff after play in
an end zone) and period breaks.

A faceoff is a few seconds where almost nobody moves fast and the skaters stand in
a tight group, ending when play restarts.  Its place is the group's median camera
x: about the rink centre (max_x//2 + center_x_offset) for centre-ice faceoffs.
Centre-ice faceoffs happen after goals and at period starts; a period start
follows a long break, a goal follows play in an end zone.

  python -m af.events DIR[=CACHE] ...      evaluate against the hand-tracked path
"""

import sys

import numpy as np
from scipy.ndimage import uniform_filter1d

from .features import HZ

STILL_FAST_FRAC = 0.06   # fraction of skaters moving > 250 px/s, 1 s average
STILL_IQR = 650          # px: the group is tight
STILL_MIN_S = 2.5
MIN_PLAYERS = 7
CENTRE_TOL = 220         # px of camera x from the centre dot
BREAK_S = 90             # a gap in live play this long is a period break, not a stoppage
END_ZONE_FRAC = 0.22     # camera x within this fraction of either end counts as an end zone


def _col(g, name):
    return g.X[:, g.names.index(name)]


def centre_x(g):
    return g.max_x // 2 + g.calib["center_x_offset"]


def faceoffs(g):
    """[{t: drop time ms, x: place (camera x), centre: bool}] in time order."""
    moving = uniform_filter1d(_col(g, "frac_r@0") + _col(g, "frac_l@0"), int(HZ), mode="nearest")
    iqr = _col(g, "q75@0") - _col(g, "q25@0")
    med = _col(g, "q50@0")
    still = (moving < STILL_FAST_FRAC) & (iqr < STILL_IQR) & (_col(g, "cnt@0") >= MIN_PLAYERS)
    d = np.diff(np.concatenate([[0], still.astype(int), [0]]))
    out = []
    for a, b in zip(np.where(d > 0)[0], np.where(d < 0)[0]):
        if (b - a) / HZ < STILL_MIN_S:
            continue
        tail = slice(max(a, b - int(2 * HZ)), b)
        x = float(np.median(med[tail]))
        out.append({"t": float(g.t[min(b, len(g.t) - 1)]), "x": x, "centre": abs(x - centre_x(g)) < CENTRE_TOL})
    # a faceoff can be split by a brief shuffle; keep the last of any within 6 s
    merged = []
    for f in out:
        if merged and f["t"] - merged[-1]["t"] < 6000:
            merged[-1] = f
        else:
            merged.append(f)
    return merged


def goals(g, pred):
    """Centre faceoffs that follow play in an end zone (not a period start).

    Returns [{t: estimated goal time ms, faceoff_t, end: 'left'|'right'}]: the goal
    time is the last moment the camera was in that end zone before the faceoff."""
    live = uniform_filter1d(_col(g, "frac_r@0") + _col(g, "frac_l@0"), int(5 * HZ), mode="nearest") > 0.12
    lo, hi = END_ZONE_FRAC * g.max_x, (1 - END_ZONE_FRAC) * g.max_x
    centres = [f for f in faceoffs(g) if f["centre"]]
    # the first centre faceoff after each break (or during it) is a period start
    starts = set()
    for a, b in intermissions(g):
        after = [f["t"] for f in centres if f["t"] >= a]
        if after:
            starts.add(after[0])
            starts.update(f["t"] for f in centres if a <= f["t"] <= b)
    out = []
    for f in centres:
        if f["t"] in starts:
            continue
        i = int(np.searchsorted(g.t, f["t"]))
        back = slice(max(0, i - int(60 * HZ)), max(0, i - int(3 * HZ)))
        live_idx = np.where(live[back])[0]
        if len(live_idx) == 0:
            continue  # no play in the last minute: a period start
        last_live = back.start + live_idx[-1]
        if (i - last_live) / HZ > BREAK_S:
            continue
        window = pred[max(0, last_live - int(20 * HZ)):last_live + 1]
        in_end = np.where((window < lo) | (window > hi))[0]
        if len(in_end) == 0:
            continue
        k = max(0, last_live - int(20 * HZ)) + in_end[-1]
        goal = {"t": float(g.t[k]), "faceoff_t": f["t"], "end": "right" if pred[k] > hi else "left"}
        if out and goal["faceoff_t"] - out[-1]["faceoff_t"] < 15000:
            out[-1] = goal  # two faceoffs for one stoppage
        else:
            out.append(goal)
    return out


def intermissions(g, min_s=60):
    """[(t0, t1)] where the ice is nearly empty for a minute or more."""
    from scipy.ndimage import median_filter
    cnt = median_filter(_col(g, "cnt@0"), int(30 * HZ), mode="nearest")
    low = cnt < 8
    d = np.diff(np.concatenate([[0], low.astype(int), [0]]))
    return [(g.t[a], g.t[b - 1]) for a, b in zip(np.where(d > 0)[0], np.where(d < 0)[0])
            if (g.t[b - 1] - g.t[a]) >= min_s * 1000]


def human_restarts(g, gap_ms=5000):
    """(time, x) of every point where the operator resumed after a pause."""
    h = g.human
    dt = np.diff(h.t)
    idx = np.where((dt > gap_ms) & ~h.cut[:-1])[0]
    return [(h.t[i + 1], h.x[i + 1]) for i in idx]


def human_goals(g):
    ct = g.human.path.camera_targets
    out = []
    for a, b in zip(ct, ct[1:]):
        if b.game_state.score != a.game_state.score and (not out or b.time - out[-1] > 5000):
            out.append(b.time)
    return out


def main():
    from .game import open_game
    from .model import Model
    model = Model()
    for spec in sys.argv[1:]:
        g = open_game(spec)
        fo = faceoffs(g)
        restarts = human_restarts(g)
        c = centre_x(g)
        print(f"== {g.video_dir}: {len(fo)} faceoffs detected ({sum(f['centre'] for f in fo)} centre), "
              f"{len(restarts)} operator restarts")
        hit = hit_c = n_c = 0
        for t, x in restarts:
            near = [f for f in fo if -12000 <= f["t"] - t <= 8000]
            is_c = abs(x - c) < CENTRE_TOL
            n_c += is_c
            if near:
                hit += 1
                hit_c += is_c and near[0]["centre"]
        print(f"  restarts with a detected faceoff within -8/+12 s: {hit}/{len(restarts)}; "
              f"centre restarts found as centre: {hit_c}/{n_c}")
        gl = goals(g, model.predict(g))
        hg = human_goals(g)
        tp = [h for h in hg if any(abs(x["t"] - h) < 15000 for x in gl)]
        fp = [x for x in gl if not any(abs(x["t"] - h) < 15000 for h in hg)]
        print(f"  goals: operator {len(hg)}, detected {len(gl)}, matched {len(tp)} (within 15 s), "
              f"extra {len(fp)}: {[round(x['t'] / 1000) for x in fp]}")
        print(f"  operator goals: {[round(h / 1000) for h in hg]}")
        print(f"  detected:       {[(round(x['t'] / 1000), x['end']) for x in gl]}")


if __name__ == "__main__":
    main()
