"""Periods, goals and intermission cuts for a game, from the scoreboard and from play.

Scoreboard (af.scoreboard), when it can be found:
  periods  long stretches of the clock running (>= 8 min, gaps < 2 min bridged)
  goals    lasting steps in the home / away score digit inside the game
Play (af.events), always:
  periods  intermissions are a nearly empty ice for a minute or more
  goals    centre-ice faceoffs after play in an end zone
Goals seen by both are "confirmed"; seen by one only, "board only" / "play only".
Everything except confirmed goals is offered for review, and every centre-ice
faceoff that no goal or period start explains is too (a goal the board missed).

The score is kept the app's way: score[0] is the team that attacks the right end
in periods 1 and 3 (GameState._get_lr swaps period 2).  Home/away on the board is
mapped onto that from the confirmed goals.

  python -m af.gameinfo DIR[=CACHE] ...      evaluate (needs a hand-tracked path for the operator's goals)
"""

import os
import sys

import numpy as np

from . import events as E
from . import scoreboard as SB

MIN_PERIOD_S = 8 * 60
BRIDGE_S = 120
MATCH_S = 25


def _runs(mask):
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.where(d > 0)[0], np.where(d < 0)[0]))


def board(g):
    """Scoreboard crops for the game (cached in CACHE/scoreboard.npz), or None."""
    path = os.path.join(g.cache, "scoreboard.npz")
    if os.path.exists(path):
        d = np.load(path)
        return {k: d[k] for k in d.files}
    origin = SB.locate_clock(g.cache)
    if origin is None:
        return None
    crops = SB.read_game(g.cache, origin)
    np.savez_compressed(path, origin=origin, **crops)
    return dict(crops, origin=np.array(origin))


def board_periods(g, crops):
    """Clock running with players on the ice (the intermission countdown runs too, on empty ice)."""
    from scipy.ndimage import median_filter
    run = SB.clock_running(crops["clock"])
    cnt = median_filter(g.X[:, g.names.index("cnt@0")], int(30 * 5), mode="nearest")
    spans = []
    for a, b in _runs(run):
        if (b - a) < 60 * 5 or np.median(cnt[a:b]) < 9:
            continue
        t0, t1 = g.t[a], g.t[min(b, len(g.t)) - 1]
        if spans and t0 - spans[-1][1] < BRIDGE_S * 1000:
            spans[-1][1] = t1
        else:
            spans.append([t0, t1])
    return [(a, b) for a, b in spans if b - a >= MIN_PERIOD_S * 1000]


def play_periods(g):
    """Fallback: the stretches between intermissions (and before/after the empty ice)."""
    breaks = E.intermissions(g)
    edges = [g.t[0]] + [x for ab in breaks for x in ab] + [g.t[-1]]
    spans = [(edges[i], edges[i + 1]) for i in range(0, len(edges) - 1, 2)]
    return [(a, b) for a, b in spans if b - a >= MIN_PERIOD_S * 1000]


def period_of(periods, t):
    for k, (a, b) in enumerate(periods):
        if t <= b + 60000:
            return k + 1
    return len(periods)


def team_of_end(end, period):
    """Score index scoring at this end: the app's attribute_to_left_team, then _get_lr."""
    team = 0 if end == "right" else 1
    return 1 - team if period == 2 else team


def analyse(g, pred):
    crops = board(g)
    periods = board_periods(g, crops) if crops is not None else []
    board_ok = len(periods) >= 2
    if not board_ok:
        periods = play_periods(g)
    lo = periods[0][0] - 60000 if periods else g.t[0]
    hi = periods[-1][1] + 60000 if periods else g.t[-1]

    board_goals = []
    if board_ok:
        for side in ("home", "away"):
            for sec, strength in SB.steps(crops[side]):
                t = float(sec * 1000 + g.t[0])
                if lo <= t <= hi and strength < SB.BOARD_RESET:
                    board_goals.append({"t": t, "side": side, "strength": strength})
    play_goals = [p for p in E.goals(g, pred) if lo <= p["t"] <= hi]

    # match board goals to play goals; map home/away onto the app's teams by vote
    votes = {"home": [0, 0], "away": [0, 0]}
    goals = []
    used = set()
    for bg in sorted(board_goals, key=lambda x: x["t"]):
        near = [k for k, p in enumerate(play_goals) if k not in used and abs(p["t"] - bg["t"]) < MATCH_S * 1000]
        k = min(near, key=lambda k: abs(play_goals[k]["t"] - bg["t"])) if near else None
        if k is not None:
            used.add(k)
            team = team_of_end(play_goals[k]["end"], period_of(periods, bg["t"]))
            votes[bg["side"]][team] += 1
        goals.append(dict(bg, status="confirmed" if k is not None else "board only",
                          end=play_goals[k]["end"] if k is not None else None))
    home_team = 0 if votes["home"][0] + votes["away"][1] >= votes["home"][1] + votes["away"][0] else 1
    side_votes = sum(votes["home"]) + sum(votes["away"])
    for gl in goals:
        gl["team"] = home_team if gl["side"] == "home" else 1 - home_team
    for k, p in enumerate(play_goals):
        if k not in used:
            goals.append({"t": p["t"], "side": None, "status": "play only", "end": p["end"],
                          "team": team_of_end(p["end"], period_of(periods, p["t"]))})
    goals.sort(key=lambda x: x["t"])
    # board says nothing about a goal the board itself confirmed, play-only goals stay out of the score
    scored = [gl for gl in goals if gl["status"] != "play only"] if board_ok else goals

    all_centre = [f for f in E.faceoffs(g) if f["centre"] and lo <= f["t"] <= hi]
    openers = set()
    for a, _ in periods:  # each period opens with a centre faceoff
        after = [f["t"] for f in all_centre if f["t"] >= a - 60000]
        if after:
            openers.add(after[0])
    goal_times = [gl["t"] for gl in goals]
    centre = [f for f in all_centre if f["t"] not in openers
              and not any(0 <= f["t"] - t <= 150000 for t in goal_times)]
    cuts = [(periods[k][1] + 5000, periods[k + 1][0] - 10000) for k in range(len(periods) - 1)]
    return {"board_ok": board_ok, "periods": periods, "goals": goals, "scored": scored,
            "home_team": home_team, "side_votes": side_votes, "centre_faceoffs": centre, "cuts": cuts}


def main():
    from .game import open_game
    from .model import Model
    model = Model()
    for spec in sys.argv[1:]:
        g = open_game(spec)
        info = analyse(g, model.predict(g))
        print(f"== {g.video_dir}: board {'found' if info['board_ok'] else 'NOT usable'}, "
              f"periods {[(round(a / 1000), round(b / 1000)) for a, b in info['periods']]}")
        score = [0, 0]
        for gl in info["goals"]:
            if gl in info["scored"]:
                score[gl["team"]] += 1
            print(f"  {gl['t'] / 1000:7.0f}s  {gl['status']:11s} side {gl['side']}  end {gl['end']}  team {gl['team']}"
                  f"  -> {tuple(score) if gl in info['scored'] else ''}")
        print(f"  home team = score[{info['home_team']}] ({info['side_votes']} votes); final {tuple(score)}")
        print(f"  unexplained centre faceoffs: {[round(f['t'] / 1000) for f in info['centre_faceoffs']]}")
        print(f"  cut suggestions: {[(round(a / 1000), round(b / 1000)) for a, b in info['cuts']]}")
        if g.human is not None:
            final = g.human.path.camera_targets[-1].game_state.score
            print(f"  operator's final score {final}")


if __name__ == "__main__":
    main()
