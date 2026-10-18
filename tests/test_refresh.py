"""Refreshing an auto path from a re-run sidecar keeps the game's own edits."""
from camerapath import CameraPath
from cameratarget import CameraTarget
from gamestate import GameState
from review import apply_goal, goal_bookmarks, legacy_goal_changes, refresh_path


def auto_path(n_s, score_from=None):
    out = []
    for s in range(n_s):
        gs = GameState()
        if score_from is not None and s >= score_from:
            gs.score = (1, 0)
        out.append(CameraTarget(s * 1000.0, x=100 + s, game_state=gs, source="auto"))
    return out


def test_edits_cuts_end_and_goals_survive_a_refresh():
    old = auto_path(100)
    # start at 5 s, a cut 20-40 s, a review fix at 50 s with a hand-typed goal, end at 90 s
    start = CameraTarget(5000.0, x=7, source="human")
    cut_a, cut_b = CameraTarget(20000.0, x=8, cut_to=True, source="review"), CameraTarget(40000.0, x=9, source="review")
    fix = CameraTarget(50500.0, x=10, source="review")
    fix.game_state.score = (0, 1)  # the old app only scored this one point
    end = CameraTarget(90000.0, x=11, source="review")
    p = CameraPath()
    p.camera_targets = old
    for c in (start, cut_a, cut_b, fix, end):
        p.replace_range(c.time - 500, c.time + 500, [c])
    p.replace_range(20001, 39999, [])
    p.replace_range(90001, 1e9, [])
    changes = legacy_goal_changes(p.camera_targets)
    assert changes == [(50500.0, 1, 1)]

    new = refresh_path(p.camera_targets, auto_path(120, score_from=30), changes)
    times = [c.time for c in new]
    assert times == sorted(times)
    assert {c.time for c in new if c.source != "auto"} == {5000.0, 20000.0, 40000.0, 50500.0, 90000.0}
    assert not [t for t in times if 20000 < t < 40000]  # cut kept
    assert max(times) == 90000.0  # end kept
    gs = {c.time: c.game_state.score for c in new}
    assert gs[10000.0] == (0, 0)
    assert gs[45000.0] == (1, 0)         # the new path's goal at 30 s
    assert gs[60000.0] == (1, 1)         # plus the hand-typed one from 50.5 s
    marks = goal_bookmarks(new, [(1.0, "bookmark"), (2.0, "goal 9-9")])
    assert [m[1] for m in marks] == ["bookmark", "goal 1-0", "goal 1-1"]


def test_apply_goal_never_goes_negative():
    t = auto_path(5)
    apply_goal(t, 2000, 0, -1)
    assert t[4].game_state.score == (0, 0)


def test_typed_in_goal_near_a_board_goal_is_not_counted_twice():
    from review import drop_known_goals
    new = auto_path(100, score_from=30)  # the board's goal at 30 s
    kept, dropped = drop_known_goals([(31000.0, 1, 1), (95000.0, 0, 1)], new)
    assert dropped == [(31000.0, 1, 1)] and kept == [(95000.0, 0, 1)]
