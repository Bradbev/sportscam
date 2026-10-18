"""ReviewSession: tiers by cumulative duration, time order within a tier, navigation."""
import json

import pytest

pytest.importorskip("cv2")

from review import TIER_BUDGET_S, ReviewSession


def session(tmp_path, entries, done=()):
    (tmp_path / "review.json").write_text(json.dumps({"version": 1, "entries": entries}))
    return ReviewSession(str(tmp_path), done)


def seg(t0_s, dur_s, score):
    return {"kind": "camera", "t0": t0_s * 1000.0, "t1": (t0_s + dur_s) * 1000.0, "score": score}


def test_tiers_fill_by_duration_and_skip_before_start(tmp_path):
    # ranked best first, each 60 s long: 5 fill tier 1 (5 min), 5 tier 2, 10 tier 3, rest grey
    entries = [seg(10000 - 100 * k, 60, 1000 - k) for k in range(25)]
    entries.append(seg(5, 60, 5000))  # best of all, but in the warm-up
    s = session(tmp_path, sorted(entries, key=lambda e: -e["score"]))
    tiered = s.tiered(start_ms=600 * 1000, end_ms=1e9)
    assert all(e["t0"] >= 600 * 1000 for _, e in tiered)
    counts = [sum(1 for t, _ in tiered if t == k) for k in range(4)]
    assert counts == [5, 5, 10, 5]
    tier1 = [e["t0"] for t, e in tiered if t == 0]
    assert tier1 == sorted(tier1)  # time order within a tier
    assert sum(TIER_BUDGET_S) == 20 * 60


def test_step_visits_unreviewed_in_order_and_wraps(tmp_path):
    entries = [seg(300, 10, 3), seg(100, 10, 2), seg(200, 10, 1)]
    s = session(tmp_path, entries, done=[200000.0])
    first = s.step(0, 1e9)
    assert first["t0"] == 100000.0
    s.accept()
    assert s.step(0, 1e9)["t0"] == 300000.0  # 200 s is already reviewed
    s.accept()
    assert s.step(0, 1e9) is None  # everything reviewed
    assert s.step(0, 1e9, forward=False)["t0"] == 300000.0  # back still works
    assert s.progress(0, 1e9)[0][:2] == (3, 3)


def test_missing_review_json_is_empty(tmp_path):
    s = ReviewSession(str(tmp_path))
    assert s.step(0, 1e9) is None
    assert s.status_lines(0, 0, False, 1000)[0].startswith("REVIEW")


def test_events_come_before_camera_segments(tmp_path):
    entries = [seg(100, 10, 9), {"kind": "goal", "t0": 500000.0, "t1": 517000.0, "t": 512000.0, "team": 0},
               {"kind": "cut", "t0": 300000.0, "t1": 400000.0}]
    s = session(tmp_path, entries)
    assert [e.get("kind") for _, e in s.tiered(0, 1e9)] == ["cut", "goal", "camera"]
