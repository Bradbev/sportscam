"""Review mode: step through the stretches autotrack is least sure of.

autotrack.py writes DIR/review.json (entries ranked by expected error).  Here the
entries after the start point (and before the end) are split into tiers by
cumulative duration, so a first pass is ~5 minutes of video, and are visited in
time order within a tier.  Progress is saved in the game pkl ("review_done").
"""

import json
import os

import cv2

REVIEW_JSON = "review.json"
TIER_BUDGET_S = (5 * 60, 5 * 60, 10 * 60)  # first pass, second pass, third pass
TIER_COLOURS = ((0, 0, 255), (0, 140, 255), (0, 220, 255), (120, 120, 120))  # BGR: red, orange, yellow, grey
LEAD_IN_MS = 3000   # playback starts this long before a segment
PAUSE_AFTER_MS = 1000


def is_auto(c):
    return getattr(c, "source", "human") == "auto"


def apply_goal(targets, t, team, delta):
    """Add (+1) / take back (-1) a goal for score index `team` on every target from t on."""
    for c in targets:
        if c.time >= t:
            gs = c.game_state.copy()
            score = list(gs.score)
            score[team] = max(0, score[team] + delta)
            gs.score = tuple(score)
            c.game_state = gs


def goal_bookmarks(targets, bookmarks):
    """bookmarks with the goal entries rebuilt from where the score goes up along the path."""
    marks = [b for b in bookmarks if not b[1].startswith("goal")]
    prev = (0, 0)
    for c in targets:
        s = c.game_state.score
        if sum(s) > sum(prev):
            marks.append((c.time, f"goal {s[0]}-{s[1]}"))
        prev = s
    return sorted(marks)


def legacy_goal_changes(targets):
    """Goals typed in by hand before they were recorded as changes: a recorded point whose
    score is ahead of the point before it (the change never reached later points)."""
    out = []
    for prev, c in zip(targets, targets[1:]):
        if is_auto(c):
            continue
        for team in (0, 1):
            if c.game_state.score[team] > prev.game_state.score[team]:
                out.append((c.time, team, c.game_state.score[team] - prev.game_state.score[team]))
    return out


def drop_known_goals(changes, new_targets, window_ms=60000):
    """(kept, dropped): typed-in goals the re-run path already has (the scoreboard saw them)."""
    goal_times, prev = [], (0, 0)
    for c in new_targets:
        if sum(c.game_state.score) > sum(prev):
            goal_times.append(c.time)
        prev = c.game_state.score
    kept, dropped = [], []
    for ch in changes:
        near = ch[2] > 0 and any(abs(ch[0] - t) <= window_ms for t in goal_times)
        (dropped if near else kept).append(ch)
    return kept, dropped


def refresh_path(old_targets, new_targets, goal_changes=()):
    """A re-run sidecar's path, with this game's own edits carried over.

    Kept from the old path: every point that isn't auto (start, review fixes, cut and
    restart markers, recordings), the cuts (auto points between a cut point and the
    next edit are removed), the end (nothing after a final edit), and goal decisions
    (replayed as score changes).  Scores and periods otherwise come from the new path."""
    from camerapath import CameraPath
    path = CameraPath()
    path.camera_targets = list(new_targets)
    edits = [c for c in old_targets if not is_auto(c)]
    for c in edits:
        _, _, nxt = path.get_camera_at_time(c.time)
        c.game_state = nxt.game_state.copy()
        path.replace_range(c.time - 500, c.time + 500, [c])
    for a, b in zip(edits, edits[1:]):
        if a.cut_to:
            path.replace_range(a.time + 1, b.time - 1, [])
    if old_targets and not is_auto(old_targets[-1]):
        path.replace_range(old_targets[-1].time + 1, float("inf"), [])
    for t, team, delta in goal_changes:
        apply_goal(path.camera_targets, t, team, delta)
    return path.camera_targets


def fmt_ms(ms):
    s = int(ms // 1000)
    h, m, s = s // 3600, s // 60 % 60, s % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class ReviewSession:
    def __init__(self, base_path, done=()):
        self.entries = []
        p = os.path.join(base_path, REVIEW_JSON)
        if os.path.exists(p):
            with open(p) as f:
                self.entries = json.load(f)["entries"]
        self.done = set(done)        # t0 of reviewed entries
        self.current = None          # entry being reviewed
        self.cut_start = None        # first press of a two-step cut
        self.record_from = None      # replace-recording: targets after this get replaced
        self.at_end = False

    # ---- tiers -------------------------------------------------------------
    def tiered(self, start_ms, end_ms):
        """[(tier, entry)] for entries inside [start, end], tier 0..3, time order within a tier."""
        inside = [e for e in self.entries if e["t0"] >= start_ms and e["t1"] <= end_ms]
        # goals, faceoffs and cuts are quick to check and matter most: always first
        out = [(0, e) for e in inside if e.get("kind", "camera") != "camera"]
        used, tier = 0.0, 0
        inside = [e for e in inside if e.get("kind", "camera") == "camera"]
        for e in inside:  # review.json is ranked best first
            while tier < len(TIER_BUDGET_S) and used >= sum(TIER_BUDGET_S[:tier + 1]):
                tier += 1
            out.append((tier, e))
            used += (e["t1"] - e["t0"]) / 1000
        out.sort(key=lambda r: (r[0], r[1].get("kind", "camera") == "camera", r[1]["t0"]))
        return out

    def step(self, start_ms, end_ms, forward=True):
        """Next (or previous) segment in review order; next skips reviewed ones."""
        order = [e for tier, e in self.tiered(start_ms, end_ms) if tier < len(TIER_BUDGET_S)]
        if not order:
            return None
        if forward:
            k = order.index(self.current) + 1 if self.current in order else 0
            rest = [e for e in order[k:] if e["t0"] not in self.done] or \
                   [e for e in order if e["t0"] not in self.done]
            self.current = rest[0] if rest else None
        else:
            k = order.index(self.current) - 1 if self.current in order else len(order) - 1
            self.current = order[max(k, 0)]
        self.at_end = False
        return self.current

    def accept(self):
        if self.current is not None:
            self.done.add(self.current["t0"])

    def progress(self, start_ms, end_ms):
        """Per tier: (segments done, segments, seconds done, seconds)."""
        rows = []
        tiered = self.tiered(start_ms, end_ms)
        for tier in range(len(TIER_BUDGET_S)):
            es = [e for t, e in tiered if t == tier]
            d = [e for e in es if e["t0"] in self.done]
            dur = lambda xs: sum((e["t1"] - e["t0"]) / 1000 for e in xs)
            rows.append((len(d), len(es), dur(d), dur(es)))
        return rows

    # ---- drawing -----------------------------------------------------------
    def status_lines(self, frame_time, start_ms, start_set, end_ms):
        rows = self.progress(start_ms, end_ms)
        tiers = "  ".join(f"T{k + 1} {d}/{n} ({sd / 60:.1f}/{s / 60:.1f}m)" for k, (d, n, sd, s) in enumerate(rows))
        start = f"start {fmt_ms(start_ms)}" if start_set else "start NOT SET (T at the first faceoff)"
        line1 = f"REVIEW  {tiers}  |  {start}  end {fmt_ms(end_ms)}"
        if self.cut_start is not None:
            line2 = f"CUT from {fmt_ms(self.cut_start)}: seek to the restart and press c again (C cancels)"
        elif self.current is not None:
            e = self.current
            kind = e.get("kind", "camera")
            keys = {"camera": "Enter = OK, a = re-record",
                    "goal": "Enter = goal, Backspace = no goal" if not e.get("scored") else
                            "Enter = goal OK, Backspace = not a goal",
                    "faceoff": "Enter = OK (add a goal with middle-click if missed)",
                    "cut": "Enter = cut the intermission out, Backspace = keep it"}[kind]
            state = "reviewed" if e["t0"] in self.done else keys
            extra = f" (~{e.get('expected_px', 0)}px)" if kind == "camera" else ""
            line2 = f"{kind} {fmt_ms(e['t0'])}-{fmt_ms(e['t1'])}: {e.get('reason', '')}{extra}  {state}"
        else:
            line2 = "= next segment  - previous  Enter accept  a re-record (replaces)  T start  Y end  c cut"
        return line1, line2

    def draw_timeline(self, img, frame_time, total_ms, start_ms, end_ms, y0, height=36):
        w = img.shape[1]
        x_of = lambda t: int(t / total_ms * (w - 1))
        cv2.rectangle(img, (0, y0), (w - 1, y0 + height), (40, 40, 40), -1)
        cv2.rectangle(img, (0, y0), (x_of(start_ms), y0 + height), (15, 15, 15), -1)
        cv2.rectangle(img, (x_of(end_ms), y0), (w - 1, y0 + height), (15, 15, 15), -1)
        for tier, e in self.tiered(start_ms, end_ms):
            col = (0, 200, 0) if e["t0"] in self.done else TIER_COLOURS[min(tier, 3)]
            if e.get("kind", "camera") != "camera" and e["t0"] not in self.done:
                col = (255, 0, 255)  # events: magenta
            cv2.rectangle(img, (x_of(e["t0"]), y0 + 4), (max(x_of(e["t1"]), x_of(e["t0"]) + 2), y0 + height - 4), col, -1)
        if self.current is not None:
            cv2.rectangle(img, (x_of(self.current["t0"]) - 3, y0), (x_of(self.current["t1"]) + 3, y0 + height),
                          (255, 255, 255), 2)
        if self.cut_start is not None:
            cv2.line(img, (x_of(self.cut_start), y0), (x_of(self.cut_start), y0 + height), (255, 0, 255), 3)
        cv2.line(img, (x_of(frame_time), y0 - 6), (x_of(frame_time), y0 + height + 6), (255, 255, 255), 2)
