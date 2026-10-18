"""Where the game is live: start, period breaks, end -> the app's cuts.

live = scoreboard clock running AND a full complement of players on the ice.
Short interruptions (stoppages, timeouts) are bridged; long ones (intermissions)
become cuts.  Compared against the human's spans (pieces between cut_to flags).

  python -m af.phases g1 [g2 ...]
"""

import sys

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

from .games import Game
from .simple_methods import HZ

RUN_THRESH = 2.5     # mean |grey diff| of the clock crop over 1 s
MIN_PLAYERS = 9      # 20 s median of on-ice count; ~13 in play, ~6 in intermissions
BRIDGE_S = 120       # interruptions shorter than this stay inside one span
MIN_SPAN_S = 60
MIN_RUN_S = 20
LEAD_S, TAIL_S = 0, 0


def runs(mask):
    """[(i0, i1)] index ranges (inclusive, exclusive) where mask is True."""
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.where(d > 0)[0], np.where(d < 0)[0]))


def live_spans(t, clock_diff, count):
    run = uniform_filter1d(clock_diff, int(2 * HZ)) > RUN_THRESH
    full = median_filter(count, int(20 * HZ), mode="nearest") >= MIN_PLAYERS
    live = run & full
    # brief clock blips (setting the clock, intermission countdown starts) are not play
    spans = [[t[a], t[b - 1]] for a, b in runs(live) if t[b - 1] - t[a] >= MIN_RUN_S * 1000]
    merged = []
    for s in spans:
        if merged and s[0] - merged[-1][1] < BRIDGE_S * 1000:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [(a - LEAD_S * 1000, b + TAIL_S * 1000) for a, b in merged if b - a >= MIN_SPAN_S * 1000]


def human_spans(h, min_s=30):
    spans, start = [], h.t[0]
    for i in range(len(h.t) - 1):
        if h.cut[i]:
            spans.append((start, h.t[i]))
            start = h.t[i + 1]
    spans.append((start, h.t[-1]))
    return [s for s in spans if s[1] - s[0] >= min_s * 1000]


def main():
    for name in sys.argv[1:] or ["g1"]:
        g = Game(name)
        d = np.load(f"D:/temp/af/feats/clock_{name}.npz")["diff"]
        cnt = g.X[:, g.names.index("cnt@0")]
        auto = live_spans(g.t, d, cnt)
        hum = human_spans(g.human)
        print(f"== {name}")
        print("  human:", "  ".join(f"{a/1000:.0f}-{b/1000:.0f}" for a, b in hum))
        print("  auto: ", "  ".join(f"{a/1000:.0f}-{b/1000:.0f}" for a, b in auto))
        for a, b in hum:
            k = int(np.argmin([abs(x - a) + abs(y - b) for x, y in auto]))
            x, y = auto[k]
            print(f"  span {a/1000:6.0f}-{b/1000:6.0f}: start {(x - a)/1000:+6.1f}s  end {(y - b)/1000:+6.1f}s")


if __name__ == "__main__":
    main()
