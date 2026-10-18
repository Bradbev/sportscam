"""Per-player velocity from sample-to-sample matching of on-ice detections.

Each kept player in sample i is matched to a player in sample i+1 (200 ms
later) by minimum-cost assignment on feet position, with a distance cutoff.
Velocities are in camera-x px/s.  Per-sample summaries describe where play is
heading, which positions alone cannot.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

MAX_STEP_PX = 160.0  # full-res band px per 200 ms (~10 m/s at mid-ice)


def player_points(det, keep):
    """Per sample: (N,2) feet points in band px, and their camera x."""
    b = det.boxes[keep]
    s = det.sample[keep]
    pts = np.stack([0.5 * (b[:, 0] + b[:, 2]), b[:, 3]], 1)
    order = np.argsort(s, kind="stable")
    s, pts = s[order], pts[order]
    bounds = np.searchsorted(s, np.arange(len(det.times) + 1))
    return [pts[bounds[i]:bounds[i + 1]] for i in range(len(det.times))]


def velocities(det, keep, with_x=False):
    pts = player_points(det, keep)
    dt = np.diff(det.times, append=det.times[-1] + 200) / 1000.0
    scale = det.pan_span / det.fixed_w  # band px -> camera x px
    out = []
    for i in range(len(pts)):
        a = pts[i]
        b = pts[i + 1] if i + 1 < len(pts) else np.zeros((0, 2))
        v = np.full(len(a), np.nan)
        if len(a) and len(b) and 0 < dt[i] < 0.5:
            d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
            r, c = linear_sum_assignment(np.minimum(d, MAX_STEP_PX * 3))
            ok = d[r, c] <= MAX_STEP_PX
            v[r[ok]] = (b[c[ok], 0] - a[r[ok], 0]) * scale / dt[i]
        out.append((v, det.col_to_x(a[:, 0])) if with_x else v)
    return out


def summarise(vel, fast=250.0):
    """Per-sample motion features.  `vel` items are (v, x) from velocities(with_x=True)."""
    n = len(vel)
    f = np.full((n, 10), np.nan)
    for i, (v, x) in enumerate(vel):
        ok = ~np.isnan(v)
        v, x = v[ok], x[ok]
        if len(v) == 0:
            continue
        srt = np.sort(v)
        r, l = v > fast, v < -fast
        f[i] = [
            v.mean(),
            np.median(v),
            r.mean(),
            l.mean(),
            srt[-3:].mean(),   # fastest rightward
            srt[:3].mean(),    # fastest leftward
            x[r].max() if r.any() else np.nan,   # front of a rightward rush
            x[l].min() if l.any() else np.nan,   # front of a leftward rush
            np.average(x, weights=np.abs(v) + 1e-3),  # speed-weighted centre
            x[np.argmax(np.abs(v))],                  # fastest skater
        ]
    # positions with no movers fall back to the speed-weighted centre
    for j in (6, 7):
        f[:, j] = np.where(np.isnan(f[:, j]), f[:, 8], f[:, j])
    for j in range(f.shape[1]):
        col = f[:, j]
        idx = np.where(~np.isnan(col))[0]
        f[:, j] = np.interp(np.arange(n), idx, col[idx]) if len(idx) else 0.0
    return f, ["v_mean", "v_med", "frac_r", "frac_l", "v_top_r", "v_top_l",
               "x_front_r", "x_front_l", "x_speedw", "x_fastest"]


def tracked_velocities(det, keep, half_window=3, with_x=True):
    """Like velocities(), but chains the frame-to-frame matches into tracks and
    fits each player's velocity over +-half_window samples of its own track
    (least-squares slope), which is far less noisy than a single 200 ms step."""
    pts = player_points(det, keep)
    scale = det.pan_span / det.fixed_w
    n = len(pts)
    # next[i][j] = index in sample i+1 matched to player j of sample i, or -1
    nxt, prv = [], [np.full(len(pts[0]), -1)]
    for i in range(n):
        a = pts[i]
        b = pts[i + 1] if i + 1 < n else np.zeros((0, 2))
        m = np.full(len(a), -1)
        p_b = np.full(len(b), -1)
        if len(a) and len(b) and det.times[i + 1] - det.times[i] < 500:
            d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
            r, c = linear_sum_assignment(np.minimum(d, MAX_STEP_PX * 3))
            ok = d[r, c] <= MAX_STEP_PX
            m[r[ok]] = c[ok]
            p_b[c[ok]] = r[ok]
        nxt.append(m)
        if i + 1 < n:
            prv.append(p_b)
    out = []
    for i in range(n):
        a = pts[i]
        v = np.full(len(a), np.nan)
        for j in range(len(a)):
            ts, xs = [det.times[i]], [a[j, 0]]
            k, jj = i, j
            for _ in range(half_window):
                jj = nxt[k][jj]
                if jj < 0:
                    break
                k += 1
                ts.append(det.times[k]); xs.append(pts[k][jj, 0])
            k, jj = i, j
            for _ in range(half_window):
                jj = prv[k][jj]
                if jj < 0:
                    break
                k -= 1
                ts.insert(0, det.times[k]); xs.insert(0, pts[k][jj, 0])
            if len(ts) >= 3:
                tt = (np.array(ts) - ts[0]) / 1000.0
                v[j] = np.polyfit(tt, np.array(xs) * scale, 1)[0]
        out.append((v, det.col_to_x(a[:, 0])) if with_x else v)
    return out
