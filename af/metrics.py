"""Score an automatic camera path against the human one.

Primary metric: mean absolute error of camera x (px), over the framed time the
human actually recorded.  Supporting numbers:

  med / p90       error distribution
  within300       fraction of time within 300px (~1/6 of the 1920px crop)
  contain         human crop centre inside the auto crop
  lag             time shift (s) of the auto path that minimises MAE; positive
                  means the auto path is *behind* the human, negative means ahead
  mae@lag         MAE at that best shift
  jitter          auto path total travel / human total travel
"""

import numpy as np

OUT_W = 1920


def evaluate(times_ms, auto_x, human, mask=None, step_ms=200):
    times_ms = np.asarray(times_ms, dtype=np.float64)
    auto_x = np.asarray(auto_x, dtype=np.float64)
    grid = np.arange(times_ms[0], times_ms[-1], step_ms)
    ok = human.scored_mask(grid)
    if mask is not None:
        ok &= mask(grid)
    grid = grid[ok]
    if len(grid) == 0:
        return None
    hx = human.x_at(grid)
    ax = np.interp(grid, times_ms, auto_x)
    err = np.abs(ax - hx)

    hc = hx + OUT_W / 2
    contain = ((hc >= ax) & (hc <= ax + OUT_W)).mean()

    best = (np.inf, 0.0)
    for shift_s in np.arange(-4.0, 4.01, 0.4):
        sx = np.interp(grid + shift_s * 1000.0, times_ms, auto_x)
        m = np.abs(sx - hx).mean()
        if m < best[0]:
            best = (m, shift_s)

    # travel on a common, uniformly-sampled grid
    def travel(v):
        return np.abs(np.diff(v)).sum()

    return {
        "n_s": len(grid) * step_ms / 1000.0,
        "mae": float(err.mean()),
        "med": float(np.median(err)),
        "p90": float(np.percentile(err, 90)),
        "within300": float((err <= 300).mean()),
        "contain": float(contain),
        "lag": float(best[1]),
        "mae_lag": float(best[0]),
        "jitter": float(travel(ax) / max(travel(hx), 1.0)),
    }


def fmt(name, m):
    if m is None:
        return f"{name}: no overlap with human path"
    return (f"{name:28s} MAE {m['mae']:6.0f}  med {m['med']:5.0f}  p90 {m['p90']:5.0f}  "
            f"<300px {m['within300']*100:5.1f}%  contain {m['contain']*100:5.1f}%  "
            f"lag {m['lag']:+.1f}s (MAE {m['mae_lag']:5.0f})  travel x{m['jitter']:.2f}  "
            f"[{m['n_s']/60:.1f} min]")
