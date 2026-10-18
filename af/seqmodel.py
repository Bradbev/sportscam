"""Temporal CNN: per-sample player features -> camera x, non-causal (offline).

Dilated 1-D convolutions give about +-12 s of context.  Trained with L1 loss on
the human-labelled samples only, using the same grouped folds as the LightGBM
model so the numbers are comparable.

  python -m af.seqmodel [--holdout-from 3093] [--epochs N]
"""

import argparse

import numpy as np
import torch
import torch.nn as nn

from .human import HumanPath
from .learned import frame_features, segment_ids
from .metrics import evaluate, fmt
from .motion import summarise, velocities
from .players import Detections
from .simple_methods import DETS, EDGES, HZ, PKL, fill_nan
from scipy.ndimage import uniform_filter1d


def per_sample_features(det, max_x):
    keep = det.on_ice()
    xs = det.per_sample(keep)
    q, hist, cnt = frame_features(xs, max_x)
    mot, _ = summarise(velocities(det, keep, with_x=True))
    pos_cols = [6, 7, 8, 9]  # motion features that are x positions
    m = mot.copy()
    m[:, pos_cols] /= max_x
    vel_cols = [0, 1, 4, 5]
    m[:, vel_cols] /= 1000.0
    F = np.hstack([q / max_x, hist, (cnt / 15.0)[:, None], m]).astype(np.float32)
    return F


class TCN(nn.Module):
    def __init__(self, d_in, width=64, dilations=(1, 2, 4, 8, 16, 1), drop=0.2):
        super().__init__()
        self.inp = nn.Conv1d(d_in, width, 1)
        self.blocks = nn.ModuleList()
        for d in dilations:
            self.blocks.append(nn.Sequential(
                nn.Conv1d(width, width, 3, padding=d, dilation=d),
                nn.GELU(), nn.Dropout(drop),
                nn.Conv1d(width, width, 1)))
        self.out = nn.Conv1d(width, 1, 1)

    def forward(self, x):  # x: (B, d, T)
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h)
        return self.out(h).squeeze(1)


def train_predict(F, y, train_mask, epochs=60, win=256, batch=32, seed=0, device="cuda"):
    """Train on samples where train_mask (and y valid); return prediction for all samples."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n, d = F.shape
    Ft = torch.tensor(F.T[None], device=device)
    yt = torch.tensor(np.nan_to_num(y), device=device, dtype=torch.float32)
    mt = torch.tensor(train_mask & ~np.isnan(y), device=device)
    model = TCN(d).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    starts_ok = np.where(np.convolve(train_mask, np.ones(win), "valid") > win * 0.2)[0]
    steps = epochs * 50
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    model.train()
    for _ in range(steps):
        st = rng.choice(starts_ok, batch)
        idx = torch.tensor(st[:, None] + np.arange(win)[None], device=device)
        xb = Ft[0][:, idx].permute(1, 0, 2)  # (B, d, win)
        pred = model(xb)
        m = mt[idx]
        loss = (torch.abs(pred - yt[idx]) * m).sum() / m.sum().clamp(min=1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        return model(Ft).cpu().numpy()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--holdout-from", type=float, default=0)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--out", default=r"D:/temp/af/seq_cv.npz")
    args = ap.parse_args()

    human = HumanPath(PKL)
    human.set_frame_size(3840, 2880)
    det = Detections(DETS, EDGES, human.max_x)
    t = det.times
    F = per_sample_features(det, human.max_x)
    scored = human.scored_mask(t)
    sid = segment_ids(human, t)
    use = scored & (sid >= 0)
    y = np.full(len(t), np.nan, dtype=np.float32)
    y[use] = human.x_at(t[use]) / human.max_x

    from sklearn.model_selection import GroupKFold
    pred = np.full(len(t), np.nan)
    ui = np.where(use)[0]
    for k, (tr, te) in enumerate(GroupKFold(args.folds).split(ui, groups=sid[use])):
        train_mask = np.zeros(len(t), bool)
        train_mask[ui[tr]] = True
        p = np.mean([train_predict(F, y, train_mask, args.epochs, seed=s) for s in range(args.seeds)], 0)
        pred[ui[te]] = p[ui[te]]
    full = fill_nan(np.clip(pred * human.max_x, 0, human.max_x))
    print(fmt("tcn cv raw", evaluate(t, full, human)))
    for sm in (1, 2):
        print(fmt(f"tcn cv sm{sm}", evaluate(t, uniform_filter1d(full, int(sm * HZ), mode="nearest"), human)))
    np.savez(args.out, t=t, pred=full)

    if args.holdout_from:
        cut = args.holdout_from * 1000
        train_mask = use & (t < cut)
        p = np.mean([train_predict(F, y, train_mask, args.epochs, seed=s) for s in range(args.seeds)], 0)
        ph = uniform_filter1d(np.clip(p * human.max_x, 0, human.max_x), int(2 * HZ), mode="nearest")
        print(fmt(f"tcn holdout>= {args.holdout_from:.0f}s sm2", evaluate(t, ph, human, mask=lambda g: g >= cut)))


if __name__ == "__main__":
    main()
