"""Debug video: band proxy + detections + human vs auto crop.

  python -m af.debugvid OUT.mp4 START_S END_S --game DIR[=CACHE] [--auto PRED.npz]

Green boxes: kept (on ice).  Red: rejected.  Yellow rectangle: human crop.
Cyan: auto crop.  Bottom strip: x over time (human yellow, auto cyan).
"""

import argparse
import glob
import os

import cv2
import numpy as np

from .game import open_game
from .players import Detections

SCALE = 0.5  # proxy scale relative to full-res band
OUT_W = 1920


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("start", type=float)
    ap.add_argument("end", type=float)
    ap.add_argument("--auto", default=None)
    ap.add_argument("--game", required=True)
    args = ap.parse_args()

    g = open_game(args.game)
    human = g.human
    det = Detections(g.dets, g.edges, g.max_x)
    keep = det.on_ice()
    auto = None
    if args.auto:
        a = np.load(args.auto)
        auto = (a["t"], a["pred"])

    # map sample index -> (proxy file, frame index)
    files = sorted(glob.glob(os.path.join(g.dets, "*_band.mp4")))
    npzs = sorted(glob.glob(os.path.join(g.dets, "*.npz")))
    starts = np.cumsum([0] + [len(np.load(f)["times_ms"]) for f in npzs])

    i0 = int(np.searchsorted(det.times, args.start * 1000))
    i1 = int(np.searchsorted(det.times, args.end * 1000))
    writer = None
    cap, cap_file = None, -1
    band0 = det.band0
    k_of = lambda x: (x + OUT_W / 2) / det.pan_span * det.fixed_w  # camera-centre col
    crop_w = OUT_W / det.pan_span * det.fixed_w

    for i in range(i0, i1):
        fi = int(np.searchsorted(starts, i, side="right") - 1)
        if fi != cap_file:
            cap = cv2.VideoCapture(files[fi])
            cap.set(cv2.CAP_PROP_POS_FRAMES, i - starts[fi])
            cap_file = fi
        ok, img = cap.read()
        if not ok:
            break
        tms = det.times[i]
        sel = det.sample == i
        for b, kp in zip(det.boxes[sel], keep[sel]):
            p0 = (int(b[0] * SCALE), int((b[1] - band0) * SCALE))
            p1 = (int(b[2] * SCALE), int((b[3] - band0) * SCALE))
            cv2.rectangle(img, p0, p1, (0, 255, 0) if kp else (0, 0, 255), 1)
        cols = np.arange(0, det.fixed_w, 8)
        for edge in (det.top, det.bot):
            pts = np.stack([cols * SCALE, edge[cols] * SCALE], 1).astype(np.int32)
            cv2.polylines(img, [pts], False, (255, 0, 255), 1)
        hx = human.x_at([tms])[0] if human is not None and human.scored_mask([tms])[0] else None
        h_img = img.shape[0]
        if hx is not None:
            c = k_of(hx)
            cv2.rectangle(img, (int((c - crop_w / 2) * SCALE), 2), (int((c + crop_w / 2) * SCALE), h_img - 3), (0, 255, 255), 3)
        if auto is not None:
            ax = np.interp(tms, auto[0], auto[1])
            c = k_of(ax)
            cv2.rectangle(img, (int((c - crop_w / 2) * SCALE), 8), (int((c + crop_w / 2) * SCALE), h_img - 9), (255, 255, 0), 2)
        cv2.putText(img, f"{tms/1000:7.1f}s  kept {int(keep[sel].sum())}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        img = cv2.resize(img, (1600, int(img.shape[0] * 1600 / img.shape[1])))
        if writer is None:
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), 10, (img.shape[1], img.shape[0]))
        writer.write(img)
    if writer:
        writer.release()


if __name__ == "__main__":
    main()
