"""Side-by-side review clip: human camera (top) vs auto camera (bottom).

Crops are produced with the same geometry as sportscam.py's render path
(rotation + per-x angle, y from the calibration, rotate_point correction),
warping only the 1920x720 crop instead of the whole frame.

  python -m af.compare PRED.npz OUT.mp4 START_S DURATION_S
"""

import argparse
import json
import math
import os
import subprocess

import cv2
import numpy as np

from .human import HumanPath
from .games import GAMES
from util import rotate_point

OUT = (1920, 720)


class Cropper:
    def __init__(self, pkl, w, h):
        self.c = pkl
        a = math.radians(pkl["rotation"] + pkl["angle_right"])
        self.max_x = int(math.sin(a) * h + math.cos(a) * w - OUT[0])
        self.w, self.h = w, h
        self.xp = [0, self.max_x // 2 + pkl["center_x_offset"], self.max_x]

    def crop(self, raw, x):
        c = self.c
        x = int(np.clip(x, 0, self.max_x))
        angle = float(np.interp(x, self.xp, [c["angle_left"], 0, c["angle_right"]]))
        y_live = -c["top_of_roi"] + float(np.interp(x, self.xp, [c["top_of_roi_left"], 0, c["top_of_roi_right"]]))
        size = np.array([self.w, self.h], dtype=np.float64)
        M = cv2.getRotationMatrix2D(tuple(size / 2.0), c["rotation"] + angle, 1.0)
        new = np.abs(M[:, :2]) @ size
        M[:, -1] += (new - size) / 2.0
        rot_w = int(new[0])
        y = int(rotate_point((x + OUT[0] // 2 - rot_w // 2, y_live), -angle)[1])
        x = int(min(x, rot_w - OUT[0]))
        M[0, 2] -= x
        M[1, 2] -= y
        return cv2.warpAffine(raw, M, OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred")
    ap.add_argument("out")
    ap.add_argument("start", type=float)
    ap.add_argument("duration", type=float)
    ap.add_argument("--game", default="g1")
    args = ap.parse_args()

    cfg = GAMES[args.game]
    human = HumanPath(cfg["pkl"])
    meta = json.load(open(os.path.join(cfg["dets"], "meta.json")))
    p = np.load(args.pred)
    files = meta["files"]
    start_ms = args.start * 1000
    fi = max(i for i, f in enumerate(files) if f["base_ms"] <= start_ms)
    f = files[fi]
    w, h, fps = f["w"], f["h"], f["fps"]
    crop = Cropper(human.pkl, w, h)
    local = (start_ms - f["base_ms"]) / 1000.0
    n = int(args.duration * fps)
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-hwaccel", "cuda", "-ss", f"{local:.3f}", "-i", f["path"],
                            "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                           stdout=subprocess.PIPE, bufsize=w * h * 3 * 2)
    half = (OUT[0] // 2, OUT[1] // 2)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", f"{half[0]}x{half[1] * 2}", "-r", str(fps), "-i", "-",
                            "-c:v", "h264_nvenc", "-preset", "p5", "-cq", "23", "-pix_fmt", "yuv420p", args.out],
                           stdin=subprocess.PIPE)
    for k in range(n):
        buf = dec.stdout.read(w * h * 3)
        if len(buf) < w * h * 3:
            break
        raw = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        t = start_ms + k * 1000.0 / fps
        hx = human.x_at([t])[0]
        ax = float(np.interp(t, p["t"], p["pred"]))
        top = cv2.resize(crop.crop(raw, hx), half, interpolation=cv2.INTER_AREA)
        bot = cv2.resize(crop.crop(raw, ax), half, interpolation=cv2.INTER_AREA)
        framed = human.scored_mask([t])[0]
        cv2.putText(top, f"HUMAN x={hx:4.0f}" + ("" if framed else " (not framed)"), (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(bot, f"AUTO  x={ax:4.0f}   t={t/1000:7.1f}s", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        enc.stdin.write(np.vstack([top, bot]).tobytes())
    enc.stdin.close()
    enc.wait()
    dec.kill()


if __name__ == "__main__":
    main()
