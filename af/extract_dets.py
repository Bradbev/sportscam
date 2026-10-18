"""Full-game person detection cache.

Decodes every source file with ffmpeg (NVDEC), keeps every `stride`-th frame,
rotates it into the base-rotation frame (only the ice band is warped), and runs
YOLO on the full-resolution band.  All boxes above a low confidence floor are
kept so that filtering can be tuned offline without re-running the detector.

Also writes a half-resolution band proxy video at the same sample rate, for
debug overlays and any later pixel-based features.

Times match FastCap: file k frame n -> sum(frames of files < k) * 40ms + n * 40ms.

Usage:
  python -m af.extract_dets VIDEO_DIR OUT_DIR [--stride 5]   (normally run via autotrack.py)
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import threading
import queue
import time

import cv2
import numpy as np

ROT_BAND = (1550, 2600)  # rows of the base-rotation full-res frame to keep


def probe(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", 
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
        "-of", "json", path])
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), float(num) / float(den), int(s["nb_frames"])


def band_matrix(w, h, angle, y0):
    """rotate_image's matrix, shifted so row y0 of the rotated frame is row 0."""
    size = np.array([w, h], dtype=np.float64)
    M = cv2.getRotationMatrix2D(tuple(size / 2.0), angle, 1.0)
    MM = np.abs(M[:, :2])
    new = MM @ size
    M[:, -1] += (new - size) / 2.0
    M[1, 2] -= y0
    return M, int(new[0])


def reader(path, w, h, stride, q):
    cmd = ["ffmpeg", "-v", "error", "-hwaccel", "cuda", "-i", path,
           "-vf", f"select=not(mod(n\\,{stride}))", "-vsync", "0",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 3 * 2)
    n = 0
    nbytes = w * h * 3
    while True:
        buf = p.stdout.read(nbytes)
        if len(buf) < nbytes:
            break
        q.put((n * stride, np.frombuffer(buf, np.uint8).reshape(h, w, 3)))
        n += 1
    q.put(None)
    p.wait()


def detect(files, out_dir, rotation, weights, stride=5, conf=0.10, max_frames=0):
    """Detect people in every file (skipping files already cached) and write meta.json."""
    from ultralytics import YOLO
    os.makedirs(out_dir, exist_ok=True)
    args = argparse.Namespace(out_dir=out_dir, stride=stride, conf=conf, max_frames=max_frames, model=weights)
    model = None
    base_ms = 0.0
    meta = {"files": [], "rotation": rotation, "band": ROT_BAND, "stride": args.stride,
            "model": os.path.basename(args.model) if isinstance(args.model, str) else "yolo11x.pt", "conf": args.conf}
    for path in files:
        w, h, fps, nframes = probe(path)
        frame_ms = 1000.0 / fps
        stem = os.path.splitext(os.path.basename(path))[0]
        out_npz = os.path.join(args.out_dir, stem + ".npz")
        meta["files"].append({"path": path, "w": w, "h": h, "fps": fps,
                              "frames": nframes, "base_ms": base_ms})
        if os.path.exists(out_npz):
            print(f"skip {stem} (cached)")
            base_ms += nframes * frame_ms
            continue

        if model is None:
            model = YOLO(args.model() if callable(args.model) else args.model)
        M, rot_w = band_matrix(w, h, rotation, ROT_BAND[0])
        band_h = ROT_BAND[1] - ROT_BAND[0]
        imgsz = int(np.ceil(rot_w / 32) * 32)
        proxy_size = (rot_w // 2 // 2 * 2, band_h // 2 // 2 * 2)
        proxy = subprocess.Popen([
            "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{proxy_size[0]}x{proxy_size[1]}", "-r", str(fps / args.stride), "-i", "-",
            "-c:v", "h264_nvenc", "-preset", "p5", "-cq", "24",
            os.path.join(args.out_dir, stem + "_band.mp4")], stdin=subprocess.PIPE)

        q = queue.Queue(maxsize=8)
        th = threading.Thread(target=reader, args=(path, w, h, args.stride, q), daemon=True)
        th.start()

        times, boxes, confs, counts = [], [], [], []
        t0 = time.time()
        k = 0
        while True:
            item = q.get()
            if item is None:
                break
            n, frame = item
            band = cv2.warpAffine(frame, M, (rot_w, band_h))
            res = model.predict(band, imgsz=imgsz, classes=[0], conf=args.conf,
                                verbose=False, half=True)[0]
            b = res.boxes.xyxy.cpu().numpy().astype(np.float32)
            b[:, [1, 3]] += ROT_BAND[0]  # back to base-rotation frame rows
            c = res.boxes.conf.cpu().numpy().astype(np.float32)
            times.append(base_ms + n * frame_ms)
            boxes.append(b)
            confs.append(c)
            counts.append(len(b))
            proxy.stdin.write(cv2.resize(band, proxy_size, interpolation=cv2.INTER_AREA).tobytes())
            k += 1
            if k % 250 == 0:
                el = time.time() - t0
                print(f"{stem} {k} samples ({n / fps:.0f}s of video) {k / el:.1f} samples/s", flush=True)
            if args.max_frames and k >= args.max_frames:
                break
        proxy.stdin.close()
        proxy.wait()
        if k == 0:
            raise SystemExit(f"could not decode any frames from {path}; nothing cached")

        np.savez_compressed(
            out_npz, times_ms=np.array(times), counts=np.array(counts, dtype=np.int32),
            boxes=np.concatenate(boxes) if boxes else np.zeros((0, 4), np.float32),
            confs=np.concatenate(confs) if confs else np.zeros(0, np.float32))
        print(f"{stem}: {k} samples in {time.time() - t0:.0f}s", flush=True)
        base_ms += nframes * frame_ms
        if args.max_frames:
            break

    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--model", default="models/yolo11x.pt")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--rotation", type=float, default=None, help="default: from the calibration")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()
    from .calibration import load_calibration, video_files
    rotation = args.rotation if args.rotation is not None else load_calibration(args.video_dir)[0]["rotation"]
    detect(video_files(args.video_dir), args.out_dir, rotation, args.model, args.stride, args.conf, args.max_frames)


if __name__ == "__main__":
    main()
