"""Run a trained model over a video (2 frames per second by default) and write an annotated video.

    python src/demo_video.py --weights runs/A/weights/best.pt --video data/raw/own_video/Tunisian_Road.mp4
    python src/demo_video.py ... --rotate 20     # fake a tilted phone: every frame turned 20 degrees
    python src/demo_video.py ... --fps 6 --show pothole crack --class-conf pothole=0.4 --roi 0.35 0.8 --persist 3
        # windshield filters: only these classes, per-class confidence, ignore sky/hood, drop 1-frame flicker
"""
import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--video", required=True)
ap.add_argument("--fps", type=float, default=2.0, help="frames analysed per second of video")
ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--device", default="cpu")
ap.add_argument("--frames", action="store_true", help="also save a JPG of every frame with a detection")
ap.add_argument("--rotate", type=float, default=0, help="degrees, turns each frame before predicting (tilted-phone test)")
ap.add_argument("--show", nargs="*", default=[], help="draw only these class names, e.g. pothole crack (the rest are still predicted)")
ap.add_argument("--class-conf", nargs="*", default=[], help="per-class minimum confidence on top of --conf, e.g. pothole=0.4 crack=0.5")
ap.add_argument("--roi", nargs=2, type=float, default=(0, 1), metavar=("TOP", "BOTTOM"),
                help="keep shapes whose centre is between these fractions of the height: cuts sky/buildings and the hood/dashboard")
ap.add_argument("--persist", type=int, default=0, help="draw a shape only once it was seen in this many frames in a row (use --fps 6+)")
a = ap.parse_args()
cconf = {k: float(v) for k, v in (x.split("=") for x in a.class_conf)}
filtered = a.show or cconf or tuple(a.roi) != (0, 1) or a.persist

model = YOLO(a.weights)
cap = cv2.VideoCapture(a.video)
step = max(1, round(cap.get(cv2.CAP_PROP_FPS) / a.fps))
w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
size = (round(w * 1280 / max(w, h)), round(h * 1280 / max(w, h)))   # longest side 1280, shape kept (portrait/square phones)
out_dir = Path(a.weights).parents[1] / "demo"; out_dir.mkdir(exist_ok=True)
name = Path(a.video).stem + (f"_rot{a.rotate:g}" if a.rotate else "") + ("_filtered" if filtered else "")
turn = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), a.rotate, 1)   # same size: corners go black, like a crop
writer = cv2.VideoWriter(str(out_dir / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, size)
per_frame, classes, prev, i = [], Counter(), [], 0
while True:
    ok = cap.grab()
    if not ok:
        break
    if i % step == 0:
        _, frame = cap.retrieve()
        frame = cv2.resize(frame, size)
        if a.rotate:
            frame = cv2.warpAffine(frame, turn, size)
        r = model.predict(frame, imgsz=a.imgsz, conf=a.conf, device=a.device, verbose=False)[0]
        if filtered:
            b, keep, cur = r.boxes, [], []
            for k in range(len(b)):
                c, (x, y, bw, bh) = model.names[int(b.cls[k])], b.xywhn[k].tolist()
                # in a row: the previous frame had the same class within one box-size (Ultralytics trackers lose
                # these shapes: tracktrack needs conf >= 0.7 to start a track, and from a moving car they jump too far)
                run = 1 + max((n for pc, px, py, n in prev if pc == c and abs(px - x) < bw and abs(py - y) < bh), default=0)
                cur.append((c, x, y, run))
                keep.append((not a.show or c in a.show) and float(b.conf[k]) >= cconf.get(c, 0) and a.roi[0] <= y <= a.roi[1]
                            and run >= a.persist)
            prev = cur
            r = r[np.array(keep, bool)]
        per_frame.append(len(r.boxes)); classes.update(model.names[int(c)] for c in r.boxes.cls)
        im = r.plot()
        for y in a.roi if tuple(a.roi) != (0, 1) else ():   # the ROI limits, to calibrate them by eye
            cv2.line(im, (0, round(y * size[1])), (size[0], round(y * size[1])), (0, 255, 255), 2)
        writer.write(im)
        if a.frames and len(r.boxes):
            cv2.imwrite(str(out_dir / f"{name}_f{i:05d}.jpg"), im)
    i += 1
writer.release()
n = len(per_frame)
print(f"{n} frames analysed | frames with >=1 detection: {sum(p > 0 for p in per_frame)} ({sum(p > 0 for p in per_frame) / n:.0%}) | detections: {dict(classes)}")
print(f"annotated video: {out_dir / (name + '.mp4')}")
