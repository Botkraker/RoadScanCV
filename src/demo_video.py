"""Run a trained model over a video (2 frames per second by default) and write an annotated video.

    python src/demo_video.py --weights runs/A/weights/best.pt --video data/raw/own_video/Tunisian_Road.mp4
"""
import argparse
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--video", required=True)
ap.add_argument("--fps", type=float, default=2.0, help="frames analysed per second of video")
ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--device", default="cpu")
a = ap.parse_args()

model = YOLO(a.weights)
cap = cv2.VideoCapture(a.video)
step = max(1, round(cap.get(cv2.CAP_PROP_FPS) / a.fps))
out_dir = Path(a.weights).parents[1] / "demo"; out_dir.mkdir(exist_ok=True)
writer = cv2.VideoWriter(str(out_dir / f"{Path(a.video).stem}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (1280, 720))
per_frame, classes, i = [], Counter(), 0
while True:
    ok = cap.grab()
    if not ok:
        break
    if i % step == 0:
        _, frame = cap.retrieve()
        r = model.predict(cv2.resize(frame, (1280, 720)), imgsz=a.imgsz, conf=a.conf, device=a.device, verbose=False)[0]
        per_frame.append(len(r.boxes)); classes.update(model.names[int(c)] for c in r.boxes.cls)
        writer.write(r.plot())
        if r.boxes and len(r.boxes) > 0:
            cv2.imwrite(str(out_dir / f"{Path(a.video).stem}_f{i:05d}.jpg"), r.plot())
    i += 1
writer.release()
n = len(per_frame)
print(f"{n} frames analysed | frames with >=1 detection: {sum(p > 0 for p in per_frame)} ({sum(p > 0 for p in per_frame) / n:.0%}) | detections: {dict(classes)}")
print(f"annotated video: {out_dir / (Path(a.video).stem + '.mp4')}")
