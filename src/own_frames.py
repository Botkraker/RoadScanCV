"""Extract frames from your own video and pre-fill boxes with a trained model, ready for Label Studio.

Split by TIME (never random): the last TEST_FRAC of the clip is the test segment, separated
from the training part by a GAP_S gap, so near-identical neighbouring frames cannot leak.

    python src/own_frames.py --video data/raw/own_video/Tunisian_Road.mp4 --weights runs/A/weights/best.pt
Writes data/own/{images/, frames.csv, ls_import.json}
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
NAMES = {0: "pothole", 1: "crack", 2: "manhole"}
TEST_FRAC, GAP_S = 0.25, 5.0

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--weights", required=True)
ap.add_argument("--every", type=float, default=1.25, help="seconds between extracted frames")
ap.add_argument("--conf", type=float, default=0.15, help="low on purpose: deleting a wrong box is quicker than drawing a missing one")
ap.add_argument("--device", default="0")
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--out", default="", help="subfolder of data/own for another video (keeps the first set intact)")
ap.add_argument("--start", type=float, default=0.0, help="seconds to skip at the beginning")
ap.add_argument("--rotate180", action="store_true", help="for a phone mounted upside down")
a = ap.parse_args()

own_root = ROOT / "data" / "own"; out = own_root / a.out; (out / "images").mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(a.video)
fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
dur, step = n / fps, max(1, round(fps * a.every))
test_start = dur * (1 - TEST_FRAC)
model, rows, tasks, i = YOLO(a.weights), [], [], 0
while cap.grab():
    if i >= a.start * fps and (i - round(a.start * fps)) % step == 0:
        _, frame = cap.retrieve()
        if a.rotate180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        t = i / fps
        seg = "test" if t >= test_start else "gap" if t >= test_start - GAP_S else "train"
        name = f"f{i:05d}.jpg"
        img = cv2.resize(frame, (1920, 1080))
        cv2.imwrite(str(out / "images" / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        r = model.predict(img, imgsz=a.imgsz, conf=a.conf, device=a.device, verbose=False)[0]
        res = [{"from_name": "label", "to_name": "image", "type": "rectanglelabels", "original_width": 1920,
                "original_height": 1080, "image_rotation": 0, "score": float(c),
                "value": {"x": x0 / 19.2, "y": y0 / 10.8, "width": (x1 - x0) / 19.2, "height": (y1 - y0) / 10.8,
                          "rotation": 0, "rectanglelabels": [NAMES[int(k)]]}}
               for (x0, y0, x1, y1), c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf, r.boxes.cls)]
        tasks.append({"data": {"image": f"/data/local-files/?d={(out / 'images' / name).relative_to(own_root).as_posix()}"}, "meta": {"segment": seg, "t": round(t, 1)},
                      "predictions": [{"model_version": "runA", "result": res}]})
        rows.append({"file": name, "frame": i, "time_s": round(t, 1), "segment": seg, "predicted_boxes": len(res)})
    i += 1
(out / "ls_import.json").write_text(json.dumps(tasks))
with open(out / "frames.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
seg = {s: sum(r["segment"] == s for r in rows) for s in ("train", "gap", "test")}
print(f"{len(rows)} frames from {dur:.0f}s of video | {seg} | predicted boxes: {sum(r['predicted_boxes'] for r in rows)}")
