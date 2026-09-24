"""Tunisian-only dataset, 5 classes, MASKS (polygons), built in rounds:
annotate in Label Studio -> train -> that model pre-draws the next frames -> correct them -> train again.

    python src/tn.py seed                                  # once: the frames already labeled in data/own
    python src/tn.py frames --video data/own_video/20260920_115321.mp4 --weights runs/tn1/weights/best.pt
    python src/tn.py sync --json data/project-....json     # Label Studio JSON export -> labels + train/val
    python src/tn.py train --run tn1
    python src/tn.py serve --weights runs/tn1/weights/best.pt   # live: Label Studio -> Settings -> Model -> http://localhost:9090

Everything lives in data/tn/: images/, labels/, train.txt, val.txt, import_*.json (drop into Label Studio).
Labels are YOLO segmentation lines: class x1 y1 x2 y2 ... (normalised polygon).
Pre-drawing: SAM turns every box into an outline; YOLOE finds trash from text until our own model knows it.
Frames are named <video>_f<frame index>, so the same frame of the same video always gets the same name.
"""
import argparse
import functools
import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
TN = ROOT / "data" / "tn"
NAMES = ["pothole", "crack", "manhole", "trash", "other"]   # 0-2 as before, so old labels carry over
VAL_FRAC = 0.2
vid = lambda video: re.sub(r"[^A-Za-z0-9]", "", Path(video).stem)
SAM_W, YOLOE_W = "sam2.1_t.pt", "yoloe-11s-seg.pt"
TRASH_PROMPTS = ["trash", "garbage bag", "plastic bag", "black plastic bag", "plastic bottle", "litter", "rubbish"]


@functools.cache
def sam():
    from ultralytics import SAM
    return SAM(SAM_W)


@functools.cache
def yoloe():
    from ultralytics import YOLOE
    m = YOLOE(YOLOE_W); m.set_classes(TRASH_PROMPTS)
    return m


def outlines(masks):
    """Largest blob of each mask, in pixels (Ultralytics' default glues all blobs into one odd polygon)."""
    from ultralytics.utils import ops
    return [ops.scale_coords(masks.data.shape[1:], x, masks.orig_shape) for x in ops.masks2segments(masks.data, "largest")]


def predraw(frame, a, shapes=(), boxes=(), trash=True):
    """shapes: [(name, pixel contour)] kept; boxes: [(name, [x0, y0, x1, y1])] outlined by SAM; trash: YOLOE searches it.
    Returns [(name, normalised polygon)], simplified to a few points so they are quick to drag."""
    shapes = list(shapes)
    if boxes:
        r = sam()(frame, bboxes=[b for _, b in boxes], device=a.device, verbose=False)[0]
        shapes += [(n, c) for (n, _), c in zip(boxes, outlines(r.masks))]
    if trash:
        r = yoloe().predict(frame, imgsz=1280, conf=a.trash_conf, agnostic_nms=True, device=a.device, verbose=False)[0]
        shapes += [("trash", c) for c in (outlines(r.masks) if r.masks is not None else [])]
    h, w = frame.shape[:2]
    polys = [(n, cv2.approxPolyDP(c.astype(np.float32), a.eps, True)[:, 0] / [w, h]) for n, c in shapes if len(c) >= 3]
    return [(n, p.flatten().tolist()) for n, p in polys if len(p) >= 3]


def ls_result(polys):
    """polys: (class name, [x1, y1, x2, y2, ...]) normalised -> Label Studio shapes (points in % of the image)."""
    return [{"id": f"s{i}", "from_name": "label", "to_name": "image", "type": "polygonlabels", "original_width": 1920, "original_height": 1080,
            "value": {"points": [[x * 100, y * 100] for x, y in zip(p[::2], p[1::2])], "closed": True, "polygonlabels": [c]}}
           for i, (c, p) in enumerate(polys)]


def task(name, polys, version):
    """Pre-drawn: you drag the points onto the real outline."""
    return {"data": {"image": f"/data/local-files/?d=images/{name}"}, "predictions": [{"model_version": "predraw", "result": ls_result(polys)}]}   # one name: Label Studio pre-fills from ONE version


def seed(a):
    """Old labels are boxes without trash/other: SAM outlines the boxes, YOLOE adds trash, you check + complete."""
    tasks = []
    for sub, video in (("", "Tunisian_Road"), ("street2", "20260920_114106"), ("street3", "20260920_112134")):
        src = ROOT / "data" / "own" / sub
        for lab in sorted((src / "labels").glob("f*.txt")):
            name = f"{vid(video)}_{lab.stem}"
            shutil.copy(src / "images" / f"{lab.stem}.jpg", TN / "images" / f"{name}.jpg")
            frame = cv2.imread(str(TN / "images" / f"{name}.jpg")); h, w = frame.shape[:2]
            boxes = [(NAMES[int(c)], [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
                     for c, cx, cy, bw, bh in ((l[0], *map(float, l[1:])) for l in map(str.split, lab.read_text().splitlines()))]
            tasks.append(task(f"{name}.jpg", predraw(frame, a, boxes=boxes), "old boxes + SAM + YOLOE"))
    (TN / "import_seed.json").write_text(json.dumps(tasks))
    print(f"{len(tasks)} old frames -> data/tn/import_seed.json (they join train.txt once you export them again)")


def frames(a):
    from ultralytics import YOLO
    model = YOLO(a.weights) if a.weights else None
    cap = cv2.VideoCapture(a.video); fps = cap.get(cv2.CAP_PROP_FPS)
    step, first, tasks, i = max(1, round(fps * a.every)), round(a.start * fps), [], 0
    while cap.grab():
        name = f"{vid(a.video)}_f{i:05d}"
        if i >= first and (i - first) % step == 0 and not (TN / "labels" / f"{name}.txt").exists():   # skip done frames
            frame = cv2.resize(cap.retrieve()[1], (1920, 1080))
            if a.rotate180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            cv2.imwrite(str(TN / "images" / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            shapes, boxes = [], []
            if model:
                r = model.predict(frame, imgsz=640, conf=a.conf, device=a.device, verbose=False)[0]
                named = [model.names[int(c)] for c in r.boxes.cls]
                if r.masks is not None:   # a mask model (tn1 on) draws its own outlines
                    shapes = list(zip(named, outlines(r.masks)))
                else:                     # a box model: SAM outlines its boxes
                    boxes = list(zip(named, r.boxes.xyxy.tolist()))
            trash = not model or "trash" not in model.names.values()
            tasks.append(task(f"{name}.jpg", predraw(frame, a, shapes, boxes, trash), Path(a.weights).parents[1].name if model else "SAM"))
        i += 1
    out = TN / f"import_{vid(a.video)}.json"; out.write_text(json.dumps(tasks))
    print(f"{len(tasks)} frames, {sum(len(t['predictions'][0]['result']) for t in tasks)} pre-drawn shapes -> {out.name}")


def serve(a):
    """Label Studio ML backend (Settings -> Model -> Connect Model -> http://localhost:9090): the model draws on each frame you open."""
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from ultralytics import YOLO
    model, version = YOLO(a.weights), Path(a.weights).parents[1].name

    def clicked(frame, smart):
        """Your clicks (keypoints) or box from Label Studio's smart tools -> SAM outlines that one object."""
        h, w = frame.shape[0] / 100, frame.shape[1] / 100   # Label Studio sends % of the image
        pts = [[r["value"]["x"] * w, r["value"]["y"] * h] for r in smart if r["type"] == "keypointlabels"]
        box = [[v["x"] * w, v["y"] * h, (v["x"] + v["width"]) * w, (v["y"] + v["height"]) * h]
               for v in (r["value"] for r in smart if r["type"] == "rectanglelabels")][-1:]
        out = sam()(frame, bboxes=box, device=a.device, verbose=False) if box else             sam()(frame, points=[pts], labels=[[1] * len(pts)], device=a.device, verbose=False)   # all clicks = one object
        label = next(iter(smart[-1]["value"].get("keypointlabels") or smart[-1]["value"].get("rectanglelabels")))
        return {"model_version": "SAM", "score": 1.0, "result": ls_result(predraw(frame, a, [(label, c) for c in outlines(out[0].masks)][:1], trash=False))}

    def predict(t, context):
        frame = cv2.imread(str(TN / "images" / Path(t["data"]["image"].split("?d=")[-1]).name))
        if (context or {}).get("result"):
            return clicked(frame, context["result"])
        r = model.predict(frame, imgsz=640, conf=a.conf, device=a.device, verbose=False)[0]
        shapes = list(zip([model.names[int(c)] for c in r.boxes.cls], outlines(r.masks))) if r.masks is not None else []
        return {"model_version": version, "score": float(r.boxes.conf.mean()) if len(r.boxes) else 0.0,
                "result": ls_result(predraw(frame, a, shapes, trash="trash" not in model.names.values()))}

    class Handler(BaseHTTPRequestHandler):   # single-threaded on purpose: one GPU, one request at a time
        def reply(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self):    # /health
            self.reply({"status": "UP", "model_class": version})

        def do_POST(self):   # /predict; /setup and /webhook just get the version
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path.startswith("/predict"):
                ctx = req.get("params", {}).get("context")
                self.reply({"results": [predict(t, ctx) for t in req.get("tasks", [])], "model_version": version})
            else:
                self.reply({"model_version": version, "status": "ok"})

    print(f"{version} serving on http://localhost:{a.port} (Ctrl+C to stop)")
    HTTPServer(("localhost", a.port), Handler).serve_forever()


def sync(json_path):
    """Label Studio JSON export (points in % of the image) -> YOLO seg labels. Only submitted frames count."""
    done = {}
    for t in json.loads(Path(json_path).read_text(encoding="utf-8")):
        ann = [x for x in t.get("annotations", []) if not x.get("was_cancelled")]
        if not ann:
            continue
        lines = [f"{NAMES.index(r['value']['polygonlabels'][0])} " + " ".join(f"{v / 100:.5f}" for pt in r["value"]["points"] for v in pt)
                 for r in ann[-1]["result"] if r["type"] == "polygonlabels"]
        done[Path(t["data"]["image"].split("?d=")[-1]).stem] = lines   # a duplicate task of the same frame: last one wins
    for s, lines in done.items():
        (TN / "labels" / f"{s}.txt").write_text("".join(f"{l}\n" for l in lines))
    print(f"{len(done)} labeled frames from {Path(json_path).name}")
    split()


def split():
    """Per video, the LAST 20% of its labeled frames is val: neighbouring frames are near-copies, never split randomly."""
    by_video, counts = {}, [0] * len(NAMES)
    for lab in sorted((TN / "labels").glob("*.txt")):   # ponytail: f%05d sorts right up to 99999 frames (27 min at 60 fps)
        by_video.setdefault(lab.stem.rsplit("_f", 1)[0], []).append(f"./images/{lab.stem}.jpg")
        for l in lab.read_text().splitlines():
            counts[int(l.split()[0])] += 1
    tr, va = [], []
    for fs in by_video.values():
        k = round(len(fs) * (1 - VAL_FRAC)); tr += fs[:k]; va += fs[k:]
    (TN / "train.txt").write_text("\n".join(tr) + "\n"); (TN / "val.txt").write_text("\n".join(va) + "\n")
    print(f"train {len(tr)}, val {len(va)} frames from {len(by_video)} videos | masks {dict(zip(NAMES, counts))}")


def train(a):
    from ultralytics import YOLO
    if a.resume:   # Windows sometimes runs out of RAM mid-run: carry on from the last epoch
        return YOLO(str(ROOT / "runs" / a.run / "weights" / "last.pt")).train(resume=True)
    data = TN / "data.yaml"
    data.write_text(yaml.safe_dump({"path": str(TN), "train": "train.txt", "val": "val.txt", "names": dict(enumerate(NAMES))}))
    YOLO(a.weights).train(data=str(data), epochs=a.epochs, imgsz=640, batch=a.batch, device=a.device, workers=1,
                          project=str(ROOT / "runs"), name=a.run, exist_ok=True, plots=False, patience=a.patience,
                          cache="ram")   # ~250 MB of small copies: the CPU stops re-decoding 1080p JPEGs every step


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    draw = argparse.ArgumentParser(add_help=False)
    draw.add_argument("--trash-conf", type=float, default=0.05, help="YOLOE confidence for trash: its scores run very low (a clear bag scored 0.085)")
    draw.add_argument("--eps", type=float, default=8.0, help="outline simplification in pixels: higher = fewer points")
    draw.add_argument("--device", default="0")
    sub.add_parser("seed", parents=[draw])
    p = sub.add_parser("frames", parents=[draw])
    p.add_argument("--video", required=True)
    p.add_argument("--weights", default="", help="model that pre-draws shapes (mask or box model); empty = blank frames")
    p.add_argument("--every", type=float, default=1.25, help="seconds between frames")
    p.add_argument("--start", type=float, default=0.0, help="seconds to skip at the beginning")
    p.add_argument("--rotate180", action="store_true", help="phone mounted upside down")
    p.add_argument("--conf", type=float, default=0.15, help="low on purpose: deleting a shape is quicker than drawing one")
    p = sub.add_parser("serve", parents=[draw])
    p.add_argument("--weights", required=True)
    p.add_argument("--conf", type=float, default=0.15, help="low on purpose: deleting a shape is quicker than drawing one")
    p.add_argument("--port", type=int, default=9090)
    p = sub.add_parser("sync"); p.add_argument("--json", required=True)
    p = sub.add_parser("train")
    p.add_argument("--run", required=True)
    p.add_argument("--weights", default="yolo11s-seg.pt", help="COCO-pretrained segmentation model")
    p.add_argument("--epochs", type=int, default=60, help="tn2 peaked at epoch 49 of 100, then only memorised")
    p.add_argument("--patience", type=int, default=20, help="stop after this many epochs without a better val score")
    p.add_argument("--resume", action="store_true", help="continue a crashed run from its last epoch")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="0")
    a = ap.parse_args()
    {"seed": lambda: seed(a), "frames": lambda: frames(a), "serve": lambda: serve(a), "sync": lambda: sync(a.json), "train": lambda: train(a)}[a.cmd]()
