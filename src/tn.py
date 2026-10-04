"""Tunisian-only dataset, 5 classes, MASKS (polygons), built in rounds:
annotate in Label Studio -> train -> that model pre-draws the next frames -> correct them -> train again.

    python src/tn.py seed                                  # once: the frames already labeled in data/own
    python src/tn.py frames --video data/own_video/20260920_115321.mp4 --weights runs/tn1/weights/best.pt
    python src/tn.py sync --json data/project-....json     # Label Studio JSON export -> labels + train/val
    python src/tn.py train --run tn1
    python src/tn.py mine --weights runs/l640_freeze/weights/best.pt --videos data/raw/tn/<clip>.mp4 ...   # hard negatives
    python src/tn.py predraw --weights runs/l640_freeze/weights/best.pt   # review frames with real damage -> import_review.json
    python src/tn.py pack --extra 20260920122358           # zip for Colab -> colab/tn_data.zip (+ unlabeled frames of a video)
    python src/tn.py serve --weights runs/tn1/weights/best.pt   # live: Label Studio -> Settings -> Model -> http://localhost:9090

Progress measure, on clips that are never trained on (HOLDOUT, TEST_CLIPS):
    python src/tn.py testset --video data/raw/tn/20260926_081645.mp4 --weights runs/a/weights/best.pt runs/b/weights/best.pt
        # -> data/tn_test/<clip>/: test frames, candidates.json, overlays/ for the AI judge (Claude subagents -> ai/*.json)
    python src/tn.py testreview      # AI verdicts -> review.html: flip what it got wrong, download gt.json into the folder
    python src/tn.py score --weights runs/<run>/weights/best.pt   # recall, hallucinations, false alarms -> runs/scoreboard.csv

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
BLOCK = 8   # val = every 5th block of 8 consecutive labeled frames (~10 s of video)
HOLDOUT = {"20260926081645", "VID20260925173443"}   # the test clips (tn.py testset/score): never trained on, never a negative
CRACKS = {"crack", "lt_crack", "block_crack"}   # "any crack": uzbtrust-based models also have linear and block crack slots
TEST = ROOT / "data" / "tn_test"
# tn.py score: clip id -> (video, --roi band of the windshield filter, "gt" = recall on gt.json | {classes} = every shape of
# these classes is a false alarm). The autoroute clip has real cracks/joints but no potholes, so only potholes count there.
# --roi bands set by eye: 081645's mount moves (hood top 0.56, then 0.70 from ~f02800); autoroute's hood edge is at 0.86-0.90.
TEST_CLIPS = {"20260926081645": ("data/raw/tn/20260926_081645.mp4", (0.25, 0.68), "gt"),
              "VID20260925173443": ("data/raw/tn/VID_20260925_173443.mp4", (0.30, 0.85), {"pothole"})}
vid = lambda video: re.sub(r"[^A-Za-z0-9]", "", Path(video).stem)
fit = lambda im: cv2.resize(im, None, fx=1920 / max(im.shape[:2]), fy=1920 / max(im.shape[:2]))   # longest side 1920, shape kept (portrait phones)
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
    return merge([(n, p.flatten().tolist()) for n, p in polys if len(p) >= 3])


def merge(polys, thr=0.3, W=960, H=540):
    """Same-class shapes overlapping more than thr of the smaller one are ONE object drawn twice: replace them by their union.
    Shapes that only touch stay apart (neighbouring potholes on a rail crossing are separate objects). Chains merge too."""
    masks = []
    for _, p in polys:
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [(np.array(p).reshape(-1, 2) * [W, H]).astype(np.int32)], 1); masks.append(m)
    root = list(range(len(polys)))
    find = lambda i: i if root[i] == i else find(root[i])
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polys[i][0] == polys[j][0] and (masks[i] & masks[j]).sum() > thr * min(masks[i].sum(), masks[j].sum()) > 0:
                root[find(j)] = find(i)
    out = []
    for g in sorted({find(i) for i in range(len(polys))}):
        members = [i for i in range(len(polys)) if find(i) == g]
        if len(members) == 1:
            out.append(polys[g]); continue
        union = np.bitwise_or.reduce([masks[i] for i in members])
        c = max(cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
        out.append((polys[g][0], (cv2.approxPolyDP(c, 1.0, True)[:, 0] / [W, H]).flatten().tolist()))
    return merge(out, thr, W, H) if len(out) < len(polys) else out   # a union can grow into a new overlap


def ls_result(polys, shape=(1080, 1920)):
    """polys: (class name, [x1, y1, x2, y2, ...]) normalised -> Label Studio shapes (points in % of the image)."""
    return [{"id": f"s{i}", "from_name": "label", "to_name": "image", "type": "polygonlabels", "original_width": shape[1], "original_height": shape[0],
            "value": {"points": [[x * 100, y * 100] for x, y in zip(p[::2], p[1::2])], "closed": True, "polygonlabels": [c]}}
           for i, (c, p) in enumerate(polys)]


def task(name, polys, version, shape=(1080, 1920)):
    """Pre-drawn: you drag the points onto the real outline."""
    return {"data": {"image": f"/data/local-files/?d=images/{name}"}, "predictions": [{"model_version": "predraw", "result": ls_result(polys, shape)}]}   # one name: Label Studio pre-fills from ONE version


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
            frame = fit(cap.retrieve()[1])
            if a.rotate180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            cv2.imwrite(str(TN / "images" / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            shapes, boxes = [], []
            if model:
                r = model.predict(frame, imgsz=640, conf=a.conf, iou=a.iou, device=a.device, verbose=False)[0]
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


def mine(a):
    """Hard-negative candidates: frames of unlabeled videos where the model still fires -> data/tn/review/index.html.
    In that page, click every frame that shows a REAL pothole/crack/manhole/trash; the rest download as negatives.txt."""
    from ultralytics import YOLO
    model, rev, fired = YOLO(a.weights), TN / "review", []
    rev.mkdir(exist_ok=True)
    for v in a.videos:
        cap = cv2.VideoCapture(v); step, i = max(1, round(cap.get(cv2.CAP_PROP_FPS) * a.every)), 0
        while cap.grab():
            name = f"{vid(v)}_f{i:05d}"
            if i % step == 0 and not (TN / "labels" / f"{name}.txt").exists():   # a labeled frame is never a negative
                frame = fit(cap.retrieve()[1])
                r = model.predict(frame, imgsz=640, conf=a.conf, device=a.device, verbose=False)[0]
                if len(r.boxes):   # only frames where it fires teach anything; the same name as frames() gives, so no copies
                    cv2.imwrite(str(TN / "images" / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    im = r.plot(); s = 480 / max(im.shape[:2])
                    cv2.imwrite(str(rev / f"{name}.jpg"), cv2.resize(im, None, fx=s, fy=s), [cv2.IMWRITE_JPEG_QUALITY, 80])
                    fired.append(name)
            i += 1
        print(f"{Path(v).name}: {len(fired)} frames with predictions so far")
    (rev / "index.html").write_text(REVIEW.replace("FRAMES", json.dumps(fired)), encoding="utf-8")
    print(f"open {rev / 'index.html'}")


REVIEW = """<!doctype html><meta charset="utf-8"><title>Hard negatives</title>
<style>body{font:14px sans-serif;margin:16px;background:#111;color:#eee}#g{display:flex;flex-wrap:wrap;gap:6px}
figure{margin:0;border:4px solid #2a2;cursor:pointer}figure.real{border-color:#d22;opacity:.45}img{height:240px;display:block}
figcaption{font-size:11px;padding:2px}p{position:sticky;top:0;background:#111;padding:8px 0;margin:0}</style>
<p>Click every frame that shows a <b>real</b> pothole, crack, manhole or trash, even one the model missed (red = kept out).
All green frames become background. <button onclick="dl()">Download negatives.txt</button> <span id="n"></span></p>
<div id="g"></div><script>
const F = FRAMES, g = document.getElementById("g"), n = document.getElementById("n");
let real = new Set(); try { real = new Set(JSON.parse(localStorage.getItem("real") || "[]")); } catch (e) {}
const count = () => { n.textContent = `${F.length - real.size} negatives, ${real.size} kept out`;
  try { localStorage.setItem("real", JSON.stringify([...real])); } catch (e) {} };
for (const f of F) {
  const el = document.createElement("figure"); el.innerHTML = `<img loading="lazy" src="${f}.jpg"><figcaption>${f}</figcaption>`;
  el.classList.toggle("real", real.has(f));
  el.onclick = () => { real.has(f) ? real.delete(f) : real.add(f); el.classList.toggle("real"); count(); };
  g.append(el);
}
count();
function dl() {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([F.filter(f => !real.has(f)).map(f => f + ".jpg\\n").join("")], {type: "text/plain"}));
  a.download = "negatives.txt"; a.click();
}
</script>
"""


def predraw_review(a):
    """The review frames you kept out of negatives.txt (= real damage) -> Label Studio import with the model's shapes.
    A uzbtrust-based model also knows classes we don't label (lt_crack, patching...): those shapes are dropped."""
    from ultralytics import YOLO
    model, tasks = YOLO(a.weights), []
    neg = set(Path(a.negatives).read_text().split())
    for f in sorted((TN / "review").glob("*.jpg")):
        if f.name in neg or (TN / "labels" / f"{f.stem}.txt").exists():
            continue
        frame = cv2.imread(str(TN / "images" / f.name))
        r = model.predict(frame, imgsz=a.imgsz, conf=a.conf, device=a.device, verbose=False)[0]
        named = [model.names[int(c)] for c in r.boxes.cls]
        shapes = [(n, c) for n, c in zip(named, outlines(r.masks) if r.masks is not None else []) if n in NAMES]
        tasks.append(task(f.name, predraw(frame, a, shapes, trash=False), Path(a.weights).parents[1].name, frame.shape))
    out = TN / "import_review.json"; out.write_text(json.dumps(tasks))
    print(f"{len(tasks)} frames, {sum(len(t['predictions'][0]['result']) for t in tasks)} pre-drawn shapes -> {out}")


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
        r = model.predict(frame, imgsz=640, conf=a.conf, iou=a.iou, device=a.device, verbose=False)[0]
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
    done, shapes = {}, 0
    for t in json.loads(Path(json_path).read_text(encoding="utf-8")):
        ann = [x for x in t.get("annotations", []) if not x.get("was_cancelled")]
        if not ann:
            continue
        raw = [(r["value"]["polygonlabels"][0], [v / 100 for pt in r["value"]["points"] for v in pt])
               for r in ann[-1]["result"] if r["type"] == "polygonlabels"]
        polys = merge(raw); shapes += len(raw)
        lines = [f"{NAMES.index(n)} " + " ".join(f"{v:.5f}" for v in p) for n, p in polys]
        done[Path(t["data"]["image"].split("?d=")[-1]).stem] = lines   # a duplicate task of the same frame: last one wins
    for s, lines in done.items():
        (TN / "labels" / f"{s}.txt").write_text("".join(f"{l}\n" for l in lines))
    print(f"{len(done)} labeled frames from {Path(json_path).name} | {shapes} shapes -> {sum(map(len, done.values()))} after merging duplicates")
    split()


def split():
    """Per video, every 5th block of BLOCK consecutive labeled frames is val (20%). Blocks spread val over the whole
    video, so every class reaches it (a last-20% cut once left potholes with 0 val examples). Neighbouring frames are
    near-copies, so frames are never shuffled, and the train frame touching a val block is dropped as a gap."""
    by_video, counts = {}, [0] * len(NAMES)
    for lab in sorted((TN / "labels").glob("*.txt")):   # ponytail: f%05d sorts right up to 99999 frames (27 min at 60 fps)
        if lab.stem.rsplit("_f", 1)[0] in HOLDOUT:
            continue
        by_video.setdefault(lab.stem.rsplit("_f", 1)[0], []).append(f"./images/{lab.stem}.jpg")
        for l in lab.read_text().splitlines():
            counts[int(l.split()[0])] += 1
    tr, va = [], []
    for fs in by_video.values():
        val = lambda i: i // BLOCK % 5 == 4
        for i, f in enumerate(fs):
            if val(i):
                va.append(f)
            elif not ((i > 0 and val(i - 1)) or (i + 1 < len(fs) and val(i + 1))):
                tr.append(f)
    (TN / "train.txt").write_text("\n".join(tr) + "\n"); (TN / "val.txt").write_text("\n".join(va) + "\n")
    print(f"train {len(tr)}, val {len(va)} frames from {len(by_video)} videos | masks {dict(zip(NAMES, counts))}")


def pack(a):
    """Zip what training reads (labeled frames, labels, train/val lists) for Colab -> colab/tn_data.zip.
    --extra adds the unlabeled frames of those videos (hard-negative candidates, no label file) and demo_video.py."""
    import zipfile
    out = ROOT / "colab" / "tn_data.zip"; out.parent.mkdir(exist_ok=True)
    lists = {n: (TN / n).read_text().split() for n in ("train.txt", "val.txt")}
    held = set(a.extra) & HOLDOUT or {l for ls in lists.values() for l in ls if Path(l).stem.rsplit("_f", 1)[0] in HOLDOUT}
    assert not held, f"{sorted(held)[:3]}: the test clip, never train on it (drop it from --extra / run tn.py sync again)"
    with zipfile.ZipFile(out, "w") as z:   # JPEGs are already compressed: stored, not deflated
        for n, lines in lists.items():
            z.write(TN / n, n)
            for l in lines:
                z.write(TN / l, l); z.write(TN / "labels" / f"{Path(l).stem}.txt", f"labels/{Path(l).stem}.txt")
        extra = [f for v in a.extra for f in sorted((TN / "images").glob(f"{v}_f*.jpg")) if not (TN / "labels" / f"{f.stem}.txt").exists()]
        for f in extra:
            z.write(f, f"images/{f.name}")
        z.write(ROOT / "src" / "demo_video.py", "demo_video.py")
    print(f"{out} ({out.stat().st_size / 1e6:.0f} MB): train {len(lists['train.txt'])}, val {len(lists['val.txt'])}, unlabeled {len(extra)} frames")


dmg = lambda n: "pothole" if n == "pothole" else "crack" if n in CRACKS else None   # model class name -> what the test counts


def damage(model, frame, conf, imgsz, device):
    """[(pothole | crack, normalised flat polygon)] that a mask model draws on a frame."""
    r = model.predict(frame, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
    if r.masks is None:
        return []
    h, w = frame.shape[:2]
    named = [dmg(model.names[int(c)]) for c in r.boxes.cls]
    return [(n, (p / [w, h]).flatten().tolist()) for n, p in zip(named, outlines(r.masks)) if n and len(p) >= 3]


def grid(w, h):
    """Columns x rows of the lettered grid the AI judge uses to name missed damage (cells about square)."""
    return (6, round(6 * h / w)) if w <= h else (round(6 * w / h), 6)


def cells(rng, w, h):
    """'C7:D8' (or 'C7') -> that rectangle as a normalised flat polygon."""
    cols, rows = grid(w, h)
    (c0, r0), (c1, r1) = [(ord(s[0].upper()) - 65, int(s[1:]) - 1) for s in (rng.split(":") * 2)[:2]]
    x0, x1, y0, y1 = min(c0, c1) / cols, (max(c0, c1) + 1) / cols, min(r0, r1) / rows, (max(r0, r1) + 1) / rows
    return [x0, y0, x1, y0, x1, y1, x0, y1]


def overlay(frame, shapes):
    """For the AI judge. Left: the frame with a lettered grid. Right: the candidate shapes, numbered."""
    h, w = frame.shape[:2]; cols, rows = grid(w, h); left, right = frame.copy(), frame.copy()
    text = lambda im, s, xy, col, sc: [cv2.putText(im, s, xy, 0, sc, c, t) for c, t in (((0, 0, 0), 6), (col, 2))]
    for c in range(cols):
        for r in range(rows):
            cv2.rectangle(left, (c * w // cols, r * h // rows), ((c + 1) * w // cols, (r + 1) * h // rows), (255, 255, 255), 1)
            text(left, f"{chr(65 + c)}{r + 1}", (c * w // cols + 4, r * h // rows + 24), (0, 255, 255), 0.7)
    for k, (n, p) in enumerate(shapes):
        a = (np.array(p).reshape(-1, 2) * [w, h]).astype(np.int32)
        cv2.polylines(right, [a], True, (0, 0, 255) if n == "pothole" else (255, 160, 0), 3)
        x, y = a[a[:, 1].argmin()]
        text(right, str(k), (min(int(x), w - 50), max(30, int(y) - 8)), (255, 255, 255), 1.1)
    im = np.hstack([left, np.zeros((h, 12, 3), np.uint8), right]); s = 1400 / h
    return cv2.resize(im, None, fx=s, fy=s)


def testset(a):
    """Test frames of a held-out clip (same 1.25 s grid as frames/mine) + every pothole/crack shape that any source draws:
    our models and YOLOE prompted with text, so damage our models miss still gets a candidate. -> data/tn_test/<clip>/
    Next: Claude subagents judge overlays/*.jpg into ai/*.json, then tn.py testreview."""
    from ultralytics import YOLO, YOLOE
    v = vid(a.video); out = TEST / v
    assert v in HOLDOUT, f"{v} is not in HOLDOUT: a test clip must never be trained on"
    for d in ("images", "overlays", "ai"):
        (out / d).mkdir(parents=True, exist_ok=True)
    trained = {Path(l).name for f in (ROOT / "runs").glob("*/train_run.txt") for l in f.read_text().split()}
    models = [YOLO(w) for w in a.weights]
    ye = YOLOE(YOLOE_W); ye.set_classes(["pothole", "crack"])
    cap = cv2.VideoCapture(a.video); step, i, cands = max(1, round(cap.get(cv2.CAP_PROP_FPS) * a.every)), 0, {}
    while cap.grab():
        name = f"{v}_f{i:05d}"
        if i % step == 0 and f"{name}.jpg" not in trained:   # a frame some run trained on (r2: 44 negatives) is no test
            frame = fit(cap.retrieve()[1]); h, w = frame.shape[:2]
            cv2.imwrite(str(out / "images" / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            shapes = [s for m in models for s in damage(m, frame, a.conf, 640, a.device)] + damage(ye, frame, a.yoloe_conf, 1280, a.device)
            cands[name] = merge(shapes, W=w // 2, H=h // 2)
            cv2.imwrite(str(out / "overlays" / f"{name}.jpg"), overlay(frame, cands[name]), [cv2.IMWRITE_JPEG_QUALITY, 85])
        i += 1
    (out / "candidates.json").write_text(json.dumps(cands))
    print(f"{len(cands)} test frames, {sum(map(len, cands.values()))} candidate shapes -> {out}")


def testreview(a):
    """candidates.json + the AI judge's ai/*.json -> review.html, every shape pre-set to the AI's verdict."""
    out = TEST / a.clip
    cands = json.loads((out / "candidates.json").read_text())
    ai = {k: x for f in sorted((out / "ai").glob("*.json")) for k, x in json.loads(f.read_text()).items()}
    missing = sorted(set(cands) - set(ai))
    assert not missing, f"{len(missing)} frames have no AI verdict yet (e.g. {missing[0]})"
    frames = []
    for name, shapes in cands.items():
        v, (h, w) = ai[name], cv2.imread(str(out / "images" / f"{name}.jpg")).shape[:2]
        assert len(v["verdicts"]) == len(shapes) and set(v["verdicts"]) <= {"pothole", "crack", "no"} \
            and all(c in ("pothole", "crack") for c, _ in v["missed"]), f"bad AI verdicts for {name}"
        s = [{"cls": c, "poly": p} for (_, p), c in zip(shapes, v["verdicts"])] + [{"cls": c, "poly": cells(r, w, h), "ai": 1} for c, r in v["missed"]]
        area = lambda x: cv2.contourArea((np.array(x["poly"]).reshape(-1, 2) * [w, h]).astype(np.float32))
        frames.append({"name": name, "shapes": sorted(s, key=area, reverse=True)})   # small shapes on top, so they stay clickable
    (out / "review.html").write_text(GT_REVIEW.replace("__CLIP__", a.clip).replace("__FRAMES__", json.dumps(frames)), encoding="utf-8")
    print(f"open {out / 'review.html'}, then save gt.json into {out}")


GT_REVIEW = """<!doctype html><meta charset="utf-8"><title>Test set review</title>
<style>body{font:14px sans-serif;margin:0;background:#111;color:#eee}#bar{padding:8px 16px;display:flex;gap:16px;flex-wrap:wrap;align-items:center}
#wrap{position:relative;display:inline-block;margin:0 16px}#im{display:block;max-height:calc(100vh - 70px);max-width:calc(100vw - 32px)}
svg{position:absolute;inset:0;width:100%;height:100%}polygon{pointer-events:all;cursor:pointer;stroke-width:3;fill-opacity:.15;vector-effect:non-scaling-stroke}
.pothole{stroke:#f33;fill:#f33}.crack{stroke:#3cf;fill:#3cf}.no{stroke:#888;fill:none;stroke-width:1.5}.ai{stroke-dasharray:3 4}body.hide svg{display:none}</style>
<div id="bar"><b id="pos"></b><span>Click a shape: pothole (red) &rarr; crack (blue) &rarr; not damage (grey). &larr;/&rarr; frames, hold H to hide shapes.
Dotted = a box the AI added where no model drew.</span><span id="sum"></span><button onclick="dl()">Download gt.json</button></div>
<div id="wrap"><img id="im"><svg id="sv" viewBox="0 0 1 1" preserveAspectRatio="none"></svg></div><script>
const CLIP = "__CLIP__", F = __FRAMES__, NEXT = {pothole: "crack", crack: "no", no: "pothole"};
let st = {cls: F.map(f => f.shapes.map(s => s.cls)), seen: [], i: 0};
try { const s = JSON.parse(localStorage.getItem(CLIP)); if (s && s.cls.length === F.length) st = s; } catch (e) {}
const pts = p => p.reduce((a, v, k) => a + (k % 2 ? "," : " ") + v, "");
function show() {
  const f = F[st.i]; if (!st.seen.includes(st.i)) st.seen.push(st.i);
  im.src = `images/${f.name}.jpg`;
  sv.innerHTML = f.shapes.map((s, k) => `<polygon data-k="${k}" class="${st.cls[st.i][k]}${s.ai ? " ai" : ""}" points="${pts(s.poly)}"/>`).join("");
  const n = c => st.cls.flat().filter(x => x === c).length;
  pos.textContent = `${f.name}  ${st.i + 1}/${F.length}`;
  sum.textContent = `looked at ${st.seen.length}/${F.length} | pothole ${n("pothole")} | crack ${n("crack")}`;
  try { localStorage.setItem(CLIP, JSON.stringify(st)); } catch (e) {}
}
sv.onclick = e => { const k = e.target.dataset.k; if (k !== undefined) { st.cls[st.i][k] = NEXT[st.cls[st.i][k]]; show(); } };
document.onkeydown = e => {
  if (e.key === "ArrowRight" || e.key === " ") { st.i = Math.min(F.length - 1, st.i + 1); show(); e.preventDefault(); }
  else if (e.key === "ArrowLeft") { st.i = Math.max(0, st.i - 1); show(); }
  else if (e.key === "h") document.body.classList.add("hide");
};
document.onkeyup = e => { if (e.key === "h") document.body.classList.remove("hide"); };
function dl() {
  const left = F.length - st.seen.length;
  if (left && !confirm(`${left} frames not looked at yet. Download anyway?`)) return;
  const frames = Object.fromEntries(F.map((f, i) => [f.name, f.shapes.map((s, k) => ({cls: st.cls[i][k], poly: s.poly})).filter(s => s.cls !== "no")]));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify({clip: CLIP, frames})], {type: "application/json"}));
  a.download = "gt.json"; a.click();
}
show();
</script>
"""


def covers(p, q, W=450, H=960, thr=0.1):
    """Normalised polygons p and q overlap by at least thr of the smaller one: the model 'found' it."""
    mp, mq = (cv2.fillPoly(np.zeros((H, W), np.uint8), [(np.array(x).reshape(-1, 2) * [W, H]).astype(np.int32)], 1) for x in (p, q))
    return int((mp & mq).sum()) >= thr * max(1, min(int(mp.sum()), int(mq.sum())))


def score(a):
    """The progress measure on TEST_CLIPS. "gt" clip: share of the gt.json potholes/cracks a same-class model shape covers,
    plus hallucinations (model shapes touching no real damage). Other clips: every shape of the listed classes is a false alarm.
    raw = conf only; filtered = demo_video's windshield filters (pothole/crack, --roi band, --persist at 6 fps)."""
    from datetime import date
    from ultralytics import YOLO
    from demo_video import keep
    sq = lambda x0, y0, x1, y1: [x0, y0, x1, y0, x1, y1, x0, y1]
    assert covers(sq(.1, .1, .5, .5), sq(.45, .45, .6, .6)) and not covers(sq(.1, .1, .2, .2), sq(.5, .5, .6, .6))
    assert covers(sq(.1, .50, .9, .51), sq(.4, .49, .45, .52))   # a small model blob on a long thin crack finds it
    model, show = YOLO(a.weights), {"pothole"} | CRACKS
    row = {"date": date.today().isoformat(), "model": Path(a.weights).parents[1].name}
    res = {m: {} for m in ("raw", "filtered")}
    for clip, (video, roi, kind) in TEST_CLIPS.items():
        if kind == "gt" and not (TEST / clip / "gt.json").exists():
            print(f"{clip}: no gt.json yet (tn.py testset + testreview), skipped"); continue
        cap = cv2.VideoCapture(str(ROOT / video)); fps = cap.get(cv2.CAP_PROP_FPS); s6 = max(1, round(fps / 6))
        if kind == "gt":
            gt = json.loads((TEST / clip / "gt.json").read_text())["frames"]
            tests = {int(n.rsplit("_f", 1)[1]): g for n, g in gt.items()}
            need = {t - k * s6: t for t in tests for k in range(a.persist)}   # each test frame + the frames its persist looks back on
        prev, i, n6, tally = [], 0, 0, {m: {"pothole": [0, 0], "crack": [0, 0], "halluc": 0, "shapes": 0} for m in res}
        while cap.grab():
            if (i in need) if kind == "gt" else i % s6 == 0:
                frame = fit(cap.retrieve()[1]); h, w = frame.shape[:2]
                r = model.predict(frame, imgsz=640, conf=a.conf, device=a.device, verbose=False)[0]
                if kind == "gt" and i == need[i] - (a.persist - 1) * s6:
                    prev = []   # a fresh persist run for every test frame
                rf, prev = keep(r, model.names, prev, show, {}, roi, a.persist)
                n6 += 1
                if kind != "gt" or i in tests:
                    for m, rr in (("raw", r), ("filtered", rf)):
                        named = [dmg(model.names[int(c)]) for c in rr.boxes.cls]
                        preds = [(n, (p / [w, h]).flatten().tolist()) for n, p in zip(named, outlines(rr.masks) if rr.masks is not None else []) if n]
                        t = tally[m]; t["shapes"] += len(preds) if kind == "gt" else sum(n in kind for n, _ in preds)
                        for g in tests.get(i, []) if kind == "gt" else []:
                            t[g["cls"]][1] += 1
                            t[g["cls"]][0] += any(n == g["cls"] and covers(g["poly"], p) for n, p in preds)
                        if kind == "gt":
                            t["halluc"] += sum(not any(covers(p, g["poly"]) for g in tests[i]) for _, p in preds)
            i += 1
        for m, t in tally.items():
            if kind == "gt":
                res[m] |= {"pothole": f"{t['pothole'][0]}/{t['pothole'][1]}", "crack": f"{t['crack'][0]}/{t['crack'][1]}",
                           "halluc": f"{t['halluc']} on {len(tests)} frames"}
            else:
                res[m][f"{'+'.join(sorted(kind))}_fa_per_min_{clip}"] = round(t["shapes"] / (i / fps / 60), 1)
        print(f"{clip}: {n6} frames analysed")
    import csv
    out = ROOT / "runs" / "scoreboard.csv"; rows = [row | {"mode": m} | r for m, r in res.items()]
    old = list(csv.DictReader(out.open())) if out.exists() else []
    cols = list(dict.fromkeys(k for r in old + rows for k in r))
    with out.open("w", newline="") as f:
        wr = csv.DictWriter(f, cols); wr.writeheader(); wr.writerows(old + rows)
    for r in rows:
        print(" | ".join(f"{k} {v}" for k, v in r.items()))
    print(f"-> {out}")


def train(a):
    from ultralytics import YOLO
    if a.resume:   # Windows sometimes runs out of RAM mid-run: carry on from the last epoch
        return YOLO(str(ROOT / "runs" / a.run / "weights" / "last.pt")).train(resume=True)
    data = TN / "data.yaml"
    data.write_text(yaml.safe_dump({"path": str(TN), "train": "train.txt", "val": "val.txt", "names": dict(enumerate(NAMES))}))
    # --init-from: copy every layer whose shape matches (backbone + neck of a road-damage BOX model) into a fresh seg model
    model = YOLO("yolo11s-seg.yaml").load(a.init_from) if a.init_from else YOLO(a.weights)
    model.train(data=str(data), epochs=a.epochs, imgsz=640, batch=a.batch, device=a.device, workers=1,
                project=str(ROOT / "runs"), name=a.run, exist_ok=True, plots=False, patience=a.patience, hsv_s=a.hsv_s,
                freeze=a.freeze or None, cache="ram")   # ~250 MB of small copies: the CPU stops re-decoding 1080p JPEGs every step


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    draw = argparse.ArgumentParser(add_help=False)
    draw.add_argument("--trash-conf", type=float, default=0.05, help="YOLOE confidence for trash: its scores run very low (a clear bag scored 0.085)")
    draw.add_argument("--eps", type=float, default=8.0, help="outline simplification in pixels: higher = fewer points")
    draw.add_argument("--device", default="0")
    draw.add_argument("--iou", type=float, default=0.7, help="NMS IoU: lower = overlapping same-class masks suppressed sooner")
    sub.add_parser("seed", parents=[draw])
    p = sub.add_parser("frames", parents=[draw])
    p.add_argument("--video", required=True)
    p.add_argument("--weights", default="", help="model that pre-draws shapes (mask or box model); empty = blank frames")
    p.add_argument("--every", type=float, default=1.25, help="seconds between frames")
    p.add_argument("--start", type=float, default=0.0, help="seconds to skip at the beginning")
    p.add_argument("--rotate180", action="store_true", help="phone mounted upside down")
    p.add_argument("--conf", type=float, default=0.15, help="low on purpose: deleting a shape is quicker than drawing one")
    p = sub.add_parser("predraw", parents=[draw]); p.add_argument("--weights", required=True)
    p.add_argument("--negatives", default=str(ROOT / "data" / "negatives.txt"), help="review frames listed here are skipped")
    p.add_argument("--conf", type=float, default=0.15, help="low on purpose: deleting a shape is quicker than drawing one")
    p.add_argument("--imgsz", type=int, default=640)
    p = sub.add_parser("serve", parents=[draw])
    p.add_argument("--weights", required=True)
    p.add_argument("--conf", type=float, default=0.15, help="low on purpose: deleting a shape is quicker than drawing one")
    p.add_argument("--port", type=int, default=9090)
    p = sub.add_parser("sync"); p.add_argument("--json", required=True)
    p = sub.add_parser("mine"); p.add_argument("--weights", required=True); p.add_argument("--videos", nargs="+", required=True)
    p.add_argument("--every", type=float, default=1.25, help="seconds between frames (same grid as frames, so names match)")
    p.add_argument("--conf", type=float, default=0.25); p.add_argument("--device", default="0")
    p = sub.add_parser("pack"); p.add_argument("--extra", nargs="*", default=[], help="video ids whose unlabeled frames go in too")
    p = sub.add_parser("testset"); p.add_argument("--video", required=True)
    p.add_argument("--weights", nargs="+", required=True, help="every model whose pothole/crack shapes become candidates")
    p.add_argument("--every", type=float, default=1.25, help="seconds between test frames (same grid as frames/mine)")
    p.add_argument("--conf", type=float, default=0.15, help="low on purpose: the AI judge and you reject, nobody draws")
    p.add_argument("--yoloe-conf", type=float, default=0.05, help="YOLOE text-prompt scores run very low (see --trash-conf)")
    p.add_argument("--device", default="0")
    p = sub.add_parser("testreview"); p.add_argument("--clip", default=sorted(HOLDOUT)[0])
    p = sub.add_parser("score"); p.add_argument("--weights", required=True)
    p.add_argument("--conf", type=float, default=0.25, help="same as demo_video")
    p.add_argument("--persist", type=int, default=3, help="filtered mode: shape seen in this many frames in a row at 6 fps")
    p.add_argument("--device", default="0")
    p = sub.add_parser("train")
    p.add_argument("--run", required=True)
    p.add_argument("--weights", default="yolo11s-seg.pt", help="COCO-pretrained segmentation model")
    p.add_argument("--epochs", type=int, default=60, help="tn2 peaked at epoch 49 of 100, then only memorised")
    p.add_argument("--patience", type=int, default=20, help="stop after this many epochs without a better val score")
    p.add_argument("--resume", action="store_true", help="continue a crashed run from its last epoch")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--init-from", default="", help="e.g. runs/A/weights/best.pt: start from road-damage features instead of COCO")
    p.add_argument("--freeze", type=int, default=0, help="freeze the first N layers (10 = the YOLO11 backbone): fewer weights to fit on few frames")
    p.add_argument("--hsv-s", type=float, default=0.7, help="saturation jitter (Ultralytics default 0.7); partial desaturation A/B")
    p.add_argument("--device", default="0")
    a = ap.parse_args()
    {"seed": lambda: seed(a), "frames": lambda: frames(a), "serve": lambda: serve(a), "sync": lambda: sync(a.json), "pack": lambda: pack(a), "mine": lambda: mine(a), "predraw": lambda: predraw_review(a), "train": lambda: train(a),
     "testset": lambda: testset(a), "testreview": lambda: testreview(a), "score": lambda: score(a)}[a.cmd]()
