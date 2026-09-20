"""Audit step B: a fixed random sample of 'crack' boxes per source, as numbered review sheets.

A person labels each numbered crop (pothole / crack / unclear); the sample CSV keeps the
box identity (and, for RDD, the original code D00/D10/D20) so the error rate can be
estimated per source and per code. Random sampling makes the estimate unbiased.

    python src/audit_sample.py
"""
import csv
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from common import OUT, RAW, ROOT

N_PER_SOURCE, PER_SHEET, SEED = 40, 20, 2026
KEEP_CODES = {"D00", "D10", "D20", "D40"}

def rdd_codes(orig_path):
    """Codes of the kept objects of an RDD image, in the same order convert_rdd.py wrote them."""
    p = Path(orig_path)
    xml = RAW / p.parent.parent / "annotations" / "xmls" / f"{p.stem}.xml"
    return [o.findtext("name").strip() for o in ET.parse(xml).getroot().iter("object")
            if o.findtext("name").strip() in KEEP_CODES]

manifest = [r for r in csv.DictReader(open(OUT / "manifest.csv")) if not r["excluded"]]
boxes = {"rdd2020": [], "kaggle_pcm": []}
for r in manifest:
    if r["source"] not in boxes or int(r["n_crack"]) == 0:
        continue
    lines = [l.split() for l in (OUT / "labels" / f"{r['id']}.txt").read_text().splitlines() if l.strip()]
    codes = rdd_codes(r["orig_path"]) if r["source"] == "rdd2020" else [""] * len(lines)
    assert len(codes) == len(lines), r["id"]
    for k, (l, code) in enumerate(zip(lines, codes)):
        if l[0] == "1":
            boxes[r["source"]].append((r, k, list(map(float, l[1:])), code))

random.seed(SEED)
sample = []
for src in ("rdd2020", "kaggle_pcm"):
    sample += [(src, *b) for b in random.sample(boxes[src], N_PER_SOURCE)]
random.shuffle(sample)   # mixed order, so the reviewer cannot tell the source from the position

def tile(r, box, num):
    im = cv2.imread(str(OUT / "images" / f"{r['id']}{r['ext']}")); H, W = im.shape[:2]
    cx, cy, w, h = box
    x0, y0, x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H
    mx, my = max((x1 - x0) * 0.6, 40), max((y1 - y0) * 0.6, 40)          # generous context around the box
    a, b, c, d = int(max(0, x0 - mx)), int(max(0, y0 - my)), int(min(W, x1 + mx)), int(min(H, y1 + my))
    crop = im[b:d, a:c].copy()
    cv2.rectangle(crop, (int(x0 - a), int(y0 - b)), (int(x1 - a), int(y1 - b)), (0, 140, 255), max(2, crop.shape[1] // 120))
    s = 330 / max(crop.shape[:2]); crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)))
    canvas = np.full((340, 340, 3), 245, np.uint8); y, x = (340 - crop.shape[0]) // 2, (340 - crop.shape[1]) // 2
    canvas[y:y + crop.shape[0], x:x + crop.shape[1]] = crop
    cv2.rectangle(canvas, (0, 0), (52, 30), (255, 255, 255), -1)
    cv2.putText(canvas, str(num), (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    return canvas

rows_csv = []
for n, (src, r, k, box, code) in enumerate(sample, 1):
    rows_csv.append({"num": n, "source": src, "id": r["id"], "box_index": k, "rdd_code": code, "verdict": ""})
with open(ROOT / "reports" / "audit" / "crack_sample.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows_csv[0])); wr.writeheader(); wr.writerows(rows_csv)

for s in range(0, len(sample), PER_SHEET):
    tiles = [tile(r, box, s + i + 1) for i, (src, r, k, box, code) in enumerate(sample[s:s + PER_SHEET])]
    tiles += [np.full((340, 340, 3), 245, np.uint8)] * (PER_SHEET - len(tiles))
    sheet = np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, PER_SHEET, 5)])
    out = ROOT / "reports" / "audit" / f"sheet_{s // PER_SHEET + 1}.jpg"
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90]); print("saved", out, sheet.shape[:2])
print(f"{len(sample)} boxes sampled ({N_PER_SOURCE} per source, seed {SEED})")
