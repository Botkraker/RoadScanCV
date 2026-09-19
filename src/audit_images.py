"""Measure every image in data/processed (read-only): file hash, perceptual hash,
blur, brightness. Writes reports/image_audit.csv. Nothing is deleted or changed.

    python src/audit_images.py
"""
import csv
import hashlib
from concurrent.futures import ProcessPoolExecutor

import cv2
import imagehash
import numpy as np
from PIL import Image

from common import OUT, ROOT, read_manifest

REPORT = ROOT / "reports" / "image_audit.csv"
COLUMNS = ["id", "source", "group_id", "md5", "phash", "blur", "brightness", "dark_frac", "bright_frac"]


def measure(row):
    path = OUT / "images" / f"{row['id']}{row['ext']}"
    data = path.read_bytes()
    md5 = hashlib.md5(data).hexdigest()
    with Image.open(path) as im:
        im.draft("L", (512, 512))  # fast JPEG decode at reduced size
        phash = str(imagehash.phash(im))
    gray = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    scale = 512 / gray.shape[1]  # compare blur on a fixed width so sizes are comparable
    gray = cv2.resize(gray, (512, max(1, int(gray.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    return {"id": row["id"], "source": row["source"], "group_id": row["group_id"],
            "md5": md5, "phash": phash,
            "blur": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
            "brightness": round(float(gray.mean()), 1),
            "dark_frac": round(float((gray < 30).mean()), 3),
            "bright_frac": round(float((gray > 225).mean()), 3)}


def main():
    rows = read_manifest()
    REPORT.parent.mkdir(exist_ok=True)
    with ProcessPoolExecutor() as pool, REPORT.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=COLUMNS)
        wr.writeheader()
        for i, res in enumerate(pool.map(measure, rows, chunksize=64), 1):
            wr.writerow(res)
            if i % 2000 == 0:
                print(f"{i}/{len(rows)}", flush=True)
    print(f"done: {len(rows)} images -> {REPORT}")


if __name__ == "__main__":
    main()
