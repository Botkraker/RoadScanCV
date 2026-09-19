"""Pothole Mix (SHREC 2022) RGB masks -> pothole boxes.

The class is encoded by mask colour (red = pothole, green = crack). Only images
with red and NO green pixels are used (see classes.yaml for why). Each red blob
becomes one box. Videos in the SHREC download are skipped.

group_id is rebuilt like in convert_kaggle.py: consecutive images of a folder
stay in one group until the picture changes a lot (pHash distance > GROUP_BREAK).

    python src/convert_shrec.py --dry-run
    python src/convert_shrec.py
"""
import argparse
import shutil
from collections import Counter

import cv2
import imagehash
import numpy as np
from PIL import Image

from common import (OUT, RAW, IdAllocator, load_classes, read_manifest,
                    voc_to_yolo, write_manifest)

SOURCE = "shrec_pothole_mix"
BASE = RAW / "shrec" / "pothole-mix"
GROUP_BREAK = 24
MIN_PIXELS = 20        # colour pixels (at half size) needed to count a colour as present
MIN_BLOB_FRAC = 0.0002  # ignore blobs smaller than 0.02% of the image


def colour_masks(mask_bgr):
    b, g, r = mask_bgr[..., 0], mask_bgr[..., 1], mask_bgr[..., 2]
    return (r > 200) & (g < 80) & (b < 80), (g > 200) & (r < 80) & (b < 80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    name_to_id, cfg = load_classes()
    conf = cfg["sources"][SOURCE]
    assert conf["colours"]["red"] == "pothole"
    pothole = name_to_id["pothole"]
    existing = read_manifest()
    rows = [r for r in existing if r["source"] != SOURCE]
    ids = IdAllocator(existing)
    if not args.dry_run:
        (OUT / "images").mkdir(parents=True, exist_ok=True)
        (OUT / "labels").mkdir(parents=True, exist_ok=True)

    stats, per_folder, group = Counter(), Counter(), 0
    for folder in conf["folders"]:
        prev_hash = None
        items = sorted((sp, p) for sp in ("training", "validation")
                       for p in (BASE / sp / folder / "images").iterdir())
        items.sort(key=lambda t: t[1].stem)  # consecutive names across both splits
        for split, img_path in items:
            mask_path = BASE / split / folder / "masks" / f"{img_path.stem}.png"
            if not mask_path.exists():
                stats["missing-mask"] += 1
                continue
            mask = cv2.imread(str(mask_path))
            red_full, green_full = colour_masks(mask)
            small = mask[::2, ::2]
            red_s, green_s = colour_masks(small)
            has_red, has_green = red_s.sum() >= MIN_PIXELS, green_s.sum() >= MIN_PIXELS
            if has_green:
                stats["skipped-has-crack-pixels"] += 1
                continue
            if not has_red:
                stats["skipped-no-pothole"] += 1
                continue

            with Image.open(img_path) as im:
                w, h = im.size
                hsh = imagehash.phash(im)
            if prev_hash is None or hsh - prev_hash > GROUP_BREAK:
                group += 1
            prev_hash = hsh

            mh, mw = red_full.shape
            if (mw, mh) != (w, h):
                stats["mask-size-differs-from-image"] += 1
            n, _, st, _ = cv2.connectedComponentsWithStats(red_full.astype(np.uint8))
            lines = []
            for i in range(1, n):
                x, y, bw, bh, area = st[i]
                if area < MIN_BLOB_FRAC * mh * mw:
                    continue
                # Mask and image can differ in size; boxes are normalized, so scale is free.
                yolo = voc_to_yolo(x, y, x + bw, y + bh, mw, mh)
                if yolo:
                    lines.append(f"{pothole} " + " ".join(f"{v:.6f}" for v in yolo))
            if not lines:
                stats["skipped-blobs-too-small"] += 1
                continue

            orig = img_path.relative_to(RAW).as_posix()
            uid = ids.get(orig)
            stats["images"] += 1
            stats["boxes"] += len(lines)
            per_folder[folder] += 1
            rows.append({"id": uid, "source": SOURCE, "group_id": f"shrec-g{group:04d}",
                         "orig_path": orig, "ext": img_path.suffix, "width": w,
                         "height": h, "n_pothole": len(lines), "n_crack": 0,
                         "n_manhole": 0, "orig_split": split})
            if not args.dry_run:
                shutil.copy2(img_path, OUT / "images" / f"{uid}{img_path.suffix}")
                (OUT / "labels" / f"{uid}.txt").write_text("\n".join(lines) + "\n")

    if not args.dry_run:
        write_manifest(rows)
    print(dict(stats), "groups:", group)
    print("images per folder:", dict(per_folder))
    print("dry run, nothing written" if args.dry_run else f"wrote {OUT}")


if __name__ == "__main__":
    main()
