"""Pothole Videos (Ihsan et al.): rgb + mask mp4 pairs -> sampled frames + YOLO boxes.

Every 8th frame of each 48-frame video is kept (frames 0, 8, ..., 40) so that
near-identical frames do not swamp the other sources. The mask blob's bounding
box becomes the pothole label. The video is the group_id, so a later split can
keep all frames of one video on the same side.

    python src/convert_pothole_videos.py --dry-run --limit 5
    python src/convert_pothole_videos.py
"""
import argparse
from collections import Counter

import cv2
import numpy as np

from common import (OUT, RAW, IdAllocator, load_classes, read_manifest,
                    voc_to_yolo, write_manifest)

SOURCE = "pothole_videos"
BASE = RAW / "Pothole Videos" / "pothole_video"
STRIDE = 8
MIN_AREA_FRAC = 0.002  # ignore mask blobs smaller than 0.2% of the image (noise)
KERNEL = np.ones((5, 5), np.uint8)


def mask_to_box(mask_bgr):
    """Largest white blob of a (compressed) mask frame -> (xmin, ymin, xmax, ymax) or None."""
    gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
    binary = (gray > 127).astype(np.uint8)  # video compression blurs 0/255, so threshold
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, KERNEL)  # remove speckle
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_AREA_FRAC * binary.size:
        return None
    x, y, w, h = cv2.boundingRect(c)
    return x, y, x + w, y + h


def read_frames(path, wanted):
    """Return {frame_index: BGR image} for the wanted indices."""
    cap = cv2.VideoCapture(str(path))
    out, i = {}, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            out[i] = frame
        i += 1
    cap.release()
    return out, i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, help="only the first N videos per split")
    args = ap.parse_args()

    name_to_id, _ = load_classes()
    pothole = name_to_id["pothole"]
    existing = read_manifest()
    rows = [r for r in existing if r["source"] != SOURCE]
    ids = IdAllocator(existing)
    if not args.dry_run:
        (OUT / "images").mkdir(parents=True, exist_ok=True)
        (OUT / "labels").mkdir(parents=True, exist_ok=True)

    stats = Counter()
    for split in ("train", "val", "test"):
        videos = sorted((BASE / split / "rgb").glob("*.mp4"))[: args.limit]
        for rgb_path in videos:
            mask_path = BASE / split / "mask" / rgb_path.name
            if not mask_path.exists():
                stats["missing-mask"] += 1
                continue
            wanted = set(range(0, 48, STRIDE))
            rgb, n_rgb = read_frames(rgb_path, wanted)
            msk, n_msk = read_frames(mask_path, wanted)
            if n_rgb != n_msk:
                stats["frame-count-mismatch"] += 1
                continue
            stats["videos"] += 1
            for i in sorted(rgb):
                box = mask_to_box(msk[i])
                if box is None:
                    stats["no-box"] += 1
                    continue
                h, w = rgb[i].shape[:2]
                yolo = voc_to_yolo(*box, w, h)
                if yolo is None:
                    stats["no-box"] += 1
                    continue
                orig = f"{rgb_path.relative_to(RAW).as_posix()}#f{i:02d}"
                uid = ids.get(orig)
                stats["frames"] += 1
                rows.append({"id": uid, "source": SOURCE,
                             "group_id": f"pvid-{rgb_path.stem}", "orig_path": orig,
                             "ext": ".jpg", "width": w, "height": h,
                             "n_pothole": 1, "n_crack": 0, "n_manhole": 0,
                             "orig_split": split})
                if not args.dry_run:
                    cv2.imwrite(str(OUT / "images" / f"{uid}.jpg"), rgb[i],
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    (OUT / "labels" / f"{uid}.txt").write_text(
                        f"{pothole} " + " ".join(f"{v:.6f}" for v in yolo) + "\n")

    if not args.dry_run:
        write_manifest(rows)
    print(dict(stats))
    print("dry run, nothing written" if args.dry_run else f"wrote {OUT}")


if __name__ == "__main__":
    main()
