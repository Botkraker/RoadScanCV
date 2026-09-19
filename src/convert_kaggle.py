"""Kaggle 'Road Damage: potholes, cracks and manholes' -> unified dataset.

Labels (labels-YOLO/) are already YOLO boxes with ids 0/1/2, so this validates
and copies them, remapping ids through classes.yaml.

The dataset gives no clip ids, but most images are consecutive VLC snapshots of
videos. group_id is rebuilt from the images: sorted frames stay in one group
until the picture changes a lot (pHash distance > GROUP_BREAK), so near-copies
end up in the same group and cannot leak across a train/test split.

    python src/convert_kaggle.py --dry-run
    python src/convert_kaggle.py
"""
import argparse
import shutil
from collections import Counter

import imagehash
from PIL import Image

from common import (OUT, RAW, IdAllocator, load_classes, read_manifest,
                    write_manifest)

SOURCE = "kaggle_pcm"
SRC = RAW / "potholes, cracks, and manholes"
GROUP_BREAK = 24  # pHash distance above which two consecutive frames are a new clip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    name_to_id, cfg = load_classes()
    id_map = {str(k): name_to_id[v] for k, v in cfg["sources"][SOURCE].items()}
    id_to_name = {v: k for k, v in name_to_id.items()}
    existing = read_manifest()
    rows = [r for r in existing if r["source"] != SOURCE]
    ids = IdAllocator(existing)
    if not args.dry_run:
        (OUT / "images").mkdir(parents=True, exist_ok=True)
        (OUT / "labels").mkdir(parents=True, exist_ok=True)

    imgs = sorted((SRC / "images").glob("*.jpg"))
    group, prev_hash, stats, per_class = 0, None, Counter(), Counter()
    for img_path in imgs:
        txt = SRC / "labels-YOLO" / f"{img_path.stem}.txt"
        if not txt.exists():
            stats["missing-label-file"] += 1
            continue
        with Image.open(img_path) as im:
            w, h = im.size
            hsh = imagehash.phash(im)
        # Different naming families (vlcsnap-* vs date-stamped photos) never share a group.
        if prev_hash is None or hsh - prev_hash > GROUP_BREAK or \
                img_path.name[:3] != prev_name[:3]:
            group += 1
        prev_hash, prev_name = hsh, img_path.name

        lines, counts = [], Counter()
        for raw in txt.read_text().splitlines():
            p = raw.split()
            if not p:
                continue
            vals = list(map(float, p[1:]))
            if p[0] not in id_map or len(vals) != 4 or not all(0 <= v <= 1 for v in vals) \
                    or vals[2] <= 0 or vals[3] <= 0:
                stats["invalid-line-dropped"] += 1
                continue
            cid = id_map[p[0]]
            lines.append(f"{cid} " + " ".join(f"{v:.6f}" for v in vals))
            counts[id_to_name[cid]] += 1

        orig = img_path.relative_to(RAW).as_posix()
        uid = ids.get(orig)
        per_class.update(counts)
        stats["images"] += 1
        rows.append({"id": uid, "source": SOURCE, "group_id": f"kaggle-g{group:04d}",
                     "orig_path": orig, "ext": img_path.suffix, "width": w, "height": h,
                     "n_pothole": counts["pothole"], "n_crack": counts["crack"],
                     "n_manhole": counts["manhole"], "orig_split": ""})
        if not args.dry_run:
            shutil.copy2(img_path, OUT / "images" / f"{uid}{img_path.suffix}")
            (OUT / "labels" / f"{uid}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""))

    if not args.dry_run:
        write_manifest(rows)
    print(dict(stats), "groups:", group)
    print("boxes kept:", dict(per_class))
    print("dry run, nothing written" if args.dry_run else f"wrote {OUT}")


if __name__ == "__main__":
    main()
