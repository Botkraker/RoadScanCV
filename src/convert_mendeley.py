"""Water-filled and dry potholes (Mendeley, Dib et al.) -> unified dataset.

Labels are already YOLO txt (class 0 = pothole), so this copies them after
validating each line, and remaps the class id through classes.yaml.

    python src/convert_mendeley.py --dry-run
    python src/convert_mendeley.py
"""
import argparse
import shutil
from collections import Counter

from PIL import Image

from common import (OUT, RAW, IdAllocator, load_classes, read_manifest,
                    write_manifest)

SOURCE = "mendeley_potholes"
SRC_DIR = next(RAW.glob("An Annotated Water-Filled*"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    name_to_id, _ = load_classes()
    pothole = name_to_id["pothole"]
    existing = read_manifest()
    rows = [r for r in existing if r["source"] != SOURCE]
    ids = IdAllocator(existing)
    if not args.dry_run:
        (OUT / "images").mkdir(parents=True, exist_ok=True)
        (OUT / "labels").mkdir(parents=True, exist_ok=True)

    n_boxes, bad, n_img = 0, Counter(), 0
    for img_path in sorted((SRC_DIR / "IMG").glob("*.jpg")):
        txt_path = SRC_DIR / "TXT" / f"{img_path.stem}.txt"
        if not txt_path.exists():
            bad["missing-label-file"] += 1
            continue
        with Image.open(img_path) as im:
            w, h = im.size

        lines = []
        for raw in txt_path.read_text().splitlines():
            parts = raw.split()
            if not parts:
                continue
            try:
                cls, *vals = parts[0], *map(float, parts[1:])
            except ValueError:
                bad["non-numeric"] += 1
                continue
            if cls != "0" or len(vals) != 4 or not all(0 <= v <= 1 for v in vals) \
                    or vals[2] <= 0 or vals[3] <= 0:
                bad["invalid-line"] += 1
                continue
            lines.append(f"{pothole} " + " ".join(f"{v:.6f}" for v in vals))

        orig = img_path.relative_to(RAW).as_posix()
        uid = ids.get(orig)
        n_img += 1
        n_boxes += len(lines)
        rows.append({"id": uid, "source": SOURCE, "group_id": f"mend-{img_path.stem}",
                     "orig_path": orig, "ext": img_path.suffix, "width": w,
                     "height": h, "n_pothole": len(lines), "n_crack": 0,
                     "n_manhole": 0})
        if not args.dry_run:
            shutil.copy2(img_path, OUT / "images" / f"{uid}{img_path.suffix}")
            (OUT / "labels" / f"{uid}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""))

    if not args.dry_run:
        write_manifest(rows)
    print(f"images: {n_img}   pothole boxes: {n_boxes}   problems: {dict(bad)}")
    print("dry run, nothing written" if args.dry_run else f"wrote {OUT}")


if __name__ == "__main__":
    main()
