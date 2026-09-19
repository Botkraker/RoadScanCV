"""RDD2020 (Pascal VOC XML) -> unified YOLO dataset.

Only train/<country> folders that have annotation XMLs are used. Codes not in
classes.yaml are dropped; the image is kept (as background if nothing remains).

    python src/convert_rdd.py --dry-run      # count only, write nothing
    python src/convert_rdd.py                # copy images, write labels + manifest
"""
import argparse
import shutil
import xml.etree.ElementTree as ET
from collections import Counter

from common import (OUT, RAW, IdAllocator, load_classes, read_manifest,
                    voc_to_yolo, write_manifest)

SOURCE = "rdd2020"
RDD_DIR = RAW / "csech_india_japan" / "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    name_to_id, cfg = load_classes()
    code_map = cfg["sources"][SOURCE]  # {"D40": "pothole", ...}
    rows = [r for r in read_manifest() if r["source"] != SOURCE]
    ids = IdAllocator(read_manifest())
    if not args.dry_run:
        (OUT / "images").mkdir(parents=True, exist_ok=True)
        (OUT / "labels").mkdir(parents=True, exist_ok=True)

    boxes, dropped, images, skipped = Counter(), Counter(), Counter(), []
    for country_dir in sorted(p for p in RDD_DIR.iterdir() if p.is_dir()):
        xml_dir = country_dir / "annotations" / "xmls"
        if not xml_dir.is_dir() or not any(xml_dir.iterdir()):
            skipped.append(country_dir.name)  # e.g. Japan: images but no labels
            continue
        for xml_path in sorted(xml_dir.glob("*.xml")):
            img_path = country_dir / "images" / f"{xml_path.stem}.jpg"
            if not img_path.exists():
                print(f"WARNING no image for {xml_path.name}")
                continue
            root = ET.parse(xml_path).getroot()
            w = int(root.findtext("size/width"))
            h = int(root.findtext("size/height"))

            lines, per_class = [], Counter()
            for obj in root.iter("object"):
                code = obj.findtext("name").strip()
                target = code_map.get(code)
                if target is None:
                    dropped[code] += 1
                    continue
                bb = obj.find("bndbox")
                y = voc_to_yolo(*(float(bb.findtext(k)) for k in
                                  ("xmin", "ymin", "xmax", "ymax")), w, h)
                if y is None:
                    dropped["degenerate-box"] += 1
                    continue
                lines.append(f"{name_to_id[target]} " + " ".join(f"{v:.6f}" for v in y))
                per_class[target] += 1

            orig = img_path.relative_to(RAW).as_posix()
            uid = ids.get(orig)
            boxes.update(per_class)
            images[country_dir.name] += 1
            rows.append({"id": uid, "source": SOURCE,
                         "group_id": f"rdd-{xml_path.stem}", "orig_path": orig,
                         "ext": img_path.suffix, "width": w, "height": h,
                         "n_pothole": per_class["pothole"],
                         "n_crack": per_class["crack"],
                         "n_manhole": per_class["manhole"]})
            if not args.dry_run:
                shutil.copy2(img_path, OUT / "images" / f"{uid}{img_path.suffix}")
                # An empty .txt is a valid "background" label. Always write it.
                (OUT / "labels" / f"{uid}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""))

    if not args.dry_run:
        write_manifest(rows)
    print(f"images per country: {dict(images)}")
    print(f"boxes kept:    {dict(boxes)}")
    print(f"boxes dropped: {dict(dropped)}")
    print(f"skipped (no annotations): {skipped}")
    print("dry run, nothing written" if args.dry_run else f"wrote {OUT}")


if __name__ == "__main__":
    main()
