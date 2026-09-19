"""Shared helpers: class map, YOLO conversion, and the ID-stable manifest."""
import csv
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
MANIFEST = OUT / "manifest.csv"

COLUMNS = ["id", "source", "group_id", "orig_path", "ext", "width", "height",
           "n_pothole", "n_crack", "n_manhole", "orig_split"]


def load_classes():
    """Return ({name: id}, full config) from classes.yaml."""
    cfg = yaml.safe_load((ROOT / "classes.yaml").read_text(encoding="utf-8"))
    return {name: cid for cid, name in cfg["classes"].items()}, cfg


def voc_to_yolo(xmin, ymin, xmax, ymax, w, h):
    """Pixel corners -> normalized (x_center, y_center, width, height), clipped to the image."""
    xmin, xmax = max(0, xmin), min(w, xmax)
    ymin, ymax = max(0, ymin), min(h, ymax)
    if xmax <= xmin or ymax <= ymin:
        return None  # degenerate box
    return ((xmin + xmax) / 2 / w, (ymin + ymax) / 2 / h,
            (xmax - xmin) / w, (ymax - ymin) / h)


def read_manifest():
    if not MANIFEST.exists():
        return []
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_manifest(rows):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["id"])
    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=COLUMNS)
        wr.writeheader()
        wr.writerows(rows)


class IdAllocator:
    """IDs are assigned once and never change: known orig_paths keep their ID,
    new ones get the next free number."""

    def __init__(self, rows):
        self.by_path = {r["orig_path"]: r["id"] for r in rows}
        self.next = max((int(r["id"]) for r in rows), default=0) + 1

    def get(self, orig_path):
        if orig_path not in self.by_path:
            self.by_path[orig_path] = f"{self.next:06d}"
            self.next += 1
        return self.by_path[orig_path]
