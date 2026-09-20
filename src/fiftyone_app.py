"""Load the unified dataset into FiftyOne and open the app for the visual label audit.

Each image gets its boxes (class + size in px at 640 input) and fields to filter on:
source, split, group_id, n_boxes, dashcam (RDD + Kaggle), biggest_box_frac.

    python src/fiftyone_app.py            # build (if needed) and open the app
    python src/fiftyone_app.py --rebuild  # delete and rebuild the FiftyOne dataset
"""
import argparse

import fiftyone as fo
import pandas as pd

from common import OUT

NAME = "roadscan"
CLASSES = {0: "pothole", 1: "crack", 2: "manhole"}
IMGSZ = 640
DASHCAM = {"rdd2020", "kaggle_pcm"}


def build():
    m = pd.read_csv(OUT / "manifest.csv", dtype=str, keep_default_na=False)
    m = m[m["excluded"] == ""]
    samples = []
    for r in m.itertuples():
        W, H = int(r.width), int(r.height)
        scale = IMGSZ / max(W, H)
        dets, biggest = [], 0.0
        for line in (OUT / "labels" / f"{r.id}.txt").read_text().splitlines():
            c, cx, cy, w, h = line.split()
            cx, cy, w, h = map(float, (cx, cy, w, h))
            dets.append(fo.Detection(
                label=CLASSES[int(c)],
                bounding_box=[cx - w / 2, cy - h / 2, w, h],   # FiftyOne: top-left x, y, width, height
                size_px=round(((w * W * scale) * (h * H * scale)) ** 0.5),
            ))
            biggest = max(biggest, w * h)
        s = fo.Sample(filepath=str(OUT / "images" / f"{r.id}{r.ext}"), tags=[r.source, r.split])
        s["id"] = r.id
        s["source"] = r.source
        s["split"] = r.split
        s["group_id"] = r.group_id
        s["n_boxes"] = len(dets)
        s["dashcam"] = r.source in DASHCAM
        s["biggest_box_frac"] = round(biggest, 3)
        s["ground_truth"] = fo.Detections(detections=dets)
        samples.append(s)
    ds = fo.Dataset(NAME, persistent=True)
    ds.add_samples(samples)
    ds.compute_metadata()
    ds.add_dynamic_sample_fields()
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    if args.rebuild and fo.dataset_exists(NAME):
        fo.delete_dataset(NAME)
    ds = fo.load_dataset(NAME) if fo.dataset_exists(NAME) else build()
    print(ds)
    session = fo.launch_app(ds, port=5151, auto=False)
    print("FiftyOne is running at http://localhost:5151  (Ctrl+C to stop)")
    session.wait()


if __name__ == "__main__":
    main()
