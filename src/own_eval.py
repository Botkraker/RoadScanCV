"""Label Studio YOLO export -> data/own/labels (our class ids) + lists, then score weights on it.

Label Studio numbers classes alphabetically (classes.txt), which differs from ours, so ids are
remapped BY NAME. Lists: own_test (last part of the clip), own_train, own_all.

    python src/own_eval.py --zip C:/Users/hp/Downloads/project-3-....zip --weights runs/A/weights/best.pt runs/B/weights/best.pt
"""
import argparse
import csv
import re
import zipfile
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
OWN = ROOT / "data" / "own"
OURS = {"pothole": 0, "crack": 1, "manhole": 2}


def build(zip_path):
    z = zipfile.ZipFile(zip_path)
    names = z.read("classes.txt").decode().split()
    remap = {str(i): str(OURS[n]) for i, n in enumerate(names)}
    (OWN / "labels").mkdir(exist_ok=True)
    kept = []
    # Label Studio names a file "f06468.txt" (JSON import) or "<hash>__...%5Cf06468.txt" (storage sync).
    # Prefer the plain name: when a frame has both, the prefixed one is a duplicate task.
    files = {}
    for n in sorted(z.namelist(), key=lambda n: "__" in n):   # plain names first
        m = re.fullmatch(r"labels/(?:.*__.*?)?(f\d{5})\.txt", n)
        if m and m[1] not in files:
            files[m[1]] = n
    for f, n in files.items():
        lines = [" ".join([remap[l.split()[0]], *l.split()[1:]]) for l in z.read(n).decode().splitlines() if l.strip()]
        (OWN / "labels" / f"{f}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        kept.append(f)
    seg = {r["file"][:-4]: r["segment"] for r in csv.DictReader(open(OWN / "frames.csv"))}
    lists = {"own_test": [f for f in kept if seg[f] == "test"], "own_train": [f for f in kept if seg[f] == "train"], "own_all": kept}
    for name, fs in lists.items():
        (OWN / f"{name}.txt").write_text("\n".join(f"./images/{f}.jpg" for f in sorted(fs)) + "\n")
    print(f"classes.txt {names} -> our ids | frames: test {len(lists['own_test'])}, train {len(lists['own_train'])}, all {len(kept)}")


def score(weights, lst, single=False, imgsz=640):
    y = OWN / f"eval_{lst}.yaml"
    y.write_text(yaml.safe_dump({"path": str(OWN), "train": f"{lst}.txt", "val": f"{lst}.txt",
                                 "names": {v: k for k, v in OURS.items()}}))
    r = YOLO(str(weights)).val(data=str(y), imgsz=imgsz, batch=4, workers=1, device=0, half=False, single_cls=single,
                               agnostic_nms=single, plots=False, verbose=False)
    row = {"model": Path(weights).parents[1].name, "list": lst + (" (any damage)" if single else ""), "mAP50": round(r.box.map50, 3)}
    if not single:
        row.update({{0: "pothole", 1: "crack", 2: "manhole"}[int(c)]: round(float(r.box.ap50[i]), 3) for i, c in enumerate(r.box.ap_class_index)})
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--sub", default="", help="subfolder of data/own for another video, e.g. street2")
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()
    OWN = OWN / a.sub
    build(a.zip)
    import pandas as pd
    lists = ("own_all",) if a.sub else ("own_test", "own_all")   # another street: every frame is unseen test data
    rows = [score(w, l, s, a.imgsz) for w in a.weights for l in lists for s in (False, True)]
    print(pd.DataFrame(rows).to_string(index=False))
