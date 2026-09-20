"""Leakage-safe 70/15/15 train/val/test split of the kept images.

The split unit is group_id (merged near-duplicate groups, videos, clips), so no
group is ever on two sides. Stratified by source + dominant class so every split
sees every source and the rare classes. Excluded images get no split.

Writes manifest.csv (split column), data/processed/{train,val,test}.txt and
data/processed/data.yaml (ready for Ultralytics YOLO).

    python src/split.py --dry-run
    python src/split.py
"""
import argparse
import warnings
from collections import Counter, defaultdict

import numpy as np
import yaml
from sklearn.model_selection import StratifiedGroupKFold

from common import OUT, load_classes, read_manifest, write_manifest

N_FOLDS = 20           # 3 folds test + 3 val + 14 train = 15% / 15% / 70%
SEED = 42
TEST_FOLDS, VAL_FOLDS = {0, 1, 2}, {3, 4, 5}
DASHCAM = {"rdd2020", "kaggle_pcm"}   # sources that look like the real use case


def dominant(r):
    """Rarest class present wins, so manhole images are spread over all splits."""
    for c in ("manhole", "crack", "pothole"):
        if int(r["n_" + c]) > 0:
            return c
    return "background"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = read_manifest()
    kept = [r for r in rows if not r["excluded"]]
    y = np.array([f"{r['source']}|{dominant(r)}" for r in kept])
    groups = np.array([r["group_id"] for r in kept])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        folds = list(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED)
                     .split(np.zeros(len(kept)), y, groups))
    if w:
        print("note:", str(w[0].message).split(".")[0])
    split_of = {}
    for k, (_, idx) in enumerate(folds):
        for i in idx:
            split_of[kept[i]["id"]] = "test" if k in TEST_FOLDS else "val" if k in VAL_FOLDS else "train"
    for r in rows:
        r["split"] = split_of.get(r["id"], "")

    # ---- checks and report
    g2s = defaultdict(set)
    for r in kept:
        g2s[r["group_id"]].add(r["split"])
    leaks = [g for g, s in g2s.items() if len(s) > 1]
    print(f"groups in more than one split (must be 0): {len(leaks)}")
    print(f"\n{'split':6s} {'images':>7s} {'groups':>7s} {'pothole':>8s} {'crack':>7s} {'manhole':>8s} {'background':>11s}")
    for sp in ("train", "val", "test"):
        rs = [r for r in kept if r["split"] == sp]
        print(f"{sp:6s} {len(rs):7d} {len({r['group_id'] for r in rs}):7d} "
              f"{sum(int(r['n_pothole']) for r in rs):8d} {sum(int(r['n_crack']) for r in rs):7d} "
              f"{sum(int(r['n_manhole']) for r in rs):8d} {sum(dominant(r) == 'background' for r in rs):11d}")
    print("\nimages per source and split:")
    srcs = sorted({r["source"] for r in kept})
    for s in srcs:
        c = Counter(r["split"] for r in kept if r["source"] == s)
        n = sum(c.values())
        print(f"  {s:20s} " + "  ".join(f"{sp} {c[sp]:5d} ({c[sp] / n:4.0%})" for sp in ("train", "val", "test")))

    if args.dry_run:
        print("\ndry run, nothing written")
        return
    write_manifest(rows)
    for sp in ("train", "val", "test"):
        (OUT / f"{sp}.txt").write_text(
            "\n".join(f"./images/{r['id']}{r['ext']}" for r in kept if r["split"] == sp) + "\n")
    # extra lists for the experiments: dashcam-only evaluation, and training without Pothole Videos
    for name, keep in {"val_dashcam": lambda r: r["split"] == "val" and r["source"] in DASHCAM,
                       "test_dashcam": lambda r: r["split"] == "test" and r["source"] in DASHCAM,
                       "train_no_videos": lambda r: r["split"] == "train" and r["source"] != "pothole_videos"}.items():
        (OUT / f"{name}.txt").write_text(
            "\n".join(f"./images/{r['id']}{r['ext']}" for r in kept if keep(r)) + "\n")
    name_to_id, _ = load_classes()
    # no "path" key: Ultralytics then resolves the lists relative to this file
    (OUT / "data.yaml").write_text(yaml.safe_dump(
        {"train": "train.txt", "val": "val.txt", "test": "test.txt",
         "names": {v: k for k, v in name_to_id.items()}}, sort_keys=False))
    print(f"\nwrote manifest split column, train/val/test.txt and data.yaml in {OUT}")


if __name__ == "__main__":
    main()
