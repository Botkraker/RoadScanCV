"""Cleaning rules, applied to manifest.csv by MARKING, never deleting files.

Reads reports/image_audit.csv (run src/audit_images.py first) and updates:
  excluded       reason string, empty = image is kept
  group_id       merged so near-duplicate images share a group (leak-safe splits)
  group_id_orig  the group before merging (kept so the step can be re-run/undone)

Rules
  1. exact duplicate files (same md5): keep the lowest id, exclude the others
  2. near-duplicates (pHash distance <= NEAR_DUP): merge their groups, remove nothing
  3. blurry images (Laplacian variance < BLUR_MIN): exclude
  Dark images are kept (they are real shade/underpass scenes).

Files stay on disk, so ids remain stable and everything is reversible.

    python src/clean.py --dry-run
    python src/clean.py
"""
import argparse
import csv
from collections import Counter, defaultdict

import numpy as np

from common import ROOT, read_manifest, write_manifest

AUDIT = ROOT / "reports" / "image_audit.csv"
NEAR_DUP = 6   # 8+ chains neighbouring frames into giant clusters (763 images at 8, 4,406 at 10)
BLUR_MIN = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    audit = {r["id"]: r for r in csv.DictReader(AUDIT.open())}
    rows = read_manifest()
    assert {r["id"] for r in rows} == set(audit), "audit report is out of date: rerun audit_images.py"
    for r in rows:  # start from the un-merged state so the script is idempotent
        r["group_id"] = r.get("group_id_orig") or r["group_id"]
        r["group_id_orig"] = r["group_id"]
        r["excluded"] = ""
    ids = [r["id"] for r in rows]

    # 1. exact duplicates
    by_md5 = defaultdict(list)
    for i in ids:
        by_md5[audit[i]["md5"]].append(i)
    reasons = {}
    for members in by_md5.values():
        for extra in sorted(members)[1:]:
            reasons[extra] = f"exact-duplicate-of-{sorted(members)[0]}"

    # 3. blur
    for i in ids:
        if i not in reasons and float(audit[i]["blur"]) < BLUR_MIN:
            reasons[i] = f"blurry (laplacian var {audit[i]['blur']} < {BLUR_MIN})"

    # 2. near-duplicates -> merge groups (union-find over group ids)
    h = np.array([int(audit[i]["phash"], 16) for i in ids], dtype=np.uint64)
    parent = {r["group_id"]: r["group_id"] for r in rows}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    grp = [r["group_id"] for r in rows]
    for start in range(0, len(ids), 500):
        dist = np.bitwise_count(h[start:start + 500, None] ^ h[None, :])
        for a, b in zip(*np.nonzero(dist <= NEAR_DUP)):
            a += start
            if b > a:
                ra, rb = find(grp[a]), find(grp[b])
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)  # smallest group id becomes the name
    for r in rows:
        r["group_id"] = find(r["group_id"])
        r["excluded"] = reasons.get(r["id"], "")

    kept = [r for r in rows if not r["excluded"]]
    print(f"images: {len(rows)}   excluded: {len(rows) - len(kept)}   kept: {len(kept)}")
    print("excluded by rule:", dict(Counter(r["excluded"].split()[0].split("-of")[0] for r in rows if r["excluded"])))
    print("excluded by source:", dict(Counter(r["source"] for r in rows if r["excluded"])))
    print(f"groups: {len({r['group_id_orig'] for r in rows})} -> {len({r['group_id'] for r in rows})}"
          f"  (kept images only: {len({r['group_id'] for r in kept})})")
    print("largest merged group:", max(Counter(r["group_id"] for r in rows).values()), "images")
    box = Counter()
    for r in kept:
        for c in ("pothole", "crack", "manhole"):
            box[c] += int(r["n_" + c])
    print("boxes in kept images:", dict(box))
    if not args.dry_run:
        write_manifest(rows)
        print("manifest.csv updated")
    else:
        print("dry run, nothing written")


if __name__ == "__main__":
    main()
