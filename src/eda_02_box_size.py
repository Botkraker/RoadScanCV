"""EDA step 2: how big are the boxes, in pixels at YOLO's 640 px input size?"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import OUT, ROOT

IMGSZ = 640
NAMES = {0: "pothole", 1: "crack", 2: "manhole"}
COLORS = {"pothole": "#2a78d6", "crack": "#eb6834", "manhole": "#1baf7a"}

m = pd.read_csv(OUT / "manifest.csv", dtype=str, keep_default_na=False)
m = m[m["excluded"] == ""]
rows = []
for r in m.itertuples():
    W, H = int(r.width), int(r.height)
    scale = IMGSZ / max(W, H)                      # letterbox: the longest side becomes 640
    for line in (OUT / "labels" / f"{r.id}.txt").read_text().splitlines():
        c, _, _, w, h = line.split()
        rows.append((r.id, r.source, NAMES[int(c)], np.sqrt(float(w) * W * scale * float(h) * H * scale)))
b = pd.DataFrame(rows, columns=["id", "source", "cls", "size_px"])
b["bucket"] = pd.cut(b["size_px"], [0, 32, 96, 1e9], labels=["small (<32)", "medium (32-96)", "large (>96)"])

print(f"{len(b):,} boxes\n")
print("box size in px at 640 (sqrt of area), by class:")
print(b.groupby("cls")["size_px"].describe(percentiles=[.1, .5, .9])[["min", "10%", "50%", "90%", "max"]].round(0).to_string(), "\n")
print("share of boxes per size bucket (%), by class:")
print((pd.crosstab(b["cls"], b["bucket"], normalize="index") * 100).round(0).to_string(), "\n")
print("share of boxes per size bucket (%), by source:")
print((pd.crosstab(b["source"], b["bucket"], normalize="index") * 100).round(0).to_string())
b.to_csv(ROOT / "reports" / "box_sizes.csv", index=False)

bins = np.logspace(np.log10(3), np.log10(700), 40)
fig, axes = plt.subplots(3, 1, figsize=(7.5, 6.2), dpi=150, sharex=True)
for ax, c in zip(axes, ["pothole", "crack", "manhole"]):
    d = b[b["cls"] == c]["size_px"]
    ax.hist(d, bins=bins, color=COLORS[c], edgecolor="#fcfcfb", linewidth=1)
    for x in (32, 96):
        ax.axvline(x, color="#898781", linewidth=1, linestyle=(0, (4, 3)))
    ax.set_xscale("log")
    ax.text(0.01, 0.86, f"{c}  (median {d.median():.0f} px, {(d < 32).mean():.0%} small)",
            transform=ax.transAxes, fontsize=10, color="#0b0b0b")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7"); ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e"); ax.yaxis.grid(True, color="#e1e0d9", linewidth=0.8); ax.set_axisbelow(True)
    ax.set_ylabel("boxes", color="#52514e")
axes[-1].set_xticks([4, 8, 16, 32, 64, 128, 256, 512]); axes[-1].set_xticklabels([4, 8, 16, 32, 64, 128, 256, 512])
axes[-1].set_xlabel("box size in pixels at 640 px input (log scale); dashed lines = 32 and 96 px", color="#52514e")
fig.suptitle("How big are the boxes?", x=0.01, ha="left", fontsize=12, color="#0b0b0b")
fig.tight_layout()
out = ROOT / "reports" / "eda" / "02_box_sizes.png"; fig.savefig(out); print("\nsaved", out)
