"""EDA step 1: how many boxes does each class have? (kept images only)"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import OUT, ROOT

CLASSES = ["pothole", "crack", "manhole"]
COLORS = {"pothole": "#2a78d6", "crack": "#eb6834", "manhole": "#1baf7a"}  # palette slots 1-3

m = pd.read_csv(OUT / "manifest.csv", dtype=str, keep_default_na=False)
kept = m[m["excluded"] == ""].copy()
for c in CLASSES:
    kept[c] = kept["n_" + c].astype(int)

boxes = kept[CLASSES].sum()
print("boxes per class:\n", boxes.to_string(), "\n")
print("share of all boxes (%):\n", (boxes / boxes.sum() * 100).round(1).to_string(), "\n")
print("boxes per class and split:\n", kept.groupby("split")[CLASSES].sum().to_string())

fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150)
bars = ax.barh(CLASSES[::-1], boxes[CLASSES[::-1]], color=[COLORS[c] for c in CLASSES[::-1]], height=0.55)
for bar, c in zip(bars, CLASSES[::-1]):
    n = boxes[c]
    ax.text(n + boxes.max() * 0.012, bar.get_y() + bar.get_height() / 2,
            f"{n:,}  ({n / boxes.sum():.0%})", va="center", fontsize=10, color="#0b0b0b")
ax.set_xlim(0, boxes.max() * 1.25)
ax.set_title("Boxes per class (17,663 kept images)", loc="left", fontsize=12, color="#0b0b0b")
ax.set_xlabel("number of boxes", color="#52514e")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color("#c3c2b7"); ax.spines["bottom"].set_color("#c3c2b7")
ax.tick_params(colors="#52514e"); ax.xaxis.grid(True, color="#e1e0d9", linewidth=0.8); ax.set_axisbelow(True)
fig.tight_layout()
out = ROOT / "reports" / "eda" / "01_class_balance.png"
fig.savefig(out); print("\nsaved", out)
