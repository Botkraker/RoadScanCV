"""EDA step 3: how many boxes per image, and how many images are empty?"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import OUT, ROOT

m = pd.read_csv(OUT / "manifest.csv", dtype=str, keep_default_na=False)
m = m[m["excluded"] == ""].copy()
for c in ("pothole", "crack", "manhole"):
    m[c] = m["n_" + c].astype(int)
m["boxes"] = m[["pothole", "crack", "manhole"]].sum(axis=1)
m["classes_present"] = (m[["pothole", "crack", "manhole"]] > 0).sum(axis=1)

print(f"{len(m):,} images, {int(m.boxes.sum()):,} boxes, {(m.boxes == 0).sum():,} empty images ({(m.boxes == 0).mean():.0%})\n")
print("boxes per image (all images): mean %.2f, median %d, max %d\n" % (m.boxes.mean(), m.boxes.median(), m.boxes.max()))
print("per source: images | empty % | mean boxes | max boxes | % of images with 3+ boxes")
g = m.groupby("source").agg(images=("id", "size"), empty=("boxes", lambda x: (x == 0).mean() * 100),
                            mean=("boxes", "mean"), max=("boxes", "max"), crowded=("boxes", lambda x: (x >= 3).mean() * 100))
print(g.round(1).to_string(), "\n")
print("images that contain 2+ different classes:", int((m.classes_present >= 2).sum()))
print("empty images by split:", m[m.boxes == 0].groupby("split").size().to_dict())

# chart: number of boxes per image (0,1,2,3,4,5+), one stacked bar per source would hide detail -> share of images
order = ["rdd2020", "kaggle_pcm", "mendeley_potholes", "shrec_pothole_mix", "pothole_videos"]
cats = ["0", "1", "2", "3-4", "5+"]
def bucket(n): return "0" if n == 0 else "1" if n == 1 else "2" if n == 2 else "3-4" if n <= 4 else "5+"
m["bucket"] = m.boxes.map(bucket)
share = pd.crosstab(m.source, m.bucket, normalize="index").reindex(index=order, columns=cats).fillna(0) * 100
ramp = ["#cde2fb", "#9ec5f4", "#5598e7", "#256abf", "#0d366b"]   # sequential blue: more boxes = darker
fig, ax = plt.subplots(figsize=(8, 3.8), dpi=150)
left = np.zeros(len(order))
for c, col in zip(cats, ramp):
    v = share[c].values
    ax.barh(order[::-1], v[::-1], left=left[::-1], color=col, height=0.6, edgecolor="#fcfcfb", linewidth=2, label=c)
    for i, (l, x) in enumerate(zip(left[::-1], v[::-1])):
        if x >= 6:
            ax.text(l + x / 2, i, f"{x:.0f}%", ha="center", va="center", fontsize=9, color="#0b0b0b" if col in ramp[:2] else "#ffffff")
    left += v
ax.set_xlim(0, 100); ax.set_xlabel("share of images (%)", color="#52514e")
ax.set_title("Boxes per image, by source", loc="left", fontsize=12, color="#0b0b0b")
ax.legend(title="boxes in image", ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=9, title_fontsize=9)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.spines["left"].set_color("#c3c2b7"); ax.spines["bottom"].set_color("#c3c2b7"); ax.tick_params(colors="#52514e")
fig.tight_layout(); out = ROOT / "reports" / "eda" / "03_boxes_per_image.png"; fig.savefig(out, bbox_inches="tight"); print("\nsaved", out)
