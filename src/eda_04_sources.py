"""EDA step 4: do the sources look different? Table of image properties + a visual sample per source."""
import cv2
import numpy as np
import pandas as pd

from common import OUT, ROOT

ORDER = ["rdd2020", "kaggle_pcm", "mendeley_potholes", "shrec_pothole_mix", "pothole_videos"]
NAMES = {0: (0, 0, 255), 1: (0, 140, 255), 2: (0, 200, 0)}  # BGR box colours: pothole red, crack orange, manhole green

m = pd.read_csv(OUT / "manifest.csv", dtype=str, keep_default_na=False)
m = m[m["excluded"] == ""]
a = pd.read_csv(ROOT / "reports" / "image_audit.csv", dtype={"id": str})[["id", "blur", "brightness"]]
m = m.merge(a, on="id")
m[["width", "height"]] = m[["width", "height"]].astype(int)
m["megapixels"] = m.width * m.height / 1e6
m["aspect"] = (m.width / m.height).round(2)

rows = []
for s in ORDER:
    d = m[m.source == s]
    common = d.groupby(["width", "height"]).size().idxmax()
    rows.append({"source": s, "images": len(d), "most_common_size": f"{common[0]}x{common[1]}",
                 "sizes_used": d.groupby(["width", "height"]).ngroups, "median_brightness": round(d.brightness.median()),
                 "median_blur": round(d.blur.median()), "blur_p10": round(d.blur.quantile(.1))})
print(pd.DataFrame(rows).to_string(index=False))
print("\nbrightness: 0 = black, 255 = white. blur = Laplacian variance at 512 px width (higher = sharper).")

# visual sample: 4 random images per source, boxes drawn
rng = np.random.default_rng(7)
TW, TH = 380, 285
def tile(r):
    im = cv2.imread(str(OUT / "images" / f"{r.id}{r.ext}")); H, W = im.shape[:2]
    for l in (OUT / "labels" / f"{r.id}.txt").read_text().splitlines():
        c, cx, cy, bw, bh = l.split(); c = int(c); cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
        cv2.rectangle(im, (int((cx-bw/2)*W), int((cy-bh/2)*H)), (int((cx+bw/2)*W), int((cy+bh/2)*H)), NAMES[c], max(2, W // 160))
    s = min(TW / W, TH / H); im = cv2.resize(im, (int(W * s), int(H * s)))
    canvas = np.full((TH, TW, 3), 250, np.uint8); y0, x0 = (TH - im.shape[0]) // 2, (TW - im.shape[1]) // 2
    canvas[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
    return canvas
rows_img = []
for s in ORDER:
    d = m[m.source == s]
    d = d[d.n_pothole.astype(int) + d.n_crack.astype(int) + d.n_manhole.astype(int) > 0]   # show images with boxes
    pick = d.iloc[rng.choice(len(d), 4, replace=False)]
    row = np.hstack([tile(r) for r in pick.itertuples()])
    cv2.putText(row, s, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 4); cv2.putText(row, s, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    rows_img.append(row)
out = ROOT / "reports" / "eda" / "04_source_samples.jpg"; cv2.imwrite(str(out), np.vstack(rows_img), [cv2.IMWRITE_JPEG_QUALITY, 88]); print("saved", out)
