# RoadScan data

Everything the models are trained and scored on: where it comes from, how it was captured, how it was labelled, how it is split, and what is wrong with it.

There are three datasets, built in this order:

| Dataset | Folder | Task | Classes | Images | Labelled by |
|---|---|---|---|---|---|
| **Public merged** | `data/processed/` | detection (boxes) | pothole, crack, manhole | 17,663 kept (17,732 total) | the original dataset authors, converted by `src/convert_*.py` |
| **Own v1** (Tunisia, boxes) | `data/own/` | detection (boxes) | pothole, crack, manhole | 148 + 41 + 78 frames | me, in Label Studio, pre-filled by run A |
| **TN** (Tunisia, masks) | `data/tn/` | segmentation (polygons) | pothole, crack, manhole, trash, other | 425 labelled frames (555 extracted) | me, in Label Studio, pre-drawn by SAM / YOLOE / the previous TN model |

All data is versioned with DVC (remote: DagsHub), never with Git. Git only holds the `.dvc` pointer files:

| Pointer | Content | Size | Files |
|---|---|---|---|
| `data/raw.dvc` | original public downloads | 14.4 GB | 48,065 |
| `data/processed.dvc` | unified public dataset | 3.1 GB | 35,475 |
| `data/own_video.dvc` | my own street videos | 4.9 GB | 6 |
| `data/own.dvc` | own v1 frames + box labels | 117 MB | 559 |
| `data/tn.dvc` | TN frames + polygon labels + Label Studio JSON | 201 MB | 989 |

---

## 1. Public merged dataset (`data/processed/`)

### 1.1 Sources: how the images were taken

| Source | Capture | Original labels | Used as |
|---|---|---|---|
| **RDD2020** (Czech, India) | smartphone mounted on a car dashboard, forward view, 600x600 | Pascal VOC boxes, codes D00/D10/D20 (cracks), D40 (pothole) | 10,535 images, crack + pothole, **all background images** |
| **Water-Filled and Dry Potholes** (Mendeley) | still photos and 2 dashcam videos, mostly close-ups; some grayscale, some date-stamped | VOC XML + YOLO txt | 713 images, pothole |
| **Potholes, Cracks and Manholes** (Kaggle) | VLC snapshots (`vlcsnap-*`) of a forward car camera, 640x360, orange car hood visible at the bottom | YOLO boxes | 2,009 images, all 3 classes, **the only source of manholes** |
| **Pothole Mix / SHREC 2022** | mix of 5 public sets, mostly close-ups, some with a "DNIT" logo | RGB masks, class = colour (red pothole, green crack) | 761 images, pothole only |
| **Pothole Videos** (Mendeley) | phone held about 130 cm above the road, pointing down, 1080x1080 | per-frame masks in a `mask/` video | 3,714 frames (every 8th) from 619 videos, 1 pothole each |
| RDD2020 Japan, PathCare | | no labels | kept aside, unused |

### 1.2 How the labels were converted

`classes.yaml` is the single class map. Each source has one conversion script that writes `images/<id>.jpg` + `labels/<id>.txt` (YOLO `class cx cy w h`, normalised) and one row in `manifest.csv`.

| Source | Script | Conversion |
|---|---|---|
| RDD2020 | `src/convert_rdd.py` | VOC to YOLO. D00/D10/D20 become crack, D40 becomes pothole. Other codes (D43/D44 paint wear, etc.) are dropped, and the image is kept as background |
| Mendeley | `src/convert_mendeley.py` | YOLO txt, validated. 4 boxes lying outside the image (the same error is in the original XML) are dropped |
| Kaggle | `src/convert_kaggle.py` | YOLO boxes. 1 zero-height box dropped. Clip groups rebuilt with pHash, because the snapshots carry no clip id |
| Pothole Mix | `src/convert_shrec.py` | mask threshold 127, keep the red blobs, one box per blob. Images with any green (crack) pixel are skipped, because boxing only the red would teach "crack = background" |
| Pothole Videos | `src/convert_pothole_videos.py` | every 8th frame, box = bounding box of the largest mask blob (threshold 127, because mp4 masks are lossy) |

Excluded on purpose: Crack500, EdmCrack600, GAPs384 (thin diagonal cracks give boxes that are mostly road; GAPs384 is academic-only and EdmCrack600 is non-commercial).

### 1.3 Cleaning (`src/audit_images.py`, `src/clean.py`)

Images are **marked, never deleted** (`excluded` column in `manifest.csv`), so ids stay stable.

- 59 exact duplicates (md5), mostly Mendeley: the lowest id is kept.
- 10 blurry images (Laplacian variance < 10), all unlabelled RDD frames.
- Near-duplicates (pHash distance <= 6) are **merged into one group**, not removed, so they cannot land in two splits. This took the group count from 12,775 to 11,927.

### 1.4 Split (`src/split.py`)

70/15/15 by `group_id` (`StratifiedGroupKFold`, seed 42), stratified by source + rarest class present. No group appears in two splits.

| Split | Images | Groups | pothole | crack | manhole | background |
|---|---|---|---|---|---|---|
| train | 12,354 | 8,369 | 7,421 | 5,287 | 644 | 4,359 |
| val | 2,645 | 1,780 | 1,821 | 1,261 | 157 | 936 |
| test | 2,664 | 1,769 | 1,610 | 1,152 | 153 | 935 |

`val_dashcam` (RDD + Kaggle only) is the list to trust, because the close-up sources inflate pothole scores.

### 1.5 Known problems

- **Two kinds of potholes**: dashcam (median 30-58 px at 640) vs close-up (Mendeley 141 px, Pothole Videos 314 px).
- **Source fingerprints**: orange hood (Kaggle, so every manhole), DNIT logo, date stamps. The model can learn the shortcut instead of the object.
- **All empty images come from RDD** (6,230), so "normal road" means RDD road.
- **Manholes are rare and small**: 954 boxes, median 33 px, 49% under 32 px.
- **Label noise** (random audit of 40 crack boxes, `reports/audit/`): about 25% of Kaggle "cracks" are potholes; about 35% of RDD crack boxes show no visible damage.

The full numbers, figures and experiment results are in `README.md`.

---

## 2. Own footage (`data/own_video/`)

Recorded by me in Tunisia with a phone, handheld or mounted low and looking forward along the street. Not in Git (faces and number plates are visible).

| File | Resolution | FPS | Length | Used in |
|---|---|---|---|---|
| `Tunisian_Road.mp4` | 3840x2160 | 59.9 | 3.2 min | own v1 (main set), TN |
| `20260920_112134.mp4` | 3840x2160 | 59.8 | 5.1 min | own v1 (`street3`); frames copied to TN, not yet labelled there |
| `20260920_114106.mp4` | 1920x1080 | 59.8 | 6.0 min | own v1 (`street2`); frames copied to TN, not yet labelled there |
| `20260920_115321.mp4` | 1920x1080 | 59.9 | 5.0 min | TN |
| `20260920_120246.mp4` | 1920x1080 | 59.9 | 1.0 min | TN |
| `20260920_122358.mp4` | 1920x1080 | 59.9 | 8.2 min | not used yet |

The five `20260920_*` clips were shot on 2026-09-20 and cover urban side streets, a wide avenue, a suburban road, a highway/bridge, and one unusable clip aimed at the sky. Which file shows which scene is not recorded yet; add it to this table.

Scenes: harsh sun, pale worn asphalt, patched surface, curbs, manhole covers, litter, pedestrians and parked cars. This is very different from the public data (a different camera height, different lighting, and roads from another country), which is why run A drops from 0.45 mAP50 on `val_dashcam` to 0.08 here.

---

## 3. Own v1: box labels (`data/own/`)

Made with `src/own_frames.py`, then scored with `src/own_eval.py` and used by `src/finetune_own.py`.

**Extraction:** one frame every 1.25 s, resized to 1920x1080, JPEG quality 92, named `f<frame index>.jpg`. `frames.csv` stores the frame index, time, segment and number of pre-filled boxes.

**Pre-labelling:** run A predicts at `conf=0.15`. The threshold is low on purpose, because deleting a wrong box is quicker than drawing a missing one. The predictions go into `ls_import.json` as Label Studio predictions.

**Annotation:** Label Studio, `RectangleLabels` (`label_config.xml`) with pothole / crack / manhole on hotkeys 1-3. Each box is corrected by hand, then exported as YOLO. `own_eval.py` remaps the class ids **by name**, because Label Studio numbers classes alphabetically.

**Split by time, never at random:** the last 25% of a clip is `test`, the 5 s before it is a discarded `gap`, and the rest is `train`. Neighbouring frames are near-copies, so a random split would leak.

| Set | Video | Frames | Labelled |
|---|---|---|---|
| `data/own/` | `Tunisian_Road.mp4` | 152 | 148 (160 boxes: 65 manhole, 51 crack, 44 pothole); 110 train / 38 test |
| `data/own/street2/` | `20260920_114106.mp4` | 41 | 41, all used as test (a different street) |
| `data/own/street3/` | `20260920_112134.mp4` | 78 | 78, all used as test |

Label style: I labelled more thoroughly than the public annotators (faint cracks, patched surface). My boxes are smaller (pothole median 45 px at 640 vs 103 px in public training), and 40% are under 32 px.

---

## 4. TN: Tunisian polygon dataset (`data/tn/`)

The current dataset. Tunisian frames only, **segmentation masks**, 5 classes. Built by `src/tn.py`.

### 4.1 Classes

| ID | Class | Meaning |
|---|---|---|
| 0 | pothole | hole in the asphalt |
| 1 | crack | crack or broken surface |
| 2 | manhole | manhole / utility cover (a distractor that looks like a pothole) |
| 3 | trash | litter, plastic bags, bottles |
| 4 | other | **not defined anywhere in the code or config.** Write down what it means (patches? stains? debris?), because it is the largest class |

IDs 0-2 match the public dataset, so old labels carry over.

### 4.2 How it is built: annotate, train, pre-draw, correct, repeat

```
seed:    own v1 boxes  --SAM 2.1 tiny-->  outlines  + YOLOE (text "trash", "plastic bag", ...)  -> import_seed.json
frames:  new video --every 1.25 s--> frames --previous TN model (or SAM on box-model boxes)--> outlines -> import_<video>.json
Label Studio: drag the pre-drawn points onto the real outline, delete wrong shapes, add missing ones
         (live helper: `tn.py serve` = ML backend; click or box an object and SAM outlines it)
sync:    Label Studio JSON export -> labels/*.txt (YOLO seg) -> split -> train.txt / val.txt
train:   yolo11s-seg (COCO pretrained) -> runs/tnN  -> next round pre-draws with it
```

Details that matter:
- **Frames**: `<video id>_f<frame index>.jpg`, 1920x1080, JPEG 92, one every 1.25 s (75 frames at 60 fps). The same frame always gets the same name, and frames that already have a label are skipped.
- **Pre-drawn outlines** are simplified with `cv2.approxPolyDP` (`--eps 8` px), so a polygon has few points and is quick to drag: median 9 points for potholes, 4 to 6 for other classes.
- **YOLOE for trash** runs at `imgsz=1280, conf=0.05` (its scores run very low), only until a TN model knows the trash class itself.
- **Annotation UI** (`label_config.xml`): `PolygonLabels` on hotkeys 1-5, plus SAM smart tools (`KeyPointLabels` and `RectangleLabels`, `smartOnly`).
- **Label format**: `class x1 y1 x2 y2 ...`, normalised polygon, one object per line. Frames that were submitted with no shapes get an empty `.txt` (a background frame).
- **Only submitted frames count**. A cancelled annotation is ignored, and if a frame has two tasks, the last one wins.
- **Split**: per video, the **last 20%** of its labelled frames (in frame order) is val. It's a time split again, but with no gap.
- **Duplicate merge** (`merge()` in `tn.py`, applied in `sync` and in all pre-drawing): same-class shapes that overlap by more than 30% of the smaller one are replaced by their union, repeated until nothing changes. Shapes that only touch stay separate, so neighbouring potholes on a rail crossing remain separate objects.

### 4.3 Current content (after `sync` of `newdataset.json`, 2026-09-24)

| Video | Frames extracted | Labelled | Empty | train / val |
|---|---|---|---|---|
| `TunisianRoad` | 148 | 143 | 20 | 114 / 29 |
| `20260920115321` | 239 | 186 | 34 | 149 / 37 |
| `20260920120246` | 49 | 49 | 1 | 39 / 10 |
| `20260920114106` | 41 | 41 | 2 | 33 / 8 |
| `20260920112134` | 78 | 6 | 0 | 5 / 1 |
| **Total** | **555** | **425** | **57** | **340 / 85** |

| Class | Masks | train | val | Median size (1080p) | Median size at 640 input |
|---|---|---|---|---|---|
| pothole | 278 | 271 | **7** | 66 px | 22 px |
| crack | 124 | 110 | 14 | 105 px | 35 px |
| manhole | 139 | 120 | 19 | 50 px | 16 px |
| trash | 338 | 277 | 61 | 26 px | **8 px** |
| other | 699 | 507 | 192 | 68 px | 22 px |

"Size" is the square root of the polygon area. The export had 1,811 shapes; `merge` turned them into 1,578.

### 4.4 Known problems

1. **Overlapping and duplicate masks (fixed in `sync`).** Before merging, the export had 1,770 pairs of touching same-class shapes, 916 of them overlapping by more than 30%. `TunisianRoad_f07050` alone had 104 pothole polygons stacked into a mesh (1,449 touching pairs), almost certainly an annotation accident. After merging, 1 borderline pair is left (`TunisianRoad_f03675`, `other`, 0.33 overlap on a tiny shape). The source JSON is still fragmented, so fix `f07050` in Label Studio too, or every re-export depends on the merge.
   - Rule going forward: **one polygon per connected damaged area, tight to its edge, no overlaps with the same class.** Don't enlarge a polygon to include healthy asphalt.
2. **Val has only 7 potholes** (and 14 cracks, 19 manholes). The time split puts almost all potholes in train, so per-class pothole AP on val is close to meaningless. Fix: choose val frames by hand, or use a whole held-out street (`20260920112134`, with 72 frames still unlabelled) as the test set.
3. **Trash is tiny** (8 px median at `imgsz=640`), below what YOLO sees reliably. Train or predict it at 1280, or accept low trash recall.
4. **"other" is undefined and is 44% of all masks.** A vague class soaks up confusion from the others.
5. **Val comes from the same streets as train** (the last 20% of the same clips, no gap). The score measures "the end of the same road", not a new street.
6. **"Trash" labels on damaged pavement.** In `20260920114106_f07546`, magenta trash shapes cover broken curb and patched asphalt. Check that trash means litter only.

### 4.5 Models trained on it (`runs/tn*`, val = `data/tn/val.txt`)

| Run | Start from | Epochs run | Best epoch | Best mask mAP50 | Box mAP50 at that epoch |
|---|---|---|---|---|---|
| tn1 | yolo11s-seg | 100 | 61 | 0.205 | 0.203 |
| tn2 | yolo11s-seg | 100 | 49 | 0.205 | 0.215 |
| tn3 | yolo11s-seg | 60 | 57 | 0.241 | 0.282 |
| tn4 | tn3 best | 12 | 12 | 0.185 | 0.251 |

Trained on a local GTX 1650 (AMP disabled, batch 4, `imgsz=640`). The runs used different amounts of labelled data (the dataset grew between rounds), so they are not a clean comparison. tn1 to tn4 were scored on the old, unmerged val (including the broken frame `TunisianRoad_f07050`). Runs after 2026-09-24 use the merged labels and the new split.

---

## 5. Rebuild

```powershell
dvc pull data/processed.dvc data/own.dvc data/own_video.dvc data/tn.dvc

# public dataset (only if rebuilding from data/raw)
python src/convert_rdd.py; python src/convert_mendeley.py; python src/convert_pothole_videos.py
python src/convert_kaggle.py; python src/convert_shrec.py
python src/audit_images.py; python src/clean.py; python src/split.py

# TN, one labelling round
python src/tn.py frames --video data/own_video/<clip>.mp4 --weights runs/tn3/weights/best.pt
python src/tn.py serve --weights runs/tn3/weights/best.pt      # optional live helper for Label Studio
python src/tn.py sync --json <Label Studio JSON export>
python src/tn.py train --run tn5
```

Credits and licences for every public source are in `README.md`. Some sources forbid commercial use.
