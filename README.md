# RoadScan

A personal computer-vision project: build one clean, unified, leakage-safe dataset for **road damage detection** from six public sources, then train a detector on it.

> Status: dataset built and two baselines trained. 17,732 images from five public sources are unified, cleaned and split without leakage; EDA and a label audit are done. Run A reaches mAP@0.5 **0.45** on dashcam validation but only **0.08** on my own street footage, so the work moved to a **Tunisian-only segmentation dataset (TN)**: 425 labelled frames from 5 street videos, 5 classes, polygon masks, built in model-assisted labelling rounds. The best TN model so far reaches mask mAP@0.5 **0.29-0.30** on held-out frames. Full data documentation: [`data.md`](data.md).

Code: <https://github.com/Botkraker/RoadScanCV> | Data (DVC remote): <https://dagshub.com/Botkraker/RoadScan>

## Goal

Detect three kinds of road anomaly in images and dashcam video:

| ID | Class   |
|----|---------|
| 0  | pothole |
| 1  | crack   |
| 2  | manhole |

The sources use different label formats and class names, so most of the work is turning them into a single consistent dataset that can be rebuilt with one command.

### Two-stage idea

The end goal is to flag **severe** anomalies, not every hairline crack. No source labels severity, so the work is split in two:

1. **Stage 1 (current):** an object detector that finds potholes, cracks and manholes (bounding boxes).
2. **Stage 2 (later, optional):** a small classifier on the cropped boxes that predicts severity (low / medium / high). It needs severity labels that must be created by hand or by rule, because no dataset provides them (the SHREC videos contain no depth data either).

## Design decisions

| Decision | Choice | Why |
|---|---|---|
| Detection or segmentation | **Detection (boxes)** | Simpler; the goal is to flag severe anomalies, and thin crack masks give poor boxes |
| Class map | `classes.yaml` is the single source of truth | Every conversion script reads it, nothing is hard-coded |
| RDD2020 codes | Only the 4 official classes: D00, D10, D20 (cracks) and D40 (pothole) | Matches the paper; D43/D44 are paint wear, not damage. Images with dropped codes are kept as background |
| Excluded masks | Crack500, EdmCrack600, GAPs384 | Thin diagonal cracks make misleading boxes; GAPs384 is academic-only and EdmCrack600 non-commercial |
| Pothole Mix class encoding | Class = mask colour: red pothole, green crack. Only images with red and no green are used | Boxing only the red blobs of an image that also shows green cracks would teach "visible cracks are background" |
| Image IDs | Running number (`000001`), assigned once and never changed | Short names; source, original path and group live in `manifest.csv`. Stable IDs keep labels and splits valid across rebuilds |
| Video frames | Every 8th frame | 48-frame clips are near-identical; using all frames would make one phone-camera source 72% of the data |
| Splits | 70/15/15 by `group_id`, stratified by source + dominant class (`src/split.py`, seed 42) | No group (video, clip, merged near-duplicates) is on two sides, so test measures unseen scenes. The rarest class present decides an image's stratum (manhole > crack > pothole > background), so manholes reach every split |
| Mask to box | Threshold at 127, remove speckle, bounding box of the largest blob | Masks stored in mp4 are lossy, so exact 255 matching fails |
| Missing clip ids (Kaggle) | Rebuild groups with pHash on consecutive frames | Prevents near-duplicate frames from leaking across splits |
| Cleaning | **Mark, never delete**: `excluded` column in `manifest.csv`, files stay on disk | Image ids stay stable and the step is reversible and re-runnable |
| Exact duplicates | Keep the lowest id, exclude the other copies (59) | Same picture annotated twice, with slightly different boxes |
| Near-duplicates | Merge their groups at pHash distance <= 6; remove nothing | RDD frames come from driving videos with no video id, so neighbouring frames would leak across splits. At >= 8, similarity chains into giant clusters (763 images at 8, 4,406 at 10) |
| Blurry images | Exclude Laplacian variance < 10 (10 images) | All were unlabeled RDD background frames. Dark images are kept: they are real shade and underpass scenes |
| Video reading | OpenCV, not ffmpeg | No extra install; reads the rgb frame and its mask together |

## Data sources

| Source | Content | Label format | Classes |
|---|---|---|---|
| **RDD2020** (Czech, India, Japan) | Dashcam images | Pascal VOC XML | `D00` `D10` `D20` `D40` (extra codes dropped) |
| **Water-Filled and Dry Potholes** (Mendeley) | Images + 2 dashcam videos | Pascal VOC XML and YOLO txt | pothole |
| **Potholes, Cracks and Manholes** (Kaggle) | 640x360 images | YOLO boxes, quadrilaterals, COCO JSON | pothole, crack, manhole |
| **Pothole Mix / SHREC 2022** | Images + colour videos (no depth, not used) | RGB segmentation masks, class = colour (red pothole, green crack) | pothole, crack |
| **Pothole Videos** (Mendeley) | Videos with `rgb/` and `mask/` | Masks | pothole |
| **PathCare** (Mendeley) | Images + videos of road faults | None found | unlabeled |

Known gaps found during inspection:

- RDD2020 annotations exist only for **train/Czech** and **train/India**. Japan train images have no XML here, and the test splits are unlabeled by design.
- Manholes exist only in the Kaggle set, so class 2 will be small.
- PathCare and Japan train are unlabeled and are kept aside (possible pseudo-labeling later).

## Processing status

Each source is converted by its own script in `src/` into `data/processed/` (`images/`, `labels/`, `manifest.csv`).

| Source | Status | Script | Result |
|---|---|---|---|
| RDD2020 (Czech, India) | **done** | `src/convert_rdd.py` | 10,535 images; 3,384 pothole and 5,192 crack boxes |
| Water-Filled and Dry Potholes | **done** | `src/convert_mendeley.py` | 713 images; 1,153 pothole boxes (4 boxes dropped: coordinates far outside the image, same error in the original XML) |
| Kaggle potholes/cracks/manholes | **done** | `src/convert_kaggle.py` | 2,009 images; 1,261 pothole, 2,518 crack, 957 manhole boxes (1 zero-height box dropped). 430 rebuilt clip groups |
| Pothole Mix (pothole masks) | **done** | `src/convert_shrec.py` | 761 images, 1,431 pothole boxes (red mask blobs). Skipped: 1,875 images with crack pixels, 3 with no pothole. Its videos are not used |
| Pothole Videos (frames + masks) | **done** | `src/convert_pothole_videos.py` | 619 videos, every 8th frame: 3,714 images, 1 pothole box each (bounding box of the mask blob). `group_id` = video |
| RDD2020 Japan, PathCare | unlabeled, kept aside | | |

Current unified dataset (`data/processed/manifest.csv`): **17,732 images** = 10,535 RDD2020 + 3,714 Pothole Videos + 2,009 Kaggle + 761 Pothole Mix + 713 Mendeley.

Boxes per source, before cleaning (the 69 excluded images remove 91 pothole, 10 crack and 3 manhole boxes):

| Source | Images | pothole | crack | manhole |
|---|---|---|---|---|
| RDD2020 (Czech, India) | 10,535 | 3,384 | 5,192 | 0 |
| Pothole Videos | 3,714 | 3,714 | 0 | 0 |
| Kaggle | 2,009 | 1,261 | 2,518 | 957 |
| Pothole Mix | 761 | 1,431 | 0 | 0 |
| Mendeley | 713 | 1,153 | 0 | 0 |
| **Total** | **17,732** | **10,943** | **7,710** | **957** |

What this table says:

- **Manholes** exist only in Kaggle (957 boxes, about 5% of all boxes), so class 2 will be the weakest.
- **Cracks** come almost only from RDD (67%) and Kaggle. The other sources give no crack labels, so crack detection depends on those two.
- **6,240 images (35%) have no boxes.** They are RDD images without damage (or with only dropped codes) and serve as background examples.
- The manifest holds **11,927 groups** (`group_id`) after cleaning: one per still image, one per video, rebuilt clip groups for Kaggle and Pothole Mix, and merged near-duplicate groups (12,775 before merging).

Update this table whenever a source is converted.

### Data quality notes from conversion

- Mendeley: 4 pothole boxes in `Pothole-261/262/268/271` lie outside the image (x up to 2609 on a 392 px image). The original XML has the same error, so these boxes are dropped.
- Pothole Videos are close-up phone shots from about 130 cm above the road, unlike the dashcam views in RDD and Mendeley. Every frame contains a pothole, so they add no background examples.
- Kaggle: 1,907 of 2,009 images are consecutive VLC video snapshots (`vlcsnap-*`) with no clip id. Consecutive frames are often near-identical (median pHash distance 16, against 30 for random pairs), so `group_id` is rebuilt by starting a new group whenever the picture changes a lot (distance > 24). Most images show the orange car hood at the bottom, a dataset bias to keep in mind.
- Pothole Mix masks are RGB, not binary: a per-channel value of 0/255 hid the fact that red (255,0,0) and green (0,255,0) encode different classes. `cracks-and-potholes-in-road` is mixed (1,639 crack-only, 322 pothole-only, 236 both images), so only its 322 pothole-only images are used. Some kept images may still show faint cracks that were not annotated.
- Pothole Mix groups are rebuilt with pHash on consecutive images, as for Kaggle.
- Cleaning result: 17,732 images, of which **69 excluded** (59 exact duplicates, 10 blurry) and **17,663 kept**. Groups went from 12,775 to 11,927 after merging near-duplicates (largest merged group: 63 images). Kept boxes: 10,852 pothole, 7,700 crack, 954 manhole. Exact duplicates are mostly Mendeley (50), then Kaggle (5), Pothole Videos (4; videos `0220` and `0221` are the same clip).
- Split result (17,663 kept images, 11,927 groups; checked: 0 groups and 0 identical files in more than one split, every listed image has its label, Ultralytics loads `data.yaml`):

| Split | Images | Groups | pothole | crack | manhole | background |
|---|---|---|---|---|---|---|
| train | 12,354 (70%) | 8,369 | 7,421 | 5,287 | 644 | 4,359 |
| val | 2,645 (15%) | 1,780 | 1,821 | 1,261 | 157 | 936 |
| test | 2,664 (15%) | 1,769 | 1,610 | 1,152 | 153 | 935 |

  Every source is spread about 70/15/15 across the splits. Caveat: near-duplicates are only merged up to pHash distance 6, so a few loosely similar RDD frames can still sit on different sides. The original dataset splits (`orig_split`) are ignored.
- Box sizes (px at YOLO's 640 input, EDA step 2): potholes are bimodal (median 103 px, 17% small) because dashcam sources (RDD, Kaggle) give medium boxes while the close-up sources (Pothole Videos 96% large, Mendeley 68% large) give large ones. Cracks: median 84 px, 5% small. **Manholes: median 33 px, 49% small**, so they are both the rarest class and the hardest to see.
- Boxes per image (EDA step 3): 1.1 on average; 35% of images (6,230) are empty, and **all of them come from RDD** (59% of RDD images), so the model's only examples of a normal road come from one source. Pothole Videos always have exactly 1 box. Kaggle is the crowded source (mean 2.4 boxes, 36% of images with 3+ boxes) and holds most of the 1,647 images that mix classes.
- Sources look very different (EDA step 4, `reports/eda/04_source_samples.jpg`): RDD and Kaggle are road scenes (dashcam); Mendeley, Pothole Videos and most of Pothole Mix are close-ups. Median brightness runs from 106 (Kaggle) to 174 (Pothole Videos) and median sharpness from 246 (Pothole Mix) to 1,171 (Pothole Videos). Each source also carries an easy-to-learn fingerprint: an orange car hood in every Kaggle frame, a "DNIT" logo in part of Pothole Mix, date stamps and grayscale images in Mendeley. Since every manhole comes from Kaggle, a model could learn "orange hood" instead of "manhole". Mitigations to consider in training: colour/brightness/blur augmentation, cropping out the hood, and reporting metrics per source.
- Images differ in size per source (RDD 600x600, Mendeley 392x806, videos 1080x1080); YOLO resizes at training time.
- Each conversion is checked against an independent count (for example RDD boxes against the raw XML counts), and mask-to-box output is checked by drawing boxes on random frames.

## Pipeline plan

Progress: `[x]` done, `[ ]` to do.

1. [x] **Environment**: Python 3.11 + uv, Git, VS Code.
2. [x] **Download** the sources (Kaggle, Mendeley, RDD2020).
3. [x] **Version data**: DVC tracks `data/raw` (14.4 GB) `data/processed` (35,469 files, 3.1 GB), `data/own`, `data/own_video` and `data/tn` (989 files, 201 MB); Git tracks only the small `.dvc` pointer files. The DVC remote `origin` is DagsHub. Only `data/processed` is meant to be pushed (verify with `dvc status -c data/processed.dvc`): it is the valuable artifact (cleaning, labels, splits), while `data/raw` is the public original downloads and can be re-downloaded from the sources in the credits.
4. [x] **Inventory**: count files and read the labels of every source.
5. [x] **Unify classes**: `classes.yaml`.
6. [x] **Convert formats**: RDD VOC, Mendeley YOLO, Pothole Videos masks, Kaggle YOLO and Pothole Mix masks.
7. [x] **Clean**: `src/audit_images.py` measures every image (md5, pHash, blur, brightness); `src/clean.py` marks duplicates and blurry images as `excluded` and merges near-duplicate groups.
8. [x] **Split without leakage**: `src/split.py` (`StratifiedGroupKFold`, 20 folds: 14 train, 3 val, 3 test) on `group_id`. Writes the `split` column, `train/val/test.txt` and `data.yaml`.
9. [x] **EDA** (done; figures in `reports/eda/`): [x] class balance, [x] box sizes, [x] boxes per image and empty images, [x] per-source differences and image properties (size, brightness, blur, visual samples).
10. [ ] **Visual label audit** (in progress): [x] browse in FiftyOne (`src/fiftyone_app.py`), [x] estimate the error rate from a random sample (`src/audit_sample.py`, sheets in `reports/audit/`; results below), [ ] model-assisted cleanup after the baseline (see findings below); fix errors in CVAT or Label Studio only if needed.
11. [ ] **Automate**: a single `make data` (or `just`) rebuild, plus pre-commit and ruff.
12. [x] **Train**: `src/train.py` (YOLO11s). Runs A and B on a Colab T4 (~2 h each, batch 16). Run C (stronger scale augmentation) was dropped after B showed the close-up data does not hurt dashcam results. The local GTX 1650 (4 GB) also trains with the CUDA build of PyTorch, but AMP is disabled on this card and batch 16 runs out of memory (it falls back to 4), so 30 epochs would take ~8 h there.
13. [x] **Own-domain adaptation (boxes)**: record video, label frames (`src/own_frames.py` + Label Studio), score (`src/own_eval.py`), fine-tune (`src/finetune_own.py`). Result: recall stayed under a third, so the approach changed to step 14.
14. [ ] **Tunisian segmentation dataset (TN)** (in progress): `src/tn.py`, polygon masks, 5 classes, model-assisted labelling rounds. See the section below.
15. [ ] **Stage 2 (optional)**: severity classifier on box crops.

## Known risks and experiment plan

What the EDA showed, ranked by how much it threatens a model used on real dashcam video:

1. **Two kinds of potholes.** Pothole sizes have two humps: dashcam sources (Kaggle, RDD, Pothole Mix: medians 30-58 px) and close-up sources (Mendeley 141 px, Pothole Videos 314 px). Pothole Videos alone is 34% of pothole boxes, always has exactly 1 box per image and no empty images, so it can dominate what "pothole" means and flatter test scores.
2. **Source fingerprints (shortcut learning).** Orange car hood in every Kaggle frame (and every manhole comes from Kaggle), "DNIT" logo in part of Pothole Mix, date stamps and grayscale images in Mendeley.
3. **All empty images come from RDD** (6,230 images, 35% of the data). The empty share is above the 0-10% background-image range that Ultralytics suggests, so the concern is variety, not quantity: false alarms on non-RDD-looking roads are untested.
4. **Manholes are rare and small** (954 boxes, median 33 px, 49% under 32 px) and come from one source. Manholes are not the goal (severe damage is), but the class is kept as a distractor: unlabeled, a dark round cover would be learned as background and could turn into pothole false alarms. Report it separately and treat pothole and crack results as the main metrics.
5. **Label noise:** crowded Kaggle images (up to 13 boxes, 36% with 3+) probably have missing labels, and box styles differ per source (very large RDD crack boxes). To check in the FiftyOne audit (step 10).

### Results on public validation data

Run A (YOLO11s, 30 epochs, imgsz 640, default augmentation, Colab T4, 2.15 h; inference 12.8 ms/image), mAP@0.5 on the validation lists:

| Validation list | all | pothole | crack | manhole |
|---|---|---|---|---|
| val (all sources, 2,645 images) | 0.546 | 0.706 | 0.327 | 0.602 |
| val_dashcam (RDD + Kaggle, 1,882 images) | 0.446 | **0.406** | 0.328 | 0.605 |

- **Potholes score 0.71 on all sources but only 0.41 on dashcam images.** The close-up sources inflate the pothole number, as the EDA predicted, so `val_dashcam` is the number to trust.
- Crack is the weakest class (recall 0.26), consistent with the label noise found in the audit.
- Validation mAP was still rising at epoch 30, so all runs are under-trained; A/B/C use the same 30 epochs so they stay comparable.
- The first class-agnostic figure (0.406) was lower than the class-aware one because the same object predicted as two classes counted as a false positive; `src/train.py` now uses class-agnostic NMS for that evaluation (`--eval-only` re-evaluates saved weights).

Run B (same setup as A, trained **without** Pothole Videos, `train_no_videos`), mAP@0.5:

| Run | val (all) | val_dashcam | pothole (dashcam) | crack (dashcam) | manhole (dashcam) | dashcam, class-agnostic* |
|---|---|---|---|---|---|---|
| A (all data) | 0.546 | 0.446 | 0.406 | 0.328 | 0.605 | 0.406 |
| B (no Pothole Videos) | 0.473 | 0.452 | 0.386 | 0.325 | 0.644 | 0.395 |

\*measured with the older code (no class-agnostic NMS), the same for A and B, so the two are comparable.

- On **all sources** B is much worse (potholes 0.71 to 0.45), because the validation set contains the easy close-up frames B never saw.
- On the **dashcam-only list, which is what counts, the difference is within noise** (mAP 0.446 vs 0.452; pothole -0.02, manhole +0.04 on only 157 manhole boxes; one run each). Pothole Videos neither help nor hurt dashcam performance measurably, so it stays in the training set, but only `val_dashcam` numbers are reported.
- Confusion matrix of run A on `val_dashcam` (conf 0.25): only 3% of manholes are predicted as potholes, so the frequent manhole-as-pothole mistakes seen on the own video are **domain shift** (every training manhole comes from Kaggle: forward camera, orange hood), not a general inability to tell them apart. Recall is low: 34% of potholes and 30% of cracks are found at conf 0.25.

**Decision on Pothole Videos: keep it.** Run B (trained without it) was no better on dashcam validation and clearly worse on my own footage, so the close-up data does not distort the model. Run C (stronger scale augmentation) was therefore dropped.

Extra levers if needed: keep fewer Pothole Videos frames (every 16th instead of 8th), colour/brightness/blur augmentation, crop the hood from Kaggle, oversample RDD empties in `train.txt` only (never in val/test), and later copy-paste augmentation (paste masked close-up potholes, shrunk, onto empty RDD roads). Unlabeled data (RDD Japan, PathCare) can later supply candidate empty images: a first model screens them and candidates are verified in FiftyOne.

## Label audit findings (FiftyOne, first pass)

- **Manholes** looked correct.
- **Potholes are often labeled as cracks**, in both RDD and Kaggle. This is partly label ambiguity, not only mistakes: alligator cracking (RDD D20) is the stage before potholes and often contains them, Kaggle's "crack" covers broad damaged or patched surface, and annotators disagree. Only about 3% of crack boxes overlap a pothole box in the same image (RDD 42 of 1,292, Kaggle 41 of 924), so the problem is potholes labeled *only* as cracks.
- Several RDD crack boxes are very large rectangles over plain road with no visible damage (bad fit).
- The error rate is unknown until measured: the app browsing was not random. `src/audit_sample.py` draws a fixed random sample (seed 2026) of 40 crack boxes per source into numbered sheets (`reports/audit/sheet_*.jpg`) plus `reports/audit/crack_sample.csv` (which also records the hidden RDD code D00/D10/D20 of each box). A reviewer labels each box pothole / crack / unclear and the rates are estimated per source and per code.

  **Result** (40 boxes, 20 per source, 95% Wilson intervals; verdicts in `reports/audit/crack_sample.csv`):

  | Source (crack boxes) | fair crack | pothole | manhole | no visible damage |
  |---|---|---|---|---|
  | Kaggle | 65% (43-82%) | **25% (11-47%)** | 5% (1-24%) | 5% (1-24%) |
  | RDD | 65% (43-82%) | 0% (0-16%) | 0% | **35% (18-57%)** |

  RDD by code: D00 7 fair / 3 no damage, D10 1 / 0, D20 5 / 4. Kaggle also had boxes containing a manhole plus crack (1), a patched pothole (1), and an extra pothole inside the crack box (2). Scaled to all boxes this is roughly 600 Kaggle "crack" boxes that are potholes (280-1,200) and roughly 1,800 RDD crack boxes with no visible damage (900-3,000). Two different problems: **Kaggle "crack" often means damaged or patched surface including potholes; RDD crack boxes are often loose or drawn on plain road.** Caveats: small sample, wide intervals, and "no visible damage" can hide hairline cracks that are hard to see in small crops. It only checks boxes that exist; it does not measure missing potholes (two were noticed).

Decisions:

- **Accept the noise for now and measure it (A + B).** Hand-relabeling thousands of boxes is not realistic; after the baseline, its confident disagreements with the labels are the best candidates to review (model-assisted cleanup, C).
- **Report two numbers**: per-class results (pothole / crack / manhole and the confusion matrix) and **class-agnostic** results (any damage, ignoring which type). Stage 1 only has to flag an anomaly, so a pothole called "crack" still flags the road; the class-agnostic number is the main Stage 1 metric.

### Results on my own footage (the real test)

A 3.2-minute 4K video of a Tunisian street, recorded by hand (`data/own_video/`, not in Git: faces and number plates). 152 frames were extracted every 1.25 s (`src/own_frames.py`), prelabeled with run A's predictions and corrected by hand in Label Studio: **148 frames, 160 boxes (65 manhole, 51 crack, 44 pothole)**. Frames are split **by time**: the last 25% of the clip is the test set, separated from the training part by a 5 s gap, so near-identical neighbouring frames cannot leak.

mAP@0.5 on those frames (`src/own_eval.py`):

| Model | own_test (38 frames) | all 148 frames | any damage (class-agnostic) |
|---|---|---|---|
| A | 0.081 | 0.079 | 0.179 |
| B (no Pothole Videos) | 0.044 | 0.060 | 0.116 |

- **The domain gap is the dominant problem**: 0.45 on dashcam validation vs 0.08 here. Different camera height, harsh sun, urban Tunisian street.
- **Manholes are the clearest failure**: the model predicted 6 where I labeled 65, and reads many of them as potholes. On dashcam validation only 3% of manholes are mistaken for potholes, so this is domain shift (every training manhole comes from Kaggle: forward camera, orange hood), not an inability to tell the classes apart.
- A beats B here, which is why Pothole Videos stays in the training set: its close-up frames resemble a low, near-ground camera.
- Caveat: 30 boxes in the test segment. These are order-of-magnitude numbers, not decimals to compare.

Fine-tuning (`src/finetune_own.py`): start from run A, train on the 110 own training frames repeated 10x mixed with 2,000 random general images (mosaic off, lr0 0.001, 15 epochs), and score on the 38 held-out test frames (30 boxes).

| Model | inference | mAP50 | recall | precision | pothole | crack | manhole |
|---|---|---|---|---|---|---|---|
| A | 640, conf 0.25 | 0.037 | 0.085 | 0.30 | 0.062 | 0.048 | 0.000 |
| A | 640, conf 0.10 | 0.070 | 0.162 | 0.14 | 0.142 | 0.070 | 0.000 |
| A_ft | 640, conf 0.25 | 0.150 | 0.176 | 0.53 | 0.000 | 0.095 | 0.355 |
| A_ft | 1280, conf 0.10 | **0.210** | **0.315** | 0.24 | 0.019 | 0.168 | 0.442 |

**The headline mAP gain is misleading.** Counting boxes at conf 0.25 on the 38 test frames: I labeled 30, run A found 7, the fine-tuned model found 5. Fine-tuning did not improve detection; it changed the class of the few detections (pothole -> manhole) to match my labels. Manhole AP went 0.005 -> 0.489 and pothole AP collapsed 0.150 -> 0.018, because my own labels are 65 manhole vs 44 pothole, so the model learned "dark round thing on this street = manhole".

**The real problem is recall: the best configuration finds under a third of the damage.** Contributing causes:

- My boxes are smaller than the public training data: pothole median 45 px at 640 input vs 103 px in training, crack 51 vs 84 px (1920x1080 frames downscaled to 640). 40% of my boxes are under 32 px.
- I labelled more thoroughly than the public annotators (faint cracks, patched surface), so the bar is higher than the training data teaches.
- Only 110 training frames from a single street.

**Practical setting for my own footage: `imgsz=1280`, `conf~0.10`** (recall 0.18 -> 0.32 for the fine-tuned model; precision drops to 0.24, acceptable when the goal is flagging). Run A gets *worse* at 1280 (recall 0.162 -> 0.059) because it was trained on larger objects; only the fine-tuned model benefits. Sweep: `runs/own_sweep.csv`.

25 more minutes of Tunisian footage is available (5 videos, 2026-09-20: urban side streets, a wide avenue, a suburban road, a highway/bridge, and one unusable clip aimed at the sky). Next: label a test set from a *different* street so a score is not memorisation of this one, and use the highway clip as a false-alarm check (smooth new asphalt should produce almost no detections).

## Tunisian segmentation dataset (TN)

Boxes on the public data did not transfer to my street (section above), so I label my own footage directly, as **polygon masks** with **5 classes**: pothole, crack, manhole, trash, other. Everything is in `src/tn.py`; `data.md` documents it in full.

**Labelling loop** (each round is cheaper than the last):

```
frames every 1.25 s -> current TN model pre-draws outlines (SAM 2.1 outlines boxes, YOLOE finds trash until the model knows it)
-> I correct them in Label Studio (tn.py serve = live model backend; click an object and SAM outlines it)
-> tn.py sync (export -> YOLO seg labels, duplicates merged, per-video time split) -> tn.py train -> next round
```

**Data now:** 555 frames extracted from 5 videos, **425 labelled** (340 train / 85 val, the last 20% of each video's labelled frames), 1,578 masks: other 699, trash 338, pothole 278, manhole 139, crack 124.

**Label cleaning.** A label audit found 1,770 pairs of same-class shapes touching each other; one frame alone held 104 stacked pothole polygons. Models trained on this drew several masks on one object. `sync` now merges same-class shapes that overlap by more than 30% of the smaller one (shapes that only touch, such as neighbouring potholes on a rail crossing, stay separate): 1,811 shapes became 1,578, and 1 borderline pair is left. On 10 fixed val frames, merging at inference removed all 5 duplicate prediction pairs without losing an object; raising conf from 0.15 to 0.25 also removed them but lost 3 of 31 objects.

**Experiments** (same split, same seed, all `best.pt` scored identically; mask mAP@0.5, val object count in brackets):

| Run | Init | Frozen | `hsv_s` | all | pothole (7) | crack (14) | manhole (19) | trash (61) | other (192) |
|---|---|---|---|---|---|---|---|---|---|
| tn4 (old labels) | tn3 | no | 0.7 | 0.169 | 0.012 | 0.078 | 0.284 | 0.151 | 0.317 |
| tn5_hsv07 | COCO seg | no | 0.7 | 0.259 | 0.346 | 0.096 | 0.301 | 0.211 | 0.343 |
| tn5_hsv035 | COCO seg | no | 0.35 | 0.286 | 0.329 | 0.184 | 0.390 | 0.176 | 0.354 |
| **tn6_freeze** | COCO seg | backbone (0-9) | 0.35 | **0.296** | 0.682 | 0.035 | 0.241 | 0.193 | 0.328 |
| tn6_initA | run A (box) | no | 0.35 | 0.232 | 0.345 | 0.034 | 0.310 | 0.096 | 0.374 |
| tn6_initA_freeze | run A (box) | backbone (0-9) | 0.35 | 0.202 | 0.309 | 0.019 | 0.273 | 0.084 | 0.324 |

- **Cleaning the labels** gave the largest gain (tn4 0.17 to tn5 0.26-0.29 with the same model size).
- **Less colour jitter** (`hsv_s` 0.35 instead of the default 0.7) helped cracks and manholes. Note that lowering `hsv_s` narrows the random saturation change; it does not desaturate the images.
- **Freezing the backbone** ties on overall score and trains in half the time (23 min on a GTX 1650); its pothole gain rests on only 7 val potholes, and cracks collapse. **tn6_freeze is the current labelling model.**
- **Starting from run A** (public road-damage detector) was worse than COCO: run A is a box model, so the mask head starts untrained, and ~340 frames are too few to learn it.
- **Limit:** val has only 7 potholes and 14 cracks, so differences of a few points are noise. Next: fully label a street the model has never seen (`20260920112134`) as a fixed test set.

## Project layout

```
classes.yaml               class map: source classes -> pothole / crack / manhole
src/
  common.py                class map, VOC->YOLO conversion, stable-ID manifest helpers
  convert_rdd.py           RDD2020 VOC XML -> YOLO
  convert_mendeley.py      water-filled/dry potholes (YOLO txt, validated)
  convert_pothole_videos.py  rgb+mask videos -> sampled frames + boxes
  convert_kaggle.py        Kaggle YOLO boxes, with rebuilt clip groups
  convert_shrec.py         Pothole Mix colour masks -> pothole boxes
  audit_images.py          read-only: md5, pHash, blur, brightness per image -> reports/image_audit.csv
  eda_01_balance.py        EDA step 1: boxes per class -> reports/eda/01_class_balance.png
  eda_02_box_size.py       EDA step 2: box sizes in px at 640 input -> reports/eda/02_box_sizes.png
  eda_03_per_image.py      EDA step 3: boxes per image, empty images -> reports/eda/03_boxes_per_image.png
  fiftyone_app.py          loads the dataset into FiftyOne (boxes, source, split, size) and opens the app
  train.py                 YOLO training + evaluation (all val, dashcam-only val, class-agnostic); runs A/B via flags
  demo_video.py            run a trained model over a video -> annotated video + detection frames
  own_frames.py            extract frames from my own video, prelabel with a model -> Label Studio import
  own_eval.py              Label Studio YOLO export -> labels + lists, score any weights on them
  finetune_own.py          fine-tune on my own frames (repeated, mixed with general images)
  tn.py                    Tunisian mask dataset: seed / frames / serve (Label Studio backend) / sync / train
  audit_sample.py          random sample of crack boxes -> numbered review sheets in reports/audit/
  eda_04_sources.py        EDA step 4: image properties per source + visual samples -> reports/eda/04_source_samples.jpg
                           (reports/eda/02b_pothole_size_by_source.png: pothole sizes coloured by source)
  clean.py                 applies the cleaning rules to manifest.csv (marks, never deletes)
  split.py                 group-based 70/15/15 split -> manifest, train/val/test.txt, data.yaml
reports/eda/               EDA figures (in Git)
reports/image_audit.csv    regenerable audit output (not in Git)
data/raw/                  original downloads (tracked by DVC, not Git)
data/own/ data/own_video/  my own footage and box labels (DVC, never Git: faces and number plates)
data/tn/                   TN frames, polygon labels, train/val lists, Label Studio JSON (DVC)
data/processed/
  images/  labels/         unified dataset, one label .txt per image (same ID)
  manifest.csv             id, source, group_id, orig_path, ext, size, box counts, orig_split,
                           group_id_orig (before merging), excluded (reason, empty = kept), split
  train.txt val.txt test.txt  image lists per split (excluded images are in none)
  data.yaml                Ultralytics dataset file (classes + lists)
.dvc/                      DVC configuration
requirements.txt
README.md
data.md                    how every dataset is captured, annotated, split, and its known problems
```

## Setup

```powershell
uv venv .venv311 --python 3.11
.\.venv311\Scripts\Activate.ps1
uv pip install -r requirements.txt
dvc pull data/processed.dvc   # downloads the unified dataset from DagsHub (3.1 GB)
dvc pull data/tn.dvc          # Tunisian mask dataset (201 MB)
```

Rebuild the processed data so far (each script supports `--dry-run` to count without writing):

```powershell
python src/convert_rdd.py
python src/convert_mendeley.py
python src/convert_pothole_videos.py   # about 15 minutes
python src/convert_kaggle.py
python src/convert_shrec.py
python src/audit_images.py             # about 2 minutes
python src/clean.py
python src/split.py
```

## Credits and licences

This project builds on public datasets. All credit for collecting and annotating the data belongs to the original authors. Please cite them if you use this work.

- **RDD2020**: D. Arya, H. Maeda, S. K. Ghosh, D. Toshniwal, Y. Sekimoto, *RDD2020: An annotated image dataset for automatic road damage detection using deep learning*, Data in Brief, 2021, DOI [10.1016/j.dib.2021.107133](https://doi.org/10.1016/j.dib.2021.107133). We use only the four official classes: D00 longitudinal crack, D10 transverse crack, D20 alligator crack, D40 pothole. Code and links: <https://github.com/sekilab/RoadDamageDetector>
- **Road Damage Dataset: Potholes, Cracks and Manholes** (Kaggle): <https://www.kaggle.com/datasets/lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes>
- **Pothole Mix** (SHREC 2022 pothole and crack segmentation data): A. Ranieri, E. Moscoso Thompson, S. Biasotti, Mendeley Data, 15 Feb 2022, DOI [10.17632/kfth5g2xk3.1](https://doi.org/10.17632/kfth5g2xk3.1), licence CC BY 4.0. It was assembled from five public datasets, which must also be credited:
  - Crack500 and GAPs384 (Yang et al., *Feature Pyramid and Hierarchical Boosting Network for Pavement Crack Detection*, IEEE T-ITS 2019). GAPs384 is **academic use only**.
  - EdmCrack600 (Mei et al., Automation in Construction 2020). **Commercial use is not allowed.**
  - Pothole-600: <https://sites.google.com/view/pothole-600>
  - Cracks and Potholes in Road Images: <https://github.com/biankatpas/Cracks-and-Potholes-in-Road-Images-Dataset>
  - CNR Road Dataset
- **An Annotated Water-Filled, and Dry Potholes Dataset for Deep Learning Applications**: J. Dib, K. Sirlantzis, G. Howells, Mendeley Data, 2 Mar 2023, DOI [10.17632/tp95cdvgm8.1](https://doi.org/10.17632/tp95cdvgm8.1), CC BY 4.0.
- **Pothole Videos**: M. Ihsan, A. Harjoko, M. A. Amrizal, Mendeley Data, 25 Mar 2024, DOI [10.17632/5bwfg4v4cd.3](https://doi.org/10.17632/5bwfg4v4cd.3), CC BY 4.0.
- **PathCare: A Dataset for Road Fault Diagnosis**: B. Abro, S. Jatoi, M. Z. Shaikh, E. Nava Baro, B. S. Chowdhry, M. Milanova, Mendeley Data, 29 Oct 2024, DOI [10.17632/6p52w7d5xd.2](https://doi.org/10.17632/6p52w7d5xd.2), CC BY 4.0.

**Licence note:** some sources restrict commercial use. This project is for personal and academic use. Check each source's terms before redistributing the data or a model trained on it.

Tools used: DVC, Ultralytics, supervision, OpenCV, FiftyOne, CleanVision, scikit-learn, pandas, ruff.
