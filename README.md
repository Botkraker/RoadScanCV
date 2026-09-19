# RoadScan

A personal computer-vision project: build one clean, unified, leakage-safe dataset for **road damage detection** from six public sources, then train a detector on it.

> Status: data preparation phase. Raw data is downloaded and versioned with DVC. Four of five labeled sources are converted into one unified dataset (**16,971 images**). The Pothole Mix pothole masks are next, then cleaning, splitting and training.

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
| Image IDs | Running number (`000001`), assigned once and never changed | Short names; source, original path and group live in `manifest.csv`. Stable IDs keep labels and splits valid across rebuilds |
| Video frames | Every 8th frame | 48-frame clips are near-identical; using all frames would make one phone-camera source 72% of the data |
| Splits | By group (video), stratified by class, made later from the manifest | Frames of one video must not straddle train and test (leakage) |
| Mask to box | Threshold at 127, remove speckle, bounding box of the largest blob | Masks stored in mp4 are lossy, so exact 255 matching fails |
| Missing clip ids (Kaggle) | Rebuild groups with pHash on consecutive frames | Prevents near-duplicate frames from leaking across splits |
| Video reading | OpenCV, not ffmpeg | No extra install; reads the rgb frame and its mask together |

## Data sources

| Source | Content | Label format | Classes |
|---|---|---|---|
| **RDD2020** (Czech, India, Japan) | Dashcam images | Pascal VOC XML | `D00` `D10` `D20` `D40` (extra codes dropped) |
| **Water-Filled and Dry Potholes** (Mendeley) | Images + 2 dashcam videos | Pascal VOC XML and YOLO txt | pothole |
| **Potholes, Cracks and Manholes** (Kaggle) | 640x360 images | YOLO boxes, quadrilaterals, COCO JSON | pothole, crack, manhole |
| **Pothole Mix / SHREC 2022** | Images + RGB-D videos | Binary segmentation masks | pothole, crack |
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
| Pothole Mix (pothole masks) | todo | | |
| Pothole Videos (frames + masks) | **done** | `src/convert_pothole_videos.py` | 619 videos, every 8th frame: 3,714 images, 1 pothole box each (bounding box of the mask blob). `group_id` = video |
| RDD2020 Japan, PathCare | unlabeled, kept aside | | |

Current unified dataset (`data/processed/manifest.csv`): **16,971 images** = 10,535 RDD2020 + 3,714 Pothole Videos + 2,009 Kaggle + 713 Mendeley.

Box totals so far: **9,512 pothole, 7,710 crack, 957 manhole**. Manholes come only from Kaggle, so class 2 is the weakest.

Update this table whenever a source is converted.

### Data quality notes from conversion

- Mendeley: 4 pothole boxes in `Pothole-261/262/268/271` lie outside the image (x up to 2609 on a 392 px image). The original XML has the same error, so these boxes are dropped.
- Pothole Videos are close-up phone shots from about 130 cm above the road, unlike the dashcam views in RDD and Mendeley. Every frame contains a pothole, so they add no background examples.
- Kaggle: 1,907 of 2,009 images are consecutive VLC video snapshots (`vlcsnap-*`) with no clip id. Consecutive frames are often near-identical (median pHash distance 16, against 30 for random pairs), so `group_id` is rebuilt by starting a new group whenever the picture changes a lot (distance > 24). Most images show the orange car hood at the bottom, a dataset bias to keep in mind.
- Images differ in size per source (RDD 600x600, Mendeley 392x806, videos 1080x1080); YOLO resizes at training time.
- Each conversion is checked against an independent count (for example RDD boxes against the raw XML counts), and mask-to-box output is checked by drawing boxes on random frames.

## Pipeline plan

Progress: `[x]` done, `[ ]` to do.

1. [x] **Environment**: Python 3.11 + uv, Git, VS Code.
2. [x] **Download** the sources (Kaggle, Mendeley, RDD2020).
3. [x] **Version data**: DVC tracks `data/raw`; Git tracks only the small `.dvc` pointer files. (`data/processed` is not versioned yet.)
4. [x] **Inventory**: count files and read the labels of every source.
5. [x] **Unify classes**: `classes.yaml`.
6. [ ] **Convert formats** (4 of 5 labeled sources done): RDD VOC, Mendeley YOLO, Pothole Videos masks and Kaggle YOLO are done. Only the Pothole Mix pothole masks (`findContours`) remain.
7. [ ] **Clean**: remove duplicates (imagehash), flag blurry/dark images (CleanVision).
8. [ ] **Split without leakage**: `StratifiedGroupKFold` / `GroupShuffleSplit` on `group_id` from the manifest.
9. [ ] **EDA**: class balance, box sizes, per-source statistics (pandas, matplotlib).
10. [ ] **Visual label audit**: browse labels per source in FiftyOne; fix errors in CVAT or Label Studio only if needed.
11. [ ] **Automate**: a single `make data` (or `just`) rebuild, plus pre-commit and ruff.
12. [ ] **Train**: baseline YOLO on Kaggle Notebooks or Colab GPU.
13. [ ] **Stage 2 (optional)**: severity classifier on box crops.

## Project layout

```
classes.yaml               class map: source classes -> pothole / crack / manhole
src/
  common.py                class map, VOC->YOLO conversion, stable-ID manifest helpers
  convert_rdd.py           RDD2020 VOC XML -> YOLO
  convert_mendeley.py      water-filled/dry potholes (YOLO txt, validated)
  convert_pothole_videos.py  rgb+mask videos -> sampled frames + boxes
  convert_kaggle.py        Kaggle YOLO boxes, with rebuilt clip groups
data/raw/                  original downloads (tracked by DVC, not Git)
data/processed/
  images/  labels/         unified dataset, one label .txt per image (same ID)
  manifest.csv             id, source, group_id, orig_path, ext, size, box counts, orig_split
.dvc/                      DVC configuration
requirements.txt
README.md
```

## Setup

```powershell
uv venv .venv311 --python 3.11
.\.venv311\Scripts\Activate.ps1
uv pip install -r requirements.txt
dvc pull      # once a remote is configured
```

Rebuild the processed data so far (each script supports `--dry-run` to count without writing):

```powershell
python src/convert_rdd.py
python src/convert_mendeley.py
python src/convert_pothole_videos.py   # about 15 minutes
python src/convert_kaggle.py
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
