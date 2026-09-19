# RoadScan

A personal computer-vision project: build one clean, unified, leakage-safe dataset for **road damage detection** from six public sources, then train a detector on it.

> Status: data preparation phase. Raw data is downloaded and versioned with DVC; conversion, cleaning and splitting come next.

## Goal

Detect three kinds of road anomaly in images and dashcam video:

| ID | Class   |
|----|---------|
| 0  | pothole |
| 1  | crack   |
| 2  | manhole |

The sources use different label formats and class names, so most of the work is turning them into a single consistent dataset that can be rebuilt with one command.

## Data sources

| Source | Content | Label format | Classes |
|---|---|---|---|
| **RDD2020** (Czech, India, Japan) | Dashcam images | Pascal VOC XML | `D00` `D10` `D20` `D40` (+ rare codes) |
| **Water-Filled and Dry Potholes** (Mendeley) | Images + 2 dashcam videos | Pascal VOC XML and YOLO txt | pothole |
| **Potholes, Cracks and Manholes** (Kaggle) | 640x360 images | YOLO boxes, quadrilaterals, COCO JSON | pothole, crack, manhole |
| **Pothole Mix / SHREC 2022** | Images + RGB-D videos | Binary segmentation masks | pothole, crack |
| **Pothole Videos** (Mendeley) | Videos with `rgb/` and `mask/` | Masks | pothole |
| **PathCare** (Mendeley) | Images + videos of road faults | None found | unlabeled |

Known gaps found during inspection:

- RDD2020 annotations exist only for **train/Czech** and **train/India**. Japan train images have no XML here, and the test splits are unlabeled by design.
- Manholes exist only in the Kaggle set, so class 2 will be small.
- PathCare and Japan train are unlabeled and are kept aside (possible pseudo-labeling later).

## Pipeline plan

1. **Environment**: Python + uv, Git, VS Code.
2. **Download**: Kaggle CLI for Kaggle, browser/wget for Mendeley, GitHub links for RDD2020.
3. **Version data**: DVC tracks `data/raw`; Git tracks only the small `.dvc` pointer files.
4. **Inventory**: count files and read labels per source.
5. **Unify classes**: one `classes.yaml` mapping every source class to `pothole` / `crack` / `manhole`.
6. **Convert formats**: VOC to YOLO (supervision), masks to YOLO via OpenCV `findContours`, COCO/quads to YOLO. Videos to frames with `ffmpeg -i in.mp4 -vf fps=4 out/%05d.jpg`.
7. **Clean**: remove duplicates (imagehash), flag blurry/dark images (CleanVision).
8. **Split without leakage**: `StratifiedGroupKFold` / `GroupShuffleSplit`, grouping by video or clip id so frames of one clip never straddle train and test.
9. **EDA**: class balance, box sizes, per-source statistics (pandas, matplotlib).
10. **Visual label audit**: browse labels per source in FiftyOne; fix errors in CVAT or Label Studio only if needed.
11. **Automate**: a single `make data` (or `just`) rebuild, plus pre-commit and ruff.
12. **Train**: baseline YOLO on Kaggle Notebooks or Colab GPU.

Open design decision: **boxes (detection) vs polygons (segmentation)**. Crack masks are thin and diagonal, so boxes from them are poor labels. This is decided before the conversion step.

## Project layout

```
data/raw/        original downloads (tracked by DVC, not Git)
data/processed/  unified YOLO dataset (planned)
.dvc/            DVC configuration
requirements.txt
README.md
```

## Setup

```powershell
uv venv --python 3.11
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
git init
dvc init
dvc pull      # once a remote is configured
```

## Credits and licences

This project builds on public datasets. All credit for collecting and annotating the data belongs to the original authors. Please cite them if you use this work.

- **RDD2020**: D. Arya, H. Maeda, S. K. Ghosh, D. Toshniwal, H. Omata, T. Kashiyama, Y. Sekimoto, *RDD2020: An annotated image dataset for automatic road damage detection using deep learning*, Data in Brief, 2021. Code and links: <https://github.com/sekilab/RoadDamageDetector>
- **Road Damage Dataset: Potholes, Cracks and Manholes** (Kaggle): <https://www.kaggle.com/datasets/lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes>
- **Pothole Mix / SHREC 2022 (pothole and crack detection using images and RGB-D data)**: assembled by the SHREC 2022 organisers from five public datasets, which must also be credited:
  - Crack500 and GAPs384 (Yang et al., *Feature Pyramid and Hierarchical Boosting Network for Pavement Crack Detection*, IEEE T-ITS 2019). GAPs384 is **academic use only**.
  - EdmCrack600 (Mei et al., Automation in Construction 2020). **Commercial use is not allowed.**
  - Pothole-600: <https://sites.google.com/view/pothole-600>
  - Cracks and Potholes in Road Images: <https://github.com/biankatpas/Cracks-and-Potholes-in-Road-Images-Dataset>
  - CNR Road Dataset
- **An Annotated Water-Filled, and Dry Potholes Dataset for Deep Learning Applications** (Mendeley Data). Authors and DOI: TODO, copy from the dataset page.
- **Pothole Videos** (Mendeley Data). Authors and DOI: TODO.
- **PathCare: A Dataset for Road Fault Diagnosis** (Mendeley Data). Authors and DOI: TODO.

**Licence note:** some sources restrict commercial use. This project is for personal and academic use. Check each source's terms before redistributing the data or a model trained on it.

Tools used: DVC, Ultralytics, supervision, OpenCV, FiftyOne, CleanVision, scikit-learn, pandas, ruff.
