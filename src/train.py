"""Train a YOLO baseline and evaluate it the way the experiment plan needs.

Runs (see README):
  A  python src/train.py --run A
  B  python src/train.py --run B --train train_no_videos
  C  python src/train.py --run C --scale 0.9
  smoke test (CPU, tiny, ~minutes):  python src/train.py --run smoke --smoke

Evaluation after training, on the best weights:
  val            all sources
  val_dashcam    RDD + Kaggle only (the real use case)  <- the number that decides A vs B vs C
  val_dashcam, class-agnostic (single class 'damage')   <- the main Stage 1 metric
Add --test to evaluate the test lists too (do this once, at the very end).

On Kaggle/Colab pass --data-dir to the folder that holds images/, labels/ and the *.txt lists.
"""
import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
NAMES = {0: "pothole", 1: "crack", 2: "manhole"}


def make_yaml(data_dir, train, val, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump({"path": str(data_dir), "train": f"{train}.txt", "val": f"{val}.txt",
                                   "names": NAMES}, sort_keys=False))
    return str(out)


def limit(data_dir, name, n, tmp):
    """First n lines of a list (smoke test only)."""
    lines = (data_dir / f"{name}.txt").read_text().split()[:n]
    (data_dir / f"{name}_{tmp}.txt").write_text("\n".join(lines) + "\n")
    return f"{name}_{tmp}"


def evaluate(model, data_dir, run_dir, val, imgsz, batch, device, single_cls=False):
    r = model.val(data=make_yaml(data_dir, "train", val, run_dir / f"eval_{val}.yaml"), imgsz=imgsz,
                  batch=batch, device=device, single_cls=single_cls, plots=False, verbose=False)
    row = {"list": val + (" (class-agnostic)" if single_cls else ""), "mAP50": r.box.map50, "mAP50-95": r.box.map}
    if not single_cls:
        for i, c in enumerate(r.box.ap_class_index):
            row[NAMES[int(c)]] = r.box.ap50[i]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--train", default="train", help="train list name: train or train_no_videos")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--scale", type=float, default=0.5, help="augmentation zoom range (Ultralytics default 0.5)")
    ap.add_argument("--device", default=None, help="0 for the first GPU, cpu, or empty for auto")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run to check the pipeline")
    ap.add_argument("--test", action="store_true", help="also evaluate the test lists")
    a = ap.parse_args()

    dd, run_dir = a.data_dir.resolve(), ROOT / "runs" / a.run
    lists = {"val": "val", "val_dashcam": "val_dashcam"}
    fraction = 1.0
    if a.smoke:
        a.epochs, a.imgsz, a.batch, a.device = 1, 320, 8, "cpu"
        lists = {k: limit(dd, v, 60, "smoke") for k, v in lists.items()}
        a.train = limit(dd, a.train, 400, "smoke")

    model = YOLO(a.model)
    model.train(data=make_yaml(dd, a.train, lists["val"], run_dir / "data.yaml"), epochs=a.epochs,
                imgsz=a.imgsz, batch=a.batch, scale=a.scale, device=a.device, fraction=fraction,
                project=str(ROOT / "runs"), name=a.run, exist_ok=True, workers=2, plots=False)

    best = YOLO(str(ROOT / "runs" / a.run / "weights" / "best.pt"))
    rows = [evaluate(best, dd, run_dir, lists["val"], a.imgsz, a.batch, a.device),
            evaluate(best, dd, run_dir, lists["val_dashcam"], a.imgsz, a.batch, a.device),
            evaluate(best, dd, run_dir, lists["val_dashcam"], a.imgsz, a.batch, a.device, single_cls=True)]
    if a.test:
        for t in ("test", "test_dashcam"):
            rows.append(evaluate(best, dd, run_dir, t, a.imgsz, a.batch, a.device))
    import pandas as pd
    df = pd.DataFrame(rows).round(3)
    print(f"\nRUN {a.run}: train={a.train} scale={a.scale} epochs={a.epochs} imgsz={a.imgsz}\n{df.to_string(index=False)}")
    df.to_csv(run_dir / "results.csv", index=False)


if __name__ == "__main__":
    main()
