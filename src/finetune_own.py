"""Fine-tune run A on your own labeled frames, then score on your held-out test frames.

Your 110 training frames would vanish among 12k general images, so they are repeated REPEAT
times and mixed with a random sample of the general train list (so the model does not forget
the rest). Your test frames (last part of the clip, after a time gap) are never trained on.

    python src/finetune_own.py --device 0
"""
import argparse
import random
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
OWN, PROC = ROOT / "data" / "own", ROOT / "data" / "processed"
NAMES = {0: "pothole", 1: "crack", 2: "manhole"}

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="A_ft")
ap.add_argument("--train-lists", nargs="+", default=["own_train.txt"], help="frame lists, relative to data/own")
ap.add_argument("--val-list", default="own_test.txt", help="held-out frame list, relative to data/own")
ap.add_argument("--weights", default="runs/A/weights/best.pt")
ap.add_argument("--repeat", type=int, default=3, help="10 made own frames a third of the mix and the model collapsed onto 'manhole'")
ap.add_argument("--freeze", type=int, default=10, help="freeze the backbone: keeps general knowledge, learns only the new appearance")
ap.add_argument("--general", type=int, default=2000, help="random general images mixed in (0 = own frames only)")
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--lr0", type=float, default=0.001, help="lower than training from scratch: keep what run A learned")
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--device", default="0")
a = ap.parse_args()

run_dir = ROOT / "runs" / a.run; run_dir.mkdir(parents=True, exist_ok=True)
absolute = lambda lst: [f"{(OWN / lst).parent}/{l.lstrip('./')}" for l in (OWN / lst).read_text().split()]
own_train = [x for lst in a.train_lists for x in absolute(lst)]
general = [f"{PROC}/{l.lstrip('./')}" for l in (PROC / "train.txt").read_text().split()]
random.seed(0)
mix = own_train * a.repeat + random.sample(general, min(a.general, len(general)))
random.shuffle(mix)
(run_dir / "train_mix.txt").write_text("\n".join(mix) + "\n")
(run_dir / "own_test_abs.txt").write_text(
    "\n".join(absolute(a.val_list)) + "\n")

data = run_dir / "data.yaml"
data.write_text(yaml.safe_dump({"path": str(run_dir), "train": "train_mix.txt", "val": "own_test_abs.txt", "names": NAMES}))
print(f"train: {len(own_train)} own frames x{a.repeat} + {min(a.general, len(general))} general = {len(mix)} images")

if __name__ == "__main__":
    YOLO(a.weights).train(data=str(data), epochs=a.epochs, imgsz=640, batch=a.batch, lr0=a.lr0, device=a.device,
                          freeze=a.freeze, workers=1, project=str(ROOT / "runs"), name=a.run, exist_ok=True, plots=False,
                          mosaic=0.0,      # mosaic shrinks objects; your frames already show damage at real scale
                          close_mosaic=0)
    best = YOLO(str(run_dir / "weights" / "best.pt"))
    import pandas as pd
    rows = []
    for single in (False, True):
        r = best.val(data=str(data), imgsz=640, batch=a.batch, workers=1, device=a.device, half=False,
                     single_cls=single, agnostic_nms=single, plots=False, verbose=False)
        row = {"list": a.val_list + (" (any damage)" if single else ""), "mAP50": round(r.box.map50, 3)}
        if not single:
            row.update({NAMES[int(c)]: round(float(r.box.ap50[i]), 3) for i, c in enumerate(r.box.ap_class_index)})
        rows.append(row)
    df = pd.DataFrame(rows); print(df.to_string(index=False)); df.to_csv(run_dir / "metrics.csv", index=False)
