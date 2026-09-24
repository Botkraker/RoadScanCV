"""Assemble the Hugging Face Space in space_build/: container files + one model + the frames + your annotations.

    python space/stage.py --weights runs/tn3/weights/best.pt --export data/tn/newdataset.json
    uvx --from huggingface_hub hf upload <you>/<space> space_build . --repo-type space

import_all.json = your Label Studio export reduced to frame + last submitted annotation, ready to import
into the hosted project (user ids from this PC would not exist there).
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "space_build"

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True, help="runs/<run>/weights/best.pt")
ap.add_argument("--export", required=True, help="latest Label Studio JSON export")
a = ap.parse_args()

shutil.rmtree(OUT, ignore_errors=True)
run = Path(a.weights).parents[1].name   # tn.py serve names the model after its run folder
copies = {"space/Dockerfile": "Dockerfile", "space/start.sh": "start.sh", "space/README.md": "README.md", "src/tn.py": "src/tn.py",
          a.weights: f"runs/{run}/weights/best.pt", "sam2.1_t.pt": "sam2.1_t.pt", "data/tn/label_config.xml": "data/tn/label_config.xml"}
for src, dst in copies.items():
    (OUT / dst).parent.mkdir(parents=True, exist_ok=True); shutil.copy(ROOT / src, OUT / dst)
shutil.copytree(ROOT / "data/tn/images", OUT / "data/tn/images")

tasks = []
for t in json.loads(Path(a.export).read_text(encoding="utf-8")):
    ann = [x for x in t.get("annotations", []) if not x.get("was_cancelled")]
    tasks.append({"data": t["data"], **({"annotations": [{"result": ann[-1]["result"]}]} if ann else {})})
(OUT / "data/tn/import_all.json").write_text(json.dumps(tasks))
print(f"space_build/: model {run}, {len(list((OUT / 'data/tn/images').glob('*.jpg')))} frames, "
      f"{len(tasks)} tasks ({sum('annotations' in t for t in tasks)} annotated) in data/tn/import_all.json")
