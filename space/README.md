---
title: RoadScan Label
emoji: 🛣️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

Private labeling Space for RoadScan: Label Studio + the current tn model (pre-drawing and SAM click-to-mask).
Built by `python space/stage.py` in the RoadScanCV repo. Open it at the direct URL `https://<owner>-<space>.hf.space`.

Secrets (Settings -> Variables and secrets):
- `LABEL_STUDIO_USERNAME` (an email) and `LABEL_STUDIO_PASSWORD`: the only account; sign-up is disabled.
- `POSTGRE_HOST`, `POSTGRE_PORT` (5432), `POSTGRE_NAME`, `POSTGRE_USER`, `POSTGRE_PASSWORD`: a free Postgres
  (e.g. Neon). Without them the Space refuses to start, because its disk is wiped on every restart.
