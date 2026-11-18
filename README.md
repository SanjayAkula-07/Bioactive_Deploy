---
title: BioPredict
emoji: 🧬
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# BioPredict – COX-2 bioactivity prediction (XGBoost + RDKit)

FastAPI backend + vanilla JS frontend. Deploy on Hugging Face Spaces (Docker) or Render.
Local run: `pip install -r requirements.txt && uvicorn backend.main:app --port 8000`

## Pages
`/app/` → login.html → predict.html (single molecule + 3D) · batch.html · benchmarks.html · docs.html. The previous plain UI is kept at `/app/classic.html`.
Login is a front-end demo gate (any email/password/org); there is no server-side authentication.
