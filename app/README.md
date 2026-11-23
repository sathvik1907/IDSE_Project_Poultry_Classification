---
title: Poultry Fecal-Image Screening
emoji: 🐔
colorFrom: yellow
colorTo: green
sdk: gradio
sdk_version: 6.31.0
python_version: "3.13"
app_file: app.py
pinned: false
---
# Poultry Disease Screening from Fecal Images
Four-class screening (Coccidiosis / Healthy / Newcastle disease / Salmonellosis) with a poultry-vs-other-animal rejection gate.
Backbone: **EfficientNetB0** - test accuracy 98.47% (leakage-free split, n=983).
Research prototype - not a substitute for laboratory diagnosis.

## Deploy
* **Hugging Face Spaces (free CPU):** create a *Gradio* Space and upload this whole folder (web UI "Upload files" handles the large model file via LFS automatically), or run the final notebook cell with `CONFIG['push_to_hf_space']` set.
* **Docker (Render / Railway / Cloud Run / any VM):** `docker build -t poultry . && docker run -p 7860:7860 poultry`
* **Local:** `pip install -r requirements.txt && python app.py` -> http://localhost:7860
* Set `USE_CLIP=0` to run without the CLIP gate (faster start, ~1 GB less RAM).
