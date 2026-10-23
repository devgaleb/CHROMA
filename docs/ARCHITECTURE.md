# Architecture

`app.py` exposes CHROMA's Gradio workspace. `app_key.py` provides color keying. `chroma_pipeline.py` performs per-frame finishing and export. `chroma_advanced.py` supplies manual correction and detail processing. Optional neural backends live in `chroma_ai.py`, `chroma_pretrained.py` and `chroma_hair.py`. Render farm submission and workers use `chroma_farm.py`, with hardware probing in `chroma_gpu.py`. Quality and model tools are separated into `chroma_quality.py` and `chroma_models.py`.

For new contributions, keep inference backends optional, report errors rather than silently falling back, and preserve source bit depth and color metadata where feasible. Tests should include transparent pixels, hair-edge gradients, moving foregrounds, scene cuts, failed workers and frame-boundary conditions.
