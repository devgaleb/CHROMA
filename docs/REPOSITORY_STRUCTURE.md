# Repository layout

- `app.py` — CHROMA user interface and basic keying workflow
- `app_key.py` — foreground hint generation
- `chroma_pipeline.py` — compositing and export stages
- `chroma_advanced.py` — rotoscoping tools
- `chroma_pretrained.py` / `chroma_temporal_export.py` — optional temporal inference
- `chroma_farm.py` / `chroma_gpu.py` — distributed rendering and GPU scheduling
- `chroma_models.py` / `chroma_quality.py` — model registry and quality checks
- `assets/` — CHROMA interface styling
- `tests/` — automated checks
- `docs/` — technical documentation

No historical application repositories are bundled. Optional models must be installed separately under their respective licences.
