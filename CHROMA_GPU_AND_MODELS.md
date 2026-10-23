# CHROMA GPU scheduling and model validation

## GPU scheduling

`chroma_gpu.py` queries NVIDIA free VRAM using `nvidia-smi`. The queue now has a `required_vram_mib` column. Jobs can be submitted from Python with `chroma_farm.submit(..., required_gpu=True, required_vram_mib=6000)` and workers started with `python chroma_farm.py worker --root /shared/chroma --gpu 0 --reserve-vram-mib 1024`.

The scheduler checks currently reported free memory before claiming a job. This is **admission control, not a reservation**; concurrent GPU processes can still consume VRAM. It supports NVIDIA `nvidia-smi` only. Actual neural GPU execution depends on the installed inference runtime and its providers. Workers should be configured one per physical GPU; shared SQLite on network filesystems may require a dedicated database service for robust production use.

## Model provenance and validation

`python chroma_models.py register --model /models/matting.onnx --manifest model.json --source 'your model origin' --license 'your license' --training-dataset 'your dataset' --validation-dataset 'your validation dataset'`

`python chroma_models.py verify --manifest model.json`

`python chroma_models.py evaluate --pred /path/to/predicted_png_masks --truth /path/to/reference_png_masks --output metrics.json`

A model is **not** marked trained or validated just because it is registered. Training a model requires an appropriate licensed dataset, GPU compute, a compatible training architecture, and held-out testing. The existing CHROMA neural matting interface accepts a compatible ONNX model; no newly trained weights are included in this package.
