# CHROMA — Production validation and render operations

## What is actually delivered
- Render chunks are leased atomically in SQLite and retried up to `--max-attempts` (default 3). Expired leases may be reassigned. Shared-storage workers should use a trusted filesystem with reliable SQLite locking.
- `--gpu 0` advertises a GPU-affinity worker and sets `CUDA_VISIBLE_DEVICES=0` before the rendering pipeline executes. This is **GPU eligibility scheduling**, not VRAM-aware allocation or proof that every processing node runs on GPU.
- `--warmup 3` processes up to three preceding frames before writing a chunk, reducing temporal discontinuities but not guaranteeing identical results to a single uninterrupted render.
- Output frames are committed with atomic same-filesystem rename. The collector refuses missing/ambiguous frames.
- `chroma_quality.py` measures matte MAE/MSE/IoU, real video *decode-only* FPS, and optional OCIO roundtrip numerical error.

## Render workers
```
python chroma_farm.py worker --root /shared/chroma --gpu 0 --max-attempts 3 --warmup 6
python chroma_farm.py worker --root /shared/chroma --max-attempts 3
python chroma_farm.py status --root /shared/chroma --job JOB_ID
python chroma_farm.py collect --root /shared/chroma --job JOB_ID --output frames.zip
```
Keep the render share accessible on all machines with identical footage paths and processing dependencies. Do not expose the shared database to untrusted users. Chunk retries are bounded; investigate permanently failed chunks before resubmitting.

## Validation with your footage and ground truth
```
python chroma_quality.py --pred output_mattes --truth verified_reference_mattes --output matte_report.json
python chroma_quality.py --video your_actual_4k_or_8k_clip.mov --frames 120 --output decode_report.json
python chroma_quality.py --ocio-config /path/to/config.ocio --source-space 'ACEScg' --destination-space 'sRGB' --output color_report.json
```
Only claim 4K/8K performance after testing genuine clips on your target hardware. Decode FPS is **not** end-to-end render throughput. OCIO tests require PyOpenColorIO and a valid user-provided configuration.

## Neural model readiness
CHROMA supports loading compatible ONNX matte models, but **this archive does not contain newly trained or independently validated neural weights**. Training and validation require legally usable datasets, labelled hair mattes, a specified model architecture, GPU compute, held-out test scenes, and published quantitative metrics. Do not represent untested third-party weights as a validated CHROMA model.

## Known limits
There is no cluster-wide VRAM monitoring, network queue service, model training pipeline, production-grade authentication, or end-to-end 8K stress-test result. The existing UI and some processing paths still depend on OpenCV's decoding behavior. Verify OCIO transforms, color gamut, alpha premultiplication, temporal behavior, and codec metadata before production delivery.
