# CHROMA Render Farm and Neural Detail

The Render Farm tab creates real frame-chunk jobs in a SQLite-backed queue. Workers running on multiple machines use a common mounted filesystem and claim jobs with transaction-protected leases. Start each worker with `python chroma_farm.py worker --root /shared/chroma`. Submit jobs from the app with that exact shared path, check progress, and collect a ZIP of rendered frames when complete.

**Limitations:** Shared filesystem required, not a network service. The input footage and background must be available at the same absolute paths on all machines. Workers process 8-bit decoded RGB frames even for 16-bit output, so this farm path does not preserve high-bit-depth source precision. Tracking state resets at chunk boundaries; verify those frames. Long chunks can exceed the 30-minute lease and be reassigned; size chunks accordingly. No GPU scheduling, retry UI, cloud provisioning, or automatic video encoding.

The Neural Detail tab accepts a locally supplied ONNX model with NCHW RGB input and a single-channel alpha output. The output is fused into uncertain boundaries and edge-filtered. Install `onnxruntime`. The ONNX model must use [0,1] RGB normalization; other models may need an adapter. **This is model-based hair matte refinement, not reconstruction of lost strands**, and no model weights are bundled.

These additions are useful production building blocks, but the app has not passed a studio-grade validation suite or long-duration distributed load testing.
