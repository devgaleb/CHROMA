# Changelog

## Open Source Edition — 2026-10-23

- Replaced the root README with installation, feature and architecture guidance.
- Added contributor, security, third-party attribution and capability documentation.
- Added dependency diagnostics, repository housekeeping, CI and basic tests.
- Preserved existing processing modules and bundled engine source without claiming new model training.

## UI and release testing update — 2026-10-23
- Reworked studio visual identity with a responsive dark interface and separate stylesheet.
- Isolated still and video output paths to avoid overwriting files from other jobs.
- Added automated still/video alpha export integration tests.
- Added documented release acceptance checklist and limitations.

### Production acceptance update
- Added environment-specific JSON acceptance report with optional footage decoding and OCIO config checks.
- Added GPU admission, lease retry and VRAM-reserve regression tests.
- Documented required visual, neural, color and distributed-hardware acceptance criteria.

## Sequence QC update
- Added an interactive sequence-inspection workspace for matte-change review.
- Added shot-change detection and per-frame uncertainty metrics with downloadable JSON.
- Added a regression test for three-frame report generation.

## Studio delivery gate
- Added cryptographic verification of packaged frame contents.
- Added configurable shot review, quality-threshold, frame-count and color-provenance checks.
- Added regression tests for rejected approvals and corrupted frames.

### Finishing and sequence integrity update
- Added optional edge-colour decontamination stage to the CHROMA node pipeline.
- Added per-frame SHA-256 sequence manifests and reference-matte quality reports.
- Added regression tests for edge processing, matte metrics, and sequence manifests.
