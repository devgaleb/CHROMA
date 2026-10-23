# CHROMA

**Open-source tools for green-screen keying, rotoscoping, compositing and render automation.**

CHROMA is a Python-based VFX workstation for image and video keying. It combines an interactive Gradio interface with frame processing, matte refinement, compositing, video export and optional neural processing. The application runs locally; footage does not need to be uploaded to an external service.

> **Project status:** active development. This repository is a working source release, not a certified film-production finishing product. Some advanced workflows require optional dependencies, licensed pretrained weights, and manual configuration. Read [Capabilities](docs/CAPABILITIES.md) before using it in a production pipeline.

## What you can do

- Key green and blue screens; adjust matte tolerance, feathering, edge contraction and spill suppression.
- Refine frames with painted correction mattes and guided edge filtering.
- Compose with exposure, contrast, saturation, light wrapping and background replacement.
- Export PNG matte sequences, EXR when supported, MP4 and ProRes workflows where FFmpeg codecs are available.
- Run optional temporal neural matte inference with separately obtained compatible model weights.
- Distribute frame chunks to shared-storage workers; schedule GPU-labelled work with memory admission checks.
- Inspect output quality and run repeatable performance diagnostics.

## Quick start

**Requirements:** Python 3.10–3.12 recommended; FFmpeg and FFprobe for video export; a modern browser for the local UI.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open the local address printed by Gradio. Optional AI, EXR and OCIO features require separate installation; see [Installation](docs/INSTALLATION.md). The UI can run without an NVIDIA GPU; performance varies with resolution and hardware.

## Project layout

```text
app.py                      Main CHROMA interface
app_key.py                  Core chroma-key operations
chroma_pipeline.py          Compositing and export
chroma_advanced.py          Rotoscoping and edge tools
chroma_pretrained.py        Temporal model adapter
chroma_hair.py              Optional neural detail refinement
chroma_farm.py              Render farm coordinator and worker
chroma_gpu.py               GPU memory discovery and scheduling
chroma_models.py            Model registry and validation
chroma_quality.py           Quality and performance diagnostics
chroma_temporal_export.py   High-bit-depth matte sequences
docs/                       Setup, architecture and limitations
tests/                      Lightweight source checks
.github/workflows/          Continuous integration
```

## Quality checks

```bash
python -m compileall -q app.py app_key.py chroma_*.py
python -m unittest discover -s tests -v
python -m pip check
```

These checks catch basic regressions, but do **not** substitute for end-to-end GPU, neural-model, or 4K/8K footage validation.

## Contributing

Bug fixes, documentation, tests and reproducible footage issues are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) before opening a pull request.

## Open-source licensing

CHROMA includes code originating in multiple source projects. **There is no blanket relicensing of the entire archive.** Each included engine retains its original notices and licence terms; consult [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Before public distribution or commercial deployment, audit each component and model checkpoint and choose an appropriate licence for CHROMA-owned code. Source availability alone does not grant open-source redistribution rights.

## Known limitations

See [docs/CAPABILITIES.md](docs/CAPABILITIES.md) for feature status. Pretrained weights are not bundled, not all formats work on all systems, and the render farm is not a fully managed cluster. No unverified film-grade quality claims are made.

## Interface and release validation

The studio includes a responsive dark UI in `assets/studio.css`. Run `python -m unittest discover -s tests -v` for automated keying/export smoke tests. See [release testing](docs/RELEASE_TESTING.md) for the manual acceptance checklist and known limitations.

## Release acceptance

Run `python chroma_release_check.py --output chroma_acceptance.json` to execute regression tests and record local GPU, neural, FFmpeg and OCIO capabilities. For footage checks use `--video path/to/footage.mov`; for OCIO config loading use `--ocio-config path/to/config.ocio`. See [production acceptance](docs/PRODUCTION_ACCEPTANCE.md). Passing the local checks does not certify production hardware, pretrained models or color accuracy.

## Shot quality inspector

Run `python chroma_inspect.py footage_frame.png --output inspection.json` to
measure screen-border uniformity, uncertain matte area and edge spill. These
metrics are heuristic diagnostics, not a substitute for shot review.

## Reuse policy

This distribution is **source-available, not open source**. See [LICENSE](LICENSE)
and [COPYING_POLICY.md](COPYING_POLICY.md) before sharing or modifying CHROMA-owned
code. External packages and models retain their separate licences.

## Sequence quality control

Generate per-frame matte diagnostics and side-by-side original/matte/composite
previews for a folder of image frames:

```bash
python chroma_qc.py /path/to/frames --output /path/to/qc
```

The output includes `quality_report.json` and individual `_qc.jpg` previews.
This is a heuristic review tool, not a replacement for supervised visual QC.

## Rights and permissions

See [RIGHTS_AND_PERMISSIONS.md](RIGHTS_AND_PERMISSIONS.md) for permitted use,
third-party licence boundaries and reporting suspected unauthorised reposts.
CHROMA-owned code is source-available, not open source under the current licence.

## Motion QC Pro

The **CHROMA · Motion QC Pro** workspace compares alpha mattes after optical-flow
motion compensation, highlights potential temporal instability, and produces a
visual review sheet and JSON report. Supply a directory of consecutively named
frames. This tool is a diagnostic, not a replacement for artist review.

Run automated checks with `python -m unittest discover -s tests -v`.

### Temporal matte repair (new)

CHROMA can stabilize an existing matte sequence using backward optical flow, scene-cut
reset, and edge-disagreement protection. This is a post-processing tool, not neural
hair reconstruction. Input RGB frames and matching 8/16-bit grayscale PNG mattes
must share basenames. The output is 16-bit PNG alpha mattes.

```bash
python chroma_temporal_repair.py --frames ./frames --mattes ./mattes --output ./stable_mattes --strength 0.45
```

Strength ranges from 0 to 1. Review motion-blurred and fast-moving edges manually.

## Shot Review & Repair

The **CHROMA · Shot Review & Repair** panel adds frame selection, brush correction and optical-flow stabilization of 16-bit alpha sequences. See [shot review and validation](docs/SHOT_REVIEW_AND_VALIDATION.md). The production benchmark command measures CPU keying throughput on user-provided footage. Neural and GPU benchmarks require compatible hardware and weights.

## CHROMA finishing tools

The configurable compositor now supports an **Edge decontaminate** stage, designed to
reduce green or blue edge contamination using nearby confident foreground colours.
The stage is controlled by `decontaminate` (0–1) and `decontaminate_radius`
(pixel radius). It is not a neural hair reconstruction model and cannot recover
missing detail from an unusable plate.

For a frame-sequence integrity report (SHA-256 per frame):

```bash
python chroma_finish_cli.py ./renders/SH010 --output ./renders/SH010/report.json
```

For reference matte comparisons (MAE, MSE, RMSE, and bad-pixel ratio):

```bash
python chroma_finish_cli.py ./renders/SH010 --reference ./reference/SH010 --output evaluation.json
```

The new tools operate on actual frames, not synthetic benchmark claims. For long
films, split work by shots, use the render queue, validate each delivery, and retain
original camera media and OCIO configurations. Model checkpoints, calibrated studio
color tests, and production GPU performance evidence are **not included**.


## Swipe video comparison

The **Swipe comparison · video** workspace supports an original-vs-keyed draggable divider. Three user-provided videos are included as local demonstration media. To keep public source releases small and respect media licensing, consider excluding `assets/showcase/` from public distributions unless you have redistribution rights. See `docs/SWIPE_COMPARISON.md`.


## Large image and video uploads

CHROMA accepts image and video uploads up to **20 GB per file by default** across the editor, shot review and swipe comparison. Configure before launching:

```bash
CHROMA_MAX_UPLOAD_SIZE=100gb python app.py
```

On Windows PowerShell: `$env:CHROMA_MAX_UPLOAD_SIZE="100gb"; python app.py`. Set `CHROMA_MAX_UPLOAD_SIZE=0` to disable the **application-level** upload limit. Disk space, RAM, Gradio, reverse proxies, and hosting providers may impose other limits. A successful upload does not guarantee the entire file can be processed in memory. For large 4K/8K footage, ensure enough temporary disk space and use frame-sequence workflows where practical.
