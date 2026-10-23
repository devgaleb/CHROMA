# CHROMA release testing

## Automated checks

```bash
python -m unittest discover -s tests -v
python -m compileall -q .
python chroma_doctor.py
```

The integration suite generates a green-screen foreground, verifies alpha separation, exports a still, and encodes/decodes a short test video to verify a transparent PNG sequence. These tests use synthetic footage and do **not** validate neural models, GPU rendering, professional color transforms, or distributed workers.

## Manual release acceptance checklist

- [ ] Launch the UI and inspect layouts on desktop and mobile.
- [ ] Upload an image, adjust key controls, and inspect hair and semi-transparent edges.
- [ ] Upload a video and render a transparent PNG sequence.
- [ ] Test the manual roto workspace on real footage.
- [ ] Validate OCIO transforms with a known reference chart.
- [ ] Test EXR and ProRes 4444 round trips with FFmpeg.
- [ ] Test GPU neural inference using legally obtained model checkpoints.
- [ ] Test distributed retries and recovery with multiple machines.
- [ ] Record 4K/8K rendering throughput and peak memory on target hardware.
- [ ] Audit third-party licences before publication.

## Known limitations

The UI is built with Gradio, not a full native node-graph editor. The neural-model pipeline needs model weights and appropriate dependencies. Real-footage acceptance testing remains necessary before production deployment.
