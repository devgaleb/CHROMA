# CHROMA production acceptance

This release adds a repeatable local acceptance command and GPU queue regression tests. Passing these tests is **not** certification of 4K/8K, neural hair quality, OCIO/ACES correctness, or a multi-node render farm.

## Run

```bash
python -m pip install -r requirements.txt
python chroma_release_check.py --output chroma_acceptance.json
python chroma_release_check.py --video /path/to/real/4k_or_8k.mov --ocio-config /path/to/config.ocio --output workstation.json
```

The report records exact environment and measured decoding speed. It deliberately marks full render, model inference, and source bit depth unverified unless separate tests are conducted.

## Visual acceptance protocol

1. Capture the actual running UI at 1366×768, 1920×1080 and a narrow viewport. Check clipping, contrast, keyboard focus, disabled controls and file picker behavior.
2. Test footage containing flyaway hair, transparent fabric, fast motion, motion blur, green spill, low light and a subject crossing the frame edge.
3. Compare foreground detail against human-reviewed reference mattes, checking edge chatter and temporal flicker at normal playback speed.
4. Render the same sequence on one workstation and two independent workers; compare frame hashes and inspect chunk boundaries.
5. Verify OCIO transforms with a known studio configuration and calibrated reference frames; check linear/nonlinear conversions and round trips within a specified tolerance.
6. Run complete 4K and 8K renders on target GPUs, record peak VRAM, throughput, errors, thermal throttling and output validation.
7. Test real licensed neural checkpoints; record model hash, training data provenance, accuracy metrics and failure cases.

Do not publish performance or quality claims without the resulting evidence and reproducible configurations.
