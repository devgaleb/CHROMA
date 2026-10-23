# CHROMA — shot delivery and production readiness

## Sequence validation and packaging

```bash
python chroma_delivery.py /shots/010/comp/v003 --start 1001 --end 1240
python chroma_delivery.py /shots/010/comp/v003 --start 1001 --end 1240 --shot SH010 --version v003 --fps 24 --output SH010_v003.zip
```

The package contains a manifest with frame numbering, byte lengths, and SHA-256 checksums. Missing frames and ambiguous naming block delivery. Archives are written atomically and use ZIP64. This is a **shot delivery utility**, not a render-farm scheduler or proof of film-studio certification.

## Production acceptance gates

1. **Footage**: use real difficult 4K/8K plates, hair flyaways, motion blur, smoke, glass, and changing illumination; document permissions and source color metadata.
2. **Matte quality**: compare against independently prepared ground-truth alpha mattes; report MAE and boundary-region metrics per shot and per scene.
3. **Temporal**: visually inspect every frame and measure motion-compensated alpha instability, especially around cuts and occlusions.
4. **Color**: pin a versioned OCIO/ACES config; verify reference patches and EXR round trips in linear scene-referred space; reject clipped highlights or wrong transfer functions.
5. **Rendering**: run complete sequence tests on the intended production GPU fleet, with restarts, failed worker recovery, VRAM contention, and checksum verification.
6. **Delivery**: validate consecutive frame numbering, expected frame range, naming, frame rate, source provenance, and package integrity.

No neural weights, external validation datasets, or studio GPU certification are bundled with this release. These require real model licensing, training, and testing.
