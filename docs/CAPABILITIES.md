# Feature status

| Area | Current implementation | Limits |
|---|---|---|
| Green/blue keying | Local color-based matte processing | Difficult hair and reflections require manual work |
| Rotoscoping | Still-image paint corrections and per-frame mask import | No full spline/keyframe editor |
| Temporal mattes | Optical flow and optional recurrent inference | Model checkpoints and hardware required |
| Hair refinement | Guided filtering and optional ONNX model adapter | Does not reconstruct absent image detail |
| Compositing | Configurable key/track/refine/despill/grade/wrap/composite stages | Not a full Nuke-equivalent node graph |
| Color | Optional OCIO transform support | No certified ACES reference pipeline |
| Export | PNG, EXR where supported, MP4/ProRes with FFmpeg | Codec and bit-depth support vary |
| Farm | Shared-directory workers, chunk retry, GPU memory admission | No enterprise orchestration or guaranteed VRAM reservation |
| Validation | Matte metrics and video decode benchmarking | No published 4K/8K benchmark results |

Do not claim production validation until tests have been run on actual production footage, target GPUs and verified model weights.
