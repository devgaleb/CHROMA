# Installation and optional features

Create a Python 3.10–3.12 environment, then install `requirements.txt` and run `python app.py`.

Install FFmpeg/FFprobe separately for video output. To enable optional capabilities, install only the relevant packages and their compatible system libraries:

```bash
python -m pip install onnxruntime
python -m pip install opencolorio
```

Temporal PyTorch inference additionally requires a compatible PyTorch installation, the supported model implementation and legitimately acquired checkpoint weights. GPU builds vary by OS, drivers and CUDA version; follow upstream PyTorch installation guidance. EXR availability depends on the OpenCV/OpenEXR environment. Do not assume every video codec preserves transparency or bit depth.

Run `python chroma_doctor.py` to check local dependencies and supported tools.
