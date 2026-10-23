"""Local dependency diagnostics for CHROMA. Run: python chroma_doctor.py"""
import importlib.util
import json
import platform
import shutil
import sys


def collect():
    modules = ('gradio', 'cv2', 'numpy', 'PIL', 'torch', 'onnxruntime', 'PyOpenColorIO')
    return {
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'modules': {name: importlib.util.find_spec(name) is not None for name in modules},
        'executables': {name: shutil.which(name) for name in ('ffmpeg', 'ffprobe', 'nvidia-smi')},
    }


if __name__ == '__main__':
    print(json.dumps(collect(), indent=2))
