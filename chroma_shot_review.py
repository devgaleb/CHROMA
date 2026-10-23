"""Interactive frame correction and temporal repair for CHROMA."""
import json
import tempfile
import zipfile
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from chroma_temporal_repair import run as temporal_run


def load_frame(video_path, frame_index):
    capture=cv2.VideoCapture(str(video_path))
    try:
        count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count <= 0: raise ValueError('Video contains no readable frames')
        index=max(0,min(int(frame_index),count-1))
        capture.set(cv2.CAP_PROP_POS_FRAMES,index)
        ok,frame=capture.read()
        if not ok: raise ValueError('Cannot decode selected frame')
        return Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),count
    finally: capture.release()


def corrected_matte(editor, mode, screen, tolerance, softness):
    from chroma_advanced import roto_preview
    # Reuse the original CHROMA rotoscoping engine for consistent corrections.
    result, alpha, file_path=roto_preview(editor,screen,tolerance,softness,mode,5)
    return result,alpha,file_path


def repair_sequence(frame_dir,matte_dir,strength,cut_threshold):
    dest=Path(tempfile.mkdtemp(prefix='chroma_repair_'))
    report=temporal_run(frame_dir,matte_dir,dest,float(strength),float(cut_threshold))
    archive=dest.parent/(dest.name+'.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for file in sorted(dest.iterdir()): z.write(file,file.name)
    return str(archive),report
