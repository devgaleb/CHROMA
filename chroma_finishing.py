"""CHROMA finishing: edge decontamination, matte QC and deterministic render manifests."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np


def decontaminate_edges(rgb: np.ndarray, alpha: np.ndarray, screen='#00ff00', strength=0.7, radius=5):
    """Replace contaminated edge colour with nearby opaque foreground colour.

    Input RGB is straight (unassociated), linear or display RGB in [0, 1].
    The algorithm does not invent missing hair strands; alpha remains unchanged.
    """
    rgb=np.asarray(rgb,dtype=np.float32)
    alpha=np.asarray(alpha,dtype=np.float32)
    if rgb.ndim!=3 or rgb.shape[2]!=3 or alpha.shape!=rgb.shape[:2]:
        raise ValueError('Expected HxWx3 RGB and HxW alpha')
    radius=max(1,min(31,int(radius)))
    strength=float(np.clip(strength,0,1))
    opaque=(alpha>=0.97).astype(np.uint8)
    if not np.any(opaque):return rgb.copy()
    # Inpaint missing edge colour from confident opaque neighbours, preserving texture.
    missing=(1-opaque).astype(np.uint8)*255
    src=np.uint8(np.clip(rgb,0,1)*255)
    reconstructed=cv2.inpaint(src,missing,radius,cv2.INPAINT_TELEA).astype(np.float32)/255
    edge=np.clip((alpha-0.005)/0.4,0,1)*np.clip((0.995-alpha)/0.4,0,1)
    weight=(edge*strength)[:,:,None]
    return np.clip(rgb*(1-weight)+reconstructed*weight,0,1).astype(np.float32)


def matte_metrics(prediction,reference):
    p=np.asarray(prediction,np.float32); r=np.asarray(reference,np.float32)
    if p.shape!=r.shape or p.ndim!=2:raise ValueError('Mattes must have identical 2D shapes')
    if p.size==0:raise ValueError('Empty matte')
    p=np.clip(p,0,1);r=np.clip(r,0,1)
    d=np.abs(p-r)
    return {'mae':float(d.mean()),'mse':float(np.mean((p-r)**2)),
            'rmse':float(np.sqrt(np.mean((p-r)**2))),
            'bad_pixels_0_1':float(np.mean(d>0.1))}


def sequence_manifest(folder,output=None):
    folder=Path(folder)
    files=sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {'.png','.exr'})
    if not files:raise ValueError('No image frames found')
    manifest={'schema':'chroma.sequence.v1','frame_count':len(files),'frames':[]}
    for p in files:
        with p.open('rb') as stream:
            digest=hashlib.file_digest(stream,'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(stream.read()).hexdigest()
        manifest['frames'].append({'file':p.name,'bytes':p.stat().st_size,'sha256':digest})
    if output:Path(output).write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return manifest
