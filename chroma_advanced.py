"""CHROMA advanced finishing: interactive paint mattes, guided fine edges, OCIO and 16-bit decoding."""
import os, subprocess, tempfile
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def paint_matte(editor, mode='Add foreground', threshold=12):
    """Convert Gradio ImageEditor painted RGBA layers into an editable alpha matte."""
    if not editor or editor.get('background') is None:
        raise ValueError('Load an image into the roto editor.')
    base=np.asarray(editor['background'].convert('RGB'))
    h,w=base.shape[:2]
    mask=np.zeros((h,w),np.uint8)
    for layer in editor.get('layers',[]):
        rgba=np.asarray(layer.convert('RGBA').resize((w,h)))
        mask=np.maximum(mask,rgba[:,:,3])
    mask[mask < threshold]=0
    return base,mask if mode=='Add foreground' else 255-mask


def guided_hair_refine(rgb,alpha,radius=5,epsilon=.001):
    """Edge-aware guided-filter alpha refinement (not neural hair synthesis)."""
    guide=cv2.cvtColor(np.float32(np.clip(rgb,0,1)),cv2.COLOR_RGB2GRAY)
    p=np.float32(np.clip(alpha,0,1)); r=max(1,int(radius)); k=(2*r+1,2*r+1)
    mean_i=cv2.boxFilter(guide,-1,k);mean_p=cv2.boxFilter(p,-1,k)
    corr_i=cv2.boxFilter(guide*guide,-1,k);corr_ip=cv2.boxFilter(guide*p,-1,k)
    a=(corr_ip-mean_i*mean_p)/(corr_i-mean_i*mean_i+epsilon)
    b=mean_p-a*mean_i
    return np.clip(cv2.boxFilter(a,-1,k)*guide+cv2.boxFilter(b,-1,k),0,1)


def ocio_transform(rgb,config_path,src,dst):
    """Apply a user-supplied OCIO configuration without silently substituting a look."""
    if not config_path or src==dst:return rgb
    try:import PyOpenColorIO as ocio
    except ImportError as e:raise RuntimeError('Install optional dependency: pip install opencolorio') from e
    config=ocio.Config.CreateFromFile(str(config_path))
    processor=config.getProcessor(src,dst).getDefaultCPUProcessor()
    out=np.ascontiguousarray(rgb.astype(np.float32).copy())
    processor.applyRGB(out.reshape(-1,3))
    return out


def decode_rgb48(video,max_frames=1000):
    """Stream RGB48LE through FFmpeg. Preserves decoded 16-bit precision; memory bounded by one frame."""
    probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,avg_frame_rate','-of','default=noprint_wrappers=1',str(video)],capture_output=True,text=True,check=True)
    props=dict(line.split('=',1) for line in probe.stdout.splitlines() if '=' in line)
    w,h=int(props['width']),int(props['height'])
    if w*h>100_000_000:raise ValueError('Frame dimensions exceed safety limit')
    cmd=['ffmpeg','-v','error','-i',str(video),'-map','0:v:0','-frames:v',str(int(max_frames)),'-f','rawvideo','-pix_fmt','rgb48le','-']
    proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    size=w*h*6
    try:
        for _ in range(int(max_frames)):
            chunks=[];remaining=size
            while remaining:
                chunk=proc.stdout.read(remaining)
                if not chunk:break
                chunks.append(chunk);remaining-=len(chunk)
            if remaining:break
            yield np.frombuffer(b''.join(chunks),dtype='<u2').reshape(h,w,3).astype(np.float32)/65535
    finally:
        proc.stdout.close();proc.terminate()
        try:proc.wait(timeout=3)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()


def roto_preview(editor,screen,tolerance,softness,brush_mode,hair_radius):
    from chroma_pipeline import matte_key
    base,brush=paint_matte(editor,brush_mode)
    rgb=base.astype(np.float32)/255
    alpha=matte_key(rgb,screen,tolerance,softness)
    strength=brush.astype(np.float32)/255
    if brush_mode=='Add foreground':alpha=np.maximum(alpha,strength)
    else:alpha=np.minimum(alpha,1-strength)
    alpha=guided_hair_refine(rgb,alpha,hair_radius)
    rgba=np.dstack((base,np.uint8(np.clip(alpha,0,1)*255)))
    folder=Path(tempfile.mkdtemp(prefix='chroma_roto_'))
    path=folder/'corrected_matte.png'
    Image.fromarray(np.uint8(alpha*255),'L').save(path)
    return Image.fromarray(rgba,'RGBA'),Image.fromarray(np.uint8(alpha*255),'L'),str(path)
