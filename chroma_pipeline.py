"""CHROMA high precision compositing pipeline. Float32 linear-light RGB throughout."""
import os
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
from pathlib import Path
import json, tempfile, zipfile, subprocess
import cv2
import numpy as np
from PIL import Image

NODES=('Key','Track matte','Manual correction','Edge refine','Despill','Edge decontaminate','Grade','Light wrap','Composite')

def linear(x):
    return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
def srgb(x):
    x=np.maximum(x,0)
    return np.where(x<=.0031308,x*12.92,1.055*np.power(x,1/2.4)-.055)
def load_rgb(path):
    if isinstance(path,Image.Image):
        return np.asarray(path.convert('RGB')).astype(np.float32)/255
    a=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if a is None: raise ValueError('Cannot decode image: '+str(path))
    if a.ndim==2:a=cv2.cvtColor(a,cv2.COLOR_GRAY2BGR)
    if a.shape[2]==4:a=a[:,:,:3]
    if a.dtype==np.uint8:a=a.astype(np.float32)/255
    elif a.dtype==np.uint16:a=a.astype(np.float32)/65535
    else:a=a.astype(np.float32)
    return cv2.cvtColor(a,cv2.COLOR_BGR2RGB)
def save_exr(path,rgba):
    # OpenCV's EXR support is optional and must be present in the installed wheel.
    if not cv2.imwrite(str(path),cv2.cvtColor(rgba.astype(np.float32),cv2.COLOR_RGBA2BGRA)):
        raise RuntimeError('OpenEXR encoder unavailable in OpenCV')
def matte_key(rgb,screen,tolerance,softness):
    target=np.array([int(screen[i:i+2],16) for i in (1,3,5)],np.float32)/255
    pixels=cv2.cvtColor(np.uint8(np.clip(rgb,0,1)*255),cv2.COLOR_RGB2HSV).astype(np.float32)
    t=cv2.cvtColor(np.uint8([[target*255]]),cv2.COLOR_RGB2HSV)[0,0].astype(np.float32)
    hue=np.abs(pixels[:,:,0]-t[0]);hue=np.minimum(hue,180-hue)/90
    dist=np.sqrt((hue*1.6)**2+((pixels[:,:,1]-t[1])/255*.35)**2)
    alpha=np.clip((dist-tolerance)/max(softness,.001),0,1)
    alpha=np.maximum(alpha,np.clip((.18-pixels[:,:,1]/255)*3,0,1))
    alpha=np.maximum(alpha,np.clip((.12-pixels[:,:,2]/255)*3,0,1))
    return alpha.astype(np.float32)
def track_mask(prev_gray,gray,prev_alpha):
    """Dense Farneback backward optical flow: advects the previous alpha into current coordinates."""
    flow=cv2.calcOpticalFlowFarneback(gray,prev_gray,None,.5,3,21,3,5,1.2,0)
    h,w=gray.shape
    xx,yy=np.meshgrid(np.arange(w,dtype=np.float32),np.arange(h,dtype=np.float32))
    return cv2.remap(prev_alpha,xx+flow[:,:,0],yy+flow[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
def load_mask(path,size):
    m=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if m is None:raise ValueError('Cannot decode correction mask '+str(path))
    if m.ndim==3:m=cv2.cvtColor(m[:,:,:3],cv2.COLOR_BGR2GRAY)
    m=cv2.resize(m,size,interpolation=cv2.INTER_LINEAR)
    return m.astype(np.float32)/(65535 if m.dtype==np.uint16 else 255) if m.dtype!=np.float32 else np.clip(m,0,1)

def apply_nodes(rgb,background,settings,state=None,frame_index=0,corrections=None):
    """Configurable operation graph, each node executed once in user-defined order."""
    h,w=rgb.shape[:2]; alpha=np.ones((h,w),np.float32)
    work=linear(np.clip(rgb,0,None)); bg=linear(np.clip(cv2.resize(background,(w,h)),0,1)) if background is not None else None
    current_gray=cv2.cvtColor(np.uint8(np.clip(rgb,0,1)*255),cv2.COLOR_RGB2GRAY)
    tracking_alpha=None
    for node in settings['nodes']:
        if node=='Key':alpha=matte_key(rgb,settings['screen'],settings['tolerance'],settings['softness'])
        elif node=='Track matte' and state is not None and state.get('gray') is not None:
            tracked=track_mask(state['gray'],current_gray,state['alpha'])
            alpha=np.clip((1-settings['track_strength'])*alpha+settings['track_strength']*tracked,0,1)
        elif node=='Manual correction' and corrections:
            # User-provided grayscale mask images named 000000.png etc, or explicit frame index.
            names=[f'{frame_index:06d}.png',f'{frame_index}.png']
            mask=next((corrections/n for n in names if (corrections/n).exists()),None)
            if mask is not None:
                correction=load_mask(mask,(w,h))
                alpha=np.clip(alpha*(1-settings['correction_strength'])+correction*settings['correction_strength'],0,1)
        elif node=='Edge refine':
            choke=int(settings['choke']);
            if choke:
                k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*abs(choke)+1,)*2)
                alpha=cv2.erode(alpha,k) if choke>0 else cv2.dilate(alpha,k)
            if settings['feather']>0:alpha=cv2.GaussianBlur(alpha,(0,0),settings['feather'])
            # Joint bilateral guided smoothing helps avoid excessive detail erosion.
            if settings['detail']>0:
                guide=cv2.cvtColor(np.uint8(np.clip(rgb,0,1)*255),cv2.COLOR_RGB2GRAY)
                smooth=cv2.bilateralFilter(alpha,5,max(.01,settings['detail']*.15),3)
                edges=cv2.Laplacian(guide,cv2.CV_32F)/255
                blend=np.clip(np.abs(edges)*settings['detail']*4,0,1)
                alpha=smooth*(1-blend)+alpha*blend
            alpha=np.clip(alpha,0,1)
        elif node=='Despill':
            target=np.array([int(settings['screen'][i:i+2],16) for i in (1,3,5)])
            channel=int(np.argmax(target));others=[i for i in range(3) if i!=channel]
            spill=np.maximum(0,work[:,:,channel]-np.maximum(work[:,:,others[0]],work[:,:,others[1]]))
            work[:,:,channel]=np.maximum(0,work[:,:,channel]-spill*settings['despill']*np.clip(1-alpha+.2,0,1))
        elif node=='Edge decontaminate':
            from chroma_finishing import decontaminate_edges
            work=linear(decontaminate_edges(srgb(np.clip(work,0,1)),alpha,settings['screen'],settings['decontaminate'],settings['decontaminate_radius']))
        elif node=='Grade':
            work=np.maximum(0,work*2**settings['exposure'])
            work=np.maximum(0,(work-.18)*settings['contrast']+.18)
            luma=np.sum(work*np.array([.2126,.7152,.0722]),axis=2,keepdims=True)
            work=np.maximum(0,luma+(work-luma)*settings['saturation'])
        elif node=='Light wrap' and bg is not None and settings['wrap']>0:
            blur=cv2.GaussianBlur(bg,(0,0),max(1,settings['wrap_radius']))
            boundary=np.clip(cv2.dilate(alpha,np.ones((5,5),np.uint8))-cv2.erode(alpha,np.ones((5,5),np.uint8)),0,1)
            work=work*(1-boundary[:,:,None]*settings['wrap'])+blur*boundary[:,:,None]*settings['wrap']
        elif node=='Composite' and bg is not None:
            tracking_alpha=alpha.copy()
            work=work*alpha[:,:,None]+bg*(1-alpha[:,:,None]);alpha=np.ones_like(alpha)
    if state is not None:state.update(gray=current_gray,alpha=(tracking_alpha if tracking_alpha is not None else alpha).copy())
    rgba=np.dstack((srgb(np.clip(work,0,100)),alpha)).astype(np.float32)
    if settings.get('ocio_config') and settings.get('ocio_input') != settings.get('ocio_output'):
        from chroma_advanced import ocio_transform
        rgba[:,:,:3]=ocio_transform(rgba[:,:,:3],settings['ocio_config'],settings['ocio_input'],settings['ocio_output'])
    return rgba

def settings_from_json(s):
    d=json.loads(s) if isinstance(s,str) else dict(s)
    nodes=d.get('nodes',list(NODES))
    if not isinstance(nodes,list) or len(set(nodes))!=len(nodes) or any(n not in NODES for n in nodes):raise ValueError('Invalid node graph')
    d['nodes']=nodes
    defaults=dict(screen='#00ff00',tolerance=.22,softness=.17,track_strength=.25,correction_strength=1.,choke=0,feather=.5,detail=.3,despill=1.,exposure=0.,contrast=1.,saturation=1.,wrap=.15,wrap_radius=8,decontaminate=.65,decontaminate_radius=5)
    for k,v in defaults.items():d.setdefault(k,v)
    return d

def export_still(foreground,background,settings,format):
    rgb=load_rgb(foreground);bg=load_rgb(background) if background is not None else None
    rgba=apply_nodes(rgb,bg,settings_from_json(settings))
    path=Path(tempfile.mkdtemp(prefix='chroma_still_'))/('result.exr' if format=='EXR 32-bit float' else 'result.png')
    if format=='EXR 32-bit float':save_exr(path,rgba)
    elif format=='PNG RGBA 16-bit':cv2.imwrite(str(path),cv2.cvtColor(np.uint16(np.clip(rgba,0,1)*65535),cv2.COLOR_RGBA2BGRA))
    else:Image.fromarray(np.uint8(np.clip(rgba,0,1)*255),'RGBA').save(path)
    return str(path)

def export_video(video,background,settings,format,max_frames,correction_zip=None,progress=None):
    cfg=settings_from_json(settings);bg=load_rgb(background) if background is not None else None
    work=Path(tempfile.mkdtemp(prefix='chroma_render_'));frames=work/'frames';frames.mkdir()
    corrections=None
    if correction_zip:
        corrections=work/'corrections';corrections.mkdir()
        with zipfile.ZipFile(correction_zip) as z:
            for f in z.infolist():
                # No directory traversal or oversized archive extraction.
                if Path(f.filename).name==f.filename and f.filename.lower().endswith('.png') and f.file_size<100_000_000:
                    (corrections/f.filename).write_bytes(z.read(f))
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():raise ValueError('Unable to open video')
    fps=cap.get(cv2.CAP_PROP_FPS) or 24;state={};n=0
    if cfg.get('high_bit_depth'):
        from chroma_advanced import decode_rgb48
        source=decode_rgb48(video,max_frames)
    else:
        def frames_bgr():
            while True:
                ok,bgr=cap.read()
                if not ok:break
                yield cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).astype(np.float32)/255
        source=frames_bgr()
    try:
        for rgb in source:
            if n>=int(max_frames):break
            rgba=apply_nodes(rgb,bg,cfg,state,n,corrections)
            name=f'{n:06d}.'+('exr' if format=='EXR sequence' else 'png')
            path=frames/name
            if format=='EXR sequence':save_exr(path,rgba)
            elif format=='PNG RGBA 16-bit':cv2.imwrite(str(path),cv2.cvtColor(np.uint16(np.clip(rgba,0,1)*65535),cv2.COLOR_RGBA2BGRA))
            else:Image.fromarray(np.uint8(np.clip(rgba,0,1)*255),'RGBA').save(path)
            n+=1
    finally:cap.release()
    if not n:raise ValueError('No frames decoded')
    if format=='ProRes 4444 MOV' or format=='H.264 MP4':
        dest=work/('render.mov' if format=='ProRes 4444 MOV' else 'render.mp4')
        cmd=['ffmpeg','-y','-loglevel','error','-framerate',str(fps),'-i',str(frames/'%06d.png')]
        if format=='ProRes 4444 MOV':cmd+=['-c:v','prores_ks','-profile:v','4','-pix_fmt','yuva444p10le']
        else:cmd+=['-c:v','libx264','-pix_fmt','yuv420p']
        cmd+=['-i',str(video),'-map','0:v:0','-map','1:a?','-c:a','aac' if format=='H.264 MP4' else 'pcm_s16le','-shortest',str(dest)]
        result=subprocess.run(cmd,capture_output=True,text=True)
        if result.returncode:raise RuntimeError(result.stderr[-1600:])
    else:
        dest=work/('exr_sequence.zip' if format=='EXR sequence' else 'png_sequence.zip')
        with zipfile.ZipFile(dest,'w',zipfile.ZIP_STORED) as z:
            for p in sorted(frames.iterdir()):z.write(p,p.name)
    return str(dest),f'Rendered {n} frames at {fps:.3f} fps. Format: {format}'
