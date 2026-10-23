"""CHROMA temporal AI matte exporter. Runs only with user-installed licensed model code and weights."""
from pathlib import Path
import tempfile, zipfile, json, time
import cv2
import numpy as np
from PIL import Image
from chroma_pretrained import TemporalMatte

def export_temporal(video, checkpoint, architecture='mobilenetv3', device='auto', downsample=0.25, max_frames=300, edge_blur=0.0, reset_interval=0):
    if not video or not Path(video).is_file(): raise ValueError('Provide an existing video')
    if not checkpoint or not Path(checkpoint).is_file(): raise ValueError('Provide licensed pretrained weights')
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened(): raise ValueError('Could not decode input video')
    engine=TemporalMatte(checkpoint,architecture, None if device=='auto' else device,downsample)
    out=Path(tempfile.mkdtemp(prefix='chroma_temporal_'))
    start=time.monotonic();count=0
    try:
        while count<int(max_frames):
            ok,bgr=cap.read()
            if not ok:break
            if reset_interval and count and count%int(reset_interval)==0:engine.reset()
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            matte=engine.predict(rgb)
            if edge_blur>0:matte=cv2.GaussianBlur(matte,(0,0),float(edge_blur))
            Image.fromarray(np.uint16(np.clip(matte,0,1)*65535)).save(out/f'{count:08d}.png')
            count+=1
    finally:cap.release()
    if not count:raise RuntimeError('No frames decoded')
    manifest={'frames':count,'fps':cap.get(cv2.CAP_PROP_FPS),'width':int(rgb.shape[1]),'height':int(rgb.shape[0]),'format':'16-bit grayscale PNG','elapsed_seconds':round(time.monotonic()-start,3),'architecture':architecture}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    archive=out/'CHROMA_temporal_mattes.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for path in sorted(out.glob('*.png')):z.write(path,path.name)
        z.write(out/'manifest.json','manifest.json')
    return str(archive),json.dumps(manifest,indent=2)
