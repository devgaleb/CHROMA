"""CHROMA temporal matte stabilization with scene-cut and motion safeguards."""
from pathlib import Path
import argparse
import json
import cv2
import numpy as np
from PIL import Image

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}

def stabilize(rgb_frames, alpha_frames, strength=0.45, cut_threshold=0.24,
              disagreement=0.18, flow_scale=0.5):
    """Yield temporally stabilized float32 mattes in [0,1]. Frames are RGB, same size."""
    if not 0 <= strength <= 1 or not 0 < flow_scale <= 1:
        raise ValueError('strength must be [0,1] and flow_scale (0,1]')
    previous_rgb = previous_alpha = None
    for rgb, matte in zip(rgb_frames, alpha_frames):
        rgb = np.asarray(rgb, dtype=np.float32)
        alpha = np.clip(np.asarray(matte, dtype=np.float32), 0, 1)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[:2] != alpha.shape:
            raise ValueError('RGB and alpha dimensions must match')
        if previous_rgb is None or previous_rgb.shape != rgb.shape or float(np.mean(np.abs(rgb-previous_rgb))) > cut_threshold:
            result = alpha.copy()
        else:
            h,w = alpha.shape
            sw,sh = max(8,round(w*flow_scale)),max(8,round(h*flow_scale))
            prev = cv2.cvtColor((np.clip(previous_rgb,0,1)*255).astype('uint8'),cv2.COLOR_RGB2GRAY)
            curr = cv2.cvtColor((np.clip(rgb,0,1)*255).astype('uint8'),cv2.COLOR_RGB2GRAY)
            # Backward flow: current pixel -> previous pixel.
            flow = cv2.calcOpticalFlowFarneback(cv2.resize(curr,(sw,sh)),cv2.resize(prev,(sw,sh)),None,.5,3,21,3,5,1.2,0)
            flow = cv2.resize(flow,(w,h),interpolation=cv2.INTER_LINEAR)
            flow[...,0] *= w/sw; flow[...,1] *= h/sh
            x,y = np.meshgrid(np.arange(w,dtype=np.float32),np.arange(h,dtype=np.float32))
            warped = cv2.remap(previous_alpha,x+flow[...,0],y+flow[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            difference = np.abs(alpha-warped)
            # Protect moving boundaries: don't smear large disagreements into the matte.
            confidence = np.clip(1-difference/max(disagreement,1e-6),0,1)
            blend = strength*confidence
            result = np.clip(alpha*(1-blend)+warped*blend,0,1)
        previous_rgb,previous_alpha = rgb.copy(),result.copy()
        yield result

def run(source, mattes, output, strength=0.45, cut_threshold=0.24):
    src, masks, dest = Path(source),Path(mattes),Path(output)
    frames = sorted(p for p in src.iterdir() if p.suffix.lower() in EXTENSIONS)
    if not frames: raise ValueError('No input frames')
    pairs = [(p,masks/(p.stem+'.png')) for p in frames]
    missing = [str(m) for _,m in pairs if not m.exists()]
    if missing: raise FileNotFoundError('Missing matching alpha masks: '+', '.join(missing[:5]))
    def rgb_iter():
        for p,_ in pairs:
            with Image.open(p) as im: yield np.asarray(im.convert('RGB'),dtype=np.float32)/255
    def alpha_iter():
        for _,p in pairs:
            with Image.open(p) as im: yield np.asarray(im.convert('L'),dtype=np.float32)/255
    dest.mkdir(parents=True,exist_ok=True)
    for (p,_),a in zip(pairs,stabilize(rgb_iter(),alpha_iter(),strength,cut_threshold)):
        tmp=dest/(p.stem+'.tmp.png'); final=dest/(p.stem+'.png')
        Image.fromarray(np.round(a*65535).astype('uint16')).save(tmp)
        tmp.replace(final)
    report={'frames':len(pairs),'strength':strength,'cut_threshold':cut_threshold,'output':str(dest)}
    (dest/'report.json').write_text(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='CHROMA motion-aware temporal matte repair')
    parser.add_argument('--frames',required=True);parser.add_argument('--mattes',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--strength',type=float,default=.45)
    parser.add_argument('--cut-threshold',type=float,default=.24)
    args=parser.parse_args()
    print(json.dumps(run(args.frames,args.mattes,args.output,args.strength,args.cut_threshold),indent=2))
