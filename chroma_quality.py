"""Reproducible validation of mattes, OCIO round trips and real-source decode speed."""
import argparse,json,time
from pathlib import Path
import numpy as np
import cv2

def matte_metrics(prediction,ground_truth):
    p=np.clip(np.asarray(prediction,dtype=np.float32),0,1)
    g=np.clip(np.asarray(ground_truth,dtype=np.float32),0,1)
    if p.shape!=g.shape:raise ValueError('Matte dimensions mismatch')
    return {'mae':float(np.mean(np.abs(p-g))),'mse':float(np.mean((p-g)**2)),'threshold_iou':float(np.logical_and(p>=.5,g>=.5).sum()/max(1,np.logical_or(p>=.5,g>=.5).sum()))}

def validate_directory(pred_dir,truth_dir):
    pred_dir=Path(pred_dir);truth_dir=Path(truth_dir);scores=[]
    for file in sorted(pred_dir.glob('*.png')):
        match=truth_dir/file.name
        if not match.exists():continue
        p=cv2.imread(str(file),cv2.IMREAD_UNCHANGED);g=cv2.imread(str(match),cv2.IMREAD_UNCHANGED)
        if p is None or g is None:continue
        if p.ndim==3:p=p[:,:,-1] if p.shape[2]==4 else cv2.cvtColor(p,cv2.COLOR_BGR2GRAY)
        if g.ndim==3:g=g[:,:,-1] if g.shape[2]==4 else cv2.cvtColor(g,cv2.COLOR_BGR2GRAY)
        scores.append(matte_metrics(p.astype('float32')/np.iinfo(p.dtype).max,g.astype('float32')/np.iinfo(g.dtype).max))
    if not scores:raise ValueError('No matching PNG mattes')
    return {'frames':len(scores),**{k:float(np.mean([v[k] for v in scores])) for k in scores[0]}}

def benchmark_video(path,frames=60):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():raise ValueError('Cannot open source video')
    start=time.perf_counter();n=0;dimensions=None
    while n<frames:
        ok,frame=cap.read()
        if not ok:break
        dimensions=[frame.shape[1],frame.shape[0]];n+=1
    cap.release();elapsed=time.perf_counter()-start
    return {'frames_decoded':n,'resolution':dimensions,'elapsed_seconds':round(elapsed,3),'decode_fps':round(n/max(elapsed,1e-9),2),'note':'Decode-only benchmark, not a full VFX render'}

def ocio_roundtrip(config,source,destination):
    from chroma_advanced import ocio_transform
    rng=np.random.default_rng(42);rgb=rng.uniform(.05,.9,(32,32,3)).astype('float32')
    result=ocio_transform(ocio_transform(rgb,config,source,destination),config,destination,source)
    return {'max_absolute_error':float(np.max(np.abs(result-rgb))),'mean_absolute_error':float(np.mean(np.abs(result-rgb)))}

def main():
    p=argparse.ArgumentParser();p.add_argument('--pred');p.add_argument('--truth');p.add_argument('--video');p.add_argument('--frames',type=int,default=60);p.add_argument('--ocio-config');p.add_argument('--source-space');p.add_argument('--destination-space');p.add_argument('--output')
    a=p.parse_args();report={}
    if a.pred and a.truth:report['matte_validation']=validate_directory(a.pred,a.truth)
    if a.video:report['video_benchmark']=benchmark_video(a.video,a.frames)
    if a.ocio_config:report['ocio_roundtrip']=ocio_roundtrip(a.ocio_config,a.source_space,a.destination_space)
    if not report:p.error('Supply matte pair, video, or OCIO configuration')
    result=json.dumps(report,indent=2)
    if a.output:Path(a.output).write_text(result)
    print(result)
if __name__=='__main__':main()
