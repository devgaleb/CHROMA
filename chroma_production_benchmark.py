"""Benchmark full CHROMA key processing using actual video footage."""
import argparse
import json
import time
import cv2
import numpy as np
from app import key_frame

def benchmark(path,max_frames=120):
    cap=cv2.VideoCapture(path)
    if not cap.isOpened(): raise ValueError('Cannot open source video')
    samples=[]; dimensions=None
    try:
        for _ in range(max_frames):
            ok,frame=cap.read()
            if not ok:break
            dimensions=(frame.shape[1],frame.shape[0]);start=time.perf_counter()
            key_frame(frame,[0,255,0],.22,.17,0,1,.5,0)
            samples.append(time.perf_counter()-start)
    finally:cap.release()
    if not samples:raise ValueError('No frames decoded')
    return {'frames':len(samples),'resolution':dimensions,'mean_processing_ms':round(float(np.mean(samples))*1000,2),'p95_processing_ms':round(float(np.percentile(samples,95))*1000,2),'processing_fps':round(len(samples)/sum(samples),2),'scope':'CPU key_frame only; excludes decoding, GPU neural inference, OCIO and encoding'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('video');p.add_argument('--frames',type=int,default=120);p.add_argument('--output',default='chroma_benchmark.json');a=p.parse_args()
    result=benchmark(a.video,a.frames)
    with open(a.output,'w') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
