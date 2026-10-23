"""CHROMA shot analysis: inspect screen uniformity, spill and matte edges."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from chroma_pipeline import load_rgb, matte_key

def inspect_shot(path, screen='#00ff00', tolerance=.22, softness=.17):
    rgb=np.clip(load_rgb(path),0,1)
    h,w=rgb.shape[:2]
    border=max(1,min(h,w)//15)
    samples=np.concatenate([rgb[:border].reshape(-1,3),rgb[-border:].reshape(-1,3),rgb[:, :border].reshape(-1,3),rgb[:,-border:].reshape(-1,3)])
    screen_rgb=np.median(samples,axis=0)
    alpha=matte_key(rgb,screen,tolerance,softness)
    uncertain=(alpha>.05)&(alpha<.95)
    channel=int(np.argmax(np.array([int(screen[i:i+2],16) for i in (1,3,5)])))
    others=[i for i in range(3) if i!=channel]
    spill=np.maximum(0,rgb[:,:,channel]-np.maximum(rgb[:,:,others[0]],rgb[:,:,others[1]]))
    edge=cv2.morphologyEx((alpha>.5).astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8)).astype(bool)
    return {'width':w,'height':h,'border_screen_rgb':np.round(screen_rgb,4).tolist(),
            'screen_uniformity_std':round(float(np.mean(np.std(samples,axis=0))),5),
            'estimated_foreground_fraction':round(float(np.mean(alpha>.5)),5),
            'uncertain_matte_fraction':round(float(np.mean(uncertain)),5),
            'mean_spill_near_edge':round(float(spill[edge].mean()) if edge.any() else 0.,5),
            'suggestions':_suggest(float(np.mean(np.std(samples,axis=0))),float(np.mean(uncertain)),float(spill[edge].mean()) if edge.any() else 0.)}

def _suggest(uniformity, uncertainty, spill):
    notes=[]
    if uniformity>.10:notes.append('Uneven border lighting detected; consider a clean plate or additional screen correction.')
    if uncertainty>.10:notes.append('Large semi-transparent matte region; inspect fine details at full resolution.')
    if spill>.12:notes.append('Possible screen spill around edges; adjust despill strength.')
    return notes or ['No major issues flagged by heuristic analysis; review the matte visually.']

if __name__=='__main__':
    p=argparse.ArgumentParser(description='CHROMA shot quality inspector')
    p.add_argument('image');p.add_argument('--screen',default='#00ff00');p.add_argument('--output')
    args=p.parse_args();result=inspect_shot(args.image,args.screen)
    output=json.dumps(result,indent=2)
    if args.output:Path(args.output).write_text(output+'\n',encoding='utf-8')
    print(output)
