"""Fast per-frame coarse mattes for CHROMA AI input hints."""
import cv2
import numpy as np

def automatic_hint(rgb, screen='auto'):
    rgb=np.asarray(rgb,dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Expected RGB frame')
    f=rgb.astype(np.float32) / 255.0
    green=f[:,:,1]-np.maximum(f[:,:,0],f[:,:,2])
    blue=f[:,:,2]-np.maximum(f[:,:,0],f[:,:,1])
    if screen=='auto':
        border=np.concatenate((green[0],green[-1],green[:,0],green[:,-1]))
        border_blue=np.concatenate((blue[0],blue[-1],blue[:,0],blue[:,-1]))
        screen='blue' if np.percentile(border_blue,75)>np.percentile(border,75) else 'green'
    dominance=blue if screen=='blue' else green
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    hue=hsv[:,:,0].astype(np.float32)
    target=120 if screen=='blue' else 60
    distance=np.minimum(abs(hue-target),180-abs(hue-target))
    screen_conf=np.clip((dominance-0.035)/0.18,0,1)*np.clip((24-distance)/17,0,1)
    alpha=np.uint8(np.clip(1-screen_conf,0,1)*255)
    return cv2.GaussianBlur(alpha,(3,3),0.5)
