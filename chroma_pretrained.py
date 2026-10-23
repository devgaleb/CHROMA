"""CHROMA pretrained temporal matting adapter. Requires licensed model weights."""
from pathlib import Path
import numpy as np
import cv2

class TemporalMatte:
    def __init__(self, checkpoint, architecture='mobilenetv3', device=None, downsample=0.25):
        import torch
        self.torch=torch
        self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.downsample=float(downsample)
        if not 0.05 <= self.downsample <= 1: raise ValueError('downsample must be 0.05..1')
        try:
            from model import MattingNetwork
        except ImportError as exc:
            raise RuntimeError('Install the licensed Robust Video Matting model package and make its model module importable') from exc
        checkpoint=Path(checkpoint)
        if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
        self.model=MattingNetwork(architecture)
        # Checkpoints must be obtained from a trusted source.
        state=torch.load(str(checkpoint),map_location='cpu',weights_only=True)
        self.model.load_state_dict(state,strict=True)
        self.model.to(self.device).eval()
        self.reset()
    def reset(self):
        self.state=[None]*4
    def predict(self,rgb):
        arr=np.asarray(rgb)
        if arr.ndim!=3 or arr.shape[2]!=3: raise ValueError('Expected RGB HxWx3')
        arr=np.clip(arr.astype(np.float32)/ (255. if arr.max()>1 else 1.),0,1)
        x=self.torch.from_numpy(np.ascontiguousarray(arr.transpose(2,0,1)))[None].to(self.device)
        with self.torch.inference_mode():
            foreground,alpha,*self.state=self.model(x,*self.state,self.downsample)
        a=alpha[0,0].float().cpu().numpy()
        return np.clip(a,0,1)

def refine_video_frames(frames,checkpoint,architecture='mobilenetv3',device=None,downsample=0.25):
    engine=TemporalMatte(checkpoint,architecture,device,downsample)
    for frame in frames:
        yield engine.predict(frame)
