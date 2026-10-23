"""Optional ONNX neural alpha prediction, fused with chroma matte around uncertain hair edges."""
import numpy as np
import cv2
from chroma_advanced import guided_hair_refine

def neural_hair_matte(rgb,base_alpha,model_path,input_size=512,edge_width=12,blend=0.8):
    try:import onnxruntime as ort
    except ImportError as exc:raise RuntimeError('Install onnxruntime to enable CHROMA Neural Detail') from exc
    rgb=np.asarray(rgb,dtype=np.float32)
    if rgb.max()>1:rgb=rgb/255.0
    base=np.clip(np.asarray(base_alpha,dtype=np.float32),0,1)
    if rgb.shape[:2]!=base.shape:raise ValueError('RGB and alpha sizes differ')
    session=ort.InferenceSession(str(model_path),providers=['CPUExecutionProvider'])
    inp=session.get_inputs()[0];shape=inp.shape
    if len(shape)!=4 or shape[1] not in (3,'3',None):raise ValueError('Expected NCHW RGB ONNX model')
    h,w=rgb.shape[:2];size=int(input_size)
    tensor=cv2.resize(rgb,(size,size),interpolation=cv2.INTER_AREA).transpose(2,0,1)[None].astype(np.float32)
    prediction=np.asarray(session.run(None,{inp.name:tensor})[0]).squeeze()
    if prediction.ndim!=2:raise ValueError('ONNX output must be one-channel alpha image')
    if prediction.min()<0 or prediction.max()>1:prediction=1/(1+np.exp(-np.clip(prediction,-30,30)))
    neural=cv2.resize(prediction.astype(np.float32),(w,h),interpolation=cv2.INTER_LINEAR)
    edge=cv2.morphologyEx(np.uint8(base>0.5),cv2.MORPH_GRADIENT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(max(3,int(edge_width)*2+1),)*2))
    weight=cv2.GaussianBlur(edge.astype(np.float32)/255,(0,0),max(1,edge_width/2))*float(blend)
    matte=base*(1-weight)+neural*weight
    return guided_hair_refine(rgb,np.clip(matte,0,1),radius=3)
