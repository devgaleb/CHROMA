"""Model provenance, hash validation and optional ground-truth evaluation."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def register(model,manifest,source,license_name,training_dataset,validation_dataset):
    p=Path(model)
    if not p.is_file() or p.suffix.lower()!='.onnx':raise ValueError('Provide an existing ONNX model')
    if not source or not license_name:raise ValueError('Model source and license required')
    record={'format':'onnx','file':str(p.resolve()),'sha256':sha256(p),'source':source,'license':license_name,'training_dataset':training_dataset,'validation_dataset':validation_dataset,'validated':False}
    Path(manifest).write_text(json.dumps(record,indent=2));return record

def verify(manifest):
    r=json.loads(Path(manifest).read_text());p=Path(r['file'])
    if not p.is_file() or sha256(p)!=r['sha256']:raise RuntimeError('Model missing or SHA-256 mismatch')
    return r

def evaluate(pred_dir,truth_dir):
    a,b=Path(pred_dir),Path(truth_dir);files=sorted(a.glob('*.png'))
    if not files:raise ValueError('No prediction PNG masks')
    mae=[];mse=[]
    for p in files:
        t=b/p.name
        if not t.is_file():raise FileNotFoundError(t)
        x=np.asarray(Image.open(p).convert('L'),dtype=np.float32)/255
        y=np.asarray(Image.open(t).convert('L'),dtype=np.float32)/255
        if x.shape!=y.shape:raise ValueError(f'Shape mismatch: {p.name}')
        mae.append(float(np.mean(np.abs(x-y))));mse.append(float(np.mean((x-y)**2)))
    return {'frames':len(files),'mae':float(np.mean(mae)),'mse':float(np.mean(mse)),'rmse':float(np.sqrt(np.mean(mse)))}

def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    r=s.add_parser('register');r.add_argument('--model',required=True);r.add_argument('--manifest',required=True);r.add_argument('--source',required=True);r.add_argument('--license',required=True);r.add_argument('--training-dataset',default='unknown');r.add_argument('--validation-dataset',default='unknown')
    v=s.add_parser('verify');v.add_argument('--manifest',required=True)
    e=s.add_parser('evaluate');e.add_argument('--pred',required=True);e.add_argument('--truth',required=True);e.add_argument('--output')
    a=p.parse_args()
    if a.command=='register':out=register(a.model,a.manifest,a.source,a.license,a.training_dataset,a.validation_dataset)
    elif a.command=='verify':out=verify(a.manifest)
    else:
        out=evaluate(a.pred,a.truth)
        if a.output:Path(a.output).write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
