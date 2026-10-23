"""Generate integrity manifests and compare alpha mattes for a CHROMA delivery."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from chroma_finishing import matte_metrics,sequence_manifest

def read_alpha(path):
    a=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if a is None:raise ValueError(f'Unable to decode {path}')
    if a.ndim==3:a=a[:,:,3] if a.shape[2]==4 else cv2.cvtColor(a[:,:,:3],cv2.COLOR_BGR2GRAY)
    if a.dtype==np.uint8:return a.astype(np.float32)/255
    if a.dtype==np.uint16:return a.astype(np.float32)/65535
    return np.asarray(a,np.float32)

def main():
    parser=argparse.ArgumentParser(description='CHROMA matte evaluation and sequence integrity')
    parser.add_argument('sequence',type=Path)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--output',type=Path,default=Path('chroma_finish_report.json'))
    args=parser.parse_args()
    report=sequence_manifest(args.sequence)
    if args.reference:
        scores=[]
        for frame in report['frames']:
            source=args.sequence/frame['file'];truth=args.reference/frame['file']
            if not truth.exists():raise FileNotFoundError(truth)
            scores.append(matte_metrics(read_alpha(source),read_alpha(truth)))
        report['validation']={key:float(np.mean([s[key] for s in scores])) for key in scores[0]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f'CHROMA report: {args.output}')
if __name__=='__main__':main()
