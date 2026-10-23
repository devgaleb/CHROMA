"""CHROMA studio delivery: reproducible frame-sequence inspection and packaging."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile
import zipfile
from datetime import datetime, timezone

IMAGE_EXTENSIONS={'.png','.exr','.tif','.tiff'}
FRAME_PATTERN=re.compile(r'^(.*?)(\d+)$')

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def inspect_sequence(directory, expected_start=None, expected_end=None, min_size=1):
    root=Path(directory).resolve()
    if not root.is_dir():raise ValueError('Sequence directory not found')
    frames=sorted((p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS))
    if not frames:raise ValueError('No image sequence frames found')
    grouped={}
    for p in frames:
        match=FRAME_PATTERN.match(p.stem)
        if not match:continue
        key=(match.group(1),p.suffix.lower())
        grouped.setdefault(key,[]).append((int(match.group(2)),p))
    if not grouped:raise ValueError('Frames must end with numeric frame indices')
    if len(grouped)!=1:raise ValueError('Ambiguous sequence: multiple prefixes or formats')
    (prefix,extension),items=next(iter(grouped.items()))
    numbers=[n for n,_ in items]
    if len(set(numbers))!=len(numbers):raise ValueError('Duplicate frame numbers')
    start=min(numbers) if expected_start is None else int(expected_start)
    end=max(numbers) if expected_end is None else int(expected_end)
    if end<start:raise ValueError('Invalid expected frame range')
    missing=sorted(set(range(start,end+1))-set(numbers))
    undersized=[p.name for _,p in items if p.stat().st_size<min_size]
    return {'sequence':prefix,'extension':extension,'start':start,'end':end,
            'frame_count':len(items),'missing_frames':missing,'undersized_frames':undersized,
            'valid':not missing and not undersized and min(numbers)>=start and max(numbers)<=end,
            'frames':[{'name':p.name,'frame':n,'bytes':p.stat().st_size,'sha256':sha256(p)} for n,p in sorted(items)]}

def package_delivery(directory,output,shot,version='v001',fps=24,expected_start=None,expected_end=None):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',shot):raise ValueError('Shot name must use letters, numbers, _ or -')
    if fps<=0:raise ValueError('FPS must be positive')
    report=inspect_sequence(directory,expected_start,expected_end)
    if not report['valid']:raise ValueError('Delivery blocked: missing or invalid frames: '+str(report['missing_frames']))
    manifest={'format':'CHROMA-delivery-1','shot':shot,'version':version,'fps':fps,
              'created_utc':datetime.now(timezone.utc).isoformat(),'sequence':report}
    destination=Path(output).resolve();destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent,suffix='.zip',delete=False) as tmp:temp=Path(tmp.name)
    try:
        with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as archive:
            archive.writestr('manifest.json',json.dumps(manifest,indent=2))
            for frame in report['frames']:
                archive.write(Path(directory)/frame['name'],arcname='frames/'+frame['name'])
        temp.replace(destination)
    finally:
        temp.unlink(missing_ok=True)
    return manifest

def main():
    parser=argparse.ArgumentParser(description='Validate and package CHROMA shot sequences')
    parser.add_argument('directory');parser.add_argument('--output');parser.add_argument('--shot',default='shot')
    parser.add_argument('--version',default='v001');parser.add_argument('--fps',type=float,default=24)
    parser.add_argument('--start',type=int);parser.add_argument('--end',type=int)
    args=parser.parse_args()
    report=(package_delivery(args.directory,args.output,args.shot,args.version,args.fps,args.start,args.end)
            if args.output else inspect_sequence(args.directory,args.start,args.end))
    print(json.dumps(report,indent=2))
    if not args.output and not report['valid']:raise SystemExit(2)
if __name__=='__main__':main()
