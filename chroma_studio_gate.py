"""CHROMA shot-delivery gate: checks provenance, frame integrity and approval evidence.

No synthetic pass results: absent measurements are explicitly unverified.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

REQUIRED_COLOR = ('working_space', 'display_space', 'ocio_config_sha256')

def validate_delivery(archive_path, specification=None):
    problems=[]; warnings=[]; metrics={}
    path=Path(archive_path)
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        if len(names)!=len(set(names)): problems.append('Duplicate ZIP entries')
        if any(n.startswith('/') or '..' in Path(n).parts for n in names): problems.append('Unsafe archive path')
        if 'manifest.json' not in names: return {'passed':False,'errors':['Missing manifest.json'],'warnings':[]}
        manifest=json.loads(z.read('manifest.json'))
        sequence=manifest.get('sequence',{})
        if not sequence.get('valid'): problems.append('Sequence manifest is not valid')
        frames=sequence.get('frames',[])
        for frame in frames:
            name='frames/'+frame['name']
            if name not in names:
                problems.append('Missing frame '+name);continue
            digest=hashlib.sha256()
            with z.open(name) as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
            if digest.hexdigest()!=frame['sha256']: problems.append('Checksum mismatch '+name)
        metrics['verified_frames']=len(frames)
        spec=specification or {}
        if spec.get('expected_frames') is not None and len(frames)!=spec['expected_frames']:
            problems.append('Frame count does not match production specification')
        color=spec.get('color_pipeline',{})
        for key in REQUIRED_COLOR:
            if not color.get(key): warnings.append('Unverified color pipeline: '+key)
        if spec.get('require_color_evidence') and any(not color.get(k) for k in REQUIRED_COLOR):
            problems.append('Required OCIO provenance is missing')
        checks=spec.get('quality_checks',{})
        for name, config in checks.items():
            actual=config.get('measured')
            if actual is None:
                warnings.append('Not measured: '+name)
                if config.get('required',True): problems.append('Required measurement missing: '+name)
                continue
            if 'maximum' in config and actual>config['maximum']: problems.append('Quality limit exceeded: '+name)
            if 'minimum' in config and actual<config['minimum']: problems.append('Quality limit not reached: '+name)
        approvals=spec.get('approvals',{})
        for role in spec.get('required_approvals',[]):
            if not approvals.get(role): problems.append('Missing approval: '+role)
        metrics['shot']=manifest.get('shot');metrics['version']=manifest.get('version')
    return {'passed':not problems,'errors':problems,'warnings':warnings,'metrics':metrics}

def main():
    p=argparse.ArgumentParser(description='CHROMA production delivery verification')
    p.add_argument('delivery');p.add_argument('--spec',help='JSON production requirements')
    p.add_argument('--report',help='Write machine-readable gate report')
    a=p.parse_args()
    spec=json.loads(Path(a.spec).read_text()) if a.spec else None
    result=validate_delivery(a.delivery,spec)
    if a.report:Path(a.report).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(2)
if __name__=='__main__':main()
