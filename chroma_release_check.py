"""CHROMA release acceptance runner. Reports evidence, not inferred capability."""
import argparse, json, os, platform, shutil, subprocess, sys, time, unittest
from pathlib import Path

def command_version(command):
    path=shutil.which(command)
    if not path:return {'available':False}
    try:
        p=subprocess.run([path,'-version' if command.startswith('ff') else '--version'],capture_output=True,text=True,timeout=8)
        return {'available':True,'path':path,'version':(p.stdout or p.stderr).splitlines()[0][:200]}
    except Exception as exc:return {'available':True,'error':str(exc)}

def report(video=None, ocio_config=None, output=None):
    root=Path(__file__).resolve().parent
    r={'app':'CHROMA','timestamp_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(), 'platform':platform.platform(),'python':sys.version.split()[0], 'checks':{}}
    r['checks']['ffmpeg']=command_version('ffmpeg');r['checks']['ffprobe']=command_version('ffprobe')
    try:
        from chroma_gpu import discover
        r['checks']['gpu']={'devices':discover(),'hardware_verified':False}
    except Exception as exc:r['checks']['gpu']={'error':str(exc)}
    try:
        import PyOpenColorIO as ocio
        r['checks']['ocio']={'installed':True,'version':getattr(ocio,'__version__','unknown')}
        if ocio_config:
            cfg=ocio.Config.CreateFromFile(str(Path(ocio_config).resolve()))
            r['checks']['ocio']['config_loaded']=True
            r['checks']['ocio']['color_spaces']=[cfg.getColorSpaceNameByIndex(i) for i in range(cfg.getNumColorSpaces())]
    except Exception as exc:r['checks']['ocio']={'installed':False,'error':str(exc)}
    try:
        import torch
        r['checks']['neural']={'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),'model_inference_verified':False}
    except Exception as exc:r['checks']['neural']={'available':False,'error':str(exc)}
    if video:
        import cv2
        cap=cv2.VideoCapture(str(Path(video).resolve()))
        if not cap.isOpened():r['checks']['footage']={'error':'Cannot open footage'}
        else:
            frames=0;start=time.perf_counter();shape=None
            while frames<60:
                ok,frame=cap.read()
                if not ok:break
                shape=[int(frame.shape[1]),int(frame.shape[0])];frames+=1
            seconds=time.perf_counter()-start
            r['checks']['footage']={'frames_decoded':frames,'resolution':shape,'decode_fps':round(frames/max(seconds,1e-9),2),'source_bit_depth_verified':False,'full_vfx_render_verified':False}
        cap.release()
    suite=unittest.defaultTestLoader.discover(str(root/'tests'),pattern='test_*.py',top_level_dir=str(root))
    import io
    stream=io.StringIO(); result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    r['checks']['tests']={'run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful(),'details':stream.getvalue()[-8000:]}
    r['release_ready']=False
    r['reason']='Neural model, GPU farm, visual acceptance and production footage require separate hardware/asset verification.'
    if output:Path(output).write_text(json.dumps(r,indent=2),encoding='utf-8')
    return r

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video');p.add_argument('--ocio-config');p.add_argument('--output',default='chroma_acceptance.json');a=p.parse_args()
    r=report(a.video,a.ocio_config,a.output)
    print(json.dumps({k:v for k,v in r.items() if k!='checks'},indent=2))
    print('Tests:',r['checks']['tests']['run'],'passed:',r['checks']['tests']['passed'])
    print('Report:',a.output)
    sys.exit(0 if r['checks']['tests']['passed'] else 1)
