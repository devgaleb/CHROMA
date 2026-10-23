"""CHROMA shared-storage render queue with retry leases and GPU-affinity workers."""
import argparse, json, os, sqlite3, time, uuid, zipfile, socket
from pathlib import Path
import cv2
from chroma_gpu import available
import numpy as np
from PIL import Image
from chroma_pipeline import apply_nodes, load_rgb, save_exr

SCHEMA='''CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,video TEXT,background TEXT,settings TEXT,format TEXT,frames INTEGER,fps REAL,status TEXT,created REAL);
CREATE TABLE IF NOT EXISTS chunks(job TEXT,idx INTEGER,start INTEGER,stop INTEGER,status TEXT,owner TEXT,lease REAL,error TEXT,attempts INTEGER DEFAULT 0,required_gpu INTEGER DEFAULT 0,required_vram_mib INTEGER DEFAULT 0,PRIMARY KEY(job,idx));'''

def connect(root):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(root/'chroma_farm.sqlite',timeout=30,isolation_level=None)
    db.execute('PRAGMA journal_mode=WAL');db.executescript(SCHEMA)
    cols={r[1] for r in db.execute('PRAGMA table_info(chunks)')}
    for name,typ in [('attempts','INTEGER DEFAULT 0'),('required_gpu','INTEGER DEFAULT 0'),('required_vram_mib','INTEGER DEFAULT 0')]:
        if name not in cols:db.execute(f'ALTER TABLE chunks ADD COLUMN {name} {typ}')
    return db

def submit(root,video,background,settings,format,max_frames=1000,chunk_size=48,required_gpu=False,required_vram_mib=0):
    if int(required_vram_mib)<0:raise ValueError("Negative VRAM request")
    if required_vram_mib:required_gpu=True
    video=Path(video).resolve()
    if not video.is_file():raise FileNotFoundError(video)
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():raise ValueError('Cannot decode footage')
    total=min(int(max_frames),int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps=cap.get(cv2.CAP_PROP_FPS) or 24;cap.release()
    if total<=0 or chunk_size<1:raise ValueError('Invalid frame count or chunk size')
    if format not in ('PNG RGBA 8-bit','PNG RGBA 16-bit','EXR sequence'):raise ValueError('Unsupported output format')
    jid=uuid.uuid4().hex[:16];db=connect(root)
    with db:
        db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?)',(jid,str(video),str(Path(background).resolve()) if background else '',json.dumps(settings),format,total,fps,'queued',time.time()))
        for idx,start in enumerate(range(0,total,int(chunk_size))):
            db.execute('INSERT INTO chunks(job,idx,start,stop,status,owner,lease,error,attempts,required_gpu,required_vram_mib) VALUES (?,?,?,?,?,?,?,?,?,?,?)',(jid,idx,start,min(total,start+int(chunk_size)),'queued','',0,'',0,int(required_gpu),int(required_vram_mib)))
    db.close();return jid

def claim(db,worker,has_gpu=False,lease=1800,max_attempts=3,free_vram_mib=0):
    db.execute('BEGIN IMMEDIATE')
    try:
        row=db.execute("SELECT job,idx,start,stop FROM chunks WHERE attempts<? AND required_gpu<=? AND required_vram_mib<=? AND (status='queued' OR (status='running' AND lease<?)) ORDER BY attempts, rowid LIMIT 1",(max_attempts,int(has_gpu),int(free_vram_mib),time.time())).fetchone()
        if row:
            db.execute("UPDATE chunks SET status='running',owner=?,lease=?,error='',attempts=attempts+1 WHERE job=? AND idx=?",(worker,time.time()+lease,row[0],row[1]))
        db.commit();return row
    except Exception:
        db.rollback();raise

def render_chunk(root,db,task,warmup=3):
    job,idx,start,stop=task
    video,background,settings,fmt=db.execute('SELECT video,background,settings,format FROM jobs WHERE id=?',(job,)).fetchone()
    folder=Path(root).resolve()/job/f'{idx:06d}';folder.mkdir(parents=True,exist_ok=True)
    # Decode preceding frames to warm up optical-flow/temporal state across chunk boundaries.
    cap=cv2.VideoCapture(video);begin=max(0,start-int(warmup));cap.set(cv2.CAP_PROP_POS_FRAMES,begin)
    bg=load_rgb(background) if background else None;state={};cfg=json.loads(settings)
    try:
        for frame in range(begin,stop):
            ok,bgr=cap.read()
            if not ok:raise RuntimeError(f'Could not decode frame {frame}')
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).astype(np.float32)/255
            rgba=apply_nodes(rgb,bg,cfg,state,frame)
            if frame<start:continue
            dest=folder/f'{frame:08d}.{ "exr" if fmt=="EXR sequence" else "png"}'
            temp=folder/f'{frame:08d}.partial.{ "exr" if fmt=="EXR sequence" else "png"}'
            if fmt=='EXR sequence':save_exr(temp,rgba)
            elif fmt=='PNG RGBA 16-bit':
                if not cv2.imwrite(str(temp),cv2.cvtColor(np.uint16(np.clip(rgba,0,1)*65535),cv2.COLOR_RGBA2BGRA)):raise RuntimeError('16-bit PNG write failed')
            else:Image.fromarray(np.uint8(np.clip(rgba,0,1)*255),'RGBA').save(temp, format='PNG')
            os.replace(temp,dest)
    finally:cap.release()

def work(root,once=False,poll=3,gpu=None,max_attempts=3,warmup=3,reserve_vram_mib=1024):
    if gpu is not None:os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    worker=f'{socket.gethostname()}-{os.getpid()}-gpu{gpu if gpu is not None else "cpu"}'
    db=connect(root)
    try:
        while True:
            task=claim(db,worker,gpu is not None,max_attempts=max_attempts,free_vram_mib=available(gpu,reserve_vram_mib) if gpu is not None else 0)
            if not task:
                if once:return
                time.sleep(poll);continue
            try:
                render_chunk(root,db,task,warmup)
                with db:db.execute("UPDATE chunks SET status='done',lease=0 WHERE job=? AND idx=? AND owner=?",(task[0],task[1],worker))
            except Exception as exc:
                with db:
                    db.execute("UPDATE chunks SET status=CASE WHEN attempts>=? THEN 'failed' ELSE 'queued' END,error=?,lease=0 WHERE job=? AND idx=? AND owner=?",(max_attempts,str(exc)[:1000],task[0],task[1],worker))
                if once:raise
    finally:db.close()

def status(root,job):
    db=connect(root)
    rows=db.execute('SELECT status,count(*) FROM chunks WHERE job=? GROUP BY status',(job,)).fetchall();db.close()
    return dict(rows)

def collect(root,job,output):
    db=connect(root)
    rows=db.execute('SELECT idx,start,stop,status FROM chunks WHERE job=? ORDER BY idx',(job,)).fetchall()
    if not rows or any(r[3]!='done' for r in rows):raise RuntimeError('Job is not fully rendered')
    with zipfile.ZipFile(output,'w',zipfile.ZIP_STORED,allowZip64=True) as archive:
        for idx,start,stop,_ in rows:
            for frame in range(start,stop):
                folder=Path(root).resolve()/job/f'{idx:06d}'
                matches=[p for p in folder.glob(f'{frame:08d}.*') if '.partial.' not in p.name]
                if len(matches)!=1:raise RuntimeError(f'Missing or ambiguous frame {frame}')
                archive.write(matches[0],matches[0].name)
    db.close();return str(output)

def main():
    p=argparse.ArgumentParser(description='CHROMA distributed render workers (trusted shared filesystem)')
    p.add_argument('action',choices=['worker','status','collect']);p.add_argument('--root',required=True);p.add_argument('--job');p.add_argument('--output');p.add_argument('--once',action='store_true');p.add_argument('--gpu',type=int);p.add_argument('--max-attempts',type=int,default=3);p.add_argument('--warmup',type=int,default=3);p.add_argument('--reserve-vram-mib',type=int,default=1024)
    a=p.parse_args()
    if a.action=='worker':work(a.root,a.once,gpu=a.gpu,max_attempts=a.max_attempts,warmup=a.warmup,reserve_vram_mib=a.reserve_vram_mib)
    elif a.action=='status':print(json.dumps(status(a.root,a.job),indent=2))
    else:print(collect(a.root,a.job,a.output))
if __name__=='__main__':main()
