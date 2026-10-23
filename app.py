"""CHROMA professional compositing workspace."""
import os, io, zipfile, tempfile, subprocess
from pathlib import Path
import cv2
import numpy as np
import gradio as gr
from PIL import Image
from chroma_pipeline import export_still, export_video, NODES
from chroma_advanced import roto_preview

ROOT=Path(__file__).resolve().parent

def key_frame(frame, color, tolerance, softness, shrink, feather, despill, min_area, bg=None):
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB).astype(np.float32)/255
    target=np.array(color,dtype=np.float32)/255
    hsv=cv2.cvtColor(np.uint8([[np.round(target*255).astype(np.uint8)]]),cv2.COLOR_RGB2HSV)[0,0]
    pixels=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV).astype(np.float32)
    hue_delta=np.abs(pixels[:,:,0]-float(hsv[0])); hue_delta=np.minimum(hue_delta,180-hue_delta)/90
    sat=np.maximum(pixels[:,:,1]/255,0.01)
    val=pixels[:,:,2]/255
    chroma_dist=np.sqrt((hue_delta*1.6)**2+((pixels[:,:,1]-float(hsv[1]))/255*.35)**2)
    confidence=np.clip((chroma_dist-tolerance)/max(softness,.001),0,1)
    confidence=np.maximum(confidence,np.clip((.18-sat)*3,0,1))
    confidence=np.maximum(confidence,np.clip((.12-val)*3,0,1))
    matte=np.uint8(np.clip(confidence,0,1)*255)
    if min_area>0:
        n,labels,stats,_=cv2.connectedComponentsWithStats((matte>160).astype(np.uint8),8)
        for i in range(1,n):
            if stats[i,cv2.CC_STAT_AREA]<min_area: matte[labels==i]=0
    if shrink>0:
        k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*shrink+1,2*shrink+1))
        matte=cv2.erode(matte,k)
    if feather>0: matte=cv2.GaussianBlur(matte,(0,0),feather*.6)
    alpha=matte.astype(np.float32)/255
    channel=int(np.argmax(target))
    others=[x for x in range(3) if x!=channel]
    spill=np.maximum(0,rgb[:,:,channel]-np.maximum(rgb[:,:,others[0]],rgb[:,:,others[1]]))
    edge=(1-alpha)*alpha*4
    rgb[:,:,channel]-=spill*despill*edge
    rgb=np.clip(rgb,0,1)
    rgba=np.dstack((np.uint8(rgb*255),matte))
    if bg is None: return rgba
    back=cv2.resize(bg,(frame.shape[1],frame.shape[0]))
    result=rgb*alpha[:,:,None]+cv2.cvtColor(back,cv2.COLOR_BGR2RGB).astype(np.float32)/255*(1-alpha[:,:,None])
    return np.uint8(np.clip(result,0,1)*255)

def rgb_hex(hexcolor):
    h=hexcolor.lstrip('#'); return [int(h[i:i+2],16) for i in (0,2,4)]

def process_image(image, background, color, tolerance, softness, shrink, feather, despill, min_area, output):
    if image is None: raise gr.Error('Upload a foreground image first.')
    fg=cv2.cvtColor(np.array(image.convert('RGB')),cv2.COLOR_RGB2BGR)
    bg=cv2.cvtColor(np.array(background.convert('RGB')),cv2.COLOR_RGB2BGR) if background else None
    if output=='Composite' and bg is None: raise gr.Error('Upload a background for composite mode.')
    result=key_frame(fg,rgb_hex(color),tolerance,softness,int(shrink),feather,despill,int(min_area),bg if output=='Composite' else None)
    if output=='Alpha matte': result=np.repeat(result[:,:,3:4],3,axis=2)
    path=Path(tempfile.mkdtemp(prefix='chroma_still_'))/'output.png'; Image.fromarray(result).save(path)
    return Image.fromarray(result),str(path)

def process_video(video, background, color, tolerance, softness, shrink, feather, despill, min_area, output, frame_limit):
    if not video: raise gr.Error('Upload a video first.')
    cap=cv2.VideoCapture(video)
    if not cap.isOpened(): raise gr.Error('Cannot decode the video.')
    fps=cap.get(cv2.CAP_PROP_FPS) or 24
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width<1 or height<1: raise gr.Error('Invalid video dimensions.')
    bg=cv2.cvtColor(np.array(background.convert('RGB')),cv2.COLOR_RGB2BGR) if background else None
    if output=='Composite' and bg is None: raise gr.Error('Upload a background for composite mode.')
    out_dir=Path(tempfile.mkdtemp(prefix='chroma_'))
    count=0
    try:
        while count<int(frame_limit):
            ok,frame=cap.read()
            if not ok: break
            result=key_frame(frame,rgb_hex(color),tolerance,softness,int(shrink),feather,despill,int(min_area),bg if output=='Composite' else None)
            if output=='Alpha matte': result=np.repeat(result[:,:,3:4],3,axis=2)
            Image.fromarray(result).save(out_dir/f'{count:06d}.png')
            count+=1
    finally: cap.release()
    if not count: raise gr.Error('No frames could be decoded.')
    if output=='Transparent PNG sequence':
        dest=out_dir/'transparent_frames.zip'
        with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(out_dir.glob('*.png')): archive.write(file,file.name)
        return str(dest),f'Exported {count} RGBA frames at {fps:.2f} fps.'
    dest=out_dir/'render.mp4'
    cmd=['ffmpeg','-y','-loglevel','error','-framerate',str(fps),'-i',str(out_dir/'%06d.png'),'-c:v','libx264','-pix_fmt','yuv420p',str(dest)]
    try: subprocess.run(cmd,check=True,capture_output=True)
    except FileNotFoundError: raise gr.Error('Install FFmpeg to export MP4. PNG sequence mode works without FFmpeg.')
    except subprocess.CalledProcessError as exc: raise gr.Error(exc.stderr.decode(errors='replace')[-700:])
    return str(dest),f'Exported {count} frames at {fps:.2f} fps. Video is silent; source audio is not copied.'


# Interactive before/after video comparison. The preview is intentionally low-resolution;
# use the standard render workflow for full-resolution output.
def swipe_preview(source, color, tolerance, softness, shrink, feather, despill):
    import html, uuid
    if not source: raise gr.Error('Choose a video or upload footage first.')
    cap=cv2.VideoCapture(str(source))
    if not cap.isOpened(): raise gr.Error('Could not open video.')
    fps=cap.get(cv2.CAP_PROP_FPS) or 24.0
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width<1 or height<1: cap.release();raise gr.Error('Invalid video dimensions.')
    scale=min(1.0,640/max(width,height))
    w=max(2,int(width*scale)//2*2);h=max(2,int(height*scale)//2*2)
    directory=Path(tempfile.mkdtemp(prefix='chroma_swipe_'))
    before=directory/'before.mp4';after=directory/'transparent_preview.mp4'
    codec=cv2.VideoWriter_fourcc(*'mp4v')
    first=cv2.VideoWriter(str(before),codec,fps,(w,h))
    second=cv2.VideoWriter(str(after),codec,fps,(w,h))
    if not first.isOpened() or not second.isOpened():
        cap.release();first.release();second.release();raise gr.Error('Could not initialize preview encoders.')
    tile=24
    yy,xx=np.indices((h,w)); checker=(((xx//tile+yy//tile)%2)*38+65).astype(np.uint8)
    bg=np.stack([checker]*3,axis=-1)
    count=0
    try:
        while count < min(120,int(fps*4)):
            ok,frame=cap.read()
            if not ok:break
            frame=cv2.resize(frame,(w,h),interpolation=cv2.INTER_AREA)
            rgba=key_frame(frame,rgb_hex(color),tolerance,softness,int(shrink),feather,despill,0)
            alpha=rgba[:,:,3:4].astype(np.float32)/255.0
            fg=cv2.cvtColor(rgba[:,:,:3],cv2.COLOR_RGB2BGR).astype(np.float32)
            comp=np.uint8(np.clip(fg*alpha+bg*(1-alpha),0,255))
            first.write(frame);second.write(comp);count+=1
    finally:
        cap.release();first.release();second.release()
    if not count:raise gr.Error('No decodable frames found.')
    # Browser-compatible H.264 for cross-platform playback; ffmpeg is an existing dependency.
    for f in (before,after):
        tmp=f.with_name(f.stem+'_h264.mp4')
        result=subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(f),'-an','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(tmp)],capture_output=True)
        if result.returncode==0:tmp.replace(f)
    a='/gradio_api/file='+str(before);b='/gradio_api/file='+str(after)
    ident='chroma'+uuid.uuid4().hex[:10]
    return f"""<div id='{ident}' class='chroma-compare'>
      <div class='compare-stage'>
        <video class='compare-base' src='{html.escape(b,quote=True)}' autoplay muted loop playsinline></video>
        <video class='compare-top' src='{html.escape(a,quote=True)}' autoplay muted loop playsinline></video>
        <div class='compare-line'></div><div class='compare-handle'>↔</div>
        <span class='compare-label compare-left'>ORIGINAL</span><span class='compare-label compare-right'>BACKGROUND REMOVED</span>
        <input aria-label='Swipe between original and background removed video' type='range' min='0' max='100' value='50' class='compare-slider' oninput="this.parentElement.style.setProperty('--split',this.value+'%')">
      </div><p class='compare-note'>Drag to compare • Checkerboard represents transparency • Preview limited to 4 seconds / 640 px</p>
      <script>/* Video elements share the same loop length and start together. */</script>
    </div>"""

def demo_video(name):
    files={'Portrait 4K':'9558525-uhd_2160_4096_25fps.mp4','Cinematic before':'cinematic-before2.mp4','Landscape 4K':'8693323-uhd_3840_2160_30fps.mp4'}
    return str(ROOT/'assets'/'showcase'/files[name])

LOGO_URL=str(ROOT/'assets'/'chroma-logo.png')
css=(ROOT/'assets'/'studio.css').read_text(encoding='utf-8')
with gr.Blocks(title='CHROMA | Professional Keying',css=css,theme=gr.themes.Base(primary_hue='lime',neutral_hue='slate')) as demo:
    gr.HTML(f"""<header class='chroma-header'><div class='chroma-mark'><img src='/gradio_api/file={LOGO_URL}' alt='CHROMA emblem'></div><div class='chroma-wordmark'>CHROMA <small>VFX STUDIO</small></div><div class='chroma-header-right'><span class='status-dot'></span> LOCAL WORKSPACE <span class='version'>STUDIO 2026</span></div></header><section id='hero'><div class='eyebrow'>POST-PRODUCTION / COMPOSITING SUITE</div><h1>Every frame. <em>Under control.</em></h1><p>Precision chroma keying, rotoscoping, motion-aware mattes and film finishing. One unified creative workspace.</p><div class='hero-chips'><span>◉ CHROMA KEY</span><span>◇ ROTO LAB</span><span>▣ NODE COMPOSITOR</span><span>↗ RENDER FARM</span></div></section>""")
    with gr.Row():
        with gr.Column(scale=1, elem_classes=['chroma-sidebar']):
            gr.Markdown('### ◈  KEY CONTROLS')
            color=gr.ColorPicker(value='#00ff00',label='Screen color')
            tolerance=gr.Slider(0,1,value=.22,step=.005,label='Color tolerance')
            softness=gr.Slider(.005,.7,value=.17,step=.005,label='Edge softness')
            shrink=gr.Slider(0,12,value=0,step=1,label='Matte choke (pixels)')
            feather=gr.Slider(0,12,value=.5,step=.25,label='Edge feather')
            despill=gr.Slider(0,2,value=1,step=.05,label='Color spill suppression')
            min_area=gr.Slider(0,10000,value=0,step=25,label='Remove small foreground islands')
            gr.Markdown('**Tip:** Adjust tolerance before choking edges. Excessive choke removes hair and fine detail.')
        with gr.Column(scale=3, elem_classes=['chroma-workspace']):
            with gr.Tabs():
                with gr.Tab('Still image studio'):
                    with gr.Row():
                        img=gr.Image(type='pil',label='Foreground / green screen')
                        back=gr.Image(type='pil',label='Replacement background (optional)')
                    mode=gr.Radio(['Transparent PNG','Composite','Alpha matte'],value='Transparent PNG',label='Output mode')
                    run=gr.Button('Render image',variant='primary')
                    with gr.Row():
                        preview=gr.Image(label='Rendered result',type='pil')
                        download=gr.File(label='Download PNG')
                    run.click(process_image,[img,back,color,tolerance,softness,shrink,feather,despill,min_area,mode],[preview,download])
                with gr.Tab('Swipe comparison · video'):
                    gr.Markdown('### Before / after swipe viewer\nCompare original footage against a keyed preview with a draggable divider. You can use your own video or one of the three included clips.')
                    demo_choice=gr.Dropdown(['Portrait 4K','Cinematic before','Landscape 4K'],value='Cinematic before',label='Included footage')
                    load_demo=gr.Button('Load selected footage')
                    swipe_source=gr.Video(label='Source video (or upload your own)',value=demo_video('Cinematic before'))
                    load_demo.click(demo_video,[demo_choice],[swipe_source])
                    swipe_button=gr.Button('Generate swipe comparison',variant='primary')
                    swipe_result=gr.HTML(label='Interactive comparison')
                    swipe_button.click(swipe_preview,[swipe_source,color,tolerance,softness,shrink,feather,despill],[swipe_result])
                with gr.Tab('Motion / video studio'):
                    video=gr.Video(label='Foreground footage')
                    video_bg=gr.Image(type='pil',label='Replacement background (image)')
                    video_mode=gr.Radio(['Transparent PNG sequence','Composite','Alpha matte'],value='Transparent PNG sequence',label='Export')
                    frame_limit=gr.Slider(1,20000,value=1000,step=1,label='Maximum frames to process')
                    render=gr.Button('Render video',variant='primary')
                    video_file=gr.File(label='Download rendered video / PNG sequence')
                    status=gr.Textbox(label='Render status')
                    render.click(process_video,[video,video_bg,color,tolerance,softness,shrink,feather,despill,min_area,video_mode,frame_limit],[video_file,status])
            with gr.Tab('CHROMA · Rotoscope Lab'):
                gr.Markdown('### Interactive rotoscoping and edge refinement')
                gr.Markdown('Paint directly on the image using the brush tool. Choose foreground to protect hair, hands or clothing; choose erase to remove unwanted screen regions. Export the corrected matte for use in the node compositor. The guided edge filter preserves image-aligned detail; it does not invent missing hair strands.')
                roto_editor=gr.ImageEditor(type='pil',label='Paint matte corrections on foreground',brush=gr.Brush(colors=['#ffffff'],default_size=20),height=510)
                with gr.Row():
                    roto_mode=gr.Radio(['Add foreground','Erase foreground'],value='Add foreground',label='Brush operation')
                    roto_screen=gr.ColorPicker(value='#00ff00',label='Screen color')
                    roto_tol=gr.Slider(0,1,value=.22,step=.01,label='Key tolerance')
                    roto_soft=gr.Slider(.005,.7,value=.17,step=.005,label='Softness')
                    roto_hair=gr.Slider(1,18,value=5,step=1,label='Fine-edge guided radius')
                roto_button=gr.Button('Generate corrected matte',variant='primary')
                with gr.Row():
                    roto_result=gr.Image(label='Corrected transparent foreground')
                    roto_alpha=gr.Image(label='Alpha inspection')
                roto_file=gr.File(label='Download matte PNG')
                def run_roto(*args):
                    try:return roto_preview(*args)
                    except Exception as exc:raise gr.Error(str(exc))
                roto_button.click(run_roto,[roto_editor,roto_screen,roto_tol,roto_soft,roto_mode,roto_hair],[roto_result,roto_alpha,roto_file])
            with gr.Tab('CHROMA VFX · Node compositor'):
                gr.Markdown('### CHROMA VFX Pipeline — float32 processing, motion tracking, correction mattes, grading, light wrap, EXR and ProRes')
                gr.Markdown('**Graph order:** each selected node runs top to bottom. For a transparent export, disable Composite. Track matte uses dense optical flow; supply frame-specific PNG correction masks in a ZIP for difficult frames.')
                with gr.Row():
                    vfx_input=gr.Image(type='pil',label='Still foreground')
                    vfx_bg=gr.Image(type='pil',label='Background plate')
                vfx_clip=gr.Video(label='Motion footage (optional)')
                with gr.Row():
                    vfx_nodes=gr.CheckboxGroup(list(NODES),value=list(NODES),label='Enabled nodes (fixed production order)')
                    vfx_screen=gr.ColorPicker(value='#00ff00',label='Key color')
                with gr.Accordion('Keying · Tracking · Edge detail',open=True):
                    with gr.Row():
                        vt=gr.Slider(0,1,value=.22,step=.005,label='Tolerance')
                        vs=gr.Slider(.005,.7,value=.17,step=.005,label='Softness')
                        vtrack=gr.Slider(0,1,value=.25,step=.05,label='Optical-flow tracking blend')
                        vcorr=gr.Slider(0,1,value=1,step=.05,label='Correction mask strength')
                    with gr.Row():
                        vchoke=gr.Slider(-12,12,value=0,step=1,label='Matte expand / choke')
                        vf=gr.Slider(0,8,value=.5,step=.1,label='Feather')
                        vdetail=gr.Slider(0,1,value=.3,step=.05,label='Fine edge preservation')
                        vd=gr.Slider(0,2,value=1,step=.05,label='Despill')
                with gr.Accordion('Film color · Lighting',open=False):
                    with gr.Row():
                        vex=gr.Slider(-5,5,value=0,step=.1,label='Exposure stops')
                        vcontrast=gr.Slider(.1,3,value=1,step=.05,label='Contrast')
                        vsat=gr.Slider(0,3,value=1,step=.05,label='Saturation')
                        vwrap=gr.Slider(0,1,value=.15,step=.05,label='Light wrap strength')
                        vwr=gr.Slider(1,80,value=8,step=1,label='Light wrap radius')
                with gr.Accordion('Color management · Precision decoding',open=False):
                    gr.Markdown('OCIO transforms require a supplied config and optional PyOpenColorIO. Source and destination color-space names must match the config. 16-bit decode uses FFmpeg RGB48LE, rather than OpenCV 8-bit frames.')
                    ocio_config=gr.File(label='OCIO configuration (.ocio)',file_types=['.ocio'],type='filepath')
                    ocio_input=gr.Textbox(value='scene_linear',label='Input color space')
                    ocio_output=gr.Textbox(value='scene_linear',label='Output color space')
                    high_depth=gr.Checkbox(value=False,label='Decode motion footage at 16-bit RGB precision (FFmpeg)')
                vfx_corrections=gr.File(label='Frame correction masks ZIP (000000.png, 000001.png, ...)',file_types=['.zip'],type='filepath')
                with gr.Row():
                    vfx_still_format=gr.Dropdown(['PNG RGBA 8-bit','PNG RGBA 16-bit','EXR 32-bit float'],value='PNG RGBA 8-bit',label='Still format')
                    vfx_video_format=gr.Dropdown(['PNG sequence','EXR sequence','ProRes 4444 MOV','H.264 MP4'],value='PNG sequence',label='Motion format')
                    vfx_limit=gr.Slider(1,50000,value=1000,step=1,label='Frame limit')
                vfx_render_still=gr.Button('Render VFX still',variant='primary')
                vfx_render_motion=gr.Button('Render VFX motion',variant='primary')
                vfx_file=gr.File(label='VFX output')
                vfx_message=gr.Textbox(label='Render log')
                def vfx_config(nodes,screen,tol,soft,track,corr,choke,feather,detail,despill,exposure,contrast,sat,wrap,radius,ocio_file,ocio_src,ocio_dst,high_precision):
                    return dict(nodes=[n for n in NODES if n in nodes],screen=screen,tolerance=tol,softness=soft,track_strength=track,correction_strength=corr,choke=int(choke),feather=feather,detail=detail,despill=despill,exposure=exposure,contrast=contrast,saturation=sat,wrap=wrap,wrap_radius=radius,ocio_config=ocio_file,ocio_input=ocio_src,ocio_output=ocio_dst,high_bit_depth=high_precision)
                controls=[vfx_nodes,vfx_screen,vt,vs,vtrack,vcorr,vchoke,vf,vdetail,vd,vex,vcontrast,vsat,vwrap,vwr,ocio_config,ocio_input,ocio_output,high_depth]
                def render_vfx_still(fg,bg,fmt,*args):
                    if fg is None:raise gr.Error('Upload a foreground still.')
                    try:return export_still(fg,bg,vfx_config(*args),fmt),'Still render complete.'
                    except Exception as e:raise gr.Error(str(e))
                def render_vfx_motion(clip,bg,fmt,limit,masks,*args):
                    if not clip:raise gr.Error('Upload motion footage.')
                    try:return export_video(clip,bg,vfx_config(*args),fmt,limit,masks)
                    except Exception as e:raise gr.Error(str(e))
                vfx_render_still.click(render_vfx_still,[vfx_input,vfx_bg,vfx_still_format]+controls,[vfx_file,vfx_message])
                vfx_render_motion.click(render_vfx_motion,[vfx_clip,vfx_bg,vfx_video_format,vfx_limit,vfx_corrections]+controls,[vfx_file,vfx_message])
            with gr.Tab('CHROMA · Render Farm'):
                gr.Markdown('### Distributed rendering · Shared storage\nSubmit a video and run workers on other computers that mount the **same shared directory**. Jobs are divided into frame chunks, leased through SQLite and assembled into an image-sequence ZIP. Motion tracking resets at chunk boundaries; inspect those boundaries for temporal discontinuities.')
                with gr.Row():
                    farm_video=gr.Video(label='Source video')
                    farm_bg=gr.Image(type='filepath',label='Optional background')
                farm_root=gr.Textbox(label='Shared render directory (absolute path, accessible to all workers)',value=str(ROOT/'farm_jobs'))
                with gr.Row():
                    farm_fmt=gr.Dropdown(['PNG RGBA 8-bit','PNG RGBA 16-bit','EXR sequence'],value='PNG RGBA 8-bit',label='Sequence format')
                    farm_max=gr.Number(value=1000,precision=0,label='Maximum frames')
                    farm_chunk=gr.Number(value=48,precision=0,label='Frames per worker chunk')
                farm_settings=gr.Textbox(label='Pipeline settings JSON',value='{"nodes":["Key","Edge refine","Despill"],"screen":"#00ff00","tolerance":0.12,"softness":0.2}')
                farm_submit=gr.Button('Submit distributed job',variant='primary')
                farm_job=gr.Textbox(label='Job ID')
                farm_check=gr.Button('Check job status')
                farm_status=gr.JSON(label='Worker progress')
                farm_collect=gr.Button('Collect finished frames')
                farm_output=gr.File(label='Rendered sequence')
                def submit_farm(root,video,bg,settings,fmt,max_frames,chunk):
                    import json
                    from chroma_farm import submit
                    if not video:raise gr.Error('Select a source video')
                    return submit(root,video,bg,json.loads(settings),fmt,int(max_frames),int(chunk))
                def status_farm(root,job):
                    from chroma_farm import status
                    return status(root,job)
                def collect_farm(root,job):
                    from chroma_farm import collect
                    return collect(root,job,str(Path(root)/f'{job}_frames.zip'))
                farm_submit.click(submit_farm,[farm_root,farm_video,farm_bg,farm_settings,farm_fmt,farm_max,farm_chunk],farm_job)
                farm_check.click(status_farm,[farm_root,farm_job],farm_status)
                farm_collect.click(collect_farm,[farm_root,farm_job],farm_output)
                gr.Code('python chroma_farm.py worker --root /path/to/shared/render-directory',language='shell',label='Start on each worker machine')
            with gr.Tab('CHROMA · Neural Detail'):
                gr.Markdown('### Neural fine-detail alpha fusion\nLoad a **compatible, locally supplied ONNX alpha-matting model**. Neural predictions are blended near keyed edges, helping preserve fine hair detail when the model supports it. No pretrained model is bundled; results depend on its training and input normalization.')
                hair_image=gr.Image(type='pil',label='Foreground image')
                hair_base=gr.Image(type='pil',label='Initial alpha matte (grayscale)')
                hair_model=gr.File(label='ONNX matting model',file_types=['.onnx'],type='filepath')
                with gr.Row():
                    hair_size=gr.Slider(128,1024,value=512,step=64,label='Model input size')
                    hair_width=gr.Slider(2,32,value=12,step=1,label='Edge region width')
                    hair_blend=gr.Slider(0,1,value=0.8,step=0.05,label='Neural blend')
                hair_run=gr.Button('Refine alpha with neural model',variant='primary')
                hair_result=gr.Image(type='pil',label='Refined matte')
                def run_hair(rgb,alpha,model,size,width,blend):
                    from chroma_hair import neural_hair_matte
                    if rgb is None or alpha is None or not model:raise gr.Error('Upload image, alpha and ONNX model')
                    arr=np.asarray(rgb.convert('RGB')).astype(np.float32)/255
                    matte=np.asarray(alpha.convert('L').resize((arr.shape[1],arr.shape[0]))).astype(np.float32)/255
                    out=neural_hair_matte(arr,matte,model,int(size),int(width),float(blend))
                    return Image.fromarray(np.uint8(np.clip(out,0,1)*255),'L')
                hair_run.click(run_hair,[hair_image,hair_base,hair_model,hair_size,hair_width,hair_blend],hair_result)
            with gr.Tab('CHROMA · Temporal AI Render'):
                gr.Markdown('### Temporal AI mattes · 16-bit sequence\nGenerate high-precision alpha frames with recurrent neural inference. Requires installed model source and legitimate pretrained weights; no checkpoint is bundled.')
                temporal_video=gr.Video(label='Source footage')
                temporal_weights=gr.File(label='Model weights (.pth)',type='filepath',file_types=['.pth'])
                with gr.Row():
                    temporal_arch=gr.Dropdown(['mobilenetv3','resnet50'],value='mobilenetv3',label='Neural architecture')
                    temporal_device=gr.Dropdown(['auto','cpu','cuda'],value='auto',label='Compute device')
                    temporal_ratio=gr.Slider(.05,1,value=.25,step=.05,label='Inference resolution ratio')
                with gr.Row():
                    temporal_limit=gr.Slider(1,10000,value=240,step=1,label='Frame limit')
                    temporal_blur=gr.Slider(0,3,value=0,step=.1,label='Optional matte smoothing')
                    temporal_reset=gr.Number(value=0,precision=0,label='Reset temporal state every N frames (0 = never)')
                temporal_button=gr.Button('Render 16-bit AI matte sequence',variant='primary')
                temporal_archive=gr.File(label='16-bit matte sequence')
                temporal_report=gr.Code(label='Render manifest',language='json')
                def temporal_run(video,weights,arch,device,ratio,limit,blur,reset):
                    from chroma_temporal_export import export_temporal
                    try:return export_temporal(video,weights,arch,device,ratio,int(limit),blur,int(reset))
                    except Exception as exc:raise gr.Error(str(exc))
                temporal_button.click(temporal_run,[temporal_video,temporal_weights,temporal_arch,temporal_device,temporal_ratio,temporal_limit,temporal_blur,temporal_reset],[temporal_archive,temporal_report])
            with gr.Tab('CHROMA · Shot Review & Repair'):
                gr.Markdown('### Frame-by-frame review and temporal stabilization\nSelect a video frame, paint matte corrections, and export corrected masks. For sequence-wide optical-flow stabilization, supply matching source and alpha-frame folders.')
                review_video=gr.Video(label='Review footage')
                review_index=gr.Number(value=0,precision=0,label='Frame index (zero-based)')
                review_load=gr.Button('Load selected frame')
                review_editor=gr.ImageEditor(type='pil',label='Paint corrections',brush=gr.Brush(colors=['#ffffff'],default_size=12),height=460)
                review_count=gr.Textbox(label='Frame information')
                def load_review(video,index):
                    from chroma_shot_review import load_frame
                    if not video: raise gr.Error('Select footage first')
                    frame,count=load_frame(video,index)
                    return {'background':frame,'layers':[],'composite':frame},f'Frame {max(0,int(index))} / {count-1}'
                review_load.click(load_review,[review_video,review_index],[review_editor,review_count])
                with gr.Row():
                    review_mode=gr.Radio(['Add foreground','Erase foreground'],value='Add foreground',label='Brush mode')
                    review_color=gr.ColorPicker(value='#00ff00',label='Key color')
                    review_tol=gr.Slider(0,1,value=.22,step=.01,label='Tolerance')
                    review_soft=gr.Slider(.005,.7,value=.17,step=.005,label='Softness')
                review_apply=gr.Button('Export corrected frame matte',variant='primary')
                review_alpha=gr.Image(label='Corrected alpha')
                review_rgba=gr.Image(label='Corrected foreground')
                review_mask=gr.File(label='Save correction PNG')
                def review_correct(*args):
                    from chroma_shot_review import corrected_matte
                    try:return corrected_matte(*args)
                    except Exception as exc:raise gr.Error(str(exc))
                review_apply.click(review_correct,[review_editor,review_mode,review_color,review_tol,review_soft],[review_rgba,review_alpha,review_mask])
                gr.Markdown('### Optical-flow temporal repair · 16-bit matte sequence')
                with gr.Row():
                    repair_frames=gr.Textbox(label='Source frame directory')
                    repair_masks=gr.Textbox(label='Matching alpha PNG directory')
                with gr.Row():
                    repair_strength=gr.Slider(0,1,value=.45,step=.05,label='Temporal strength')
                    repair_cut=gr.Slider(.01,.8,value=.24,step=.01,label='Scene cut threshold')
                repair_button=gr.Button('Stabilize and package sequence',variant='primary')
                repair_zip=gr.File(label='16-bit repaired matte ZIP')
                repair_report=gr.JSON(label='Repair report')
                def repair_submit(*args):
                    from chroma_shot_review import repair_sequence
                    try:return repair_sequence(*args)
                    except Exception as exc:raise gr.Error(str(exc))
                repair_button.click(repair_submit,[repair_frames,repair_masks,repair_strength,repair_cut],[repair_zip,repair_report])
            with gr.Tab('CHROMA · Sequence QC'):
                gr.Markdown('### Sequence quality review\nInspect frame-to-frame matte changes, uncertain edges, and possible shot cuts. Motion can trigger review flags; this is a diagnostic, not automatic defect detection.')
                qc_source = gr.Textbox(label='Source image-sequence folder (local path)')
                qc_output = gr.Textbox(label='Report output folder', value='chroma_qc_output')
                with gr.Row():
                    qc_screen = gr.ColorPicker(label='Screen color', value='#00ff00')
                    qc_limit = gr.Slider(1, 2000, value=250, step=1, label='Maximum frames')
                    qc_flicker = gr.Slider(.01, .5, value=.09, step=.01, label='Matte change review threshold')
                qc_button = gr.Button('Analyze sequence', variant='primary')
                qc_report = gr.JSON(label='Sequence diagnostics')
                qc_file = gr.File(label='Download JSON report')
                def qc_run(source, output, screen, limit, flicker):
                    from chroma_sequence import inspect_sequence
                    try:
                        path, report = inspect_sequence(source, output, screen=screen, limit=int(limit), flicker_threshold=float(flicker))
                        return report, path
                    except Exception as exc:
                        raise gr.Error(str(exc))
                qc_button.click(qc_run, [qc_source, qc_output, qc_screen, qc_limit, qc_flicker], [qc_report, qc_file])
            with gr.Tab('CHROMA · Motion QC Pro'):
                gr.Markdown('### Motion-compensated matte inspection\nDetect possible matte flicker after accounting for subject movement. Export a visual contact sheet and machine-readable diagnostics.')
                mq_source = gr.Textbox(label='Source frame directory')
                mq_output = gr.Textbox(label='Output directory', value='chroma_motion_qc')
                with gr.Row():
                    mq_color = gr.ColorPicker(label='Key color', value='#00ff00')
                    mq_limit = gr.Slider(1, 2000, value=250, step=1, label='Maximum frames')
                    mq_threshold = gr.Slider(.01, .5, value=.09, step=.01, label='Review threshold')
                mq_run = gr.Button('Run motion-aware QC', variant='primary')
                mq_summary = gr.JSON(label='QC results')
                mq_preview = gr.Image(label='Visual review contact sheet', type='filepath')
                mq_download = gr.File(label='Download QC report')
                def motion_qc_run(source, output, color, limit, threshold):
                    from chroma_motion_qc import analyze
                    try:
                        report_file, preview_file, report = analyze(source, output, screen=color, limit=int(limit), flicker_threshold=float(threshold))
                        return report, preview_file, report_file
                    except Exception as exc:
                        raise gr.Error(str(exc))
                mq_run.click(motion_qc_run,[mq_source,mq_output,mq_color,mq_limit,mq_threshold],[mq_summary,mq_preview,mq_download])
            gr.Markdown('''### Professional workflow
**1.** Shoot an evenly lit green or blue screen. **2.** Set the key color and refine tolerance. **3.** Inspect the alpha matte at full resolution. **4.** Adjust choke, feather and despill. **5.** Export transparent PNG frames for downstream VFX or a composite MP4.\n\n**CHROMA AI:** The temporal matte workflow accepts compatible separately installed model weights. The standard keyer runs independently.''')
if __name__=='__main__':
    from chroma_uploads import upload_limit
    demo.queue(default_concurrency_limit=2).launch(
        inbrowser=True,
        max_file_size=upload_limit(),
        allowed_paths=[str(ROOT/'assets'/'showcase'), str(ROOT/'assets'/'chroma-logo.png'), tempfile.gettempdir()],
    )
