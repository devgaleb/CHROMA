"""Motion-compensated matte QC and visual review sheets for CHROMA."""
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw
from chroma_pipeline import load_rgb, matte_key

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.exr'}

def analyze(folder, output, screen='#00ff00', tolerance=.22, softness=.17,
            limit=250, flicker_threshold=.09, cut_threshold=.24, preview_size=320):
    source = Path(folder).expanduser().resolve()
    if not source.is_dir():
        raise ValueError('Source must be an existing local directory')
    frames = sorted(p for p in source.iterdir() if p.suffix.lower() in EXTENSIONS and p.is_file())[:int(limit)]
    if not frames:
        raise ValueError('No supported frames found')
    dest = Path(output).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    previous_gray = previous_alpha = None
    previous_rgb = None
    results, thumbnails = [], []
    for index, frame in enumerate(frames):
        rgb = np.clip(load_rgb(str(frame)), 0, 1).astype(np.float32)
        alpha = np.clip(matte_key(rgb, screen, tolerance, softness), 0, 1).astype(np.float32)
        w = min(int(preview_size), rgb.shape[1]); h = max(1, round(rgb.shape[0]*w/rgb.shape[1]))
        small = cv2.resize(rgb, (w,h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor((small*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        small_alpha = cv2.resize(alpha,(w,h),interpolation=cv2.INTER_AREA)
        row = {'index':index, 'frame':frame.name,
               'uncertain_edge_fraction':round(float(np.mean((alpha>.05)&(alpha<.95))),6)}
        if previous_gray is not None:
            if previous_gray.shape != gray.shape:
                previous_gray = cv2.resize(previous_gray,(w,h))
                previous_alpha = cv2.resize(previous_alpha,(w,h))
                previous_rgb = cv2.resize(previous_rgb,(w,h))
            scene_delta = float(np.mean(np.abs(small-previous_rgb)))
            row['scene_change_score'] = round(scene_delta,6)
            row['scene_cut'] = scene_delta > cut_threshold
            if not row['scene_cut']:
                # Flow maps pixels in the current frame back into the preceding frame.
                flow = cv2.calcOpticalFlowFarneback(gray,previous_gray,None,.5,3,21,3,5,1.2,0)
                x,y = np.meshgrid(np.arange(w,dtype=np.float32),np.arange(h,dtype=np.float32))
                warped = cv2.remap(previous_alpha,x+flow[...,0],y+flow[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
                error = np.abs(small_alpha-warped)
                row['motion_compensated_matte_error'] = round(float(np.mean(error)),6)
                row['review_matte_change'] = row['motion_compensated_matte_error'] > flicker_threshold
        previous_gray,previous_alpha,previous_rgb = gray,small_alpha,small
        checker = np.indices((h,w)).sum(axis=0)//16 % 2
        background = np.where(checker[...,None]==0,.22,.36).astype(np.float32)
        composite = small*small_alpha[...,None]+background*(1-small_alpha[...,None])
        side = np.concatenate((small,np.repeat(small_alpha[...,None],3,axis=2),composite),axis=1)
        thumbnails.append(Image.fromarray((side*255).clip(0,255).astype('uint8')))
        results.append(row)
    columns=2; margin=14; label_h=28; tile_w=thumbnails[0].width; tile_h=thumbnails[0].height+label_h
    # Bound preview size for long sequences; reports still cover all frames.
    preview_count=min(len(thumbnails),60)
    sheet=Image.new('RGB',(columns*tile_w+(columns+1)*margin,((preview_count+columns-1)//columns)*tile_h+(((preview_count+columns-1)//columns)+1)*margin),'#15191e')
    draw=ImageDraw.Draw(sheet)
    for i in range(preview_count):
        x=margin+(i%columns)*(tile_w+margin); y=margin+(i//columns)*(tile_h+margin)
        sheet.paste(thumbnails[i],(x,y+label_h))
        label=f'{i:04d}  {frames[i].name[:32]}'
        if results[i].get('review_matte_change'): label+='  REVIEW'
        draw.text((x+4,y+7),label,fill='#b8f570' if results[i].get('review_matte_change') else '#e5e9ed')
    sheet_path=dest/'review_sheet.jpg';sheet.save(sheet_path,quality=88)
    report={'application':'CHROMA','algorithm':'Farneback motion-compensated matte difference',
            'frame_count':len(results),'review_count':sum(bool(r.get('review_matte_change')) for r in results),
            'preview_count':preview_count,'results':results,
            'limitations':'Optical flow can fail on occlusion, cuts, fine hair, and rapid movement. Review flags are diagnostics, not proof of defects.'}
    json_path=dest/'motion_qc.json';json_path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return str(json_path),str(sheet_path),report
