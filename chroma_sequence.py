"""Sequence-level matte diagnostics for CHROMA."""
from pathlib import Path
import json
import cv2
import numpy as np
from chroma_pipeline import load_rgb, matte_key

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.exr'}

def inspect_sequence(folder, output, screen='#00ff00', tolerance=.22, softness=.17, limit=250, cut_threshold=.24, flicker_threshold=.09):
    files = sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)[:int(limit)]
    if not files:
        raise ValueError('No image frames found')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    previous_image = previous_alpha = None
    rows = []
    for path in files:
        image = np.clip(load_rgb(str(path)), 0, 1).astype(np.float32)
        alpha = np.clip(matte_key(image, screen, tolerance, softness), 0, 1).astype(np.float32)
        record = {'frame': path.name, 'uncertain_edge_fraction': round(float(np.mean((alpha > .05) & (alpha < .95))), 6)}
        if previous_image is not None:
            h, w = previous_image.shape[:2]
            small = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
            change = float(np.mean(np.abs(small - previous_image)))
            record['scene_change_score'] = round(change, 6)
            record['scene_cut'] = change > cut_threshold
            if not record['scene_cut']:
                prior = cv2.resize(previous_alpha, (alpha.shape[1], alpha.shape[0]), interpolation=cv2.INTER_LINEAR)
                # Diagnostic only: moving objects can legitimately change mattes.
                delta = float(np.mean(np.abs(alpha - prior)))
                record['matte_change_score'] = round(delta, 6)
                record['review_matte_change'] = delta > flicker_threshold
        previous_image = cv2.resize(image, (160, 90), interpolation=cv2.INTER_AREA)
        previous_alpha = alpha
        rows.append(record)
    report = {'application': 'CHROMA', 'frames': len(rows), 'review_count': sum(bool(r.get('review_matte_change')) for r in rows), 'results': rows, 'note': 'Matte changes may be caused by legitimate motion; review flagged frames visually.'}
    destination = out / 'sequence_report.json'
    destination.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return str(destination), report
