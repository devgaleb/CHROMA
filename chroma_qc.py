"""CHROMA quality-control contact sheets and shot reports."""
from pathlib import Path
import argparse
import json
import cv2
import numpy as np
from chroma_inspect import inspect_shot
from chroma_pipeline import load_rgb, matte_key

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.exr'}

def analyze_folder(source, output, screen='#00ff00', tolerance=.22, softness=.17, limit=250):
    source, output = Path(source), Path(output)
    if not source.is_dir():
        raise NotADirectoryError(source)
    files = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)[:limit]
    if not files:
        raise ValueError('No supported image frames found')
    output.mkdir(parents=True, exist_ok=True)
    frames = []
    for path in files:
        rgb = np.clip(load_rgb(str(path)), 0, 1)
        alpha = np.clip(matte_key(rgb, screen, tolerance, softness), 0, 1)
        report = inspect_shot(str(path), screen, tolerance, softness)
        report['filename'] = path.name
        preview = (rgb * 255).astype(np.uint8)
        checker = np.indices(alpha.shape).sum(axis=0) // 16 % 2
        background = np.where(checker[..., None] == 0, 60, 105).astype(np.float32)
        composite = preview * alpha[..., None] + background * (1 - alpha[..., None])
        mask = (alpha * 255).astype(np.uint8)
        tile = np.concatenate([preview, np.repeat(mask[..., None], 3, axis=2), composite.astype(np.uint8)], axis=1)
        tile = cv2.resize(tile, (900, max(1, int(900 * tile.shape[0] / tile.shape[1]))), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(output / (path.stem + '_qc.jpg')), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
        frames.append(report)
    result = {'product': 'CHROMA', 'frame_count': len(frames), 'mean_uncertain_matte_fraction': float(np.mean([r['uncertain_matte_fraction'] for r in frames])), 'frames': frames}
    (output / 'quality_report.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CHROMA sequence quality-control reports and previews')
    parser.add_argument('frames', help='Folder of source image frames')
    parser.add_argument('--output', required=True)
    parser.add_argument('--screen', default='#00ff00')
    parser.add_argument('--limit', type=int, default=250)
    args = parser.parse_args()
    print(json.dumps(analyze_folder(args.frames, args.output, args.screen, limit=args.limit), indent=2))
