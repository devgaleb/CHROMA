# CHROMA pretrained temporal matting

CHROMA supports importing official, previously trained human-video matting checkpoints through `chroma_pretrained.py`. This is **not** a newly trained CHROMA model. Model weights are not bundled in this distribution.

## Source and license

Official model: https://github.com/PeterL1n/RobustVideoMatting

The upstream code is GPL-3.0. Review its license before combining or distributing code, and retain all required notices. Download checkpoints only from the official project links. The checkpoint must match the selected MobileNetV3 or ResNet50 architecture. Use the upstream `model.py` and its model package on your Python path.

## Usage

```python
from chroma_pretrained import TemporalMatte
from PIL import Image
import numpy as np
engine = TemporalMatte('/path/to/rvm_mobilenetv3.pth')
frame = np.asarray(Image.open('/path/to/frame.png').convert('RGB'))
alpha = engine.predict(frame)
Image.fromarray((alpha*65535).astype('uint16')).save('matte_16bit.png')
```

Keep one `TemporalMatte` instance per shot, calling `predict` sequentially on frames. Call `reset()` on scene cuts. For distributed chunks, overlap preceding frames to warm up recurrent state; never assume independently initialized chunks are temporally identical. This adapter is for human matting; it is not a universal object or hair reconstruction model.

## Validation

Run `python chroma_quality.py --pred <predictions> --truth <ground_truth>` using a held-out ground-truth dataset. No accuracy, 4K/8K throughput, or neural training claims are made without actual measured results.
