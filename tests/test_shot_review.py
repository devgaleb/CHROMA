import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from chroma_shot_review import load_frame,repair_sequence

class ShotReviewTests(unittest.TestCase):
    def test_repair_outputs_16bit(self):
        with tempfile.TemporaryDirectory() as t:
            base=Path(t); src=base/'src'; masks=base/'masks';src.mkdir();masks.mkdir()
            for i in range(3):
                rgb=np.zeros((32,32,3),dtype=np.uint8);rgb[:,:,1]=255;rgb[8:24,8+i:24+i]=[255,0,0]
                Image.fromarray(rgb).save(src/f'{i:04d}.png')
                Image.fromarray(np.where(rgb[:,:,0]>0,255,0).astype('uint8')).save(masks/f'{i:04d}.png')
            archive,report=repair_sequence(src,masks,.4,.3)
            import zipfile
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(len([x for x in z.namelist() if x.endswith('.png')]),3)
            self.assertEqual(report['frames'],3)

if __name__=='__main__':unittest.main()
