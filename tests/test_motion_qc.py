import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from chroma_motion_qc import analyze

class MotionQCTest(unittest.TestCase):
    def test_reports_and_sheet(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=root/'frames';source.mkdir()
            for i in range(3):
                frame=np.zeros((48,64,3),dtype=np.uint8);frame[:,:,1]=255
                frame[10:35,12+i*2:30+i*2]=[255,0,0]
                Image.fromarray(frame).save(source/f'{i:04d}.png')
            report, sheet, data=analyze(source,root/'output',limit=3)
            self.assertEqual(data['frame_count'],3)
            self.assertTrue(Path(report).is_file())
            self.assertTrue(Path(sheet).is_file())
            self.assertIn('motion_compensated_matte_error',data['results'][1])
