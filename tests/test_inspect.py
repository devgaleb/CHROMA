import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from chroma_inspect import inspect_shot

class ShotInspectionTests(unittest.TestCase):
    def test_inspection_reports_dimensions_and_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            img=np.zeros((64,96,3),dtype=np.uint8);img[:,:,1]=255
            img[16:48,30:65]=[230,20,20]
            path=Path(d)/'shot.png';Image.fromarray(img).save(path)
            report=inspect_shot(path)
            self.assertEqual((report['width'],report['height']),(96,64))
            self.assertGreater(report['estimated_foreground_fraction'],.05)
            self.assertLess(report['estimated_foreground_fraction'],.5)
            self.assertIsInstance(report['suggestions'],list)

if __name__=='__main__':unittest.main()
