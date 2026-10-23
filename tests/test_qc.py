import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from chroma_qc import analyze_folder

class QualityControlTests(unittest.TestCase):
    def test_sequence_reports_and_previews(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            src = root / 'frames'
            src.mkdir()
            for n in range(2):
                frame = np.zeros((48, 64, 3), np.uint8)
                frame[:] = (0, 255, 0)
                frame[12:36, 20+n:44+n] = (0, 0, 255)
                cv2.imwrite(str(src / f'{n:04}.png'), frame)
            report = analyze_folder(src, root / 'qc')
            self.assertEqual(report['frame_count'], 2)
            self.assertTrue((root / 'qc' / 'quality_report.json').is_file())
            self.assertEqual(len(list((root / 'qc').glob('*_qc.jpg'))), 2)
