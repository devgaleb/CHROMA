import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from chroma_sequence import inspect_sequence

class SequenceTests(unittest.TestCase):
    def test_sequence_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'frames'
            folder.mkdir()
            for i in range(3):
                arr = np.zeros((32, 32, 3), dtype=np.uint8)
                arr[:, :, 1] = 255
                arr[8:24, 8+i:24+i] = [255, 0, 0]
                Image.fromarray(arr).save(folder / f'{i:03}.png')
            report_file, result = inspect_sequence(folder, Path(tmp) / 'report')
            self.assertEqual(result['frames'], 3)
            self.assertTrue(Path(report_file).exists())
            self.assertEqual(len(result['results']), 3)

if __name__ == '__main__': unittest.main()
