import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from chroma_delivery import inspect_sequence,package_delivery

class DeliveryTests(unittest.TestCase):
    def test_missing_frame_blocks_delivery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for i in (1,3):(root/f'shot.{i:04d}.png').write_bytes(b'valid')
            self.assertEqual(inspect_sequence(root)['missing_frames'],[2])
            with self.assertRaises(ValueError):package_delivery(root,root/'out.zip','shot')
    def test_package_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for i in (1,2):(root/f'shot.{i:04d}.png').write_bytes(b'frame'+bytes([i]))
            archive=root/'delivery.zip'
            package_delivery(root,archive,'scene_001')
            with zipfile.ZipFile(archive) as z:
                manifest=json.loads(z.read('manifest.json'))
                self.assertTrue(manifest['sequence']['valid'])
                self.assertEqual(len(manifest['sequence']['frames']),2)
                self.assertIn('frames/shot.0001.png',z.namelist())
    def test_ambiguous_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'a0001.png').write_bytes(b'a')
            (root/'b0001.png').write_bytes(b'b')
            with self.assertRaises(ValueError):inspect_sequence(root)
if __name__=='__main__':unittest.main()
