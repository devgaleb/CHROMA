import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chroma_doctor import collect

class DoctorTests(unittest.TestCase):
    def test_report_shape(self):
        report = collect()
        self.assertIn('python', report)
        self.assertIn('modules', report)
        self.assertIn('ffmpeg', report['executables'])

    def test_core_source_present(self):
        root = Path(__file__).resolve().parents[1]
        for name in ('app.py', 'app_key.py', 'chroma_pipeline.py', 'chroma_farm.py'):
            with self.subTest(name=name):
                self.assertTrue((root / name).is_file())
