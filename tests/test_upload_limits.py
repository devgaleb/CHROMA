import os
import unittest
from unittest.mock import patch
from chroma_uploads import upload_limit

class UploadLimitTests(unittest.TestCase):
    def test_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(upload_limit(), "20gb")
    def test_custom(self):
        with patch.dict(os.environ, {"CHROMA_MAX_UPLOAD_SIZE": "100gb"}):
            self.assertEqual(upload_limit(), "100gb")
    def test_unlimited(self):
        with patch.dict(os.environ, {"CHROMA_MAX_UPLOAD_SIZE": "0"}):
            self.assertIsNone(upload_limit())
    def test_invalid(self):
        with patch.dict(os.environ, {"CHROMA_MAX_UPLOAD_SIZE": "hello"}):
            with self.assertRaises(ValueError):
                upload_limit()
