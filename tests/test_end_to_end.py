"""Functional smoke tests for the CHROMA still and video export paths."""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from PIL import Image

class StudioIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app import key_frame, process_image, process_video
        cls.key_frame=staticmethod(key_frame)
        cls.process_image=staticmethod(process_image)
        cls.process_video=staticmethod(process_video)

    def test_key_preserves_foreground_and_removes_green(self):
        frame=np.zeros((96,128,3),dtype=np.uint8)
        frame[:]=(0,255,0)
        cv2.rectangle(frame,(35,25),(90,80),(0,0,255),-1)
        result=self.key_frame(frame,[0,255,0],.22,.17,0,.5,1,0)
        self.assertEqual(result.shape,(96,128,4))
        self.assertLess(int(result[5,5,3]),20)
        self.assertGreater(int(result[50,60,3]),240)

    def test_still_export_creates_unique_png(self):
        a=np.full((80,100,3),(0,255,0),dtype=np.uint8)
        a[20:60,30:70]=(255,0,0)
        _,path=self.process_image(Image.fromarray(a),None,'#00ff00',.22,.17,0,.5,1,0,'Transparent PNG')
        self.assertTrue(Path(path).exists())
        self.assertEqual(Image.open(path).mode,'RGBA')

    def test_video_png_sequence(self):
        with tempfile.TemporaryDirectory() as d:
            video=Path(d)/'input.avi'
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),12,(96,64))
            if not writer.isOpened():self.skipTest('OpenCV video encoder unavailable')
            for x in (10,20,30):
                f=np.zeros((64,96,3),dtype=np.uint8);f[:]=(0,255,0)
                cv2.rectangle(f,(x,15),(x+20,45),(0,0,255),-1);writer.write(f)
            writer.release()
            output,msg=self.process_video(str(video),None,'#00ff00',.22,.17,0,.5,1,0,'Transparent PNG sequence',3)
            import zipfile
            with zipfile.ZipFile(output) as z:
                self.assertEqual(len(z.namelist()),3)
                self.assertEqual(Image.open(z.open('000000.png')).mode,'RGBA')

if __name__=='__main__':unittest.main()
