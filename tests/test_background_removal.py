import unittest
import numpy as np
import cv2
from chroma_remove import remove_frame

class BackgroundRemovalTests(unittest.TestCase):
    def test_green_screen_is_transparent(self):
        frame=np.zeros((80,80,3),np.uint8)
        frame[:]=(0,255,0)
        frame[20:60,20:60]=(0,0,255)
        result=remove_frame(frame,'Green / blue screen','#00ff00',.22,.17,0,0,1,0)
        self.assertEqual(result.shape,(80,80,4))
        self.assertLess(int(result[5,5,3]),10)
        self.assertGreater(int(result[40,40,3]),240)
    def test_composite_replaces_green(self):
        frame=np.full((40,40,3),(0,255,0),np.uint8)
        bg=np.full((40,40,3),(255,0,0),np.uint8)
        result=remove_frame(frame,'Green / blue screen','#00ff00',.22,.17,0,0,1,0,bg)
        self.assertEqual(result.shape,(40,40,3))
        self.assertGreater(int(result[10,10,2]),240)

if __name__=='__main__':unittest.main()
