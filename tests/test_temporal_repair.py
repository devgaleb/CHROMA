import unittest
import numpy as np
from chroma_temporal_repair import stabilize

class TemporalRepairTests(unittest.TestCase):
    def test_constant_frames(self):
        rgb=np.zeros((32,32,3),np.float32)
        alpha=np.full((32,32),.5,np.float32)
        result=list(stabilize([rgb,rgb],[alpha,alpha]))
        self.assertEqual(len(result),2)
        self.assertTrue(np.allclose(result[1],alpha))
    def test_scene_cut_resets(self):
        a=np.zeros((32,32,3),np.float32)
        b=np.ones((32,32,3),np.float32)
        mattes=[np.zeros((32,32),np.float32),np.ones((32,32),np.float32)]
        result=list(stabilize([a,b],mattes))
        self.assertTrue(np.allclose(result[1],1))
    def test_invalid_strength(self):
        with self.assertRaises(ValueError): list(stabilize([],[],strength=2))
