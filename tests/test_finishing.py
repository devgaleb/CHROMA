import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from chroma_finishing import decontaminate_edges,matte_metrics,sequence_manifest
from chroma_pipeline import apply_nodes,settings_from_json

class FinishingTests(unittest.TestCase):
    def test_edge_decontamination_keeps_alpha_and_dimensions(self):
        rgb=np.zeros((24,24,3),np.float32);rgb[:,:,1]=1
        rgb[6:18,6:18]=[1,0,0]
        alpha=np.zeros((24,24),np.float32);alpha[6:18,6:18]=1;alpha[5:19,5:19]=np.maximum(alpha[5:19,5:19],0.3)
        out=decontaminate_edges(rgb,alpha)
        self.assertEqual(out.shape,rgb.shape)
        self.assertTrue(np.isfinite(out).all())
        self.assertTrue(np.array_equal(out[10,10],rgb[10,10]))
    def test_metrics(self):
        a=np.zeros((3,3),np.float32)
        self.assertEqual(matte_metrics(a,a)['mae'],0)
    def test_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            Image.new('RGBA',(2,2)).save(Path(tmp)/'000001.png')
            m=sequence_manifest(tmp,Path(tmp)/'manifest.json')
            self.assertEqual(m['frame_count'],1)
            self.assertEqual(len(m['frames'][0]['sha256']),64)
    def test_node_pipeline(self):
        rgb=np.zeros((32,32,3),np.float32);rgb[:,:,1]=1;rgb[8:24,8:24]=[1,0,0]
        cfg=settings_from_json({'nodes':['Key','Edge decontaminate']})
        rgba=apply_nodes(rgb,None,cfg)
        self.assertEqual(rgba.shape,(32,32,4))
        self.assertGreater(rgba[16,16,3],rgba[0,0,3])
if __name__=='__main__':unittest.main()
