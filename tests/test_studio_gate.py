import json
import tempfile
import unittest
from pathlib import Path
from chroma_delivery import package_delivery
from chroma_studio_gate import validate_delivery

class StudioGateTests(unittest.TestCase):
    def test_delivery_integrity_and_review(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);frames=root/'frames';frames.mkdir()
            for i in range(1001,1004):(frames/f'shot.{i}.png').write_bytes(b'frame'+str(i).encode())
            archive=root/'shot.zip'
            package_delivery(frames,archive,'SH010',expected_start=1001,expected_end=1003)
            self.assertTrue(validate_delivery(archive)['passed'])
            spec={'expected_frames':3,'required_approvals':['supervisor'], 'approvals':{},'quality_checks':{'alpha_mae':{'maximum':0.02,'required':True}}}
            failed=validate_delivery(archive,spec)
            self.assertFalse(failed['passed'])
            self.assertTrue(any('Missing approval' in x for x in failed['errors']))
            spec['approvals']={'supervisor':'approved'};spec['quality_checks']['alpha_mae']['measured']=0.01
            self.assertTrue(validate_delivery(archive,spec)['passed'])
    def test_tampered_frame_rejected(self):
        import zipfile
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);frames=root/'frames';frames.mkdir()
            (frames/'a.0001.png').write_bytes(b'original')
            archive=root/'delivery.zip';package_delivery(frames,archive,'SH020')
            with zipfile.ZipFile(archive) as z: content={n:z.read(n) for n in z.namelist()}
            content['frames/a.0001.png']=b'tampered'
            with zipfile.ZipFile(archive,'w') as z:
                for n,b in content.items():z.writestr(n,b)
            self.assertFalse(validate_delivery(archive)['passed'])
if __name__=='__main__':unittest.main()
