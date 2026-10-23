"""Tests for queue claims, retry admission, and GPU memory constraints."""
import tempfile, unittest
from chroma_farm import connect, claim
from chroma_gpu import available
from unittest.mock import patch

class FarmAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.db=connect(self.temp.name)
        self.db.execute("INSERT INTO jobs VALUES ('j','v','','{}','PNG RGBA 8-bit',1,24,'queued',0)")
        self.db.execute("INSERT INTO chunks(job,idx,start,stop,status,owner,lease,error,attempts,required_gpu,required_vram_mib) VALUES ('j',0,0,1,'queued','',0,'',0,1,6000)")
    def tearDown(self):self.db.close();self.temp.cleanup()
    def test_cpu_cannot_claim_gpu_job(self):self.assertIsNone(claim(self.db,'cpu',False,free_vram_mib=99999))
    def test_low_vram_cannot_claim(self):self.assertIsNone(claim(self.db,'gpu',True,free_vram_mib=5000))
    def test_gpu_claim_with_sufficient_vram(self):self.assertEqual(claim(self.db,'gpu',True,free_vram_mib=7000)[:2],('j',0))
    def test_exhausted_retry_budget(self):
        self.db.execute("UPDATE chunks SET attempts=3,status='queued'")
        self.assertIsNone(claim(self.db,'gpu',True,max_attempts=3,free_vram_mib=7000))
    def test_expired_lease_reclaimed(self):
        self.db.execute("UPDATE chunks SET status='running',lease=1,attempts=1")
        self.assertIsNotNone(claim(self.db,'gpu2',True,max_attempts=3,free_vram_mib=7000))
    def test_available_memory_reserve(self):
        with patch('chroma_gpu.discover',return_value=[{'index':0,'free_mib':8192}]):
            self.assertEqual(available(0,2048),6144)
            self.assertEqual(available(0,10000),0)
if __name__=='__main__':unittest.main()
