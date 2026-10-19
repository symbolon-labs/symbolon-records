import json,os,signal,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from job import supervise
class Lifecycle(unittest.TestCase):
 def test_normal_and_timeout(self):
  with tempfile.TemporaryDirectory() as d:
   a=supervise([sys.executable,'-c','pass'],Path(d)/'normal',5)
   self.assertEqual(a['exit_code'],0);self.assertTrue(a['child_reaped']);self.assertFalse(a['interrupted'])
   z=supervise([sys.executable,'-c','import time;time.sleep(30)'],Path(d)/'timeout',.2)
   self.assertTrue(z['timeout']);self.assertEqual(z['exit_code'],-signal.SIGKILL);self.assertTrue(z['child_reaped'])
 def test_signal_forward_and_reap(self):
  for sig in [signal.SIGTERM,signal.SIGINT]:
   with self.subTest(signal=sig),tempfile.TemporaryDirectory() as d:
    dest=Path(d)/'job'
    script='from job import supervise;from pathlib import Path;import sys;supervise([sys.executable,"-c","import time;time.sleep(30)"],Path(sys.argv[1]),20,.5)'
    p=subprocess.Popen([sys.executable,'-B','-c',script,str(dest)],cwd=Path(__file__).parent,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
     end=time.monotonic()+5
     while not (dest/'launch.json').exists():
      if time.monotonic()>end:self.fail('launch missing')
      time.sleep(.02)
     p.send_signal(sig);self.assertEqual(p.wait(timeout=5),0)
     x=json.loads((dest/'completion.json').read_text());self.assertEqual(x['signal'],sig);self.assertTrue(x['child_reaped']);self.assertTrue(x['interrupted']);self.assertFalse(x['timeout']);self.assertNotEqual(x['exit_code'],0)
     pid=json.loads((dest/'launch.json').read_text())['pid']
     with self.assertRaises(ProcessLookupError):os.kill(pid,0)
    finally:
     if p.poll() is None:p.kill();p.wait()
if __name__=='__main__':unittest.main()
