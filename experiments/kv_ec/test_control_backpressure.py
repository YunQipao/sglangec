import concurrent.futures,threading,time,unittest
from unittest.mock import patch
from .remote_rs_client import RemoteRSClient

class ControlBackpressureTests(unittest.TestCase):
 def test_shared_gate_bounds_calls_and_releases_after_failure(self):
  gate=threading.BoundedSemaphore(1);lock=threading.Lock();active=0;maximum=0
  def action(self,op,**kw):
   nonlocal active,maximum
   with lock:active+=1;maximum=max(maximum,active)
   try:
    time.sleep(.005)
    if op=='fail':raise ValueError('expected')
    return {'ok':True},b''
   finally:
    with lock:active-=1
  with patch.object(RemoteRSClient,'_call_unbounded',action):
   clients=[RemoteRSClient('unused',timeout=1,control_semaphore=gate) for _ in range(16)]
   with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
    results=list(pool.map(lambda c:c.call('ok'),clients))
   self.assertEqual(maximum,1);self.assertEqual(len(results),16)
   with self.assertRaises(ValueError):clients[0].call('fail')
   self.assertEqual(clients[1].call('ok')[0],{'ok':True})
 def test_gate_timeout_does_not_release_foreign_holder(self):
  gate=threading.BoundedSemaphore(1);gate.acquire()
  c=RemoteRSClient('unused',timeout=.01,control_semaphore=gate)
  with self.assertRaises(TimeoutError):c.call('unused')
  self.assertFalse(gate.acquire(blocking=False));gate.release()
