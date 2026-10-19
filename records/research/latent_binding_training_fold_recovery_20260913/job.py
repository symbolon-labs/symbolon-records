"""One exact child; bounded signal cleanup, no automatic retries."""
import fcntl,json,os,resource,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def supervise(command,dest,cap,grace=10):
 dest.mkdir();start=time.monotonic();received=[];child=None;old={};timedout=False;killed=False
 def interrupted(signum,frame):
  if not received:received.append((signum,time.monotonic()))
  if child is not None and child.poll() is None:child.send_signal(signum)
 for sig in (signal.SIGTERM,signal.SIGINT):old[sig]=signal.signal(sig,interrupted)
 try:
  with (dest/'run.log').open('x') as log:
   child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OPENBLAS_NUM_THREADS':'4'})
   if received:child.send_signal(received[0][0])
   write(dest/'launch.json',dict(pid=child.pid,wrapper_pid=os.getpid(),pid_namespace=os.readlink('/proc/self/ns/pid'),cap_s=cap,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
   while child.poll() is None:
    now=time.monotonic()
    if now-start>=cap and not received:timedout=True;child.kill();killed=True;break
    if received and now-received[0][1]>=grace:child.kill();killed=True;break
    try:child.wait(timeout=.1)
    except subprocess.TimeoutExpired:pass
   code=child.wait()
  value=dict(exit_code=code,timeout=timedout,interrupted=bool(received),signal=received[0][0] if received else None,forced_kill=killed,child_reaped=True,restoration_verified=False,elapsed_s=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),new_external_spend_usd=0)
  write(dest/'completion.json',value)
  return value
 finally:
  if child is not None and child.poll() is None:child.kill();child.wait()
  for sig,handler in old.items():signal.signal(sig,handler)
if __name__=='__main__':
 cap=json.loads((ROOT/'FREEZE.json').read_text())['cap_s'];assert cap==37800
 with (ROOT/'job.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  value=supervise([sys.executable,'-B',str(ROOT/'runtime.py'),'run'],ROOT/'job',cap)
 print(json.dumps(value));raise SystemExit(124 if value['timeout'] else 128+value['signal'] if value['interrupted'] else value['exit_code'])
