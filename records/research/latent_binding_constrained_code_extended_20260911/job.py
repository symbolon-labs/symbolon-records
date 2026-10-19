"""One exclusive capped child, reaped inside its original PID namespace."""
import fcntl
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
if __name__=='__main__':
    cap=json.loads((ROOT/'FREEZE.json').read_text())['cap_s'];assert cap==2400
    with (ROOT/'job.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);dest=ROOT/'job';dest.mkdir();start=time.monotonic()
        with (dest/'run.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-B',str(ROOT/'runtime.py'),'run'],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OPENBLAS_NUM_THREADS':'4'})
            write(dest/'launch.json',dict(pid=child.pid,wrapper_pid=os.getpid(),pid_namespace=os.readlink('/proc/self/ns/pid'),cap_s=cap,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
            timedout=False
            try:code=child.wait(timeout=cap)
            except subprocess.TimeoutExpired:timedout=True;child.kill();child.wait();code=124
            except BaseException:child.kill();child.wait();raise
        value=dict(exit_code=code,timeout=timedout,child_reaped=True,elapsed_s=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),new_external_spend_usd=0)
        write(dest/'completion.json',value);print(json.dumps(value));raise SystemExit(code)
