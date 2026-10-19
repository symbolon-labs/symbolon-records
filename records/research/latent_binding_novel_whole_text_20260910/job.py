"""One exclusive process per endpoint; user-authorized no wall-time kill."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent
def write(p,value):
    with p.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('role',choices=['sender','receiver']);role=p.parse_args().role;cap=json.loads((ROOT/'FREEZE.json').read_text())['caps'][role]
    assert cap is None
    with (ROOT/'job.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);dest=ROOT/f'job_{role}';dest.mkdir();start=time.monotonic()
        with (dest/'run.log').open('x') as log:
            proc=subprocess.Popen([sys.executable,'-B',str(ROOT/'runtime.py'),role],stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1'})
            write(dest/'launch.json',dict(pid=proc.pid,wrapper_pid=os.getpid(),cap_s=cap,role=role,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
            timeout=False
            try:code=proc.wait(timeout=cap)
            except subprocess.TimeoutExpired:timeout=True;proc.kill();proc.wait();code=124
            except BaseException:proc.kill();proc.wait();raise
        value=dict(exit_code=code,timeout=timeout,elapsed_s=time.monotonic()-start,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),new_external_spend_usd=0)
        write(dest/'completion.json',value);print(json.dumps(value));raise SystemExit(code)
