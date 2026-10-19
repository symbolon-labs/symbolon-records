"""Post-freeze closing checks; no inference or scientific changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def inventory():return {str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p!=ROOT/'seal.json'}
def verify():
    assert read(ROOT/'seal.json')['files']==inventory();print(json.dumps(dict(status='PASS',files=len(inventory()),seal_sha256=sha(ROOT/'seal.json'))))
def close():
    assert not (ROOT/'seal.json').exists();job=read(ROOT/'job/completion.json');assert job['exit_code']==0 and not job['timeout'] and job['child_reaped']
    assert read(ROOT/'run/status.json')['completed'] and read(ROOT/'ANALYSIS.json')['independent_audit']=='PASS'
    # Child completion/reaping was recorded in its original PID namespace.
    commands=[['-m','unittest','discover','-s',str(ROOT),'-p','test_seam.py','-v'],[str(ROOT/'runtime.py'),'check'],[str(ROOT/'analysis.py')]]
    checks=[]
    for cmd in commands:
        start=time.monotonic();p=subprocess.run([sys.executable,'-B',*cmd],capture_output=True,text=True,timeout=300,env={**os.environ,'OPENBLAS_NUM_THREADS':'4'})
        checks.append(dict(command=cmd,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr,elapsed_s=time.monotonic()-start));print(json.dumps(dict(command=cmd,exit_code=p.returncode)),flush=True);assert p.returncode==0,checks[-1]
    with (ROOT/'closing_checks.json').open('x') as f:json.dump(checks,f,indent=2);f.write('\n')
    with (ROOT/'seal.json').open('x') as f:json.dump(dict(files=inventory(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),verdict=read(ROOT/'ANALYSIS.json')['verdict']),f,indent=2);f.write('\n')
    verify()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['close','verify']);q=p.parse_args();{'close':close,'verify':verify}[q.command]()
