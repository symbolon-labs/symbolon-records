"""Post-freeze closing checks and immutable inventory; no inference/training."""
import argparse
import hashlib
import json
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
    value=read(ROOT/'seal.json');assert value['files']==inventory();print(json.dumps(dict(status='PASS',files=len(value['files']),seal_sha256=sha(ROOT/'seal.json'))))
def close():
    assert not (ROOT/'seal.json').exists();job=read(ROOT/'job/completion.json');assert job['exit_code']==0 and not job['timeout']
    launch=read(ROOT/'job/launch.json');assert not any(Path(f'/proc/{launch[k]}').exists() for k in ['pid','wrapper_pid'])
    commands=[['-m','unittest','discover','-s',str(ROOT),'-p','test_core.py','-v'],
        ['-m','unittest','discover','-s',str(ROOT.parent/'latent_binding_value_span_20260908'),'-p','test_span.py','-v'],
        [str(ROOT/'runtime.py'),'check'],[str(ROOT/'analysis.py')],[str(ROOT/'audit.py')],[str(ROOT/'report.py'),'verify']]
    result=[]
    for cmd in commands:
        start=time.monotonic();p=subprocess.run([sys.executable,'-B',*cmd],capture_output=True,text=True,timeout=600)
        result.append(dict(command=cmd,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr,elapsed_s=time.monotonic()-start))
        print(json.dumps(dict(command=cmd,exit_code=p.returncode)),flush=True)
        if p.returncode:raise AssertionError(result[-1])
    with (ROOT/'closing_checks.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    with (ROOT/'seal.json').open('x') as f:json.dump(dict(files=inventory(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),verdict=read(ROOT/'analysis.json')['verdict']),f,indent=2);f.write('\n')
    verify()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['close','verify']);args=p.parse_args();{'close':close,'verify':verify}[args.command]()
