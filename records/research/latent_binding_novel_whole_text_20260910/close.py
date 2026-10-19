"""Post-freeze read-only validation and final inventory, no model execution."""
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
    assert not (ROOT/'seal.json').exists()
    for role in ['sender','receiver']:
        job=read(ROOT/f'job_{role}/completion.json');launch=read(ROOT/f'job_{role}/launch.json')
        assert job['exit_code']==0 and not job['timeout'] and not any(Path(f'/proc/{launch[k]}').exists() for k in ['pid','wrapper_pid'])
    for name,h in read(ROOT/'ANALYSIS_DEPENDENCIES.json')['files'].items():assert sha(Path(name))==h
    commands=[['-m','unittest','discover','-s',str(ROOT),'-p','test_text.py','-v'],[str(ROOT/'runtime.py'),'check'],
        *[[str(ROOT/'audit.py'),stage] for stage in ['inputs','sender','receiver','summary']],[str(ROOT/'report.py'),'verify']]
    result=[]
    for cmd in commands:
        start=time.monotonic();p=subprocess.run([sys.executable,'-B',*cmd],capture_output=True,text=True,timeout=600)
        row=dict(command=cmd,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr,elapsed_s=time.monotonic()-start);result.append(row)
        print(json.dumps(dict(command=cmd,exit_code=p.returncode)),flush=True)
        assert p.returncode==0,row
    with (ROOT/'closing_checks.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    with (ROOT/'seal.json').open('x') as f:json.dump(dict(files=inventory(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),verdict=read(ROOT/'summary.json')['verdict']),f,indent=2);f.write('\n')
    verify()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['close','verify']);q=p.parse_args();{'close':close,'verify':verify}[q.command]()
