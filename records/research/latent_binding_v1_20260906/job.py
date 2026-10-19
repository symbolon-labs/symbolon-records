"""Bounded sequential first gate; no paid compute and no automatic retry."""
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def main():
    dest=ROOT/'tiny_job'; assert not dest.exists(); dest.mkdir()
    start=time.monotonic()
    state=dict(pid=os.getpid(),status='running',utc_start=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),stages=[],external_spend_usd=0)
    commands=[('prepare',['prepare']),('text_sender',['text-sender']),('text_receiver',['text-receiver']),('train_tiny',['train-tiny'])]
    for name,args in commands:
        cmd=[sys.executable,str(ROOT/'run.py'),*args]
        t=time.monotonic(); stage=dict(name=name,command=cmd,status='running',utc_start=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        with (dest/f'{name}.log').open('x') as stream:
            child=subprocess.Popen(cmd,cwd=ROOT.parents[1],stdout=stream,stderr=subprocess.STDOUT)
            stage['pid']=child.pid; state['stages'].append(stage)
            (dest/'state.json').write_text(json.dumps(state,indent=2)+'\n')
            print(json.dumps(dict(event='stage_started',**stage)),flush=True)
            code=child.wait()
        stage.update(exit_code=code,status='complete' if code==0 else 'failed',elapsed_s=time.monotonic()-t,
                     cumulative_child_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
        state['elapsed_s']=time.monotonic()-start
        if code: state['status']='failed'
        if name=='text_receiver' and code==0:
            summary=json.loads((ROOT/'text_tiny/summary.json').read_text())
            if not summary['oracle_gate']: state['status']='oracle_no_go'
        (dest/'state.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps(dict(event='stage_finished',**stage)),flush=True)
        if state['status']!='running': return
    state['status']='complete'; state['tiny_pass']=json.loads((ROOT/'run_tiny/gates.json').read_text())['pass']
    (dest/'state.json').write_text(json.dumps(state,indent=2)+'\n')


if __name__=='__main__': main()
