"""Small sequential job driver; explicit gates, no blind restarts."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def write(path, value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(); p.add_argument('phase',choices=['tiny','development','confirmation'])
    args=p.parse_args()
    if args.phase!='tiny':
        gates=json.loads((ROOT/'run_tiny/semantic_gates.json').read_text())
        assert all(gates[k] for k in ['gate0','gate1','gate2'])
    if args.phase=='tiny':
        commands=[('prepare_tiny',['channel_cf.py','prepare','--split','tiny']),
                  ('train_tiny',['channel_cf.py','train','--mode','tiny'])]
    elif args.phase=='development':
        commands=[('text_tiny',['text_cf.py','--split','tiny']),
                  ('prepare_train',['channel_cf.py','prepare','--split','train']),
                  ('prepare_dev',['channel_cf.py','prepare','--split','dev']),
                  ('train_full',['channel_cf.py','train','--mode','full'])]
    else:
        commands=[('reproduce_dev',['final_cf.py','reproduce-dev']),
                  ('text_dev',['text_cf.py','--split','dev']),
                  ('freeze',['final_cf.py','freeze']),
                  ('prepare_final',['channel_cf.py','prepare','--split','final']),
                  ('score_final',['final_cf.py','score']),
                  ('text_final',['text_cf.py','--split','final']),
                  ('sender_reproduce',['sender_reproduce.py']),
                  ('analyze_final',['final_cf.py','analyze'])]
    dest=ROOT/f'{args.phase}_job'; assert not dest.exists(); dest.mkdir()
    start=time.monotonic()
    state={'pid':os.getpid(),'status':'running','stages':[],
           'utc_start':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'external_spend_usd':0}
    for name,command in commands:
        cmd=[sys.executable,str(ROOT/command[0]),*command[1:]]
        stage={'name':name,'command':cmd,'status':'running','utc_start':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
        t=time.monotonic()
        with (dest/f'{name}.log').open('x') as log:
            child=subprocess.Popen(cmd,cwd=ROOT.parents[1],stdout=log,stderr=subprocess.STDOUT)
            stage['pid']=child.pid; state['stages'].append(stage); write(dest/'state.json',state)
            print(json.dumps({'event':'stage_started',**stage}),flush=True)
            code=child.wait()
        stage.update(exit_code=code,elapsed_s=time.monotonic()-t,status='complete' if code==0 else 'failed',
                     peak_child_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
        state['elapsed_s']=time.monotonic()-start
        if code!=0: state['status']='failed'
        write(dest/'state.json',state); print(json.dumps({'event':'stage_finished',**stage}),flush=True)
        if code!=0: raise SystemExit(1)
        if name=='text_tiny':
            assert json.loads((ROOT/'text_tiny/summary.json').read_text())['oracle_text_sanity_pass'], 'Receiver text sanity failed; do not scale'
    state['status']='complete'; write(dest/'state.json',state)


if __name__=='__main__': main()
