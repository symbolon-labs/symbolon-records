import argparse
import common as h
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['close','verify']);a=p.parse_args()
 if a.mode=='close':
  h.check();assert h.read(h.ROOT/'ANALYSIS.json')['status']=='PASS'
  assert h.read(h.ROOT/'job/completion.json')['child_reaped']
  assert (h.ROOT/'REPORT.md').exists() and (h.ROOT/'closing_checks.json').exists()
  h.write(h.ROOT/'seal.json',dict(files=h.inventory(h.ROOT)))
 else:assert h.inventory(h.ROOT)==h.read(h.ROOT/'seal.json')['files']
 print(dict(status='PASS',seal_sha256=h.sha(h.ROOT/'seal.json')))
