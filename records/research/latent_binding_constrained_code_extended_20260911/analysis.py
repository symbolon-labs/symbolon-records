"""Independent scalar score and oracle-stage audit; no inference."""
import argparse,json,math
import numpy as np
import torch
import runtime as r
ROOT=r.ROOT

def metrics(out,targets,tokens):
 a=out['token_lp'].numpy();lp=np.array([[np.float32(v[0]+v[1]) for v in row] for row in a]);assert np.isfinite(a).all() and np.array_equal(lp,out['lp'].numpy())
 joint=np.argmax(lp,axis=1).tolist();assert joint==out['prediction'].tolist();reps=[];seen=set()
 for j,(p,d) in enumerate(tokens):
  if p not in seen:reps.append(j);seen.add(p)
 seq=[]
 for row in a:
  for j in range(128):assert row[j,0]==row[next(k for k in reps if tokens[k][0]==tokens[j][0]),0]
  first=max(reps,key=lambda j:float(row[j,0]));siblings=[j for j in range(128) if tokens[j][0]==tokens[first][0]];seq.append(max(siblings,key=lambda j:float(row[j,1])))
 labels=[r.b.IDS.index(t) for t in targets]
 return {'joint':sum(p==t for p,t in zip(joint,labels)),'sequential':sum(p==t for p,t in zip(seq,labels))}, {'joint':[r.b.IDS[p] for p in joint],'sequential':[r.b.IDS[p] for p in seq],'targets':targets}
def analyze():
 r.check();s=r.read(ROOT/'run/status.json');job=r.read(ROOT/'job/completion.json');assert s['completed'] and job['exit_code']==0 and job['child_reaped'] and not job['timeout']
 assert s['before']==s['after'] and s['restored_exact'] and s['no_hooks'] and s['no_parameter_gradients'];x=r.read(ROOT/'inputs.json');initial=r.load(ROOT/'initial.pt');tokens=x['leading_tokens'];results={};fits={}
 stock=r.load(ROOT/'run/stock.pt');assert torch.equal(stock['stock'],stock['disabled']) and torch.equal(stock['stock'],stock['self']) and torch.equal(stock['stock'],r.load(ROOT/'run/restored.pt'))
 for path in (ROOT/'run').glob('*.pt'):
  z=r.load(path)
  if isinstance(z,dict) and 'first' in z and 'repeat' in z:
   r.r.equal(z['first'],z['repeat'])
 for pool,conditions in [('familiar',['original']),('novel',['original','changed'])]:
  info=x['banks'][pool]
  for condition in conditions:
   old=r.load(r.PARENT/'run'/f'{pool}_{condition}_sham.pt')['first'];new=r.load(ROOT/'run'/f'{pool}_{condition}_baseline.pt')['first'];r.r.equal(old,new);metrics(new,info['targets'][condition],tokens)
  path=ROOT/'run'/f'{pool}_fit.pt'
  if not path.exists():assert pool=='novel' and not s['novel_fit_opened'];continue
  fit=r.load(path);assert fit['steps']==200 and len(fit['history'])==200 and torch.equal(fit['initial'],initial[pool]) and torch.isfinite(fit['terminal']).all()
  assert [z['step_before_update'] for z in fit['history']]==list(range(200))
  assert all(math.isfinite(z['loss']) and math.isfinite(z['gradient_norm']) and z['gradient_norm']>0 for z in fit['history'])
  parent=r.load(r.R/'latent_binding_constrained_code_oracle_20260911/run'/f'{pool}_fit.pt');prefix=r.load(ROOT/'run'/f'{pool}_prefix40.pt')
  assert prefix['exact_parent_replay'] and torch.equal(prefix['coordinates'],parent['coordinates']) and torch.equal(prefix['value'],parent['terminal']) and fit['history'][:40]==parent['history']
  optimizer=r.load(ROOT/'run'/f'{pool}_optimizer200.pt');assert all(float(v['step'])==200 for v in optimizer['state'].values())
  terminal=fit['terminal'];coords=r.load(ROOT/'coordinates.pt');assert coords['lr']==r.LR
  assert torch.equal(terminal,initial[pool]+(fit['coordinates']/coords['s'])@coords['vh'])
  st=r.state();a=float(st['reader.gate'].double()*st['reader.norm_scale'].double())*st['reader.output.weight'].double().numpy()
  reconstructed=(coords['u'].numpy()*coords['s'].numpy())@coords['vh'].numpy();assert np.linalg.norm(a-reconstructed)/np.linalg.norm(a)<1e-10
  npmapped=(terminal.numpy()@st['reader.output.weight'].double().numpy().T).astype('float32');npboundary=st['reader.query_center'].numpy()[None]+((st['reader.gate'].numpy()*npmapped)*st['reader.norm_scale'].numpy())[:,None,:]
  assert np.linalg.norm(npboundary-r.render(terminal).numpy())/np.linalg.norm(npboundary)<1e-5
  assert fit['boundary_movement_norm']==float((r.render(terminal)-r.render(initial[pool])).norm())
  assert fit['movement_norm']==float((terminal-initial[pool]).norm()) and fit['terminal_norm']==float(terminal.norm())
  teacher=r.load(ROOT/'run'/f'{pool}_terminal_teacher.pt');assert teacher['targets'].tolist()==[tokens[r.b.IDS.index(t)] for t in info['targets']['original']]
  fits[pool]={k:v for k,v in fit.items() if k not in ['initial','terminal','coordinates']};fits[pool]['terminal_full_vocab_teacher_forced_correct']=(teacher['prediction']==teacher['targets']).sum(0).tolist()
  gen=torch.Generator().manual_seed(60113);noise=torch.randn(terminal.shape,generator=gen);noise=noise/noise.norm(dim=-1,keepdim=True)*terminal.norm(dim=-1,keepdim=True);assert torch.equal(noise,r.load(ROOT/'run'/f'{pool}_random.pt'))
  parity=r.load(ROOT/'run'/f'{pool}_parity.pt');err=float((parity['fast']['lp']-parity['direct']).abs().max());assert err==parity['error'] and err<=1e-4 and torch.equal(parity['fast']['prediction'],parity['direct'].argmax(-1))
  results[pool]={}
  for condition in conditions:
   targets=info['targets'][condition];inserted=[info['messages'][(info['messages'].index(t)+1)%8] for t in targets];ms={}
   for arm in ['matched','wrong','random']:
    out=r.load(ROOT/'run'/f'{pool}_{condition}_{arm}.pt')['first'];m,rows=metrics(out,targets,tokens)
    if arm=='wrong':m['inserted']=metrics(out,inserted,tokens)[0]
    assert m==s[pool][condition][arm];ms[arm]={'counts':m,**rows}
   minimum=31 if pool=='familiar' else 28
   passed=all(ms['matched']['counts'][k]>=minimum and ms['wrong']['counts']['inserted'][k]>=28 and ms['wrong']['counts'][k]<=8 and ms['random']['counts'][k]<=8 for k in ['joint','sequential']);assert passed==s[pool][condition]['pass']
   results[pool][condition]={'pass':passed,'arms':ms}
 if not results['familiar']['original']['pass']:
  assert not s['novel_fit_opened'] and s['verdict']=='FAMILIAR ORACLE CODE QUALIFICATION NO-GO'
 else:
  assert s['novel_fit_opened'] and 'novel' in results
  expected='ORACLE QUERY CODE REACHABLE; SOURCE TRANSFER UNPROVEN' if all(z['pass'] for z in results['novel'].values()) else 'NOVEL ORACLE QUERY CODE NO-GO';assert s['verdict']==expected
 return {'verdict':s['verdict'],'independent_audit':'PASS','supervised_activation_fit':True,'native_novel':'0/0/0 unchanged','results':results,'fits':fits,'job':job}
def report(a):
 lines=['# Query-context oracle activation codes','',f"**{a['verdict']}**",'','Supervised fitting of shared64-value codes through the unchanged output map at block15; receiver and bridge weights frozen.200fixed Adamsteps in SVD-conditioned coordinates per opened pool, no selected checkpoint. Labels supplied during fitting and evaluation lookup. This is oracle reachability, not novel-ID transfer or a trained bridge. Native d0novel0/0/0 unchanged.','','| Pool/condition | Matched joint/seq | Wrong inserted joint/seq | Wrong intended joint/seq | Random intended joint/seq | Gate |','|---|---|---|---|---|---|']
 for pool,conditions in a['results'].items():
  for condition,z in conditions.items():
   arms=z['arms'];m=arms['matched']['counts'];w=arms['wrong']['counts'];ran=arms['random']['counts'];f=lambda v:f"{v['joint']}/{v['sequential']}"
   lines.append(f"| {pool}/{condition} | {f(m)} | {f(w['inserted'])} | {f(w)} | {f(ran)} | {z['pass']} |")
 lines+=['','Sequential is candidate-constrained16prefix/8digit decoding. Terminal full-vocabulary teacher-forced token diagnostics, every loss/gradient norm, code norm movement, per-row predictions and all counts are in ANALYSIS.json. No unrestricted generation evaluated.','',f"Job {a['job']['elapsed_s']:.3f}s, peak{a['job']['peak_rss_kib']}KiB,exit0/childreaped. Independent scalar audit, exact baseline and scoring repeats, direct parity, live parameter identity/restoration and no-parameter-gradient checks pass.22model snapshots checked through frozen parent identity. No sender/reset/newdata/final/Arabic/cloud/browser/Rust/push. USD0external.",'','A successful supervised oracle code is not an automatically qualified training target for source transfer. Changed contexts reuse the same fitted codebook. No bridge training or parameter schedule search follows automatically.','']
 return '\n'.join(lines)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--save',action='store_true');q=p.parse_args();a=analyze();s=report(a)
 if q.save:r.write(ROOT/'ANALYSIS.json',a);(ROOT/'REPORT.md').write_text(s)
 else:assert a==r.read(ROOT/'ANALYSIS.json') and s==(ROOT/'REPORT.md').read_text()
 print(s)
