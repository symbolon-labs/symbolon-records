import argparse,copy,json,resource,signal,subprocess,sys,time
import torch
from transformers import AutoTokenizer
import common as h
from common import ROOT,b,c,read,write,load,sha
from seam import Boundary

def equal(a,z):
 for k in ['token_lp','lp','prediction']:assert torch.equal(a[k],z[k]),k

def prepare():
 assert not (ROOT/'FREEZE.json').exists() and not (ROOT/'inputs.json').exists()
 b.c.seed_all(h.SEED);state=load(h.CANDIDATE);bridge=c.bridge();bridge.requires_grad_(False)
 source=read(h.SOURCE/'inputs.json');ids=source['ids'];assert len(ids)==len(set(ids))==56
 raw=load(h.SOURCE/'run/features.pt');features=raw['projected'];wire=raw['wire']
 folds=read(h.R/'latent_binding_training_fold_audit_20260912/RESULT.json');assert folds['status']=='PASS'
 assert sorted(n for f in folds['folds'] for n in f['validation_ids'])==sorted(ids)
 tok=AutoTokenizer.from_pretrained(b.c.model_path('receiver'),local_files_only=True);bank=b.AnswerBank(tok)
 rows=[dict(id=n,feature=i,color=color,label=b.IDS.index(n),query=f'Which ID is {color}?') for i,n in enumerate(ids) for color in b.COLORS]
 q=bank.prompts([r['query'] for r in rows]);assert q.shape==(224,5)
 info=read(h.FAMILIAR/'run/manifest.json');data=read(h.FAMILIAR/'run/data.json')
 saved=load(h.FAMILIAR/'run/prefill_correct.pt')['first']['capture'];mem=load(h.FAMILIAR/'run/memories.pt')['memories']['correct']
 assembled=[];template_indices=[];routes=[]
 for i,row in enumerate(rows):
  matches=[j for j,r in enumerate(data['rows']) if r['color']==row['color'] and info['ids'][j]==q[i].tolist()];assert matches
  j=matches[0];hard=saved['hard'][j,3:5].argmax(-1);assert int(hard[0])==int(hard[1]);slot=int(hard[0])
  z=mem[j].clone();z[2*slot]=wire[row['feature']];assembled.append(z);template_indices.append(j);routes.append(slot)
 assembled=torch.stack(assembled)
 # Native projected source features, not labels, supply the selected value.
 with torch.no_grad():projected=bridge.projector(assembled)
 assert torch.equal(projected[torch.arange(224),2*torch.tensor(routes)],features[[r['feature'] for r in rows]])
 coords=[]
 for f in folds['folds']:
  assert set(f['fit_ids']).isdisjoint(f['validation_ids'])
  assert f['fit_rows']==[i for i,r in enumerate(rows) if r['id'] in f['fit_ids']]
  assert f['validation_rows']==[i for i,r in enumerate(rows) if r['id'] in f['validation_ids']]
  indices=[ids.index(n) for n in f['fit_ids']];co=h.coordinates(state,features[indices]);co['fit_indices']=indices;coords.append(co)
 torch.save(dict(projected=features.clone(),wire=wire.clone()),ROOT/'features.pt')
 torch.save(dict(wire=assembled,template_indices=template_indices,routes=routes),ROOT/'memories.pt');torch.save(coords,ROOT/'coordinates.pt')
 write(ROOT/'folds.json',folds)
 write(ROOT/'inputs.json',dict(ids=ids,rows=rows,query_ids=q.tolist(),candidate_tokens=bank.tokens.tolist(),
   scope='Ancestral training-ID alignment validation on selected-slot assembled memories; not new source capture or full binding evidence'))
 print(json.dumps([dict(fold=i,rank=co['rank'],lr=.05*(960/(64*co['rank']))**.5) for i,co in enumerate(coords)]))

def freeze():
 assert not (ROOT/'FREEZE.json').exists()
 test=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(ROOT),'-p','test_*.py','-v'],capture_output=True,text=True)
 assert test.returncode==0,test.stdout+test.stderr
 write(ROOT/'FREEZE.json',dict(identity=h.identity(),cap_s=h.CAP,fold_cap_s=h.FOLD_CAP,steps=h.STEPS,seed=h.SEED,tests=dict(stdout=test.stdout,stderr=test.stderr,exit_code=test.returncode)))
 h.check();print('FROZEN',sha(ROOT/'FREEZE.json'))

def historical(model,bridge,bank):
 info=read(h.FAMILIAR/'run/manifest.json');ids=torch.tensor(info['ids']);mem=load(h.FAMILIAR/'run/memories.pt')['memories']
 old=load(h.FAMILIAR/'run/evaluation/outputs.pt')['outputs'];old['canonical_only']=load(h.FAMILIAR/'run/evaluation/canonical_only.pt')['result']
 dest=ROOT/'run/historical';dest.mkdir();w=b.Receiver(model,bridge)
 try:
  for arm in ['receiver_only','correct','shuffled','wrong_binding','random_71001','random_71002','zero','canonical_only']:
   bridge.reader.canonical_only=arm=='canonical_only';z=None if arm=='receiver_only' else mem['correct' if arm=='canonical_only' else arm]
   first=b.score(w.forward,ids,bank,z);repeat=b.score(w.forward,ids,bank,z);equal(first,repeat);equal(first,old[arm]);torch.save(dict(first=first,repeat=repeat),dest/f'{arm}.pt');b.event('historical_exact',arm=arm)
 finally:bridge.reader.canonical_only=False;w.hook.remove()

def native_contract(model,bridge,bank,x,indices,state,co,theta,dest,name):
 candidate={k:v.clone() for k,v in state.items()};candidate[h.NAME]=h.weight(theta,co);h.fixed(candidate,state);bridge.load_state_dict(candidate)
 ids=torch.tensor(x['query_ids'])[indices];labels=torch.tensor([x['rows'][i]['label'] for i in indices]);tails=bank.tokens[labels];seq=torch.cat([ids,tails[:,:1]],1)
 memory=load(ROOT/'memories.pt');wires=memory['wire'][indices];features=load(ROOT/'features.pt')['projected'];m=features[[x['rows'][i]['feature'] for i in indices]]
 prior=load(h.FAMILIAR/'run/prefill_correct.pt')['first']['capture'];w=b.Receiver(model,bridge);records=[];logits=[]
 try:
  with torch.no_grad():
   for start in range(0,len(indices),8):
    ix=indices[start:start+8];template=[memory['template_indices'][i] for i in ix]
    for tokens in [ids[start:start+8],seq[start:start+8]]:
     z=w.forward(tokens,wires[start:start+8],2).clone();cap={k:v.clone() for k,v in bridge.reader.capture.items()}
     assert torch.equal(cap['hard'][:,:5],prior['hard'][template]) and torch.equal(cap['scores'][:,:5],prior['scores'][template])
     expected_routes=torch.tensor([memory['routes'][i] for i in ix]);assert torch.equal(cap['hard'][:,3:5].argmax(-1),expected_routes[:,None].expand(-1,2))
     assert torch.equal(cap['memory'][torch.arange(len(ix)),2*expected_routes],m[start:start+8])
     assert torch.equal(cap['output'][:,:3],cap['hidden'][:,:3]);assert torch.equal(cap['output'][:,5:],cap['hidden'][:,5:])
     assert not cap['used_delta'][:,5:].any()
     payload=h.render(m[start:start+8],candidate[h.NAME],state);assert torch.equal(payload,cap['output'][:,3:5])
     records.append(dict(indices=ix,tokens=tokens.clone(),capture=cap))
     if tokens.shape[1]==6:logits.append(z)
 finally:w.hook.remove()
 seam=Boundary(model);rendered=[]
 try:
  with torch.no_grad():
   for start in range(0,len(indices),8):rendered.append(seam.forward(seq[start:start+8],h.render(m[start:start+8],candidate[h.NAME],state),2).clone())
 finally:seam.close()
 native=torch.cat(logits);rendered=torch.cat(rendered);error=float((native-rendered).abs().max());assert error<=1e-4 and torch.equal(native.argmax(-1),rendered.argmax(-1))
 torch.save(dict(records=records,native=native,rendered=rendered,error=error),dest/f'{name}.pt');bridge.load_state_dict(state)
 return error

def preflight(model,bridge,bank,x,f,state,co,dest):
 theta=torch.zeros(64,co['rank'],dtype=torch.float64);assert torch.equal(h.weight(theta,co),state[h.NAME]);ix=f['fit_rows'][:8]
 gen=torch.Generator().manual_seed(h.SEED);d=torch.randn(theta.shape,generator=gen,dtype=torch.float64);d/=d.norm()
 errors=[native_contract(model,bridge,bank,x,ix,state,co,z,dest,name) for z,name in [(theta,'zero_parity'),(.01*d,'perturbed_parity')]]
 ids=torch.tensor(x['query_ids'])[ix];labels=torch.tensor([x['rows'][i]['label'] for i in ix]);t=bank.tokens[labels];seq=torch.cat([ids,t[:,:1]],1)
 features=load(ROOT/'features.pt')['projected'];m=features[[x['rows'][i]['feature'] for i in ix]];seam=Boundary(model)
 try:
  def loss(q):return h.batch_loss(seam.forward(seq,h.render(m,h.weight(q,co),state),2),t,len(ix))
  parameter=theta.clone().requires_grad_();value=loss(parameter);grad=torch.autograd.grad(value,parameter)[0];assert torch.isfinite(grad).all() and grad.norm()>0;d=grad/grad.norm()
  with torch.no_grad():finite=float((loss(theta+.01*d)-loss(theta-.01*d))/.02)
  analytic=float((grad*d).sum());relative=abs(finite-analytic)/abs(analytic);assert finite>0 and relative<=.1
  out=dict(errors=errors,analytic=analytic,finite=finite,relative=relative);write(dest/'preflight.json',out);return out
 finally:seam.close();bridge.load_state_dict(state)

def evaluate(model,bridge,bank,x,f,state,co,theta,step,dest):
 sd=dest/f'step{step}';sd.mkdir();candidate={k:v.clone() for k,v in state.items()};candidate[h.NAME]=h.weight(theta,co);h.fixed(candidate,state);torch.save(candidate,sd/'candidate.pt')
 prior=h.ORIGINAL/'run'/f'fold{f["fold"]}'/f'step{step}'
 if (prior/'candidate.pt').exists():h.exact_tree(candidate,load(prior/'candidate.pt'))
 matched=h.MATCHED/'run'/f'fold{f["fold"]}'/f'step{step}'
 if (matched/'candidate.pt').exists():h.exact_tree(candidate,load(matched/'candidate.pt'))
 ix=f['validation_rows'];err=native_contract(model,bridge,bank,x,ix,state,co,theta,sd,'contract');bridge.load_state_dict(load(sd/'candidate.pt'))
 q=torch.tensor(x['query_ids'])[ix];z=load(ROOT/'memories.pt')['wire'][ix];labels=[x['rows'][i]['label'] for i in ix];w=b.Receiver(model,bridge)
 try:
  first=b.score(w.forward,q,bank,z);repeat=b.score(w.forward,q,bank,z);equal(first,repeat)
  direct=b.direct_score(w.forward,q[:8],bank,z[:8]);de=float((first['lp'][:8]-direct).abs().max())
  single=b.direct_score(w.forward,q[:1],bank,z[:1]);cross=float((first['lp'][:1]-single).abs().max())
  # Persist diagnostics before asserting, including failure values.
  torch.save(dict(direct=direct,single=single,fast=first['lp'][:8],direct_error=de,cross_batch_error=cross),sd/'direct_diagnostic.pt')
  assert de<=1e-4 and torch.equal(first['prediction'][:8],direct.argmax(-1))
  if (prior/'scores.pt').exists():
   old=load(prior/'scores.pt');equal(first,old['first']);equal(repeat,old['repeat']);assert torch.equal(single,old['direct'])
  if f['fold']==1 and step==100:
   diag=load(h.DIAGNOSTIC/'run/batch8.pt');equal({k:v[:8] for k,v in first.items() if k in ['token_lp','lp','prediction']},diag['fast']);assert torch.equal(direct,diag['direct'])
   assert torch.equal(single,load(h.DIAGNOSTIC/'run/batch1.pt')['direct'])
  torch.save(dict(first=first,repeat=repeat,direct=direct,direct_error=de,indices=ix),sd/'scores.pt')
 finally:w.hook.remove();bridge.load_state_dict(state)
 for name in ['scores.pt','direct_diagnostic.pt']:
  if (matched/name).exists():h.exact_tree(load(sd/name),load(matched/name))
 out=dict(**h.counts(first,labels,bank.tokens),parity=err,direct_error=de);write(sd/'counts.json',out);b.event('fold_checkpoint',fold=f['fold'],step=step,**out);return out

def fit_fold(model,bridge,bank,x,f,state,co):
 started=time.monotonic();dest=ROOT/'run'/f'fold{f["fold"]}';dest.mkdir();write(dest/'start.json',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),fit_ids=f['fit_ids'],validation_ids=f['validation_ids']))
 def cap():assert time.monotonic()-started<h.FOLD_CAP,'Fold time cap; no continuation licensed'
 assert all(torch.equal(v,state[k]) for k,v in bridge.state_dict().items())
 assert co['fit_indices']==[x['ids'].index(n) for n in f['fit_ids']]
 preflight(model,bridge,bank,x,f,state,co,dest);cap()
 theta=torch.nn.Parameter(torch.zeros(64,co['rank'],dtype=torch.float64));opt=torch.optim.Adam([theta],lr=.05*(960/(64*co['rank']))**.5);assert not opt.state
 ix=f['fit_rows'];features=load(ROOT/'features.pt')['projected'];m=features[[x['rows'][i]['feature'] for i in ix]];q=torch.tensor(x['query_ids'])[ix];targets=bank.tokens[[x['rows'][i]['label'] for i in ix]]
 history=[];totals={0:evaluate(model,bridge,bank,x,f,state,co,theta.detach(),0,dest)}
 for step in range(1,201):
  cap();seam=Boundary(model)
  try:
   opt.zero_grad(set_to_none=True);total=0.;correct=torch.zeros(2,dtype=torch.long)
   for start in range(0,len(ix),8):
    t=targets[start:start+8];seq=torch.cat([q[start:start+8],t[:,:1]],1)
    logits=seam.forward(seq,h.render(m[start:start+8],h.weight(theta,co),state),2);loss=h.batch_loss(logits,t,len(ix));loss.backward();total+=float(loss.detach());correct+=(logits.detach().argmax(-1)==t).sum(0)
   assert theta.grad is not None and torch.isfinite(theta.grad).all();assert all(p.grad is None and not p.requires_grad for p in model.parameters())
   history.append(dict(step_before_update=step-1,loss=total,gradient_norm=float(theta.grad.norm()),teacher_forced=correct.tolist()));opt.step();assert torch.isfinite(theta).all()
  finally:seam.close()
  if step%5==0:b.event('fold_fit',fold=f['fold'],step=step,loss=total,teacher_forced=correct.tolist())
  if step in h.STEPS:
   totals[step]=evaluate(model,bridge,bank,x,f,state,co,theta.detach(),step,dest)
   fitstate=dict(theta=theta.detach().clone(),optimizer=copy.deepcopy(opt.state_dict()),history=copy.deepcopy(history))
   oldfit=h.ORIGINAL/'run'/f'fold{f["fold"]}'/f'step{step}/fit.pt'
   if oldfit.exists():h.exact_tree(fitstate,load(oldfit));b.event('optimizer_replay_exact',fold=f['fold'],step=step)
   matchedfit=h.MATCHED/'run'/f'fold{f["fold"]}'/f'step{step}/fit.pt'
   if matchedfit.exists():h.exact_tree(fitstate,load(matchedfit));b.event('matched_optimizer_replay_exact',fold=f['fold'],step=step)
   torch.save(fitstate,dest/f'step{step}/fit.pt')
  cap()
 write(dest/'completed.json',dict(elapsed_s=time.monotonic()-started,totals=totals));return totals

def body(status):
 h.check();b.c.seed_all(h.SEED);x=read(ROOT/'inputs.json');state=load(h.CANDIDATE);bridge=c.bridge();bridge.requires_grad_(False)
 model,tok=b.c.load('receiver');before=b.c.parameter_hash(model);assert before=='6a3e1782ebad58d1e9323d5eeaff191d112848b5f0058394a998d1ac54c492b3';bank=b.AnswerBank(tok);assert bank.tokens.tolist()==x['candidate_tokens']
 seq=torch.tensor(x['query_ids'])[:8]
 with torch.no_grad():stock=b.native_forward(model,seq).clone()
 w=b.Receiver(model,bridge)
 try:
  with torch.no_grad():disabled=w.forward(seq,None).clone();zero=w.forward(seq,torch.zeros(8,8,64)).clone()
  assert torch.equal(stock,disabled) and torch.equal(stock,zero)
 finally:w.hook.remove()
 torch.save(dict(stock=stock,disabled=disabled,zero=zero),ROOT/'run/stock.pt')
 try:
  historical(model,bridge,bank);status['historical_exact']=True
  totals={s:dict(joint=0,sequential=0,rows=0) for s in h.STEPS}
  for f,co in zip(read(ROOT/'folds.json')['folds'],load(ROOT/'coordinates.pt')):
   torch.manual_seed(h.SEED)
   def timeout(signum,frame):raise TimeoutError('Frozen per-fold 9000-second cap')
   previous=signal.signal(signal.SIGALRM,timeout);signal.alarm(h.FOLD_CAP)
   try:result=fit_fold(model,bridge,bank,x,f,state,co)
   finally:signal.alarm(0);signal.signal(signal.SIGALRM,previous)
   for s in h.STEPS:
    for k in totals[s]:totals[s][k]+=result[s][k]
   status['folds_completed']+=1
  assert all(v['rows']==224 for v in totals.values());status.update(totals=totals,selection=h.select(totals))
 finally:
  bridge.load_state_dict(state);assert all(torch.equal(v,state[k]) for k,v in bridge.state_dict().items());assert all(not m._forward_hooks for m in model.modules())
  with torch.no_grad():restored=b.native_forward(model,seq).clone()
  assert torch.equal(stock,restored);torch.save(restored,ROOT/'run/restored.pt');after=b.c.parameter_hash(model);assert after==before and all(p.grad is None and not p.requires_grad for p in model.parameters());status.update(before=before,after=after,restoration_exact=True,no_hooks=True,no_gradients=True)
 h.check()
def run():
 (ROOT/'run').mkdir();start=time.monotonic();status=dict(completed=False,folds_completed=0,novel_evaluated=False)
 try:body(status);status['completed']=True
 except BaseException as e:status.update(error_type=type(e).__name__,error=str(e));raise
 finally:write(ROOT/'run/status.json',dict(**status,elapsed_s=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','freeze','check','run']);a=ap.parse_args();dict(prepare=prepare,freeze=freeze,check=h.check,run=run)[a.mode]()
