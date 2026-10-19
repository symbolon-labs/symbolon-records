"""Supervised temporary activation fitting, frozen endpoints and bridge."""
import argparse,importlib.util,json,resource,time,subprocess,sys,functools,math
from pathlib import Path
import torch
from seam import Boundary
ROOT=Path(__file__).resolve().parent;R=ROOT.parent
spec=importlib.util.spec_from_file_location('sealed_oracle_parent',R/'latent_binding_literal_id_swap_20260910/runtime.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
b=r.b;read=r.read;write=r.write;sha=r.sha;load=r.load
PARENT=R/'latent_binding_literal_id_swap_20260910'
SEALS={'latent_binding_literal_id_swap_20260910':'7ae446cc945af7e9eb23e77f59ee078618f3d2b39877db30c4ee3afd5bb272ce','latent_binding_sequential_readout_20260911':'4ef7234281ab686780ffa3cb88c4dd02a8eb4db6910307ebf443c4a266e4f480'}
LR=.05*math.sqrt(960/64)
SEALS.update({'latent_binding_query_code_oracle_20260911':'af10421b1e3f7cbb721e2ed6ceadce03b774b308a935a23abc22cd748282d1c9','latent_binding_oracle_representability_20260911':'d6050d6f8d945b7031c99992d1d036a1408653637d8f0b6b79af2f4ce4939ca7'})
SEALS['latent_binding_constrained_code_oracle_20260911']='1cdf9f081be2327b07dd5b26cfcb9a4ec82eece2207241b302a77356e6eff525'
CODE=['PLAN.md','runtime.py','seam.py','test_seam.py','job.py','analysis.py','close.py']
def identity():
 r.parents()
 actual=r.identity();expected=read(PARENT/'FREEZE.json')['identity'];key=str(ROOT/'runtime.py')
 if key in actual['imports']:assert actual['imports'].pop(key)==sha(ROOT/'runtime.py')
 assert actual==expected
 for name,digest in SEALS.items():
  root=R/name;assert sha(root/'seal.json')==digest;assert r.inventory(root)==read(root/'seal.json')['files']
 return {'code':{n:sha(ROOT/n) for n in CODE},'parents':SEALS,'inputs':sha(ROOT/'inputs.json'),'initial':sha(ROOT/'initial.pt'),'coordinates':sha(ROOT/'coordinates.pt'),'candidate':sha(r.c.CANDIDATE)}
def check():assert identity()==read(ROOT/'FREEZE.json')['identity']
def freeze():
 assert not (ROOT/'FREEZE.json').exists()
 p=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(ROOT),'-p','test_seam.py','-v'],capture_output=True,text=True);assert p.returncode==0,p.stderr
 x=read(PARENT/'inputs.json');initial={};baseline=load(PARENT/'baseline.pt');state=load(r.c.CANDIDATE)
 alpha=float(state['reader.gate'].double()*state['reader.norm_scale'].double());u,sv,vh=torch.linalg.svd(alpha*state['reader.output.weight'].double(),full_matrices=False)
 assert int((sv>torch.finfo(torch.float32).eps*960*sv[0]).sum())==64
 torch.save({'s':sv,'vh':vh,'u':u,'lr':LR},ROOT/'coordinates.pt')
 for pool in ['familiar','novel']:
  info=x['banks'][pool];refs=[]
  cap=load(r.native(pool)/'run/prefill_correct.pt')['first']['capture']
  vals=torch.nn.functional.linear(cap['memory'][:,::2].double(),state['reader.value.weight'].double());selected=cap['hard'].double()@vals
  assert torch.equal(selected[:,3],selected[:,4])
  for target in info['messages']:
   inds=[i for i,t in enumerate(info['targets']['original']) if t==target]
   z=selected[inds,3];assert torch.equal(z,z[:1].expand_as(z));refs.append(z[0])
  initial[pool]=torch.stack(refs)
  ix=[info['messages'].index(t) for t in info['targets']['original']]
  assert torch.equal(render(initial[pool][ix]),baseline[pool+'_original'])
 write(ROOT/'inputs.json',x);torch.save(initial,ROOT/'initial.pt')
 write(ROOT/'FREEZE.json',{'identity':identity(),'cap_s':2400,'steps':200,'lr':LR,'seed':60113,'tests':{'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr},'supervised_oracle_activation_fit':True,'model_training':False})
 check();print('FROZEN',sha(ROOT/'FREEZE.json'))
def score(w,ids,bank,payload,name):
 a=b.score(w.forward,ids,bank,payload);z=b.score(w.forward,ids,bank,payload);r.equal(a,z)
 torch.save({'first':a,'repeat':z},ROOT/'run'/f'{name}.pt');b.event('score_repeat_pass',arm=name);return a
@functools.lru_cache(maxsize=1)
def state():return load(r.c.CANDIDATE)
def render(value):
 st=state();mapped=torch.nn.functional.linear(value.double(),st['reader.output.weight'].double()).float()
 residual=(st['reader.gate']*mapped)*st['reader.norm_scale']
 return st['reader.query_center'][None]+residual[:,None,:]
def counts(out,targets,tokens):
 parts=out['token_lp'];seq=[]
 groups=[torch.where(tokens[:,0]==p)[0] for p in dict.fromkeys(tokens[:,0].tolist())];reps=torch.tensor([int(g[0]) for g in groups])
 for i in range(len(parts)):
  g=groups[int(parts[i,reps,0].argmax())];seq.append(int(g[parts[i,g,1].argmax()]))
 target=torch.tensor([b.IDS.index(t) for t in targets])
 return {'joint':int((out['prediction']==target).sum()),'sequential':int((torch.tensor(seq)==target).sum())}
def fit(w,ids,bank,labels,index,initial,pool):
 code=torch.nn.Parameter(torch.zeros_like(initial));coordinates=load(ROOT/'coordinates.pt');sv=coordinates['s'];vh=coordinates['vh'];opt=torch.optim.Adam([code],lr=LR);history=[];targets=bank.tokens[labels]
 for step in range(200):
  opt.zero_grad(set_to_none=True);total=0.;correct=torch.zeros(2,dtype=torch.long)
  for start in range(0,32,8):
   q=ids[start:start+8];t=targets[start:start+8];seq=torch.cat([q,t[:,:1]],1)
   with torch.enable_grad():
    value=initial+(code/sv)@vh
    logits=w.forward(seq,render(value[index[start:start+8]]),2)
    loss=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),t.reshape(-1))/4
    loss.backward()
   total+=float(loss.detach());correct+=(logits.detach().argmax(-1)==t).sum(0)
  assert code.grad is not None and torch.isfinite(code.grad).all() and torch.isfinite(code).all()
  grad=float(code.grad.norm());assert grad>0
  assert all(p.grad is None and not p.requires_grad for p in w.model.parameters())
  history.append({'step_before_update':step,'loss':total,'gradient_norm':grad,'full_vocab_teacher_forced_correct':correct.tolist()})
  opt.step()
  if step+1==40:
   parent=load(R/'latent_binding_constrained_code_oracle_20260911/run'/f'{pool}_fit.pt')
   prefix_value=initial+(code.detach()/sv)@vh
   assert history==parent['history'] and torch.equal(code.detach(),parent['coordinates']) and torch.equal(prefix_value,parent['terminal'])
   torch.save({'coordinates':code.detach().clone(),'value':prefix_value.clone(),'optimizer':opt.state_dict(),'exact_parent_replay':True},ROOT/'run'/f'{pool}_prefix40.pt')
   b.event('prefix40_exact',pool=pool)
  if (step+1)%10==0:b.event('oracle_fit_step',pool=pool,step=step+1,loss=total)
 assert torch.isfinite(code).all()
 terminal=initial+(code.detach()/sv)@vh
 result={'initial':initial,'terminal':terminal.clone(),'coordinates':code.detach().clone(),'boundary_movement_norm':float((render(terminal)-render(initial)).norm()),'history':history,'movement_norm':float((terminal-initial).norm()),'initial_norm':float(initial.norm()),'terminal_norm':float(terminal.norm()),'steps':200}
 torch.save(opt.state_dict(),ROOT/'run'/f'{pool}_optimizer200.pt');torch.save(result,ROOT/'run'/f'{pool}_fit.pt');return result['terminal']
def evaluate(w,ids,bank,info,code,pool,conditions):
 results={};gen=torch.Generator().manual_seed(60113);noise=torch.randn(code.shape,generator=gen);noise=noise/noise.norm(dim=-1,keepdim=True)*code.norm(dim=-1,keepdim=True)
 torch.save(noise,ROOT/'run'/f'{pool}_random.pt')
 for condition in conditions:
  targets=info['targets'][condition];ix=torch.tensor([info['messages'].index(t) for t in targets]);wrong=(ix+1)%8;inserted=[info['messages'][i] for i in wrong]
  results[condition]={}
  for arm,payload in [('matched',code[ix]),('wrong',code[wrong]),('random',noise[ix])]:
   out=score(w,ids,bank,render(payload),f'{pool}_{condition}_{arm}');m=counts(out,targets,bank.tokens)
   if arm=='wrong':m['inserted']=counts(out,inserted,bank.tokens)
   results[condition][arm]=m
  matched=results[condition]['matched'];wr=results[condition]['wrong'];ran=results[condition]['random'];minimum=31 if pool=='familiar' else 28
  results[condition]['pass']=all(matched[k]>=minimum and wr['inserted'][k]>=28 and wr[k]<=8 and ran[k]<=8 for k in ['joint','sequential'])
 return results
def body(status):
 check();b.c.seed_all(60113);x=read(ROOT/'inputs.json');initial=load(ROOT/'initial.pt');baseline=load(PARENT/'baseline.pt')
 model,tok=b.c.load('receiver');before=b.c.parameter_hash(model);assert before=='6a3e1782ebad58d1e9323d5eeaff191d112848b5f0058394a998d1ac54c492b3';status['before']=before
 assert all(not p.requires_grad and p.grad is None for p in model.parameters()) and all(not m._forward_hooks for m in model.modules())
 bank=b.AnswerBank(tok);assert bank.tokens.tolist()==x['leading_tokens'];fi=torch.tensor(x['banks']['familiar']['query_ids']);six=torch.cat([fi[:8],bank.tokens[:8,:1]],1)
 with torch.no_grad():stock=model(input_ids=six,use_cache=False,logits_to_keep=1).logits.clone()
 w=Boundary(model)
 try:
  with torch.no_grad():
   w.observe=True;disabled=w.forward(six);own=w.capture['after'][:,3:5].clone();selfout=w.forward(six,own);w.observe=False
   assert torch.equal(stock,disabled) and torch.equal(stock,selfout)
  torch.save({'stock':stock,'disabled':disabled,'self':selfout},ROOT/'run/stock.pt')
  for pool,conditions in [('familiar',['original']),('novel',['original','changed'])]:
   ids=torch.tensor(x['banks'][pool]['query_ids'])
   for condition in conditions:
    out=score(w,ids,bank,baseline[pool+'_'+condition],f'{pool}_{condition}_baseline');r.equal(out,load(PARENT/'run'/f'{pool}_{condition}_sham.pt')['first'])
  status['baseline_exact']=True
  # Check native teacher-forced code path before any optimization.
  info=x['banks']['familiar'];labels=torch.tensor([b.IDS.index(t) for t in info['targets']['original']]);targets=bank.tokens[labels]
  with torch.no_grad():lp=w.forward(torch.cat([fi[:8],targets[:8,:1]],1),baseline['familiar_original'][:8],2).log_softmax(-1);actual=lp.gather(-1,targets[:8,:,None]).squeeze(-1)
  old=load(ROOT/'run/familiar_original_baseline.pt')['first']['token_lp'][torch.arange(8),labels[:8]];error=float((actual-old).abs().max());assert error<=1e-4
  torch.save({'actual':actual,'reference':old,'error':error},ROOT/'run/teacher_parity.pt');status['teacher_parity_error']=error
  for pool in ['familiar','novel']:
   if pool=='novel':status['novel_fit_opened']=True
   info=x['banks'][pool];ids=torch.tensor(info['query_ids']);labels=torch.tensor([b.IDS.index(t) for t in info['targets']['original']]);ix=torch.tensor([info['messages'].index(t) for t in info['targets']['original']])
   code=fit(w,ids,bank,labels,ix,initial[pool],pool)
   targets=bank.tokens[labels];fullpred=[]
   with torch.no_grad():
    for start in range(0,32,8):
     logits=w.forward(torch.cat([ids[start:start+8],targets[start:start+8,:1]],1),render(code[ix[start:start+8]]),2)
     fullpred.append(logits.argmax(-1))
   torch.save({'prediction':torch.cat(fullpred),'targets':targets},ROOT/'run'/f'{pool}_terminal_teacher.pt')
   payload=render(code[ix[:1]]);fast=b.score(w.forward,ids[:1],bank,payload);direct=b.direct_score(w.forward,ids[:1],bank,payload)
   err=float((fast['lp']-direct).abs().max());assert err<=1e-4 and torch.equal(fast['prediction'],direct.argmax(-1))
   torch.save({'fast':fast,'direct':direct,'error':err},ROOT/'run'/f'{pool}_parity.pt')
   status[pool]=evaluate(w,ids,bank,info,code,pool,['original'] if pool=='familiar' else ['original','changed'])
   if pool=='familiar' and not status[pool]['original']['pass']:status['verdict']='FAMILIAR ORACLE CODE QUALIFICATION NO-GO';return
  status['verdict']='ORACLE QUERY CODE REACHABLE; SOURCE TRANSFER UNPROVEN' if all(v['pass'] for v in status['novel'].values()) else 'NOVEL ORACLE QUERY CODE NO-GO'
 finally:
  w.close();assert all(not m._forward_hooks for m in model.modules())
  with torch.no_grad():restored=model(input_ids=six,use_cache=False,logits_to_keep=1).logits.clone()
  assert torch.equal(stock,restored);torch.save(restored,ROOT/'run/restored.pt')
  after=b.c.parameter_hash(model);assert before==after and all(not p.requires_grad and p.grad is None for p in model.parameters());status.update(after=after,restored_exact=True,no_parameter_gradients=True,no_hooks=True)
 check()
def run():
 (ROOT/'run').mkdir();start=time.monotonic();status={'completed':False,'novel_fit_opened':False,'supervised_oracle_only':True}
 try:body(status);check();status['completed']=True
 except BaseException as exc:status.update(error_type=type(exc).__name__,error=str(exc));raise
 finally:write(ROOT/'run/status.json',{**status,'elapsed_s':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','check','run']);q=p.parse_args();{'freeze':freeze,'check':check,'run':run}[q.mode]()
