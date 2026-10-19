"""Offline fixed diagnostic only. No model runtime or novel artifacts imported."""
import hashlib,json,sys,platform
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
RESEARCH=ROOT.parent
S='latent_binding_alignment_source_20260911'
R='latent_binding_training_fold_recovery_20260913'
C='latent_binding_never_success_20260914'
F='latent_binding_fold_failure_modes_20260913'
SEALS={S:'48d5c0842d864312e7cdbf59fdcbd5aa927c97648c1d321746d7c50a5305f3f4',R:'6267a6d332f8e2fddc4ec8a0ae286a9e2801029100d89b09220f0c99c402d67d',C:'746e1be76bfec472712203456144c14d3a29d6596f9812ef07e950f4ea46a109',F:'9eb3edd33237f3cb4ed08b332e90d3c8be7446f8e5cd0cc1b93a021dc12e76cf'}
FILES={S:['inputs.json','run/captures.pt','run/features.pt'],R:['inputs.json','features.pt','folds.json']+[f'run/fold{i}/step100/scores.pt' for i in range(4)],C:['RESULT.json'],F:['RESULT.json']}
CODE=['PLAN.md','runtime.py','test_runtime.py','job.py','test_job.py']
def read(p):return json.loads(p.read_text())
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(p):return torch.load(p,weights_only=True,map_location='cpu')
def targets(ids):
 y=np.zeros((len(ids),24))
 for i,s in enumerate(ids):y[i,ord(s[0])-65]=1;y[i,16+int(s[1])]=1
 return y

def identity():
 sources={}
 for name,paths in FILES.items():
  base=RESEARCH/name;assert sha(base/'seal.json')==SEALS[name],name
  inv=read(base/'seal.json')['files']
  for rel in paths:
   actual=sha(base/rel);assert actual==inv[rel],(name,rel)
   sources[f'{name}/{rel}']=actual
 return dict(sources=sources,seals=SEALS,code={p:sha(ROOT/p) for p in CODE},cap_s=300,threads=4,versions={'python':platform.python_version(),'numpy':np.__version__,'torch':str(torch.__version__)},folds=read(RESEARCH/R/'folds.json')['folds'],lambda_factor=.001,score_tolerance=1e-8,argmax='exact_first_index',conditions=24)

def data():
 src=read(RESEARCH/S/'inputs.json');ids=src['ids'];assert len(ids)==len(set(ids))==56
 assert all(len(s)==2 and 'A'<=s[0]<='P' and s[1] in '01234567' for s in ids)
 # Saved source tokenizer contract: A..P=32..47, 0..7=15..22.
 assert src['pairs']==[[32+ord(s[0])-65,15+int(s[1])] for s in ids]
 cap=load(RESEARCH/S/'run/captures.pt');assert torch.equal(cap['first'],cap['repeat']);assert tuple(cap['first'].shape)==(56,2,2048)
 a=load(RESEARCH/S/'run/features.pt');b=load(RESEARCH/R/'features.pt')
 for k,d in [('wire',64),('projected',960)]:assert torch.equal(a[k],b[k]) and tuple(a[k].shape)==(56,d)
 reps={'two_states':cap['first'].reshape(56,-1).double().numpy(),'wire':a['wire'].double().numpy(),'projected':a['projected'].double().numpy()}
 assert all(np.isfinite(x).all() for x in reps.values())
 inp=read(RESEARCH/R/'inputs.json');rows=inp['rows'];assert len(rows)==224
 tokens=inp['candidate_tokens'];assert len(tokens)==128
 letters=[tokens[8*i][0] for i in range(16)];digits=[tokens[i][1] for i in range(8)]
 assert len(set(letters))==16 and len(set(digits))==8
 assert tokens==[[l,d] for l in letters for d in digits]
 for i,s in enumerate(ids):
  rr=[r for r in rows if r['id']==s];assert len(rr)==4 and {r['color'] for r in rr}=={'red','green','blue','yellow'}
  assert all(r['feature']==i and r['label']==8*(ord(s[0])-65)+int(s[1]) for r in rr)
 folds=read(RESEARCH/R/'folds.json')['folds'];assert len(folds)==4
 held=[]
 for j,f in enumerate(folds):
  fi,va=f['fit_ids'],f['validation_ids'];assert f['fold']==j and len(fi)==[41,42,42,43][j]
  assert not(set(fi)&set(va)) and set(fi+va)==set(ids) and len(fi+va)==56
  assert (targets(fi).sum(0)>0).all()
  for role,ss in [('fit',fi),('validation',va)]:assert f[role+'_rows']==[i for i,r in enumerate(rows) if r['id'] in ss]
  held+=va
 assert len(held)==56 and set(held)==set(ids)
 return ids,reps,folds,rows

def ridge(x,y,z,backend='numpy'):
 if backend=='numpy':
  xm=x.mean(0);ym=y.mean(0);xc=x-xm;lam=.001*np.square(xc).sum()/len(x)
  w=xc.T@np.linalg.solve(xc@xc.T+lam*np.eye(len(x)),y-ym)
  return (z-xm)@w+ym,dict(xmean=xm,ymean=ym,weight=w,lam=np.array(lam))
 xx,yy,zz=[torch.from_numpy(v.copy()) for v in (x,y,z)]
 xm=xx.mean(0);ym=yy.mean(0);xc=xx-xm;lam=.001*xc.square().sum()/len(xx)
 w=xc.T@torch.linalg.solve(xc@xc.T+lam*torch.eye(len(xx),dtype=torch.float64),yy-ym)
 return ((zz-xm)@w+ym).numpy(),dict(xmean=xm.numpy(),ymean=ym.numpy(),weight=w.numpy(),lam=lam.numpy())
def pred(scores):return np.stack([scores[:,:16].argmax(1),scores[:,16:].argmax(1)],axis=1)
def prediction(s):p=pred(s);return [chr(65+int(l))+str(int(d)) for l,d in p]
def records(ids,scores):
 out=[]
 for s,z,p in zip(ids,scores,prediction(scores)):
  parts=[z[:16],z[16:]];gold=[ord(s[0])-65,int(s[1])]
  margins=[float(v[g]-np.delete(v,g).max()) for v,g in zip(parts,gold)]
  gaps=[float(np.sort(v)[-1]-np.sort(v)[-2]) for v in parts]
  out.append(dict(id=s,prediction=p,letter_correct=p[0]==s[0],digit_correct=p[1]==s[1],joint_correct=p==s,gold_margins=margins,top_two_gaps=gaps))
 return out

def counts(rr):return dict(n=len(rr),letter=sum(r['letter_correct'] for r in rr),digit=sum(r['digit_correct'] for r in rr),joint=sum(r['joint_correct'] for r in rr))
def native(ids,folds,rows):
 old=read(RESEARCH/F/'RESULT.json');assert old['selected_step']==100
 oldrows=old['rows'];assert len(oldrows)==896
 lookup={(r['fold'],r['step'],r['index']):r for r in oldrows};assert len(lookup)==896
 selected=[]
 for f in folds:
  d=load(RESEARCH/R/f"run/fold{f['fold']}/step100/scores.pt");assert d['indices']==f['validation_rows']
  lp=d['first']['token_lp'];assert torch.equal(lp,d['repeat']['token_lp'])
  assert tuple(lp.shape)==(len(d['indices']),128,2)
  for j,index in enumerate(d['indices']):
   row=rows[index];s=row['id'];a=lp[j].double().numpy().reshape(16,8,2)
   assert np.array_equal(a[:,:,0],np.repeat(a[:,0:1,0],8,axis=1))
   prefix=int(a[:,0,0].argmax());digit=int(a[prefix,:,1].argmax());joint=int(a.sum(2).reshape(-1).argmax());gold=ord(s[0])-65
   seq=chr(65+prefix)+str(digit);jp=chr(65+joint//8)+str(joint%8)
   rr=lookup[(f['fold'],100,index)]
   assert rr['id']==s and rr['color']==row['color'] and rr['sequential_prediction']==seq and rr['joint_prediction']==jp
   assert rr['prefix_prediction']==chr(65+prefix) and rr['gold_prefix_digit_prediction']==s[0]+str(int(a[gold,:,1].argmax()))
   assert rr['sequential_correct']==(seq==s) and rr['joint_correct']==(jp==s)
   selected.append(rr)
 assert len(selected)==224 and sum(r['sequential_correct'] for r in selected)==25 and sum(r['joint_correct'] for r in selected)==22
 groups={k:[] for k in ['neither_ever','prefix_only_ever','digit_only_ever','both_disjoint','ever_sequential']}
 for s in ids:
  rr=[r for r in oldrows if r['id']==s];assert len(rr)==16
  pe=any(r['prefix_correct'] for r in rr);de=any(r['gold_prefix_digit_correct'] for r in rr);se=any(r['sequential_correct'] for r in rr)
  g='ever_sequential' if se else 'both_disjoint' if pe and de else 'prefix_only_ever' if pe else 'digit_only_ever' if de else 'neither_ever'
  groups[g].append(s)
 assert groups==read(RESEARCH/C/'RESULT.json')['groups']
 return selected,groups

def summarize(ids,folds,rows,arrays):
 native_rows,groups=native(ids,folds,rows);groups={'all':ids,'persistent':[s for s in ids if s not in groups['ever_sequential']],**groups}
 result=dict(status='PASS',scope='fixed matched-fit-ID offline component probes; no native gate promotion',conditions=[],aggregate={},native={},cohort_ids=groups)
 for rep in ['two_states','wire','projected']:
  for mode in ['real','shifted']:
   allheld=[]
   for f in folds:
    key=f"f{f['fold']}_{rep}_{mode}";d=arrays[key]
    held=records(f['validation_ids'],d['numpy_scores'][len(f['fit_ids']):]);fit=records(f['fit_ids'],d['numpy_scores'][:len(f['fit_ids'])])
    assigned=f['fit_ids'] if mode=='real' else list(np.roll(np.array(f['fit_ids']),1))
    assignedfit=records(assigned,d['numpy_scores'][:len(f['fit_ids'])])
    allheld+=held
    result['conditions'].append(dict(key=key,fold=f['fold'],representation=rep,mode=mode,lambda_value=float(d['lam']),max_backend_score_error=float(np.abs(d['numpy_scores']-d['torch_scores']).max()),fit_true=counts(fit),fit_assigned=counts(assignedfit),heldout=counts(held),fit_rows=fit,heldout_rows=held))
   result['aggregate'][rep+'_'+mode]={g:counts([r for r in allheld if r['id'] in ss]) for g,ss in groups.items()}
 for g,ss in groups.items():
  rr=[r for r in native_rows if r['id'] in ss]
  result['native'][g]=dict(n_ids=len(ss),n_context_rows=len(rr),joint_rows=sum(r['joint_correct'] for r in rr),sequential_rows=sum(r['sequential_correct'] for r in rr),sequential_any_ids=sum(any(r['sequential_correct'] for r in rr if r['id']==s) for s in ss),sequential_all_ids=sum(all(r['sequential_correct'] for r in rr if r['id']==s) for s in ss),joint_any_ids=sum(any(r['joint_correct'] for r in rr if r['id']==s) for s in ss),joint_all_ids=sum(all(r['joint_correct'] for r in rr if r['id']==s) for s in ss))
 result['native_rows']=native_rows
 return result

def run():
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 assert identity()==read(ROOT/'FREEZE.json')
 ids,reps,folds,rows=data();dest=ROOT/'run';dest.mkdir();arrays={}
 for f in folds:
  fi=[ids.index(s) for s in f['fit_ids']];va=[ids.index(s) for s in f['validation_ids']]
  for rep,x in reps.items():
   for mode in ['real','shifted']:
    y=targets(f['fit_ids']);y=y if mode=='real' else np.roll(y,1,axis=0);z=x[fi+va]
    ns,p=ridge(x[fi],y,z);ts,tp=ridge(x[fi],y,z,'torch');key=f"f{f['fold']}_{rep}_{mode}"
    d={**p,'torch_weight':tp['weight'],'torch_xmean':tp['xmean'],'torch_ymean':tp['ymean'],'torch_lam':tp['lam'],'numpy_scores':ns,'torch_scores':ts,'fit_indices':np.array(fi),'validation_indices':np.array(va)}
    np.savez(dest/(key+'.npz'),**d);arrays[key]=d
    error=float(np.abs(ns-ts).max());assert np.isfinite(ns).all() and np.isfinite(ts).all()
    assert error<=1e-8 and np.array_equal(pred(ns),pred(ts)),(key,error)
    print(key,'PASS',error,flush=True)
 assert len(arrays)==24
 result=summarize(ids,folds,rows,arrays);write(ROOT/'RESULT.json',result)
 assert identity()==read(ROOT/'FREEZE.json')
 write(dest/'status.json',dict(status='PASS',conditions=24,model_loads=0,model_updates=0,native_evaluations=0,novel_access=0,external_spend_usd=0))

def verify():
 assert identity()==read(ROOT/'FREEZE.json');ids,reps,folds,rows=data();arrays={};maxerr=0.
 for f in folds:
  fi=[ids.index(s) for s in f['fit_ids']];va=[ids.index(s) for s in f['validation_ids']]
  for rep,x in reps.items():
   for mode in ['real','shifted']:
    key=f"f{f['fold']}_{rep}_{mode}";d=dict(np.load(ROOT/'run'/(key+'.npz'),allow_pickle=False));arrays[key]=d
    assert np.array_equal(d['fit_indices'],fi) and np.array_equal(d['validation_indices'],va)
    y=targets(f['fit_ids']);y=y if mode=='real' else np.roll(y,1,axis=0)
    xm=x[fi].mean(0);ym=y.mean(0);xc=x[fi]-xm;lam=.001*np.square(xc).sum()/len(fi)
    assert np.array_equal(xm,d['xmean']) and np.array_equal(ym,d['ymean']) and lam==float(d['lam'])
    # Normal-equation residual verifies the saved ridge solution without another fit.
    w=d['weight'];lhs=xc.T@(xc@w)+lam*w;rhs=xc.T@(y-ym)
    assert np.linalg.norm(lhs-rhs)/max(np.linalg.norm(rhs),1e-30)<1e-8
    ns=(x[fi+va]-xm)@w+ym
    z=torch.from_numpy(x[fi+va].copy());ts=((z-torch.from_numpy(d['torch_xmean']))@torch.from_numpy(d['torch_weight'])+torch.from_numpy(d['torch_ymean'])).numpy()
    assert np.array_equal(ns,d['numpy_scores']) and np.array_equal(ts,d['torch_scores'])
    err=float(np.abs(ns-ts).max());maxerr=max(maxerr,err);assert err<=1e-8 and np.array_equal(pred(ns),pred(ts))
 assert summarize(ids,folds,rows,arrays)==read(ROOT/'RESULT.json')
 return dict(status='PASS',conditions_verified=24,source_identity='PASS',saved_forward_replay='EXACT',normal_equations='PASS',max_backend_score_error=maxerr,no_refit=True)

if __name__=='__main__':
 command=sys.argv[1]
 if command=='preflight':data();print(json.dumps(identity(),indent=2))
 elif command=='freeze':data();write(ROOT/'FREEZE.json',identity());print(sha(ROOT/'FREEZE.json'))
 elif command=='check':assert identity()==read(ROOT/'FREEZE.json');data();print('Frozen identity PASS')
 elif command=='run':run()
 elif command=='verify':print(json.dumps(verify(),indent=2))
 else:raise ValueError(command)
