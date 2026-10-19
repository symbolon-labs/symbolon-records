"""Independent NumPy/Torch saved-logit analysis; no model runtime imports."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;R=ROOT.parent
read=lambda p:json.loads(p.read_text())
load=lambda p:torch.load(p,weights_only=True,map_location='cpu')
def scores(logits,targets):
 a=logits.double().numpy();maximum=a.max(-1,keepdims=True);lp=a-maximum-np.log(np.exp(a-maximum).sum(-1,keepdims=True));gold=np.take_along_axis(lp,targets.numpy()[...,None],-1)[...,0]
 tp=torch.log_softmax(logits.double(),-1).gather(-1,targets[...,None])[...,0].numpy();assert np.max(np.abs(tp-gold))<=1e-8
 prediction=a.argmax(-1);masked=a.copy();np.put_along_axis(masked,targets.numpy()[...,None],-np.inf,-1);margin=np.take_along_axis(a,targets.numpy()[...,None],-1)[...,0]-masked.max(-1)
 return gold,prediction,margin,float(np.max(np.abs(tp-gold)))
def compute():
 inp=read(R/'latent_binding_training_fold_recovery_20260913/inputs.json');probe=read(R/'latent_binding_matched_fold_probe_20260914/RESULT.json');correct=set(v['id'] for c in probe['conditions'] if c['mode']=='real' and c['representation']=='wire' for v in c['heldout_rows'] if v['joint_correct']);co=probe['cohort_ids'];co={**co,'persistent_decodable':sorted(correct&set(co['persistent'])),'persistent_not_decodable':sorted(set(co['persistent'])-correct)};assert len(co['persistent_decodable'])==27
 rr=[];maxerr=0.
 for f in range(4):
  d=load(ROOT/f'run/fold{f}/logits.pt');old=load(R/f'latent_binding_training_fold_recovery_20260913/run/fold{f}/step100/contract.pt');assert torch.equal(d['baseline'],old['native'])
  for a,b in [('baseline','native_repeat'),('baseline','seam'),('baseline','sham'),('treatment','treatment_repeat'),('control','control_repeat')]:assert torch.equal(d[a],d[b])
  values={k:scores(d[k],d['targets']) for k in ['baseline','treatment','control']};maxerr=max(maxerr,max(x[3] for x in values.values()))
  for j,i in enumerate(d['indices']):
   source=inp['rows'][i];row=dict(fold=f,index=i,id=source['id'],color=source['color'],targets=d['targets'][j].tolist(),arms={})
   for k,(gold,pred,margin,_) in values.items():row['arms'][k]=dict(gold_logp=gold[j].tolist(),prediction=pred[j].tolist(),gold_margin=margin[j].tolist(),correct=(pred[j]==d['targets'][j].numpy()).tolist())
   rr.append(row)
 assert len(rr)==224 and len({x['index'] for x in rr})==224
 perid=[]
 for name in inp['ids']:
  rs=[r for r in rr if r['id']==name];assert len(rs)==4
  effects={k:np.mean([np.array(r['arms'][k]['gold_logp'])-np.array(r['arms']['baseline']['gold_logp']) for r in rs],0).tolist() for k in ['treatment','control']}
  difference=(np.array(effects['treatment'])-np.array(effects['control'])).tolist();perid.append(dict(id=name,effects=effects,treatment_minus_control=difference,mean_component_contrast=float(np.mean(difference))))
 summary={}
 for group,ids in co.items():
  rs=[r for r in rr if r['id'] in ids];ps=[r for r in perid if r['id'] in ids];totals={}
  for arm in ['baseline','treatment','control']:
   flags=np.array([r['arms'][arm]['correct'] for r in rs],dtype=bool).reshape(-1,2);bf=np.array([r['arms']['baseline']['correct'] for r in rs],dtype=bool).reshape(-1,2)
   totals[arm]=dict(prefix_correct=int(flags[:,0].sum()),gold_prefix_digit_correct=int(flags[:,1].sum()),both_teacher_forced=int(flags.all(1).sum()),lost_prefix=int((bf[:,0]&~flags[:,0]).sum()),lost_digit=int((bf[:,1]&~flags[:,1]).sum()),gained_prefix=int((~bf[:,0]&flags[:,0]).sum()),gained_digit=int((~bf[:,1]&flags[:,1]).sum()))
  summary[group]=dict(n_ids=len(ids),n_context_rows=len(rs),arms=totals,treatment_minus_control_mean_components=np.mean([p['treatment_minus_control'] for p in ps],0).tolist() if ps else None,primary_mean_component_contrast=float(np.mean([p['mean_component_contrast'] for p in ps])) if ps else None,negative_contrast_ids=sum(p['mean_component_contrast']<0 for p in ps),positive_contrast_ids=sum(p['mean_component_contrast']>0 for p in ps))
 status=read(ROOT/'run/status.json');complete=read(ROOT/'run/complete.json');assert complete['completed'] and status['restoration_exact'] and status['before']==status['after'];assert torch.equal(load(ROOT/'run/stock.pt')['stock'],load(ROOT/'run/restored.pt'))
 return dict(status='AUDIT_PASS',interpretation='necessity intervention, not native repair; teacher-forced metrics not sequential whole-ID accuracy',max_numpy_torch_logp_error=maxerr,cohorts=co,summary=summary,per_id=perid,rows=rr)
if __name__=='__main__':
 r=compute()
 if sys.argv[1]=='save':
  with (ROOT/'RESULT.json').open('x') as f:json.dump(r,f,indent=2,allow_nan=False)
 elif sys.argv[1]=='verify':assert r==read(ROOT/'RESULT.json')
 else:raise ValueError(sys.argv[1])
 print(json.dumps({'status':r['status'],'primary':r['summary']['persistent_decodable'],'all':r['summary']['all']}))
