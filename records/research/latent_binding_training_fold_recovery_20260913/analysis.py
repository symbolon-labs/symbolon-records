"""Independent scalar/NumPy closing audit; no model or new scoring."""
import argparse,math
import numpy as np
import torch
import common as h

def counts(out,targets,tokens):
 p=out['token_lp'].numpy();lp=out['lp'].numpy();assert np.array_equal(p.sum(-1),lp)
 assert np.array_equal(lp.argmax(-1),out['prediction'].numpy())
 groups={}
 for i,(letter,digit) in enumerate(tokens):groups.setdefault(letter,[]).append(i)
 joint=seq=0;per=[]
 for i,target in enumerate(targets):
  j=max(range(len(tokens)),key=lambda k:float(lp[i,k]))
  g=max(groups.values(),key=lambda ks:float(p[i,ks[0],0]));s=max(g,key=lambda k:float(p[i,k,1]))
  joint+=j==target;seq+=s==target;per.append(dict(target=target,joint_prediction=j,sequential_prediction=s))
 return dict(joint=joint,sequential=seq,rows=len(targets)),per

def audit():
 h.check();status=h.read(h.ROOT/'run/status.json');assert status['completed'] and status['folds_completed']==4 and not status['novel_evaluated']
 completion=h.read(h.ROOT/'job/completion.json');assert completion['exit_code']==0 and not completion['timeout'] and completion['child_reaped']
 x=h.read(h.ROOT/'inputs.json');folds=h.read(h.ROOT/'folds.json')['folds'];coords=h.load(h.ROOT/'coordinates.pt');state=h.load(h.CANDIDATE)
 totals={s:dict(joint=0,sequential=0,rows=0) for s in h.STEPS};details=[];maximum=0.
 for f,co in zip(folds,coords):
  assert co['fit_indices']==[x['ids'].index(n) for n in f['fit_ids']]
  fit=h.load(h.ROOT/'features.pt')['projected'][co['fit_indices']].double().numpy();sing=np.linalg.svd(fit,compute_uv=False)
  assert len(fit)==co['rank'] and np.allclose(sing,co['input_singular'].numpy(),rtol=1e-10,atol=1e-10)
  lam=.001*np.square(fit-fit.mean(0)).sum()/len(fit);assert math.isclose(lam,co['input_lambda'],rel_tol=1e-12)
  assert math.isclose(float(np.square(fit@co['input_basis'].numpy()).sum(-1).mean()),1,rel_tol=1e-10)
  dest=h.ROOT/'run'/f'fold{f["fold"]}';assert h.read(dest/'preflight.json')['relative']<=.1
  for step in h.STEPS:
   sd=dest/f'step{step}';a=h.load(sd/'scores.pt');assert a['indices']==f['validation_rows']
   for k in ['token_lp','lp','prediction']:assert torch.equal(a['first'][k],a['repeat'][k])
   targets=[x['rows'][i]['label'] for i in a['indices']];result,per=counts(a['first'],targets,x['candidate_tokens']);recorded=h.read(sd/'counts.json')
   for k in result:assert result[k]==recorded[k];totals[step][k]+=result[k]
   err=float((a['first']['lp'][:8]-a['direct']).abs().max());assert err==a['direct_error'] and err<=1e-4
   candidate=h.load(sd/'candidate.pt');h.fixed(candidate,state)
   matched=h.MATCHED/'run'/f'fold{f["fold"]}'/f'step{step}'
   for name in ['candidate.pt','scores.pt','direct_diagnostic.pt','fit.pt']:
    if (matched/name).exists():h.exact_tree(h.load(sd/name),h.load(matched/name))
   prior=h.ORIGINAL/'run'/f'fold{f["fold"]}'/f'step{step}'
   if (prior/'candidate.pt').exists():h.exact_tree(candidate,h.load(prior/'candidate.pt'))
   if (prior/'scores.pt').exists():
    old=h.load(prior/'scores.pt')
    for key in ['token_lp','lp','prediction']:assert torch.equal(a['first'][key],old['first'][key])
   diagnostic=h.load(sd/'direct_diagnostic.pt');assert torch.equal(diagnostic['direct'],a['direct'])
   assert torch.equal(a['first']['prediction'][:8],a['direct'].argmax(-1))
   if step:
    fitstate=h.load(sd/'fit.pt');theta=fitstate['theta'];assert len(fitstate['history'])==step
    if (prior/'fit.pt').exists():h.exact_tree(fitstate,h.load(prior/'fit.pt'))
    reconstructed=(co['base'].double().numpy()+co['output_basis'].numpy()@theta.numpy()@co['input_basis'].numpy().T).astype(np.float32)
    weight=candidate[h.NAME].numpy();relative=float(np.linalg.norm(reconstructed.astype(float)-weight)/np.linalg.norm(weight));assert relative<=1e-6;maximum=max(maximum,relative)
    assert torch.equal(h.weight(theta,co),candidate[h.NAME])
   else:assert all(torch.equal(v,state[k]) for k,v in candidate.items())
   contract=h.load(sd/'contract.pt');assert contract['error']<=1e-4
   for r in contract['records']:
    cap=r['capture'];hidden=cap['hidden'];assert torch.equal(cap['output'][:,:3],hidden[:,:3]) and torch.equal(cap['output'][:,5:],hidden[:,5:])
    bridge=h.c.bridge();bridge.requires_grad_(False);bridge.load_state_dict(candidate)
    with torch.no_grad():out=bridge.reader(hidden,cap['memory'])
    assert torch.equal(out,cap['output']) and torch.equal(bridge.reader.capture['hard'],cap['hard']) and torch.equal(bridge.reader.capture['scores'],cap['scores'])
   details.append(dict(fold=f['fold'],step=step,counts=result,per_row=per))
 selected=min([50,100,200],key=lambda s:(-totals[s]['sequential'],s));qualified=all(totals[selected][k]>totals[0][k] for k in ['joint','sequential'])
 assert status['selection']['selected_step']==selected and status['selection']['qualifies']==qualified
 assert {str(k):v for k,v in totals.items()}==status['totals'];assert status['restoration_exact'] and status['no_hooks'] and status['no_gradients'] and status['before']==status['after']
 stock=h.load(h.ROOT/'run/stock.pt');assert torch.equal(stock['stock'],h.load(h.ROOT/'run/restored.pt'))
 h.check();return dict(status='PASS',totals=totals,selected_step=selected,qualifies=qualified,maximum_numpy_weight_relative=maximum,details=details,scope='Developmental alignment validation, not pristine novel binding; no automatic all56 refit')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--save',action='store_true');a=p.parse_args();result=audit()
 if a.save:h.write(h.ROOT/'ANALYSIS.json',result)
 else:assert h.read(h.ROOT/'ANALYSIS.json')==__import__('json').loads(__import__('json').dumps(result))
 print({k:v for k,v in result.items() if k!='details'})
