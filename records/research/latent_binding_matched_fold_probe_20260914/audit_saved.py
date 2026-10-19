"""Independent saved-array outcome audit. Does not fit or import runtime.py."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
r=read(ROOT/'RESULT.json');freeze=read(ROOT/'FREEZE.json')
bykey={c['key']:c for c in r['conditions']};assert len(bykey)==24
aggregate={};errors={};gaps={};foldcounts={};assigned_null_overlap={};fitcounts={}
for rep in ['two_states','wire','projected']:
 for mode in ['real','shifted']:
  mapping={};fit_sum=np.zeros(4,dtype=int);assigned_sum=np.zeros(4,dtype=int)
  for f in freeze['folds']:
   key=f"f{f['fold']}_{rep}_{mode}";d=np.load(ROOT/'run'/(key+'.npz'));c=bykey[key]
   n=len(f['fit_ids']);p=d['numpy_scores'];preds=[];goldids=f['fit_ids']+f['validation_ids']
   cc=[];gs=[]
   for s,z in zip(goldids,p):
    l=max(range(16),key=lambda k:z[k]);di=max(range(8),key=lambda k:z[16+k]);out=chr(65+l)+str(di)
    preds.append(out);cc.append([int(out[0]==s[0]),int(out[1]==s[1]),int(out==s)])
    lm=float(z[ord(s[0])-65]-max(z[k] for k in range(16) if k!=ord(s[0])-65))
    dm=float(z[16+int(s[1])]-max(z[16+k] for k in range(8) if k!=int(s[1])))
    gap=[float(sorted(z[:16])[-1]-sorted(z[:16])[-2]),float(sorted(z[16:])[-1]-sorted(z[16:])[-2])]
    gs.append(gap)
    reported=(c['fit_rows']+c['heldout_rows'])[len(preds)-1]
    assert reported['prediction']==out and reported['gold_margins']==[lm,dm] and reported['top_two_gaps']==gap
   for role,start,stop in [('fit_true',0,n),('heldout',n,len(goldids))]:
    vals=np.asarray(cc[start:stop]).sum(0);expected=dict(n=stop-start,letter=int(vals[0]),digit=int(vals[1]),joint=int(vals[2]));assert c[role]==expected
   assigned=f['fit_ids'] if mode=='real' else f['fit_ids'][-1:]+f['fit_ids'][:-1]
   aa=[sum(p[i]==s[i] for p,s in zip(preds[:n],assigned)) for i in [0,1]]+[sum(p==s for p,s in zip(preds[:n],assigned))]
   assert c['fit_assigned']==dict(n=n,letter=aa[0],digit=aa[1],joint=aa[2])
   fit_sum+=np.array([n,*np.asarray(cc[:n]).sum(0)]);assigned_sum+=np.array([n,*aa])
   for s,pp,g in zip(f['validation_ids'],preds[n:],gs[n:]):assert s not in mapping;mapping[s]=(pp,g)
   foldcounts[key]=c['heldout']
   if mode=='shifted':assigned_null_overlap[str(f['fold'])]={'n':n,'unchanged_letter_labels':sum(s[0]==a[0] for s,a in zip(f['fit_ids'],assigned)),'unchanged_digit_labels':sum(s[1]==a[1] for s,a in zip(f['fit_ids'],assigned)),'unchanged_whole_labels':sum(s==a for s,a in zip(f['fit_ids'],assigned))}
  assert len(mapping)==56
  name=rep+'_'+mode;aggregate[name]={}
  for group,ss in r['cohort_ids'].items():
   value=dict(n=len(ss),letter=sum(mapping[s][0][0]==s[0] for s in ss),digit=sum(mapping[s][0][1]==s[1] for s in ss),joint=sum(mapping[s][0]==s for s in ss));assert value==r['aggregate'][name][group];aggregate[name][group]=value
  errors[name]={s:v[0] for s,v in mapping.items() if s!=v[0]}
  gaparr=np.array([v[1] for v in mapping.values()]);gaps[name]={'minimum_letter_gap':float(gaparr[:,0].min()),'minimum_digit_gap':float(gaparr[:,1].min()),'component_gaps_le_1e_8':int((gaparr<=1e-8).sum())}
  fitcounts[name]={'true_labels':fit_sum.tolist(),'assigned_labels':assigned_sum.tolist(),'columns':['fit_occurrences','letter','digit','whole_id'],'note':'168 overlapping fit occurrences, not independent heldout IDs'}
# Independently join saved selected rows to unique IDs and count outcomes.
rr=r['native_rows'];assert len(rr)==224 and len({x['index'] for x in rr})==224
for group,ids in r['cohort_ids'].items():
 rows=[x for x in rr if x['id'] in ids];out=dict(n_ids=len(ids),n_context_rows=len(rows))
 for kind in ['joint','sequential']:
  out[kind+'_rows']=sum(x[kind+'_prediction']==x['id'] for x in rows)
  ns=[sum(x[kind+'_prediction']==s for x in rows if x['id']==s) for s in ids]
  out[kind+'_any_ids']=sum(n>0 for n in ns);out[kind+'_all_ids']=sum(n==4 for n in ns)
 assert out==r['native'][group]
result=dict(status='PASS',conditions=24,heldout_predictions=336,aggregate=aggregate,fit_counts=fitcounts,errors=errors,top_two_gaps=gaps,shifted_control_label_overlap=assigned_null_overlap,fold_counts=foldcounts,native=r['native'],no_refit=True)
if __name__=='__main__':
 import sys
 if sys.argv[1]=='save':
  with (ROOT/'INDEPENDENT_AUDIT.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 elif sys.argv[1]=='verify':assert read(ROOT/'INDEPENDENT_AUDIT.json')==result
 else:raise ValueError(sys.argv[1])
 print('Independent outcome/margin/count audit PASS; no fits')
