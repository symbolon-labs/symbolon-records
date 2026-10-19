"""Post-result independent NumPy geometry audit; no fitted decoder or new gate."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def audit():
    banks=torch.load(ROOT/'stages.pt',weights_only=True);result=read(ROOT/'RESULT.json');checks=0
    for name,b in banks.items():
        labels=[r['id'] for r in b['meta']];ids=sorted(set(labels));ri=[labels.index(i) for i in ids]
        for stage,saved in b['stages'].items():
            x=saved.numpy().astype(np.float64);c=x[ri];record=result['results'][name][stage]
            spread=max(np.linalg.norm(row-c[ids.index(label)]) for row,label in zip(x,labels))
            assert np.isclose(spread,record['within_id_max_l2'],rtol=1e-10,atol=1e-12)
            for pair in record['pairs']:
                a,bb=c[ids.index(pair['a'])],c[ids.index(pair['b'])]
                assert np.isclose(np.linalg.norm(a-bb),pair['euclidean'],rtol=1e-10,atol=1e-12)
                assert np.isclose(1-np.dot(a,bb)/(np.linalg.norm(a)*np.linalg.norm(bb)),pair['cosine'],rtol=1e-8,atol=1e-12)
            s=np.linalg.svd(c-c.mean(0),compute_uv=False)
            for dtype,key in [(np.float32,'torch.float32'),(np.float64,'torch.float64')]:
                assert int((s>np.finfo(dtype).eps*max(c.shape)*s[0]).sum())==record['centered_rank'][key]
            for metric,rec in record['privileged_reference_decode'].items():
                pred=[ids[int(np.argmin([np.linalg.norm(row-v) if metric=='euclidean' else 1-np.dot(row,v)/(np.linalg.norm(row)*np.linalg.norm(v)) for v in c]))] for row in x]
                assert pred==rec['predictions']
                eligible=[i for i,r in enumerate(b['meta']) if r['group']!=b['meta'][ri[ids.index(r['id'])]]['group']]
                assert eligible==rec['eligible_indices'] and len(eligible)==rec['n']
                assert sum(pred[i]==labels[i] for i in eligible)==rec['correct']
                assert sum(ids[(ids.index(pred[i])+1)%len(ids)]==labels[i] for i in eligible)==rec['shifted_label_control_correct']
            for metric,rec in record['familiar_nearest_neighbor'].items():
                f=banks['familiar'];fl=[r['id'] for r in f['meta']];fi=sorted(set(fl));fc=f['stages'][stage].numpy().astype(np.float64)[[fl.index(i) for i in fi]]
                pred=[fi[int(np.argmin([np.linalg.norm(row-v) if metric=='euclidean' else 1-np.dot(row,v)/(np.linalg.norm(row)*np.linalg.norm(v)) for v in fc]))] for row in x]
                assert pred==rec['predictions']
                agreement=sum(p==r['prediction'] for p,r in zip(pred,b['meta']));assert agreement==rec['agreement']
                strong=agreement>=56 and all(sum(p==r['prediction'] for p,r in zip(pred,b['meta']) if r['id']==label)>=.75*labels.count(label) for label in ids)
                assert strong==rec['strong_agreement']
            checks+=1
    models=read(ROOT.parent/'latent_binding_order_ladder_20260910/ARCHITECTURE_LOCK.json')['models'];files=0
    for model in models.values():
        for name,v in model['files'].items():assert sha(Path(model['path'])/name)==v['sha256'];files+=1
    assert files==22
    return dict(status='PASS',bank_stages=checks,model_tokenizer_files=files,result_sha256=sha(ROOT/'RESULT.json'),stages_sha256=sha(ROOT/'stages.pt'),post_result_audit=True,no_new_hypothesis_or_threshold=True)
if __name__=='__main__':
    torch.set_num_threads(4)
    p=argparse.ArgumentParser();p.add_argument('--save',action='store_true');q=p.parse_args();result=audit()
    if q.save:
        with (ROOT/'AUDIT.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    else:assert result==read(ROOT/'AUDIT.json')
    print(json.dumps(result))
