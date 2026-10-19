"""Frozen native bridge with explicitly joined novel whole-ID inputs."""
import copy
import importlib.util
from pathlib import Path
import torch
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'latent_binding_shared_id_native_corrected_20260908'
spec=importlib.util.spec_from_file_location('sealed_shared_native_for_novel',PARENT/'core.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
b,s,u,n,evaluation=p.b,p.s,p.u,p.n,p.evaluation
DATA,FIT,ORIGIN=p.DATA,p.FIT,p.ORIGIN
SOURCE=ROOT.parent/'latent_binding_novel_whole_source_20260910'
ISO=p.SOURCE
CANDIDATE=p.PARENT/'run/candidate.pt'
NATIVE=PARENT
CHECKPOINT=p.CHECKPOINT
VIEW=0
ORDER=[0,1,2,3]
load,bridge,fixed,last_logits=p.load,p.bridge,p.fixed,p.last_logits
def gate(metrics,paired):
    counts={k:round(v['accuracy']*32) for k,v in metrics.items()}
    donor=round(metrics['shuffled']['donor_following']*32)
    changed=round(metrics['wrong_binding']['donor_following']*32)
    count=(counts['correct']>=31 and donor>=31 and changed>=28 and counts['receiver_only']<=4
        and counts['zero']==counts['receiver_only'] and counts['canonical_only']<=4
        and all(counts[k]<=8 for k in ['shuffled','wrong_binding','random_71001','random_71002']))
    uncertainty=all(v[m]['ci95'][0]>0 for part in paired.values() for v in part.values() for m in ['accuracy','margin'])
    passed=count and uncertainty
    return dict(native_view_pass=passed,count_pass=count,paired_pass=uncertainty,view=VIEW,
        thresholds=dict(original=31,shuffled_donor=31,changed_donor=28),
        verdict='NOVEL WHOLE-ID NATIVE CONTROLS PASS; TEXT PENDING' if passed else 'NOVEL WHOLE-ID BINDING NO-GO; TEXT PENDING',
        text_measured=False,full_context_gate_completed=False,novel_binding_established=False)

def query_join(current,previous):
    """Match historical controls by actual query tokens; reject absent queries."""
    assert current.ndim==previous.ndim==2 and current.shape[1]==previous.shape[1]
    rows=[]
    for tokens in current:
        matches=torch.where((previous==tokens).all(-1))[0]
        assert len(matches)>0
        rows.append(int(matches[0]))
    return torch.tensor(rows)
def load_data():
    data=b.read(SOURCE/'data/examples.json');joined=load(SOURCE/'run/hybrid.pt')
    inputs=b.read(SOURCE/'inputs.json')
    assert inputs['seed']==191003 and not inputs['receiver_evaluated']
    tok=AutoTokenizer.from_pretrained(b.read(SOURCE/'FREEZE.json')['identity']['models']['sender']['path'],local_files_only=True)
    values={};tokens={};texts={}
    for field,key,arm,offset in [('source','hidden','original',0),('wrong_source','wrong_hidden','changed',8)]:
        texts[key]=[r[field] for r in data['records']];ids=b.c.tokens(tok,texts[key]);tokens[key]=ids
        assert torch.equal(ids,joined['source_ids'][offset:offset+8])
        assert texts[key]==inputs['sources'][offset:offset+8]
        values[key]=joined[arm].clone()
    assert torch.equal(values['hidden'][:,[0,1,5,6,10,11,15,16]],values['wrong_hidden'][:,[0,1,5,6,10,11,15,16]])
    return data,dict(**values,source_ids=torch.cat(list(tokens.values()))),dict(view=VIEW,order=ORDER,seed=191003,
        source_texts=texts,source_ids={k:v.tolist() for k,v in tokens.items()},cache_rows=joined['cache_rows'].tolist(),
        input_exact_cache_reuse=True,source_layer=13,source_interface='isolated source ID states plus fresh condition-specific whole-table non-ID states')
def source_parity(memories,data):
    # Saved-data diagnostic only; inference memories were already computed.
    lookup={}
    for i,row in enumerate(data['rows']):
        for j,line in enumerate(data['records'][row['group']]['source'].splitlines()):
            name=line.split(' = ')[0];wire=memories['correct'][i,2*j]
            if name in lookup:assert torch.equal(lookup[name],wire)
            else:lookup[name]=wire
    assert set(lookup)==set(b.read(SOURCE/'inputs.json')['selected_ids'])
    assert torch.equal(memories['correct'][:,::2],memories['wrong_binding'][:,::2])
