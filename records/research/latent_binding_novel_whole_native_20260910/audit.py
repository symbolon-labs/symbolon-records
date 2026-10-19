"""Independent scalar/NumPy score and state audit, no runtime/analysis imports."""
import argparse
import platform
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'latent_binding_context_diversity_20260907'
FIT=ROOT.parent/'latent_binding_context_span_fit_20260908'
NATIVE=ROOT.parent/'latent_binding_shared_id_native_corrected_20260908'
FRESH_SOURCE=ROOT.parent/'latent_binding_novel_whole_source_20260910'
CANDIDATE=ROOT.parent/'latent_binding_shared_id_query_mechanics_20260908'
SOURCE=ROOT.parent/'latent_binding_isolated_source_mechanics_20260908'
from transformers import AutoTokenizer
spec_source=importlib.util.spec_from_file_location('independent_source_wire_audit',SOURCE/'audit.py');source_audit=importlib.util.module_from_spec(spec_source);spec_source.loader.exec_module(source_audit)
import scalar
IDS=scalar.IDS
NAMES=['receiver_only','correct','shuffled','wrong_binding','random_71001','random_71002','zero','canonical_only']
def read(p):return json.loads(p.read_text())
def load(p):return torch.load(p,weights_only=True)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def decision(counts,donor,changed,paired):
    count=(counts['correct']>=31 and donor>=31 and changed>=28 and counts['receiver_only']<=4 and counts['zero']==counts['receiver_only']
        and all(counts[k]<=8 for k in ['shuffled','wrong_binding','random_71001','random_71002']) and counts['canonical_only']<=4)
    uncertainty=all(v[m]['ci95'][0]>0 for part in paired.values() for v in part.values() for m in ['accuracy','margin'])
    passed=count and uncertainty
    return dict(native_view_pass=passed,count_pass=count,paired_pass=uncertainty,
        verdict='NOVEL WHOLE-ID NATIVE CONTROLS PASS; TEXT PENDING' if passed else 'NOVEL WHOLE-ID BINDING NO-GO; TEXT PENDING')
def audit():
    freeze=read(ROOT/'FREEZE.json');identity=freeze['identity'];files=0
    for path,digest in read(ROOT/'ANCHORS.json').items():
        parent=Path(path);assert sha(parent/'seal.json')==digest
        inventory=read(parent/'seal.json')['files']
        assert set(inventory)=={str(p.relative_to(parent)) for p in parent.rglob('*') if p.is_file() and p!=parent/'seal.json'}
        for name,h in inventory.items():assert sha(parent/name)==h
    for name,h in identity['code'].items():assert sha(ROOT/name)==h
    for kind in ['inputs','imports']:
        for p,h in identity[kind].items():assert sha(Path(p))==h
    for model in identity['models'].values():
        for name,v in model['files'].items():
            p=Path(model['path'])/name;assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256'];files+=1
    historical=read(ROOT.parent/'latent_binding_value_span_20260908/FREEZE.json')['identity']
    current=copy.deepcopy(identity['qualified_span'])
    host=current['parent_identity']['parent_identity']
    assert host['platform']==platform.platform()=='Linux-7.2.3-arch1-2-x86_64-with-glibc2.44'
    assert historical['parent_identity']['parent_identity']['platform']=='Linux-7.2.2-arch1-1-x86_64-with-glibc2.44'
    host['platform']=historical['parent_identity']['parent_identity']['platform']
    assert current==historical
    state=load(CANDIDATE/'run/candidate.pt');unchanged=load(ROOT/'run/unchanged_state.pt')
    assert len(state)==len(unchanged)==11 and all(np.array_equal(v,unchanged[k]) for k,v in state.items())
    assert sha(CANDIDATE/'run/candidate.pt')==freeze['checkpoint']=='d0f16cb9705ba53daf6cc8f2bfad03a08efa96cbb4f77dd836c9533e948cba87'
    result=read(ROOT/'analysis.json');done=read(ROOT/'run/status.json');job=read(ROOT/'job/completion.json');manifest=read(ROOT/'run/manifest.json')
    assert job['exit_code']==0 and not job['timeout'] and done['completed'] and 'error' not in done
    assert freeze['cap_s']==read(ROOT/'job/launch.json')['cap_s']==2400
    assert done['receiver_before']==done['receiver_after']=='6a3e1782ebad58d1e9323d5eeaff191d112848b5f0058394a998d1ac54c492b3'
    assert not manifest['training'] and manifest['dev_evaluated'] and manifest['novel_whole_id_evaluated'] and not manifest['final_evaluated'] and not manifest['text_measured']
    assert manifest['view']==0 and manifest['order']==[0,1,2,3] and manifest['data_seed']==191003
    assert not (DATA/'DEV_LOCK.json').exists() and not (DATA/'dev').exists()
    data=read(ROOT/'run/data.json');assert data==read(FRESH_SOURCE/'data/examples.json')
    lock=read(ROOT/'NOVEL_LOCK.json');assert lock['burned'] and lock['seed']==191003 and lock['checkpoint']==freeze['checkpoint']
    assert lock['freeze_sha256']==sha(ROOT/'FREEZE.json') and lock['data_sha256']==sha(FRESH_SOURCE/'data/examples.json')
    history=load(FIT/'train/inputs.pt')['ids'][:32].numpy();current_ids=np.array(manifest['ids']);history_rows=[]
    tok_receiver=AutoTokenizer.from_pretrained(identity['models']['receiver']['path'],local_files_only=True)
    assert current_ids.tolist()==[tok_receiver.encode(r['query'],add_special_tokens=False) for r in data['rows']]
    for row in current_ids:
        ix=np.flatnonzero((history==row).all(-1));assert len(ix)>0;history_rows.append(int(ix[0]))
    assert history_rows==manifest['history_rows']
    rows=data['rows'];targets={k:[] for k in ['original','changed']};positions={k:[] for k in targets}
    for row in rows:
        record=data['records'][row['group']]
        for key,field,label in [('original','source','answer'),('changed','wrong_source','wrong_answer')]:
            pairs=[s.split(' = ') for s in record[field].splitlines()];assert len(pairs)==4 and len({p[0] for p in pairs})==len({p[1] for p in pairs})==4
            pos=next(i for i,(_,color) in enumerate(pairs) if color==row['color']);assert pairs[pos][0]==row[label]
            targets[key].append(IDS.index(pairs[pos][0]));positions[key].append(pos)
    assert targets==manifest['targets'] and manifest['example_ids']==[r['id'] for r in rows]
    outputs={};raws={}
    for phase in ['evaluation','repeat']:
        raw=load(ROOT/'run'/phase/'outputs.pt');canon=load(ROOT/'run'/phase/'canonical_only.pt');raws[phase]=raw
        assert np.array_equal(canon['result']['token_lp'],canon['opposite_wire']['token_lp'])
        assert np.array_equal(canon['wire'],raw['memories']['correct']) and np.array_equal(canon['labels'],raw['labels'])
        assert raw['labels'].tolist()==targets['original'] and raw['wrong_labels'].tolist()==targets['changed']
        outputs[phase]={**raw['outputs'],'canonical_only':canon['result']}
    raw=raws['evaluation'];current=outputs['evaluation'];tokens=raw['candidate_tokens'].numpy();donors=raw['donor_rows']
    assert set(current)==set(outputs['repeat'])==set(NAMES)
    assert tokens.tolist()==manifest['candidate_tokens']==read(DATA/'inputs.json')['candidate_tokens']
    assert all(np.array_equal(v['token_lp'],outputs['repeat'][k]['token_lp']) for k,v in current.items())
    assert donors==manifest['donors']==raws['repeat']['donor_rows']
    assert all(rows[i]['query']==rows[j]['query'] and i!=j and rows[i]['answer']!=rows[j]['answer'] for i,j in enumerate(donors))
    assert np.array_equal(current['shuffled']['token_lp'],current['correct']['token_lp'][donors])
    assert np.array_equal(current['zero']['token_lp'],current['receiver_only']['token_lp'])
    prior=load(NATIVE/'run/evaluation/outputs.pt')['outputs'];prior['canonical_only']=load(NATIVE/'run/evaluation/canonical_only.pt')['result']
    assert all(np.array_equal(current[k]['token_lp'],prior[k]['token_lp'][history_rows]) for k in ['receiver_only','zero','canonical_only'])
    memory=load(ROOT/'run/memories.pt');source=load(ROOT/'run/source.pt');isolated=dict(first=load(FRESH_SOURCE/'run/first.pt')['isolated'],repeat=load(FRESH_SOURCE/'run/repeat.pt')['isolated'])
    source_record=dict(pairs=load(FRESH_SOURCE/'run/hybrid.pt')['isolated_pairs']);positions_id=source_audit.POS;group=np.array([r['group'] for r in rows])
    assert np.array_equal(isolated['first'],isolated['repeat'])
    tokenizer=AutoTokenizer.from_pretrained(identity['models']['sender']['path'],local_files_only=True)
    fresh=load(FRESH_SOURCE/'run/first.pt');fresh_repeat=load(FRESH_SOURCE/'run/repeat.pt');hybrid=load(FRESH_SOURCE/'run/hybrid.pt')
    assert np.array_equal(fresh['whole'],fresh_repeat['whole'])
    pair=source_record['pairs'].numpy();indices=[]
    for field,key,arm,offset in [('source','hidden','correct',0),('wrong_source','wrong_hidden','wrong_binding',8)]:
        texts=[record[field] for record in data['records']];ids=np.array([tokenizer.encode(t,add_special_tokens=False) for t in texts])
        assert np.array_equal(ids,hybrid['source_ids'][offset:offset+8])
        actual_pairs=np.array([tokenizer.encode(line.split(' = ')[0],add_special_tokens=False) for text in texts for line in text.splitlines()])
        assert np.array_equal(actual_pairs,ids[:,positions_id].reshape(32,2))
        expected=fresh['whole'][offset:offset+8].numpy().copy()
        for i in range(8):
            selected=[]
            for j in range(4):
                ix=np.flatnonzero((pair==actual_pairs[4*i+j]).all(-1));assert len(ix)>0
                values=isolated['first'].numpy()[ix];assert np.array_equal(values,np.repeat(values[:1],len(ix),axis=0))
                expected[i,positions_id[2*j:2*j+2]]=values[0];selected.append(int(ix[0]))
            indices.append(selected)
        assert np.array_equal(expected,source['payload'][key])
        assert np.array_equal(expected,hybrid['original' if offset==0 else 'changed'])
        independent=source_audit.wire(torch.from_numpy(expected),state)[group];assert np.allclose(independent,memory['memories'][arm],rtol=1e-5,atol=1e-5)
        assert source['source_join']['source_texts'][key]==texts and source['source_join']['source_ids'][key]==ids.tolist()
    assert source['source_join']==manifest['source_join'] and source['source_join']['cache_rows']==indices==hybrid['cache_rows'].tolist()
    assert np.array_equal(source['payload']['source_ids'],hybrid['source_ids'])
    lookup={}
    for i,row in enumerate(rows):
        for j,line in enumerate(data['records'][row['group']]['source'].splitlines()):
            name=line.split(' = ')[0];wire=memory['memories']['correct'][i,2*j].numpy()
            if name in lookup:assert np.array_equal(lookup[name],wire)
            else:lookup[name]=wire
    exp=read(ROOT.parent/'latent_binding_current_exposure_corrected_20260910/RESULT.json')
    assert set(lookup)==set(exp['training_unseen_development_ids']) and not set(lookup)&set(exp['positive_ids'])
    assert np.array_equal(memory['memories']['correct'][:,::2],memory['memories']['wrong_binding'][:,::2])
    for name,z in raw['memories'].items():
        assert np.array_equal(z,raws['repeat']['memories'][name]) and np.array_equal(z,memory['memories'][name])
        if name.startswith('random_'):assert np.max(np.abs(np.linalg.norm(z.numpy(),axis=-1)-np.linalg.norm(raw['memories']['correct'].numpy(),axis=-1)))<1e-5
    assert not np.any(raw['memories']['zero'].numpy())
    mech=load(ROOT/'run/mechanics.pt');stock=load(ROOT/'run/stock.pt');oldstock=load(FIT/'train/stock.pt')
    for k,row in stock.items():
        assert np.array_equal(row['first'],row['repeat'])
        assert all(np.array_equal(v,row['first']) for v in mech['controls'][k].values())
    for k,row in load(ROOT/'run/historical_stock.pt').items():
        assert np.array_equal(row['first'],row['repeat']) and np.array_equal(row['first'],oldstock[k]['first'])
    assert np.array_equal(load(ROOT/'run/restored_stock.pt'),stock['6']['first'])
    parity=load(ROOT/'run/legacy_parity.pt');reference=load(NATIVE/'run/prefill_correct.pt')['first']['logits'][:8]
    assert parity['raw'].ndim==3 and parity['raw'].shape[1]==1 and reference.ndim==2
    assert parity['raw'][:,-1].shape==reference.shape
    assert np.array_equal(mech['legacy'],parity['raw']) and np.array_equal(parity['reference'],reference)
    assert np.array_equal(parity['last'],parity['raw'][:,-1]) and np.array_equal(parity['last'],reference)
    assert np.array_equal(mech['active'],mech['repeat']) and not np.array_equal(mech['active'],stock['5']['first'])
    error=float(np.max(np.abs(mech['fast']['lp'].numpy()-mech['direct'].numpy())));assert error<=1e-4 and error==result['direct_score_error']
    assert np.array_equal(mech['fast']['prediction'],mech['direct'].numpy().argmax(-1)) and np.array_equal(mech['fast']['lp'],mech['reordered']['lp'].numpy()[:,::-1])
    for name,key in [('correct','original'),('wrong_binding','changed')]:
        for length,filename in [(5,f'prefill_{name}.pt'),(6,f'six_{name}.pt')]:
            saved=load(ROOT/'run'/filename);first=saved['first'];rep=saved['repeat'];cap=first['capture']
            assert np.array_equal(first['logits'],rep['logits']) and all(np.array_equal(v,rep['capture'][k]) for k,v in cap.items())
            h=cap['hidden'].numpy();center=h.copy();center[:,3:5]=state['reader.query_center'].numpy()
            delta=np.zeros_like(h);delta[:,3:5]=cap['raw_delta'].numpy()[:,3:5]*state['reader.norm_scale'].numpy()
            assert np.array_equal(center,cap['base']) and np.array_equal(delta,cap['used_delta'])
            assert np.array_equal(np.where(cap['active'].numpy(),center+delta,h),cap['output'])
            lp=first['logits'].log_softmax(-1)
            if length==5:
                assert torch.equal(lp[:,raw['candidate_tokens'][:,0]],current[name]['token_lp'][:,:,0])
                previous=load(NATIVE/f'run/prefill_{name}.pt')['first']['capture'];assert np.array_equal(cap['hidden'],previous['hidden'][history_rows])
            else:
                assert np.array_equal(cap['output'][:,5],cap['hidden'][:,5]) and not np.any(cap['used_delta'][:,5].numpy())
                assert saved['targets'].tolist()==targets[key] and saved['ids'][:,:5].tolist()==manifest['ids']
                assert np.array_equal(saved['ids'][:,5],tokens[np.asarray(targets[key]),0])
                for i,t in enumerate(targets[key]):
                    ix=np.flatnonzero(tokens[:,0]==tokens[t,0]);assert torch.equal(lp[i,tokens[ix,1]],current[name]['token_lp'][i,ix,1])
    measured={key:{name:scalar.verify_summary(out,targets[key],tokens,rows,result[key+'_target'][name]) for name,out in current.items()} for key in targets}
    donor_targets=[targets['original'][j] for j in donors]
    shuffled=scalar.verify_summary(current['shuffled'],donor_targets,tokens,rows,result['shuffled_donor_target'])
    paired={key:{name:{m:scalar.interval(measured[key][active][i]-measured[key][name][i],rows) for i,m in enumerate(['accuracy','margin'])} for name in controls}
        for key,active,controls in [('original','correct',['zero','shuffled','wrong_binding','random_71001','random_71002']),('changed','wrong_binding',['zero','correct','random_71001','random_71002'])]}
    assert paired==result['paired_controls']
    counts={k:int(v[0].sum()) for k,v in measured['original'].items()};gate=decision(counts,int(shuffled[0].sum()),int(measured['changed']['wrong_binding'][0].sum()),paired)
    assert gate['native_view_pass']==result['gate']['native_view_pass'] and gate['verdict']==result['verdict']==done['gate']['verdict']
    assert result['gate']==done['gate'] and {**result['gate'],'paired':paired}==read(ROOT/'run/gate.json')
    for key in targets:
        for name,(hit,_) in measured[key].items():assert [int(hit[np.asarray(positions[key])==p].sum()) for p in range(4)]==result['per_record'][key][name]
    assert result['paired_previous_order'] is None
    assert result['text']==read(ROOT/'run/text_status.json')==dict(status='PENDING',new_order_text_measured=False,historical_original_order_not_substituted=True)
    assert result['paired_latent_minus_text'] is None and not result['text_measured']
    assert not result['training'] and not result['full_context_gate_completed'] and not result['novel_binding_established'] and result['dev_evaluated']
    return dict(status='PASS',verdict=gate['verdict'],original_counts=counts,shuffled_donor=int(shuffled[0].sum()),changed_donor=int(measured['changed']['wrong_binding'][0].sum()),
        rows=32,tables=8,conditions=8,model_tokenizer_files=files,scalar_bootstrap_decision_exact=True,numpy_state_equations_exact=True,
        replays_stock_and_source_export_exact=True,analysis_sha256=sha(ROOT/'analysis.json'),checkpoint=freeze['checkpoint'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--save',action='store_true');args=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);result=audit()
    if args.save:
        with (ROOT/'audit.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    else:assert result==read(ROOT/'audit.json')
    print(json.dumps(result))
