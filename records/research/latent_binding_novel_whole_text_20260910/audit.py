"""Independent text input/scalar-score/actual-message audit; no inference."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'latent_binding_novel_whole_native_20260910'
PREVIOUS=ROOT.parent/'latent_binding_order1_text_extended_20260908'
IDS=[a+str(n) for a in 'ABCDEFGHIJKLMNOP' for n in range(8)]
HASHES={'sender':'deb5e69d2fcd24d139ce2927f8375c2c445867b196bc179df4688d98990e0ec4','receiver':'6a3e1782ebad58d1e9323d5eeaff191d112848b5f0058394a998d1ac54c492b3'}
def read(p):return json.loads(p.read_text())
def load(p):return torch.load(p,weights_only=True)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def inputs():
    assert sha(PREVIOUS/'seal.json')=='9a6365b8cb5a99be7748bc3043e41228675e0177d5bf2e99c00863b38f9265ee'
    prior=read(PREVIOUS/'seal.json')['files']
    assert set(prior)=={str(p.relative_to(PREVIOUS)) for p in PREVIOUS.rglob('*') if p.is_file() and p!=PREVIOUS/'seal.json'}
    for name,h in prior.items():assert sha(PREVIOUS/name)==h
    for name,h in read(ROOT/'ANALYSIS_DEPENDENCIES.json')['files'].items():assert sha(Path(name))==h
    f=read(ROOT/'FREEZE.json');x=read(ROOT/'inputs.json');identity=f['identity'];files=0
    assert f['seed']==68108 and f['caps']=={'sender':None,'receiver':None} and sha(ROOT/'inputs.json')==identity['data']
    for n,h in identity['code'].items():assert sha(ROOT/n)==h
    for kind in ['inputs','imports']:
        for n,h in identity[kind].items():assert sha(Path(n))==h
    assert sha(PARENT/'seal.json')==read(ROOT/'NATIVE_PARENT.json')['seal_sha256']
    for n,h in read(PARENT/'seal.json')['files'].items():assert sha(PARENT/n)==h
    for model in identity['models'].values():
        for n,v in model['files'].items():
            p=Path(model['path'])/n;assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256'];files+=1
    assert files==22 and identity['models']==read(PARENT/'FREEZE.json')['identity']['models']
    data=read(PARENT/'run/data.json');rows=[]
    for condition,field,label in [('original','source','answer'),('changed','wrong_source','wrong_answer')]:
        for i,row in enumerate(data['rows']):
            source=data['records'][row['group']][field];mapping={color:name for name,color in [s.split(' = ') for s in source.splitlines()]}
            assert mapping[row['color']]==row[label]
            rows.append(dict(row=i,condition=condition,query=row['query'],answer=row[label],source=source,example_id=row['id'],group=row['group'],color=row['color']))
    assert x['rows']==rows and x['view']==0 and x['order']==[0,1,2,3] and x['data_seed']==191003 and x['native_parent_sha256']==sha(PARENT/'analysis.json')
    tok={r:AutoTokenizer.from_pretrained(m['path'],local_files_only=True) for r,m in identity['models'].items()}
    def render(t,body):return t.apply_chat_template([dict(role='user',content=body)],tokenize=False,add_generation_prompt=True,enable_thinking=False,date_string='06 Sep 2026')
    sender_prompts=[render(tok['sender'],'Read the bindings and answer the question. Reply with exactly one ID and nothing else.\n\n'+row['source']+'\n\n'+row['query']) for row in rows]
    receiver_prompts=[render(tok['receiver'],'Copy the answer ID. Reply with exactly the ID and nothing else.\n\nAnswer ID: '+message) for message in IDS]
    assert sender_prompts==x['sender_prompts'] and receiver_prompts==x['receiver_prompts'] and x['receiver_messages']==IDS
    for role,prompts,length in [('sender',sender_prompts,54),('receiver',receiver_prompts,52)]:
        ids=[tok[role].encode(p,add_special_tokens=False) for p in prompts];tails=[tok[role].encode(a,add_special_tokens=False) for a in IDS]
        assert ids==x[role+'_ids'] and {len(v) for v in ids}=={length}
        assert tails==x[role+'_candidate_tokens'] and {len(v) for v in tails}=={2}
        for prompt,prefix in zip(prompts,ids):
            for answer,tail in zip(IDS,tails):assert tok[role].encode(prompt+answer,add_special_tokens=False)==prefix+tail
    return dict(status='PASS',freeze_sha256=sha(ROOT/'FREEZE.json'),inputs_sha256=sha(ROOT/'inputs.json'),model_tokenizer_files=files,sender_requests=64,possible_receiver_messages=128,view=0)
def scores(out):
    z=out['token_lp'].numpy();assert z.ndim==3 and z.shape[1:]==(128,2) and np.isfinite(z).all()
    lp=z.sum(-1,dtype=np.float32);pred=lp.argmax(-1)
    assert np.array_equal(lp,out['lp']) and np.array_equal(pred,out['prediction']);return lp,pred
def role_audit(role):
    result=inputs();x=read(ROOT/'inputs.json');dest=ROOT/role;s=read(dest/'status.json');j=read(ROOT/f'job_{role}/completion.json');launch=read(ROOT/f'job_{role}/launch.json')
    assert s['completed'] and j['exit_code']==0 and not j['timeout'] and launch['cap_s']==read(ROOT/'FREEZE.json')['caps'][role]
    assert not any(Path(f'/proc/{launch[k]}').exists() for k in ['pid','wrapper_pid'])
    assert s['endpoint_before']==s['endpoint_after']==HASHES[role] and s['endpoint_unchanged'] and s['no_hooks'] and s['repeat_exact']
    parity=load(dest/'parity.pt');fast,pred=scores(parity['fast']);scores(parity['repeat'])
    error=float(np.max(np.abs(fast-parity['direct'].numpy())));assert error<=1e-4 and error==s['parity']['max_difference']
    assert np.array_equal(pred,parity['direct'].numpy().argmax(-1)) and np.array_equal(parity['fast']['token_lp'],parity['repeat']['token_lp'])
    assert sha(dest/'outputs.pt')==s['outputs_sha256'];saved=load(dest/'outputs.pt')
    assert saved['candidate_tokens'].tolist()==x[role+'_candidate_tokens']
    if role=='sender':
        lp,pred=scores(saved);assert saved['ids'].tolist()==x['sender_ids']
        chunks=[load(dest/f'chunk_{i:04d}.pt') for i in range(0,64,4)]
        for chunk in chunks:scores(chunk)
        assert np.array_equal(np.concatenate([v['token_lp'].numpy() for v in chunks]),saved['token_lp'])
        scores(saved['repeat_first4']);assert np.array_equal(saved['token_lp'][:4],saved['repeat_first4']['token_lp'])
        counts={condition:sum(IDS[int(pred[i])]==row['answer'] for i,row in enumerate(x['rows']) if row['condition']==condition) for condition in ['original','changed']}
        result.update(sender_counts=counts,actual_text_complete=False)
    else:
        sender=load(ROOT/'sender/outputs.pt');scores(sender);lock=read(ROOT/'RECEIVER_LOCK.json')
        assert lock['sender_sha256']==sha(ROOT/'sender/outputs.pt') and lock['freeze_sha256']==sha(ROOT/'FREEZE.json')
        messages=sorted({row['answer'] for row in x['rows']}|{IDS[int(i)] for i in sender['prediction']});indices=[IDS.index(m) for m in messages]
        assert lock['messages']==messages and lock['indices']==indices and lock['ids']==[x['receiver_ids'][i] for i in indices]
        assert lock['prompts']==[x['receiver_prompts'][i] for i in indices] and saved['ids'].tolist()==lock['ids']
        lp,pred=scores(saved['output']);scores(saved['repeat']);assert np.array_equal(saved['output']['token_lp'],saved['repeat']['token_lp'])
        records=read(dest/'rows.json');assert records['messages']==messages and records['prompts']==lock['prompts']
        for row,expected,send in zip(records['rows'],x['rows'],sender['prediction'].tolist(),strict=True):
            assert all(row[k]==v for k,v in expected.items());message=IDS[send]
            assert row['sender_prediction']==message and row['actual_prediction']==IDS[int(pred[messages.index(message)])] and row['oracle_prediction']==IDS[int(pred[messages.index(expected['answer'])])]
        counts={condition:{name:sum(row[name]==row['answer'] for row in records['rows'] if row['condition']==condition) for name in ['sender_prediction','actual_prediction','oracle_prediction']} for condition in ['original','changed']}
        for m in messages:assert records['text_bytes'][m]==len(m.encode('utf8')) and records['text_token_counts'][m]==2
        result.update(counts=counts,unique_receiver_messages=len(messages),actual_text_complete=True)
    return dict(**result,role=role,outputs_sha256=sha(dest/'outputs.pt'),native_parity_error=error,elapsed_s=j['elapsed_s'],peak_rss_kib=s['peak_rss_kib'])
def summary():
    a=role_audit('sender');z=role_audit('receiver');assert a==read(ROOT/'sender_audit.json') and z==read(ROOT/'receiver_audit.json')
    records=read(ROOT/'receiver/rows.json')['rows'];native=read(PARENT/'analysis.json');data=read(PARENT/'run/data.json')
    spec=importlib.util.spec_from_file_location('independent_existing_table_bootstrap',PARENT/'scalar.py');scalar=importlib.util.module_from_spec(spec);spec.loader.exec_module(scalar)
    sender_lp,_=scores(load(ROOT/'sender/outputs.pt'));receiver_saved=load(ROOT/'receiver/outputs.pt');receiver_lp,_=scores(receiver_saved['output'])
    messages=read(ROOT/'RECEIVER_LOCK.json')['messages']
    measurements=[]
    for i,row in enumerate(records):
        target=IDS.index(row['answer']);metrics={}
        for name,lp in [('sender',sender_lp[i]),('actual_text',receiver_lp[messages.index(row['sender_prediction'])]),('oracle_text',receiver_lp[messages.index(row['answer'])])]:
            correct=float(lp[target]);other=float(np.max(np.delete(lp,target)))
            metrics[name]=dict(target_log_probability=correct,nll=-correct,margin=correct-other,correct=IDS[int(lp.argmax())]==row['answer'])
        measurements.append(dict(**row,metrics=metrics))
    paired={};per_color={};per_id={};per_table={};errors=[];latent_counts={};score_summary={}
    for condition,key,arm in [('original','original_target','correct'),('changed','changed_target','wrong_binding')]:
        chosen=[r for r in records if r['condition']==condition];text_hit=np.array([r['actual_prediction']==r['answer'] for r in chosen],dtype=float)
        n=native[key][arm];latent=np.array([p==t for p,t in zip(n['predictions'],n['target_ids'])],dtype=float)
        assert [r['answer'] for r in chosen]==n['target_ids']
        latent_counts[condition]=int(latent.sum())
        paired[condition]=scalar.interval(latent-text_hit,data['rows'])
        per_color[condition]={color:sum(r['actual_prediction']==r['answer'] for r in chosen if r['color']==color) for color in ['red','green','blue','yellow']}
        for dest,field in [(per_id,'answer'),(per_table,'group')]:
            dest[condition]={str(value):dict(n=sum(r[field]==value for r in chosen),correct=sum(r[field]==value and r['actual_prediction']==r['answer'] for r in chosen)) for value in sorted({r[field] for r in chosen})}
        measured=[r for r in measurements if r['condition']==condition]
        score_summary[condition]={arm:{metric:dict(mean=float(np.mean([r['metrics'][arm][metric] for r in measured])),median=float(np.median([r['metrics'][arm][metric] for r in measured]))) for metric in ['target_log_probability','nll','margin']} for arm in ['sender','actual_text','oracle_text']}
        errors.extend([r for r in chosen if r['actual_prediction']!=r['answer'] or r['sender_prediction']!=r['answer']])
    return dict(status='PASS',verdict=('SMALL TRAINING-UNSEEN WHOLE-ID COMBINATION VALIDATION PASS; FINAL UNPROVEN' if native['gate']['native_view_pass'] else 'NOVEL WHOLE-ID BINDING NO-GO; TEXT COMPARISON COMPLETE'),counts=z['counts'],paired_latent_minus_actual_text=paired,
        per_color_actual=per_color,per_id_actual=per_id,per_table_actual=per_table,score_summary=score_summary,per_example=measurements,errors=errors,latent_counts=latent_counts,sender_audit=a,receiver_audit=z,
        native_analysis_sha256=sha(PARENT/'analysis.json'),text_sender_has_query=True,latent_sender_has_query=False,novel_binding_established=bool(native['gate']['native_view_pass']),novelty_scope='Small training-unseen positive whole-ID combinations, with older evaluation history and new contexts',final_binding_established=False,new_external_spend_usd=0)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['inputs','sender','receiver','summary']);p.add_argument('--save',action='store_true');q=p.parse_args();torch.set_num_threads(4)
    value=inputs() if q.stage=='inputs' else summary() if q.stage=='summary' else role_audit(q.stage)
    name={'inputs':'input_audit.json','summary':'summary.json'}.get(q.stage,q.stage+'_audit.json')
    if q.save:
        with (ROOT/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    else:assert value==read(ROOT/name)
    print(json.dumps(value))
