"""Descriptive full-ID scores on novel whole IDs; no novel binding claim."""
import argparse
import json
import numpy as np
import torch
import runtime as rt
import stats
ROOT,c,b=rt.ROOT,rt.c,rt.b
def phase(name):
    root=ROOT/'run'/name;raw=c.load(root/'outputs.pt');canon=c.load(root/'canonical_only.pt')
    assert torch.equal(canon['result']['token_lp'],canon['opposite_wire']['token_lp'])
    assert torch.equal(canon['wire'],raw['memories']['correct'])
    return raw,{**raw['outputs'],'canonical_only':canon['result']}
def ranges(x):
    x=np.asarray(x,dtype=float);assert np.isfinite(x).all()
    return dict(mean=float(x.mean()),min=float(x.min()),max=float(x.max()),per_example=x.tolist())
def geometry(outputs,memories):
    result={}
    for name,out in outputs.items():
        item=dict(score_diagnostics={k:ranges([d[k] for d in out['diagnostics']]) for k in out['diagnostics'][0]} if out['diagnostics'] else {})
        if name in memories:
            z=memories[name].double().numpy();item['wire_l2']=ranges(np.linalg.norm(z.reshape(32,-1),axis=-1));item['slot_l2']=ranges(np.linalg.norm(z,axis=-1))
        if name in ['correct','wrong_binding']:
            cap=c.load(ROOT/f'run/prefill_{name}.pt')['first']['capture'];h=cap['hidden'].double().numpy()[:,3:5];o=cap['output'].double().numpy()[:,3:5]
            hn=np.linalg.norm(h,axis=-1);on=np.linalg.norm(o,axis=-1)
            item['query_sites_3_4']=dict(input_norm=ranges(hn),output_norm=ranges(on),effective_delta_norm=ranges(np.linalg.norm(o-h,axis=-1)),
                learned_delta_norm=ranges(cap['used_delta'][:,3:5].double().norm(dim=-1)),norm_ratio=ranges(on/hn),cosine=ranges((o*h).sum(-1)/(on*hn)))
            item['routes']=cap['hard'][:,3:5].argmax(-1).tolist();item['raw_scores']=cap['scores'][:,3:5].tolist()
        result[name]=item
    return result
def analyze():
    rt.check();job=b.read(ROOT/'job/completion.json');done=b.read(ROOT/'run/status.json')
    assert job['exit_code']==0 and not job['timeout'] and done['completed'] and 'error' not in done
    assert all(done[k] for k in ['replays_exact','candidate_source_exact','bridge_unchanged','hooks_removed','stock_restored_exact'])
    data,payload,source_join=c.load_data();rows=data['rows'];targets=c.n.targets(data);manifest=b.read(ROOT/'run/manifest.json')
    assert data==b.read(ROOT/'run/data.json')
    assert targets==manifest['targets'] and manifest['freeze_sha256']==b.sha(ROOT/'FREEZE.json')
    raw,outputs=phase('evaluation');rep,repeated=phase('repeat');tokens=raw['candidate_tokens'].tolist();donors=raw['donor_rows']
    assert set(outputs)==set(c.n.CONDITIONS) and all(torch.equal(v['token_lp'],repeated[k]['token_lp']) for k,v in outputs.items())
    assert torch.equal(outputs['shuffled']['token_lp'],outputs['correct']['token_lp'][donors])
    assert torch.equal(outputs['zero']['token_lp'],outputs['receiver_only']['token_lp'])
    assert source_join==manifest['source_join']
    bridge=c.bridge();memories,again=c.n.r.memories(bridge,data,payload);assert donors==again==manifest['donors']
    assert all(torch.equal(v,raw['memories'][k]) and torch.equal(v,rep['memories'][k]) for k,v in memories.items())
    summaries={key:{name:stats.summary(out,targets[key],rows,tokens) for name,out in outputs.items()} for key in targets}
    shuffled_targets=[targets['original'][j] for j in donors];shuffled=stats.summary(outputs['shuffled'],shuffled_targets,rows,tokens)
    paired=stats.paired(outputs,targets,rows);gate=c.gate(c.evaluation.load_metrics(ROOT/'run/evaluation'),paired)
    assert gate==done['gate'] and {**gate,'paired':paired}==b.read(ROOT/'run/gate.json')
    positions={key:[next(i for i,line in enumerate(data['records'][row['group']]['source' if key=='original' else 'wrong_source'].splitlines()) if line.split(' = ')[0]==b.IDS[t]) for row,t in zip(rows,targets[key],strict=True)] for key in targets}
    per_record={key:{name:[sum(s['predictions'][i]==s['target_ids'][i] for i,p in enumerate(positions[key]) if p==pos) for pos in range(4)] for name,s in sums.items()} for key,sums in summaries.items()}
    changes=None # Different examples: no paired comparison to CAL.
    text=b.read(ROOT/'run/text_status.json');assert text==dict(status='PENDING',new_order_text_measured=False,historical_original_order_not_substituted=True)
    text_comparison=None
    mechanics=c.load(ROOT/'run/mechanics.pt')
    return dict(status='PASS',verdict=gate['verdict'],gate=gate,original_target=summaries['original'],changed_target=summaries['changed'],
        shuffled_donor_target=shuffled,paired_controls=paired,paired_previous_order=changes,per_record=per_record,
        geometry=geometry(outputs,memories),text=text,paired_latent_minus_text=text_comparison,
        direct_score_error=float((mechanics['fast']['lp']-mechanics['direct']).abs().max()),checkpoint=c.CHECKPOINT,
        freeze_sha256=b.sha(ROOT/'FREEZE.json'),job=job,completion=done,
        timing={p:dict(conditions=b.read(ROOT/'run'/p/'metrics.json')['times'],canonical=b.read(ROOT/'run'/p/'canonical_only.json')['elapsed_s']) for p in ['evaluation','repeat']},
        training=False,view=0,order=[0,1,2,3],data_seed=191003,novel_whole_id_evaluated=True,text_measured=False,full_context_gate_completed=False,novel_binding_established=False,dev_evaluated=True,new_external_spend_usd=0)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--save',action='store_true');args=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    result=analyze()
    if args.save:
        assert not (ROOT/'analysis.json').exists();b.write(ROOT/'analysis.json',result)
    else:assert result==b.read(ROOT/'analysis.json')
    print(json.dumps(dict(verdict=result['verdict'],original={k:v['correct'] for k,v in result['original_target'].items()},
        changed={k:v['correct'] for k,v in result['changed_target'].items()},shuffled_donor=result['shuffled_donor_target']['correct'])))
