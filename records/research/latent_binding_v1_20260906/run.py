"""Sequential research stages for the fixed binding gate; no new model engine."""
import argparse
import copy
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch
from transformers import AutoTokenizer

import binding as b

ROOT=b.ROOT
SENDER_INSTRUCTION='Read the bindings and answer the question. Reply with exactly one ID and nothing else.\n\n'
RECEIVER_INSTRUCTION='Copy the answer ID. Reply with exactly the ID and nothing else.\n\n'


def check_code():
    acquisition=b.read(ROOT/'acquisition.json')
    assert acquisition['components']==b.components(), 'Frozen initial implementation changed'
    return acquisition


def model_hash(role,model):
    actual=b.c.parameter_hash(model)
    if role=='receiver': expected=b.read(b.PRIOR/'run_full/FINAL_FREEZE.json')['receiver_parameter_sha256']
    else: expected=b.read(b.PRIOR/'sender_reproduction/verification.json')['sender_final_parameter_hash']
    assert actual==expected, 'Endpoint parameters changed or wrong model'
    return actual


def preflight():
    assert not (ROOT/'acquisition.json').exists()
    result=subprocess.run([sys.executable,str(ROOT.parent/'latent_communication_dossier_20260906/verify.py'),'--models'],
                           capture_output=True,text=True,check=True)
    verification=json.loads(result.stdout); assert verification['status']=='PASS'
    parent=b.read(b.PRIOR/'run_full/FINAL_FREEZE.json')
    identities={role:b.c.identities(role) for role in ['sender','receiver']}
    assert identities==parent['models']
    shapes={}
    for role in identities:
        tok=AutoTokenizer.from_pretrained(b.c.model_path(role),local_files_only=True)
        bank=b.AnswerBank(tok)
        q=bank.prompts([f'Which ID is {color}?' for color in b.COLORS])
        shapes[role]=dict(query_tokens=q.tolist(),answers=bank.tokens.tolist())
    snapshot=ROOT/'initial_code'; snapshot.mkdir()
    for name in b.components():
        with (snapshot/name).open('xb') as stream: stream.write((ROOT/name).read_bytes())
    b.write(ROOT/'identity_pools.json',dict(seed=120501,**b.pools()))
    b.write(ROOT/'acquisition.json',dict(components=b.components(),models=identities,
        inherited_verification=verification,prior_seal_sha256=b.sha(b.PRIOR/'seal.json'),
        parameter_count=569857,architecture_class='exact imported parent BridgeModel',
        tokenization=shapes,versions=parent['versions'],platform=platform.platform(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],text=True),
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),external_spend_usd=0))
    b.event('preflight_pass')


def prepare(split):
    check_code(); start=time.monotonic(); b.c.seed_all(11000)
    data=b.dataset(split)
    dest=ROOT/'data'/split; assert not dest.exists(); dest.mkdir(parents=True)
    b.write(dest/'examples.json',data)
    model,tok=b.c.load('sender'); before=model_hash('sender',model)
    sources=[r['source'] for r in data['records']]+[r['wrong_source'] for r in data['records']]
    ids=b.c.tokens(tok,sources); assert ids.shape==(2*len(data['records']),19)
    captured=[]
    def capture(module,args,output): captured.append(output.detach().clone())
    handle=model.model.layers[13].register_forward_hook(capture)
    try:
        with torch.no_grad():
            for i in range(0,len(ids),4):
                model.model(input_ids=ids[i:i+4],use_cache=False)
                b.event('capture',split=split,completed=min(i+4,len(ids)),total=len(ids))
    finally: handle.remove()
    hidden=torch.cat(captured); n=len(data['records'])
    assert hidden.shape==(2*n,19,2048) and torch.isfinite(hidden).all()
    torch.save(dict(hidden=hidden[:n],wrong_hidden=hidden[n:],source_ids=ids),dest/'sender.pt')
    after=model_hash('sender',model); assert before==after
    b.write(dest/'manifest.json',dict(model=b.c.identities('sender'),examples_sha256=b.sha(dest/'examples.json'),
        capture_sha256=b.sha(dest/'sender.pt'),components=b.components(),layer_zero_based=13,
        sender_parameter_hash=after,source_shape=list(hidden.shape),source_order='original tables then counterfactual tables',
        elapsed_s=time.monotonic()-start,external_spend_usd=0))


def render(tok,body):
    return tok.apply_chat_template([dict(role='user',content=body)],tokenize=False,
        add_generation_prompt=True,enable_thinking=False,date_string='06 Sep 2026')


def native_model_forward(model):
    return lambda ids,memory=None,keep=1:b.native_forward(model,ids,memory,keep)


def serializable(result):
    return {k:v.tolist() if isinstance(v,torch.Tensor) else v for k,v in result.items()}


def validate_native_scoring(model,ids,bank):
    forward=native_model_forward(model)
    fast=b.score(forward,ids[:1],bank)
    direct=b.direct_score(forward,ids[:1],bank)
    error=(fast['lp']-direct).abs().max().item()
    assert error<=1e-4 and torch.equal(fast['prediction'],direct.argmax(-1))
    rerun=b.score(forward,ids[:1],bank)
    assert torch.equal(fast['token_lp'],rerun['token_lp'])
    b.event('native_scoring_parity',max_difference=error,tolerance=1e-4)
    return dict(max_difference=error,tolerance=1e-4,prediction_exact=True,rerun_exact=True)


def text_sender(split):
    check_code(); b.c.seed_all(53003); start=time.monotonic()
    data,_=b.load_data(split); dest=ROOT/f'text_{split}'; dest.mkdir(exist_ok=True)
    assert not (dest/'sender.json').exists()
    model,tok=b.c.load('sender'); before=model_hash('sender',model); bank=b.AnswerBank(tok,leading=False)
    prompts=[render(tok,SENDER_INSTRUCTION+data['records'][r['group']]['source']+'\n\n'+r['query']) for r in data['rows']]
    ids=bank.prompts(prompts); parity=validate_native_scoring(model,ids,bank)
    results=[]; inference=time.monotonic()
    for i in range(0,len(ids),4):
        result=b.score(native_model_forward(model),ids[i:i+4],bank,batch=4)
        results.append(result['token_lp']); b.event('text_sender',split=split,completed=min(i+4,len(ids)),total=len(ids))
    inference=time.monotonic()-inference; parts=torch.cat(results); scores=parts.sum(-1)
    outputs=[dict(id=r['id'],answer=r['answer'],prediction=b.IDS[int(scores[i].argmax())],
        token_log_probabilities=parts[i].tolist(),answer_log_probabilities=scores[i].tolist()) for i,r in enumerate(data['rows'])]
    after=model_hash('sender',model); assert before==after
    b.write(dest/'sender.json',dict(outputs=outputs,prompts=prompts,input_ids=ids.tolist(),
        candidates=b.IDS,candidate_tokens=bank.tokens.tolist(),parameter_hash=after,
        model=b.c.identities('sender'),inference_s=inference,elapsed_s=time.monotonic()-start,
        components=b.components(),scoring_parity=parity,examples_sha256=b.sha(ROOT/'data'/split/'examples.json')))


def text_receiver(split):
    check_code(); b.c.seed_all(53003); start=time.monotonic()
    data,_=b.load_data(split); dest=ROOT/f'text_{split}'; sender=b.read(dest/'sender.json')
    assert not (dest/'receiver.json').exists()
    assert [r['id'] for r in sender['outputs']]==[r['id'] for r in data['rows']]
    identities=sorted({r['answer'] for r in data['rows']}|{r['prediction'] for r in sender['outputs']})
    model,tok=b.c.load('receiver'); before=model_hash('receiver',model); bank=b.AnswerBank(tok,leading=False)
    prompts=[render(tok,RECEIVER_INSTRUCTION+'Answer ID: '+identity) for identity in identities]
    ids=bank.prompts(prompts); parity=validate_native_scoring(model,ids,bank)
    t=time.monotonic(); result=b.score(native_model_forward(model),ids,bank)
    inference_s=time.monotonic()-t
    repeat=b.score(native_model_forward(model),ids,bank)
    assert torch.equal(result['token_lp'],repeat['token_lp'])
    by_id={identity:dict(prediction=b.IDS[int(result['prediction'][i])],
        token_log_probabilities=result['token_lp'][i].tolist(),answer_log_probabilities=result['lp'][i].tolist()) for i,identity in enumerate(identities)}
    records=[]
    for row,sent in zip(data['rows'],sender['outputs']):
        records.append(dict(id=row['id'],group=row['group'],color=row['color'],answer=row['answer'],
            sender_prediction=sent['prediction'],actual_text=by_id[sent['prediction']],oracle_text=by_id[row['answer']]))
    oracle=sum(r['oracle_text']['prediction']==r['answer'] for r in records)/len(records)
    actual=sum(r['actual_text']['prediction']==r['answer'] for r in records)/len(records)
    per_color={color:sum(r['oracle_text']['prediction']==r['answer'] for r in records if r['color']==color)/sum(r['color']==color for r in records) for color in b.COLORS}
    after=model_hash('receiver',model); assert before==after
    b.write(dest/'receiver.json',dict(identities=identities,prompts=prompts,input_ids=ids.tolist(),
        result=serializable(result),records=records,candidates=b.IDS,candidate_tokens=bank.tokens.tolist(),
        parameter_hash=after,model=b.c.identities('receiver'),rerun_exact=True,inference_s=inference_s,scoring_parity=parity))
    b.write(dest/'summary.json',dict(sender_accuracy=sum(r['sender_prediction']==r['answer'] for r in records)/len(records),
        actual_text_accuracy=actual,oracle_text_accuracy=oracle,oracle_per_color=per_color,
        oracle_gate=oracle>=.90 and min(per_color.values())>=.75,unique_receiver_messages=len(identities),
        elapsed_s=time.monotonic()-start,components=b.components(),external_spend_usd=0))
    b.event('text_summary',split=split,oracle=oracle,actual=actual)


def memories(bridge,data,payload):
    groups=torch.tensor([r['group'] for r in data['rows']]); n=len(data['records'])
    with torch.no_grad():
        z=torch.cat([bridge.resampler(x) for x in payload['hidden'].split(32)])
        wrong=torch.cat([bridge.resampler(x) for x in payload['wrong_hidden'].split(32)])
    donors=b.c.derangement(n)
    values={'correct':z[groups],'shuffled':z[torch.tensor(donors)[groups]],
            'wrong_binding':wrong[groups],'zero':torch.zeros_like(z)[groups]}
    for seed in [71001,71002]:
        random_z=torch.randn(z.shape,generator=torch.Generator().manual_seed(seed))
        random_z=random_z/random_z.norm(dim=-1,keepdim=True)*z.norm(dim=-1,keepdim=True)
        values[f'random_{seed}']=random_z[groups]
        assert torch.allclose(random_z.norm(dim=-1),z.norm(dim=-1),atol=1e-5,rtol=0)
    donor_rows={(r['group'],r['color']):i for i,r in enumerate(data['rows'])}
    indices=[donor_rows[(int(donors[r['group']]),r['color'])] for r in data['rows']]
    return values,indices


def evaluate(wrapper,bridge,tok,data,payload,dest,baseline=None):
    dest.mkdir(); start=time.monotonic(); bank=b.AnswerBank(tok); ids=bank.prompts([r['query'] for r in data['rows']])
    labels=torch.tensor([b.IDS.index(r['answer']) for r in data['rows']])
    wrong_labels=torch.tensor([b.IDS.index(r['wrong_answer']) for r in data['rows']])
    values,donors=memories(bridge,data,payload); outputs={}; metrics={}; times={}
    bridge.eval()
    for cond in b.CONDITIONS:
        t=time.monotonic()
        if cond=='receiver_only' and baseline is not None:
            result=baseline; cached=True
        else:
            result=b.score(wrapper.forward,ids,bank,values.get(cond)); cached=False
        outputs[cond]=result; preds=result['prediction']
        metrics[cond]=dict(accuracy=(preds==labels).float().mean().item(),
            per_color={color:(preds[torch.tensor([r['color']==color for r in data['rows']])]==labels[torch.tensor([r['color']==color for r in data['rows']])]).float().mean().item() for color in b.COLORS},
            mean_id_nll=-result['lp'][torch.arange(len(labels)),labels].mean().item(),
            donor_following=(preds==labels[donors]).float().mean().item() if cond=='shuffled' else (preds==wrong_labels).float().mean().item() if cond=='wrong_binding' else None)
        times[cond]=dict(elapsed_s=time.monotonic()-t,cached=cached)
        b.event('condition',directory=dest.name,condition=cond,**metrics[cond],elapsed_s=times[cond]['elapsed_s'])
    assert torch.equal(outputs['zero']['token_lp'],outputs['receiver_only']['token_lp'])
    assert torch.equal(outputs['shuffled']['token_lp'],outputs['correct']['token_lp'][donors])
    torch.save(dict(outputs=outputs,memories=values,donor_rows=donors,labels=labels,wrong_labels=wrong_labels,
                    candidate_tokens=bank.tokens),dest/'outputs.pt')
    records=[]
    for i,row in enumerate(data['rows']):
        records.append(dict(**row,donor_id=data['rows'][donors[i]]['id'],donor_answer=data['rows'][donors[i]]['answer'],
            conditions={cond:dict(prediction=b.IDS[int(v['prediction'][i])],
                token_log_probabilities=v['token_lp'][i].tolist(),answer_log_probabilities=v['lp'][i].tolist(),
                diagnostics=v['diagnostics'][i] if v['diagnostics'] else {}) for cond,v in outputs.items()}))
    b.write(dest/'per_example.json',records)
    b.write(dest/'metrics.json',dict(conditions=metrics,times=times,gate=bridge.reader.gate.item(),
        output_sha256=b.sha(dest/'outputs.pt'),elapsed_s=time.monotonic()-start,candidates=b.IDS))
    return metrics,outputs


def mechanics(wrapper,bridge,tok,data,payload,dest):
    bank=b.AnswerBank(tok); rows=data['rows'][:8]; ids=bank.prompts([r['query'] for r in rows]); groups=[r['group'] for r in rows]
    labels=torch.tensor([b.IDS.index(r['answer']) for r in rows]); seq=torch.cat([ids,bank.tokens[labels,:1]],dim=1)
    zero=torch.zeros(8,8,64)
    wrapper.hook.remove()
    with torch.no_grad(): stock=wrapper.model(input_ids=seq,use_cache=False,logits_to_keep=2).logits
    wrapper.hook=wrapper.model.model.layers[15].register_forward_hook(wrapper._hook)
    with torch.no_grad():
        disabled=wrapper.forward(seq,None,2); zeroed=wrapper.forward(seq,zero,2)
        assert torch.equal(stock,disabled) and torch.equal(stock,zeroed)
        memory=bridge.resampler(payload['hidden'][groups]); active=wrapper.forward(seq,memory,2)
        assert not torch.equal(stock,active) and torch.equal(active,wrapper.forward(seq,memory,2))
        fast=b.score(wrapper.forward,ids[:1],bank,memory[:1])
        direct=b.direct_score(wrapper.forward,ids[:1],bank,memory[:1])
        maximum=(fast['lp']-direct).abs().max().item()
        assert maximum<=1e-4 and torch.equal(fast['prediction'],direct.argmax(-1)), 'Full-ID scoring seam failed'
        rev=copy.copy(bank); rev.tokens=bank.tokens.flip(0); rev.strings=bank.strings[::-1]
        rev.indices={p:torch.where(rev.tokens[:,0]==p)[0] for p in rev.prefixes}
        reordered=b.score(wrapper.forward,ids[:1],rev,memory[:1])
        assert torch.equal(fast['lp'],reordered['lp'].flip(1))
    bridge.zero_grad(set_to_none=True)
    loss=b.train_logits(wrapper,bank,ids,labels,bridge.resampler(payload['hidden'][groups])); loss.backward()
    gradients={name:p.grad.norm().item() if p.grad is not None else 0 for name,p in bridge.named_parameters()}
    assert all(np.isfinite(v) and v>0 for v in gradients.values())
    assert all(p.grad is None and not p.requires_grad for p in wrapper.model.parameters())
    bridge.zero_grad(set_to_none=True)
    b.write(dest/'mechanics.json',dict(status='PASS',stock_disabled_zero_full_logits_exact=True,
        active_changes_logits=True,active_repeat_exact=True,direct_scoring_max_difference=maximum,
        direct_scoring_tolerance=1e-4,candidate_reordering_exact=True,gradients=gradients,
        endpoint_gradients=False,full_id_nll=loss.item()))


def train_tiny():
    check_code(); assert b.read(ROOT/'text_tiny/summary.json')['oracle_gate'], 'Oracle text readiness failed'
    b.c.seed_all(51001); start=time.monotonic(); dest=ROOT/'run_tiny'; assert not dest.exists(); dest.mkdir()
    data,payload=b.load_data('tiny'); model,tok=b.c.load('receiver'); initial=model_hash('receiver',model)
    bridge=b.BridgeModel(); wrapper=b.Receiver(model,bridge)
    b.write(dest/'manifest.json',dict(components=b.components(),initial_parameter_hash=initial,
        model=b.c.identities('receiver'),optimizer=dict(name='AdamW',lr=.003,weight_decay=0,clip_norm=1),
        seed=51001,batch=8,max_steps=300,interval=20,trainable_parameters=sum(p.numel() for p in bridge.parameters())))
    mechanics(wrapper,bridge,tok,data,payload,dest)
    metrics,first=evaluate(wrapper,bridge,tok,data,payload,dest/'eval_0000')
    bank=b.AnswerBank(tok); ids=bank.prompts([r['query'] for r in data['rows']])
    labels=torch.tensor([b.IDS.index(r['answer']) for r in data['rows']]); groups=torch.tensor([r['group'] for r in data['rows']])
    optimizer=torch.optim.AdamW(bridge.parameters(),lr=.003,weight_decay=0)
    generator=torch.Generator().manual_seed(51002); order=torch.randperm(len(ids),generator=generator); cursor=0
    best_ce=float('inf'); best_step=None; passed=False
    with (dest/'curve.jsonl').open('x') as curve:
        for step in range(1,301):
            if cursor+8>len(ids): order=torch.randperm(len(ids),generator=generator); cursor=0
            ix=order[cursor:cursor+8]; cursor+=8; bridge.train(); optimizer.zero_grad(set_to_none=True); t=time.monotonic()
            z=bridge.resampler(payload['hidden'][groups[ix]])
            loss=b.train_logits(wrapper,bank,ids[ix],labels[ix],z); assert torch.isfinite(loss); loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(bridge.parameters(),1); assert torch.isfinite(grad)
            optimizer.step(); assert all(p.grad is None for p in model.parameters())
            row=dict(step=step,full_id_nll=loss.item(),gradient_norm=float(grad),gate=bridge.reader.gate.item(),step_s=time.monotonic()-t)
            curve.write(json.dumps(row)+'\n'); curve.flush(); b.event('train',**row)
            if step%20: continue
            metrics,outputs=evaluate(wrapper,bridge,tok,data,payload,dest/f'eval_{step:04d}',first['receiver_only'])
            ce=metrics['correct']['mean_id_nll']
            if ce<best_ce:
                best_ce=ce; best_step=step; torch.save(bridge.state_dict(),dest/'best.pt')
            passed=(metrics['correct']['accuracy']>=.95 and metrics['shuffled']['donor_following']>=.95 and
                all(metrics[c]['accuracy']<=.25 for c in ['shuffled','random_71001','random_71002']) and
                metrics['receiver_only']['accuracy']<=.125 and metrics['zero']['accuracy']==metrics['receiver_only']['accuracy'])
            if passed: break
    torch.save(bridge.state_dict(),dest/'last.pt'); final=model_hash('receiver',model); assert initial==final
    # Always reproduce terminal outputs, including an unfavorable terminal outcome.
    _,repeat=evaluate(wrapper,bridge,tok,data,payload,dest/'terminal_repeat')
    original=torch.load(dest/f'eval_{step:04d}/outputs.pt',weights_only=True)['outputs']
    exact=all(torch.equal(original[c]['token_lp'],repeat[c]['token_lp']) for c in b.CONDITIONS)
    assert exact
    b.write(dest/'decision.json',dict(verdict='TINY MECHANICS PASS — NOT HELD-OUT BINDING EVIDENCE' if passed else 'MECHANICS/TRAINING NO-GO',
        step=step,best_step=best_step,best_id_nll=best_ce,terminal_metrics=metrics,
        receiver_parameter_hash=final,terminal_rerun_exact=exact,best_sha256=b.sha(dest/'best.pt'),
        last_sha256=b.sha(dest/'last.pt'),elapsed_s=time.monotonic()-start,external_spend_usd=0))
    b.write(dest/'gates.json',dict(pass_mechanics=True,pass_tiny_accuracy=metrics['correct']['accuracy']>=.95,
        pass_shuffle=passed,**{'pass':passed},heldout_binding_established=False))
    b.event('tiny_decision',passed=passed,step=step)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['preflight','prepare','text-sender','text-receiver','train-tiny'])
    parser.add_argument('--split',choices=b.SPLITS,default='tiny'); args=parser.parse_args()
    {'preflight':preflight,'prepare':lambda:prepare(args.split),'text-sender':lambda:text_sender(args.split),
     'text-receiver':lambda:text_receiver(args.split),'train-tiny':train_tiny}[args.stage]()
