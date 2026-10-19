"""Cross-family one-shot confirmation; pinned inherited scoring and statistics."""
import argparse
import inspect
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
from transformers.models.llama import modeling_llama
from transformers.models.qwen3 import modeling_qwen3

import channel_cf as cf

ROOT, c = cf.ROOT, cf.c
analysis = cf.load_pinned('sealed_cf_statistics', cf.V2/'analysis_v2.py',
                         cf.V2_SEAL['files']['analysis_v2.py'])
CORE = analysis.CORE


def read(path):
    return json.loads(Path(path).read_text())


def exclusive(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def versions():
    return dict(torch=torch.__version__, numpy=np.__version__,
                transformers=c.transformers.__version__, python=sys.version)


def components():
    paths = {name: ROOT/name for name in ['channel_cf.py', 'final_cf.py',
        'text_cf.py', 'job.py', 'acquire.py', 'PREREGISTRATION.md',
        'TEXT_BASELINE_PREREGISTRATION.md', 'ENVIRONMENT_RECOVERY.md',
        'verify_environment.py', 'environment_restored.json', 'test_final_cf.py', 'AUDIT_NOTES.md']}
    paths.update(base=cf.BASE, bounded=cf.V2/'channel_v2.py',
        statistics=cf.V2/'analysis_v2.py', text=cf.V2/'text_baseline.py',
        qwen_runtime=Path(inspect.getfile(modeling_qwen3)),
        llama_runtime=Path(inspect.getfile(modeling_llama)))
    return {k:dict(path=str(p), sha256=c.sha(p)) for k,p in paths.items()}


def read_freeze(check_models=False):
    f = read(ROOT/'run_full/FINAL_FREEZE.json')
    assert f['components'] == components(), 'Frozen implementation changed'
    assert f['versions'] == versions()
    assert f['checkpoint_sha256'] == c.sha(ROOT/'run_full/best.pt')
    if check_models:
        assert f['models'] == {role:c.identities(role) for role in ['sender','receiver']}
    for name, hashes in f['development_data'].items():
        for filename, digest in hashes.items():
            assert c.sha(ROOT/'data'/name/filename) == digest
    return f


def development_evaluations():
    rows, _ = c.get_data('dev')
    labels = torch.tensor([r['label'] for r in rows])
    evaluations = []
    for path in (ROOT/'run_full').glob('eval_*/metrics.json'):
        step = int(path.parent.name.split('_')[1])
        m = read(path)
        assert m['output_sha256'] == c.sha(path.parent/'outputs.pt')
        out = torch.load(path.parent/'outputs.pt', weights_only=True)
        assert torch.equal(out['labels'], labels)
        ce = -out['results']['correct']['lp'][torch.arange(len(rows)),labels].mean().item()
        assert ce == m['conditions']['correct']['full_vocab_ce']
        evaluations.append((step,m))
    return evaluations


def reproduce_dev():
    start = time.monotonic(); c.seed_all(53003)
    assert read(ROOT/'train_full_completion.json')['status'] == 'completed'
    dest = ROOT/'dev_reproduction'; assert not dest.exists()
    step = analysis.select_checkpoint(development_evaluations())
    decision = read(ROOT/'run_full/decision.json')
    assert step == decision['best_step']
    checkpoint = ROOT/'run_full/best.pt'
    assert c.sha(checkpoint) == decision['best_checkpoint_sha256']
    dest.mkdir()
    rows, hidden = c.get_data('dev')
    model,tok = c.load('receiver'); initial = c.parameter_hash(model)
    assert initial == decision['receiver_final_parameter_sha256']
    bridge = cf.BridgeModel()
    bridge.load_state_dict(torch.load(checkpoint, weights_only=True)); bridge.eval()
    wrapper = c.ReceiverWrapper(model,bridge)
    parity = cf.stock_parity(wrapper,bridge,tok)
    _,repeats = c.evaluate(wrapper,bridge,tok,rows,hidden,dest/'evaluation')
    original = torch.load(ROOT/f'run_full/eval_{step:04d}/outputs.pt',weights_only=True)['results']
    checks = {cond:all(torch.equal(original[cond][k],repeats[cond][k])
                      for k in ['lp','unconstrained']) for cond in CORE}
    final = c.parameter_hash(model)
    exclusive(dest/'verification.json', dict(all_conditions_exact=all(checks.values()),
        conditions=checks, selected_step=step, checkpoint_sha256=c.sha(checkpoint),
        receiver_initial_hash=initial, receiver_final_hash=final, endpoint_frozen=initial==final,
        parity=parity, components=components(), elapsed_s=time.monotonic()-start))
    assert all(checks.values()) and initial == final


def freeze():
    run = ROOT/'run_full'
    assert not (run/'FINAL_FREEZE.json').exists()
    assert all(not (ROOT/p).exists() for p in ['data/final','run_final','text_final','prepare_final_extension.json'])
    gates = read(ROOT/'run_tiny/semantic_gates.json')
    assert all(gates[k] for k in ['gate0','gate1','gate2'])
    assert read(ROOT/'development_job/state.json')['status'] == 'complete'
    assert read(ROOT/'train_full_completion.json')['status'] == 'completed'
    d,m,e = (read(run/'decision.json'),read(run/'manifest.json'),read(ROOT/'train_full_extension.json'))
    assert e['extension_sha256'] == c.sha(ROOT/'channel_cf.py')
    assert e['base_sha256'] == m['code_sha256'] == c.sha(cf.BASE)
    assert e['preregistration_sha256'] == m['preregistration_sha256'] == c.sha(ROOT/'PREREGISTRATION.md')
    assert d['receiver_final_parameter_sha256'] == m['receiver_initial_parameter_sha256']
    assert d['best_checkpoint_sha256'] == c.sha(run/'best.pt')
    evaluations = development_evaluations(); selected = analysis.select_checkpoint(evaluations)
    assert selected == d['best_step']
    selected_metrics = dict(evaluations)[selected]; conditions = selected_metrics['conditions']
    readiness = dict(correct_ge90=conditions['correct']['accuracy']>=.90,
        each_class_ge85=min(conditions['correct']['per_class'])>=.85,
        baseline_zero_chance=conditions['receiver_only']['accuracy']==conditions['zero']['accuracy']==.25,
        randomized_controls_le35=all(conditions[k]['accuracy']<=.35 for k in ['shuffled','random_71001','random_71002']))
    exclusive(ROOT/'confirmation_job/dev_preflight.json',dict(checks=readiness,selected_step=selected))
    assert all(readiness.values()), 'Selected DEV fails; do not materialize final'
    reproduction = read(ROOT/'dev_reproduction/verification.json')
    assert reproduction['all_conditions_exact'] and reproduction['endpoint_frozen']
    assert reproduction['selected_step'] == selected
    assert reproduction['checkpoint_sha256'] == c.sha(run/'best.pt')
    assert reproduction['components'] == components()
    models = {role:c.identities(role) for role in ['sender','receiver']}
    assert models['receiver'] == m['receiver']
    assert models['sender'] == read(ROOT/'data/train/manifest.json')['model']
    text_dev = read(ROOT/'text_dev/summary.json')
    assert text_dev['code_sha256'] == c.sha(cf.V2/'text_baseline.py')
    assert text_dev['text_preregistration_sha256'] == c.sha(ROOT/'TEXT_BASELINE_PREREGISTRATION.md')
    exclusive(run/'FINAL_FREEZE.json',dict(components=components(),versions=versions(),models=models,
        checkpoint_sha256=c.sha(run/'best.pt'),selected_step=selected,selected_dev_metrics=selected_metrics,
        receiver_parameter_sha256=d['receiver_final_parameter_sha256'],final_seed=44003,groups=256,examples=1024,
        final_previously_materialized=False,decision_rule='PREREGISTRATION.md unchanged',
        development_data={s:{name:c.sha(ROOT/'data'/s/name) for name in ['examples.json','sender.pt','manifest.json']}
                          for s in ['tiny','train','dev']},
        text_dev_summary=text_dev,text_dev_output_sha256=c.sha(ROOT/'text_dev/per_example.json'),
        dev_reproduction_sha256=c.sha(ROOT/'dev_reproduction/verification.json'),
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),external_spend_usd=0))


def score():
    start=time.monotonic(); c.seed_all(53001); f=read_freeze(True)
    dest=ROOT/'run_final'; assert not dest.exists(), 'Final is one-shot; preserve partial attempts'
    rows,hidden=c.get_data('final')
    assert len(rows)==1024 and hidden.shape==(1024,22,2048)
    assert rows==cf.dataset('final')
    assert read(ROOT/'prepare_final_extension.json')['extension_sha256']==c.sha(ROOT/'channel_cf.py')
    assert read(ROOT/'data/final/manifest.json')['model']==f['models']['sender']
    dest.mkdir()
    exclusive(dest/'started.json',dict(final_burned=True,freeze_sha256=c.sha(ROOT/'run_full/FINAL_FREEZE.json'),
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    model,tok=c.load('receiver'); initial=c.parameter_hash(model)
    assert initial==f['receiver_parameter_sha256']
    bridge=cf.BridgeModel(); bridge.load_state_dict(torch.load(ROOT/'run_full/best.pt',weights_only=True)); bridge.eval()
    wrapper=c.ReceiverWrapper(model,bridge); parity=cf.stock_parity(wrapper,bridge,tok)
    _,original=c.evaluate(wrapper,bridge,tok,rows,hidden,dest/'evaluation')
    payload=torch.load(dest/'evaluation/outputs.pt',weights_only=True); repeats={}
    with torch.no_grad():
        for cond in CORE:
            ids=c.tokens(tok,[r['source']+'\n'+c.QUERY if cond=='text' else c.QUERY for r in rows[:32]])
            memory=payload['memories'].get(cond); lps=[]; preds=[]
            for i in range(0,32,8):
                logits=wrapper(ids[i:i+8],None if memory is None else memory[i:i+8])
                lps.append(logits.log_softmax(-1)[:,payload['answer_ids']]); preds.append(logits.argmax(-1))
            repeats[cond]=dict(lp=torch.cat(lps),unconstrained=torch.cat(preds))
            assert all(torch.equal(original[cond][k][:32],repeats[cond][k]) for k in repeats[cond])
    torch.save(repeats,dest/'rerun.pt'); final=c.parameter_hash(model); assert initial==final
    read_freeze()
    exclusive(dest/'execution.json',dict(elapsed_s=time.monotonic()-start,receiver_initial_hash=initial,
        receiver_final_hash=final,parity=parity,rerun_exact=True,rerun_n=32,freeze=f,
        core_output_sha256=c.sha(dest/'evaluation/outputs.pt'),per_example_sha256=c.sha(dest/'evaluation/per_example.json'),
        rerun_sha256=c.sha(dest/'rerun.pt'),final_burned=True,external_spend_usd=0))


def analyze():
    start=time.monotonic(); c.seed_all(53002); f=read_freeze(True); dest=ROOT/'run_final'
    assert not (dest/'analysis.json').exists()
    execution=read(dest/'execution.json'); assert execution['freeze']==f
    for name,key in [('evaluation/outputs.pt','core_output_sha256'),('evaluation/per_example.json','per_example_sha256'),('rerun.pt','rerun_sha256')]:
        assert c.sha(dest/name)==execution[key]
    examples,hidden=c.get_data('final'); assert examples==cf.dataset('final')
    records=read(dest/'evaluation/per_example.json'); assert len(examples)==len(records)==1024
    assert all((a['id'],a['group'],a['label'])==(b['id'],b['group'],b['label']) for a,b in zip(examples,records))
    payload=torch.load(dest/'evaluation/outputs.pt',weights_only=True)
    labels=torch.tensor([r['label'] for r in records]); assert torch.equal(labels,payload['labels'])
    repeats=torch.load(dest/'rerun.pt',weights_only=True)
    for cond in CORE:
        tensor=payload['results'][cond]
        assert torch.equal(tensor['lp'],torch.tensor([r['conditions'][cond]['answer_log_probabilities'] for r in records]))
        assert tensor['lp'].argmax(-1).tolist()==[r['conditions'][cond]['prediction'] for r in records]
        assert tensor['unconstrained'].tolist()==[r['conditions'][cond]['unconstrained_token'] for r in records]
        assert all(torch.equal(tensor[k][:32],repeats[cond][k]) for k in ['lp','unconstrained'])
    bridge=cf.BridgeModel(); bridge.load_state_dict(torch.load(ROOT/'run_full/best.pt',weights_only=True)); bridge.eval()
    memories,donors=c.make_memories(bridge,hidden)
    assert torch.equal(payload['donors'],torch.tensor(donors))
    assert sorted(donors)==list(range(len(records))) and all(i!=d for i,d in enumerate(donors))
    for i,d in enumerate(donors):
        assert records[i]['donor_id']==examples[d]['id'] and records[i]['donor_label']==examples[d]['label']
    donor_exact=torch.equal(payload['results']['shuffled']['lp'],payload['results']['correct']['lp'][donors])
    norms={}
    with torch.no_grad():
        for cond,memory in memories.items():
            assert torch.equal(memory,payload['memories'][cond]); norm=memory.norm(dim=-1)
            expanded=bridge.projector(memory).norm(dim=-1)
            norms[cond]=dict(min=norm.min().item(),max=norm.max().item(),mean=norm.mean().item(),
                per_class_mean=[norm[labels==y].mean().item() for y in range(4)],
                expanded_per_class_mean=[expanded[labels==y].mean().item() for y in range(4)])
        for cond in ['random_71001','random_71002']:
            assert torch.allclose(memories['correct'].norm(dim=-1),memories[cond].norm(dim=-1),atol=1e-5,rtol=0)
    zero_exact=all(torch.equal(payload['results']['zero'][k],payload['results']['receiver_only'][k]) for k in ['lp','unconstrained'])
    td=ROOT/'text_final'; ts=read(td/'summary.json'); text=read(td/'per_example.json')
    sender=read(td/'sender.json'); receiver=read(td/'receiver.json')
    assert ts['code_sha256']==c.sha(cf.V2/'text_baseline.py')
    assert ts['text_preregistration_sha256']==c.sha(ROOT/'TEXT_BASELINE_PREREGISTRATION.md')
    assert ts['data_sha256']==c.sha(ROOT/'data/final/examples.json')
    assert sender['model']==f['models']['sender'] and receiver['model']==f['models']['receiver']
    assert len(sender['outputs'])==len(sender['prompts'])==len(examples)
    assert len(receiver['outputs'])==len(receiver['prompts'])==4
    assert receiver['rerun_exact'] and receiver['frozen_parameter_hash']==f['receiver_parameter_sha256']
    for out in [*receiver['outputs'],receiver['missing_output']]:
        assert out['prediction']==int(np.argmax(out['answer_log_probabilities']))
    by_id={r['id']:r for r in text}; assert len(by_id)==len(text)==len(examples)
    for example,sent in zip(examples,sender['outputs']):
        row=by_id[example['id']]; pred=int(np.argmax(sent['answer_log_probabilities']))
        assert row['sender']==sent and pred==sent['prediction'] and row['transmitted_text']=='ABCD'[pred]
        assert row['text_sender']==receiver['outputs'][pred]
        parsed='ABCD'.index(example['source'].rsplit('Signal: ',1)[1])
        assert row['text_oracle']==receiver['outputs'][parsed] and row['receiver_chat_only']==receiver['missing_output']
    result=analysis.summarize(records,text)
    for cond in ['text_sender','text_oracle','receiver_chat_only']:
        assert result['accuracy'][cond]['mean']==ts['metrics'][cond]['accuracy']
        assert result['accuracy'][cond]['per_class']==ts['metrics'][cond]['per_class']
    mechanics=dict(zero_exact=zero_exact,stock_disabled_parity=execution['parity']['stock_disabled_full_vocab_exact'],
        stock_zero_parity=execution['parity']['stock_zero_full_vocab_exact'],actual_rerun_exact=execution['rerun_exact'],
        endpoint_frozen=execution['receiver_initial_hash']==execution['receiver_final_hash']==f['receiver_parameter_sha256'],
        constant_transport_norm=norms['correct']['min']>=8-1e-5 and norms['correct']['max']<=8+1e-5,
        swapped_probabilities_exactly_follow_donor=donor_exact,
        donor_following_matches_correct=result['shuffled_prediction_matches_donor']==result['accuracy']['correct']['mean'],
        memories_reproduced=True,hashes_verified=True,text_complete=True,final_untouched_before_freeze=f['final_previously_materialized'] is False)
    passed=all(result['statistical_checks'].values()) and all(mechanics.values())
    result.update(mechanical_checks=mechanics,norms=norms,gate4='PASS' if passed else 'NO-GO',
        verdict='A — LATENT COMMUNICATION CHANNEL ESTABLISHED' if passed else 'B — SYNTHETIC CHANNEL WORKS, CROSS-FAMILY NO-GO',
        scope='Cross-family synthetic two-bit channel only; no natural, Arabic or generation claim.',
        analysis_elapsed_s=time.monotonic()-start,external_spend_usd=0,text_summary=ts,
        freeze_sha256=c.sha(ROOT/'run_full/FINAL_FREEZE.json'))
    exclusive(dest/'analysis.json',result)
    c.event('crossfamily_final_decision',gate4=result['gate4'])


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['reproduce-dev','freeze','score','analyze'])
    args=p.parse_args()
    try:
        {'reproduce-dev':reproduce_dev,'freeze':freeze,'score':score,'analyze':analyze}[args.command]()
    except BaseException:
        if args.command=='score' and (ROOT/'run_final').exists():
            c.write_json(ROOT/'run_final/error.json',dict(traceback=traceback.format_exc()))
        raise
