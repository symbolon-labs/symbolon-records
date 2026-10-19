"""Audited description of a failed mechanical gate, never held-out evidence."""
import json
import statistics
from pathlib import Path

import torch
import channel_cf as cf

ROOT,c=cf.ROOT,cf.c


def read(path): return json.loads(Path(path).read_text())


def main():
    decision=read(ROOT/'run_tiny/decision.json'); assert not decision['gate1']
    audit=read(ROOT/'tiny_terminal_audit/verification.json')
    assert audit['all_conditions_exact'] and audit['endpoint_frozen'] and audit['donor_probability_equivariance']
    analysis=cf.load_pinned('sealed_tiny_statistics',cf.V2/'analysis_v2.py',cf.V2_SEAL['files']['analysis_v2.py'])
    dest=ROOT/f"run_tiny/eval_{decision['step']:04d}"
    rows,hidden=c.get_data('tiny'); records=read(dest/'per_example.json')
    text=read(ROOT/'text_tiny/per_example.json'); ts=read(ROOT/'text_tiny/summary.json')
    payload=torch.load(dest/'outputs.pt',weights_only=True)
    assert c.sha(dest/'outputs.pt')==read(dest/'metrics.json')['output_sha256']
    assert all((r['id'],r['group'],r['label'])==(x['id'],x['group'],x['label']) for r,x in zip(rows,records))
    for cond in c.CONDITIONS:
        assert torch.equal(payload['results'][cond]['lp'],torch.tensor([r['conditions'][cond]['answer_log_probabilities'] for r in records]))
    result=analysis.summarize(records,text)
    result['heldout_style_checks_DIAGNOSTIC_ONLY']=result.pop('statistical_checks')
    for cond in ['text_sender','text_oracle','receiver_chat_only']:
        assert result['accuracy'][cond]['mean']==ts['metrics'][cond]['accuracy']
    bridge=cf.BridgeModel(); bridge.load_state_dict(torch.load(ROOT/'run_tiny/last.pt',weights_only=True)); bridge.eval()
    expected,donors=c.make_memories(bridge,hidden)
    assert torch.equal(torch.tensor(donors),payload['donors'])
    norms={}; diagnostics={}
    for cond,memory in expected.items():
        assert torch.equal(memory,payload['memories'][cond])
        norm=memory.norm(dim=-1)
        norms[cond]=dict(min=norm.min().item(),max=norm.max().item(),mean=norm.mean().item(),
            per_class=[norm[payload['labels']==y].mean().item() for y in range(4)])
        d=[r['conditions'][cond]['diagnostics'] for r in records]
        diagnostics[cond]={k:dict(mean=statistics.mean(x[k] for x in d),
            min=min(x[k] for x in d),max=max(x[k] for x in d)) for k in d[0]}
    for cond in ['random_71001','random_71002']:
        assert torch.allclose(expected['correct'].norm(dim=-1),expected[cond].norm(dim=-1),atol=1e-5,rtol=0)
    assert torch.equal(payload['results']['shuffled']['lp'],payload['results']['correct']['lp'][donors])
    curve=[json.loads(line) for line in (ROOT/'run_tiny/curve.jsonl').read_text().splitlines()]
    evaluations=[dict(step=int(p.parent.name.split('_')[1]),**read(p)) for p in sorted((ROOT/'run_tiny').glob('eval_*/metrics.json'))]
    manifest=read(ROOT/'run_tiny/manifest.json'); job=read(ROOT/'tiny_job/state.json')
    result.update(verdict='F — MECHANICS / TRAINING NO-GO',scope='Single fixed cross-family configuration; 32-example training-set mechanical gate ONLY.',
        gate0='PASS',gate1='NO-GO',gate2='NOT ESTABLISHED: correct overfit failed',
        no_heldout_result=True,no_final_created=not (ROOT/'data/final').exists(),
        frozen_receiver_hash_verified=audit['receiver_initial_hash']==manifest['receiver_initial_parameter_sha256'],
        norms=norms,reader_diagnostics=diagnostics,terminal_audit=audit,
        best_observed_accuracy=max(e['conditions']['correct']['accuracy'] for e in evaluations),
        minimum_ce_checkpoint_step=decision['best_step'],evaluations=evaluations,
        trainable_parameters=manifest['trainable_parameters'],receiver_parameters=manifest['receiver_parameters'],
        parameter_fraction=manifest['parameter_fraction'],
        gradients=dict(median=statistics.median(x['gradient_norm'] for x in curve),max=max(x['gradient_norm'] for x in curve),clip_norm=1),
        gate_range=[min(x['gate'] for x in curve),max(x['gate'] for x in curve)],
        compute=dict(tiny_job_s=job['elapsed_s'],terminal_audit_s=audit['elapsed_s'],text_s=ts['elapsed_s'],
            summed_experiment_stage_s=job['elapsed_s']+audit['elapsed_s']+ts['elapsed_s'],
            peak_job_child_rss_kib=max(s['peak_child_rss_kib'] for s in job['stages']),
            peak_rss_caveat='Driver reports cumulative child peak, not isolated receiver peak.',
            median_training_step_s=statistics.median(x['step_s'] for x in curve),external_spend_usd=0),
        text_summary=ts,provenance=cf.provenance(),code_sha256=c.sha(__file__),
        uncertainty_caveat='Paired nonce-group intervals describe eight training groups; they are not generalization evidence.')
    with (ROOT/'TINY_ANALYSIS.json').open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False); f.write('\n')
    print(result['verdict'])


if __name__=='__main__': main()
