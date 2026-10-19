"""Saved-array independent audit; no endpoint or optimization."""
import argparse
import numpy as np
import torch
import common as h
from common import ROOT,read,load,write,b
import runtime


def scalar_counts(out,targets,tokens):
    parts=out['token_lp'].double().numpy();prediction=[];seq=[];groups={}
    for i,t in enumerate(tokens):groups.setdefault(int(t[0]),[]).append(i)
    for row in parts:
        totals=[float(x[0])+float(x[1]) for x in row];prediction.append(max(range(128),key=lambda j:totals[j]))
        g=max(groups.values(),key=lambda ix:float(row[ix[0],0]));seq.append(max(g,key=lambda j:float(row[j,1])))
    assert prediction==out['prediction'].tolist()
    return dict(joint=sum(a==v for a,v in zip(prediction,targets)),sequential=sum(a==v for a,v in zip(seq,targets)))


def independent_gate(outputs,info,tokens,gate):
    target=info['targets'];orig=target['original'];changed=target['changed'];donor=[orig[j] for j in info['donors']]
    count={k:scalar_counts(v,orig,tokens)['joint'] for k,v in outputs.items()}
    passed=(count['correct']>=31 and scalar_counts(outputs['shuffled'],donor,tokens)['joint']>=31 and scalar_counts(outputs['wrong_binding'],changed,tokens)['joint']>=28
        and count['receiver_only']<=4 and count['canonical_only']<=4 and count['zero']==count['receiver_only'] and all(count[k]<=8 for k in ['shuffled','wrong_binding','random_71001','random_71002']))
    assert passed==gate['count_pass'];lower=[]
    draws=np.random.default_rng(95002).integers(0,8,(10000,8))
    for group,active,controls in [('original','correct',['zero','shuffled','wrong_binding','random_71001','random_71002']),('changed','wrong_binding',['zero','correct','random_71001','random_71002'])]:
        def values(arm):
            score=outputs[arm]['lp'].tolist();hits=[];margins=[]
            for row,t in zip(score,target[group]):
                hits.append(float(max(range(128),key=lambda j:row[j])==t));margins.append(row[t]-max(row[j] for j in range(128) if j!=t))
            return hits,margins
        a=values(active)
        for arm in controls:
            z=values(arm)
            for j,key in enumerate(['accuracy','margin']):
                delta=[u-v for u,v in zip(a[j],z[j])];means=[sum(delta[k] for k,row in enumerate(info['rows']) if row['group']==g)/4 for g in range(8)]
                interval=np.quantile(np.asarray(means)[draws].mean(1),[.025,.975]);np.testing.assert_allclose(interval,gate['paired'][group][arm][key]['ci95'],rtol=0,atol=1e-12);lower.append(interval[0]>0)
    assert bool(all(lower))==gate['paired_pass'];assert bool(passed and all(lower))==gate['pass_gate']


def analyze():
    h.check();status=read(ROOT/'run/status.json');job=read(ROOT/'job/completion.json')
    assert status['completed'] and job['exit_code']==0 and not job['timeout'] and job['child_reaped']
    assert status['baseline_exact'] and status['reference_exact'] and status['restored_exact'] and status['no_hooks'] and status['no_receiver_gradients'] and status['before']==status['after']
    assert status['start']['start_step']==h.STEPS and status['start']['start_weight_relative']==0.0 and status['no_new_data']
    x=read(ROOT/'inputs.json');parent=read(h.ALIGN/'inputs.json')
    assert x['training_ids']==parent['training_ids'] and x['trainrows']==parent['trainrows'] and x['query_ids']==parent['query_ids'] and x['banks']==parent['banks']
    assert x['continuation']==dict(start_step=h.STEPS,extra_steps=h.EXTRA,total_steps=h.TOTAL,parent_seal=h.PARENT_SEAL)
    assert len(x['trainrows'])==224 and len(parent['training_ids'])==56
    coord=load(h.ALIGN/'coordinates.pt');saved=load(ROOT/'start.pt')
    assert torch.equal(saved['coordinates']['input_basis'],coord['input_basis']) and torch.equal(saved['coordinates']['output_basis'],coord['output_basis'])
    state=load(h.ALIGN/'run/candidate.pt');candidate=load(ROOT/'run/candidate.pt');fit=load(ROOT/'run/fit.pt')
    h.fixed(candidate,state);assert torch.equal(h.weight(fit['theta'],saved['coordinates']),candidate[h.NAME])
    assert len(fit['history'])==h.EXTRA and [r['step_before_update'] for r in fit['history']]==list(range(h.STEPS,h.TOTAL))
    assert all(np.isfinite(r['loss']) and np.isfinite(r['gradient_norm']) for r in fit['history'])
    assert all(int(v['step'])==h.TOTAL for v in fit['optimizer']['state'].values())
    ref=coord['base'].double().numpy()+coord['output_basis'].numpy()@fit['theta'].numpy()@coord['input_basis'].numpy().T
    actual=candidate[h.NAME].double().numpy();err=float(np.linalg.norm(ref-actual)/np.linalg.norm(actual));assert err<=1e-6
    m=load(h.SOURCE/'run/features.pt')['projected'].double().numpy();assert abs(np.square(m@coord['input_basis'].numpy()).sum(1).mean()-1)<=1e-10
    for name in ['start_parity','perturbed_parity','terminal_parity']:
        z=load(ROOT/'run'/f'{name}.pt');error=float((z['native']-z['training']).abs().max())
        assert error==z['error'] and error<=1e-4 and torch.equal(z['native'].argmax(-1),z['training'].argmax(-1))
    pre=read(ROOT/'run/preflight.json');assert pre['pass_gate'] and pre['relative_error']<=.1 and pre['finite_difference']>0
    result={}
    for label,pools in [('baseline',['familiar','novel']),('reference',['familiar','novel']),
                        ('trained',['familiar','novel'] if status['novel_evaluated'] else ['familiar'])]:
        result[label]={}
        for pool in pools:
            dest=ROOT/'run'/f'{label}_{pool}';info=x['banks'][pool];outputs={}
            for arm in ['receiver_only','correct','shuffled','wrong_binding','random_71001','random_71002','zero','canonical_only']:
                z=load(dest/f'{arm}.pt');out=z['first']
                if 'repeat' in z:runtime.equal(z['first'],z['repeat'])
                assert torch.equal(out['token_lp'].sum(-1),out['lp']) and torch.equal(out['lp'].argmax(-1),out['prediction'])
                assert scalar_counts(out,info['targets']['original'],x['candidate_tokens'])==h.counts(out,info['targets']['original'],torch.tensor(x['candidate_tokens']))
                outputs[arm]=out
            if label=='reference':
                for arm in outputs:runtime.equal(outputs[arm],load(h.reference_scores(pool)/f'{arm}.pt')['first'])
            gate=runtime.decision(outputs,info,torch.tensor(x['candidate_tokens']))
            if label!='reference':
                assert gate==read(dest/'gate.json');independent_gate(outputs,info,x['candidate_tokens'],gate)
            result[label][pool]=gate
            if label in ['baseline','trained']:
                for arm in ['correct','wrong_binding']:
                    z=load(dest/f'prefill_{arm}.pt');assert h.c.n.helpers.capture_equal(z['first'],z['repeat'])
                    prior=load(h.native(pool)/f'run/prefill_{arm}.pt')['first']['capture'];new=z['first']['capture']
                    assert torch.equal(prior['hard'],new['hard']) and torch.equal(prior['scores'],new['scores'])
                d=load(dest/'direct.pt');assert d['error']<=1e-4 and torch.equal(d['fast']['prediction'],d['direct'].argmax(-1))
            if label=='baseline':
                old=load(h.native(pool)/'run/evaluation/outputs.pt')['outputs'];old['canonical_only']=load(h.native(pool)/'run/evaluation/canonical_only.pt')['result']
                for arm,out in outputs.items():runtime.equal(old[arm],out)
            if label=='trained':
                for arm in ['receiver_only','zero','canonical_only']:runtime.equal(outputs[arm],load(ROOT/'run'/f'baseline_{pool}'/f'{arm}.pt')['first'])
    assert status['novel_evaluated']==result['trained']['familiar']['pass_gate']
    assert status['familiar']==result['trained']['familiar']
    if status['novel_evaluated']:assert status['novel']==result['trained']['novel']
    else:assert not (ROOT/'run/trained_novel').exists()
    stock=load(ROOT/'run/stock.pt');assert torch.equal(stock['stock'],stock['disabled']) and torch.equal(stock['stock'],stock['zero'])
    assert torch.equal(load(ROOT/'run/restored.pt'),stock['stock'])
    return dict(independent_audit='PASS',verdict=status['verdict'],start_step=h.STEPS,total_steps=h.TOTAL,training_ids=56,novel_training_targets=False,
        numpy_weight_relative_error=err,candidate_sha256=h.sha(ROOT/'run/candidate.pt'),results=result,job=job)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--save',action='store_true');args=parser.parse_args();result=analyze()
    if args.save:write(ROOT/'ANALYSIS.json',result)
    else:assert result==read(ROOT/'ANALYSIS.json')
    print(result['verdict'],result['independent_audit'])
