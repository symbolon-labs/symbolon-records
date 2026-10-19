"""Continue the sealed 200-step alignment to 600 steps; identical schedule, no new data."""
import argparse,resource,subprocess,sys,time
import torch
import common as h
from common import ROOT,b,c,read,write,load,sha
from seam import Boundary
import stats


def freeze():
    assert not (ROOT/'FREEZE.json').exists();b.c.seed_all(h.SEED)
    tests=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(ROOT),'-p','test_convergence.py','-v'],capture_output=True,text=True)
    assert tests.returncode==0,tests.stderr
    parent_inputs=read(h.ALIGN/'inputs.json')
    write(ROOT/'inputs.json',dict(**parent_inputs,continuation=dict(start_step=h.STEPS,extra_steps=h.EXTRA,total_steps=h.TOTAL,parent_seal=h.PARENT_SEAL)))
    fit=load(h.ALIGN/'run/fit.pt')
    saved=dict(theta=fit['theta'].clone(),optimizer=fit['optimizer'],coordinates=load(h.ALIGN/'coordinates.pt'),candidate=load(h.ALIGN/'run/candidate.pt'))
    torch.save(saved,ROOT/'start.pt')
    status=h.verify_start(saved)
    write(ROOT/'FREEZE.json',dict(identity=h.identity(),cap_s=h.CAP,seed=h.SEED,start_step=h.STEPS,extra_steps=h.EXTRA,total_steps=h.TOTAL,
        lr=h.LR,batch=8,rows=224,training_only=True,no_new_data=True,start=status,tests=dict(exit_code=tests.returncode,stdout=tests.stdout,stderr=tests.stderr)))
    assert status['start_weight_relative']==0.0
    h.check();print('FROZEN',sha(ROOT/'FREEZE.json'))


def equal(a,z):
    for k in ['token_lp','lp','prediction']:assert torch.equal(a[k],z[k]),k


def reference(model,bridge,bank,x,pool):
    """Replay the sealed 822db91d candidate scores; one call per arm."""
    info=x['banks'][pool];ids=torch.tensor(info['ids']);mem=load(h.native(pool)/'run/memories.pt')['memories']
    dest=ROOT/'run'/f'reference_{pool}';dest.mkdir();wrapper=b.Receiver(model,bridge);out={}
    try:
        for arm in ['receiver_only','correct','shuffled','wrong_binding','random_71001','random_71002','zero','canonical_only']:
            payload=None if arm=='receiver_only' else mem['correct' if arm=='canonical_only' else arm]
            bridge.reader.canonical_only=arm=='canonical_only'
            value=b.score(wrapper.forward,ids,bank,payload);out[arm]=value
            sealed=load(h.reference_scores(pool)/f'{arm}.pt')['first'];equal(value,sealed)
            torch.save(dict(first=value),dest/f'{arm}.pt');b.event('reference_replay_pass',pool=pool,arm=arm)
        bridge.reader.canonical_only=False
    finally:
        bridge.reader.canonical_only=False;wrapper.hook.remove()
    return out


def decision(outputs,info,tokens):
    targets=info['targets'];original=targets['original'];changed=targets['changed'];donor=[original[j] for j in info['donors']]
    counts={k:h.counts(v,original,tokens) for k,v in outputs.items()};follow=h.counts(outputs['shuffled'],donor,tokens);changedcount=h.counts(outputs['wrong_binding'],changed,tokens)
    countpass=(counts['correct']['joint']>=31 and follow['joint']>=31 and changedcount['joint']>=28 and counts['receiver_only']['joint']<=4 and counts['canonical_only']['joint']<=4 and counts['zero']['joint']==counts['receiver_only']['joint'] and all(counts[k]['joint']<=8 for k in ['shuffled','wrong_binding','random_71001','random_71002']))
    paired=stats.paired(outputs,targets,info['rows'])
    pairedpass=all(v[k]['ci95'][0]>0 for group in paired.values() for v in group.values() for k in ['accuracy','margin'])
    return dict(pass_gate=bool(countpass and pairedpass),count_pass=bool(countpass),paired_pass=bool(pairedpass),counts=counts,donor_following=follow,changed=changedcount,paired=paired)


def phase(model,bridge,bank,inputs,pool,label):
    info=inputs['banks'][pool];ids=torch.tensor(info['ids']);nr=h.native(pool);mem=load(nr/'run/memories.pt')['memories'];dest=ROOT/'run'/f'{label}_{pool}';dest.mkdir();out={};wrapper=b.Receiver(model,bridge)
    try:
        for arm in ['receiver_only','correct','shuffled','wrong_binding','random_71001','random_71002','zero','canonical_only']:
            payload=None if arm=='receiver_only' else mem['correct' if arm=='canonical_only' else arm]
            bridge.reader.canonical_only=arm=='canonical_only'
            first=b.score(wrapper.forward,ids,bank,payload);repeat=b.score(wrapper.forward,ids,bank,payload);equal(first,repeat)
            torch.save(dict(first=first,repeat=repeat),dest/f'{arm}.pt');out[arm]=first;b.event('native_arm_repeat',pool=pool,phase=label,arm=arm)
        bridge.reader.canonical_only=False
        fast=b.score(wrapper.forward,ids[:1],bank,mem['correct'][:1]);direct=b.direct_score(wrapper.forward,ids[:1],bank,mem['correct'][:1])
        error=float((fast['lp']-direct).abs().max());assert error<=1e-4 and torch.equal(fast['prediction'],direct.argmax(-1))
        torch.save(dict(fast=fast,direct=direct,error=error),dest/'direct.pt')
        for arm in ['correct','wrong_binding']:
            first=c.n.helpers.native_prefill(wrapper,ids,mem[arm]);repeat=c.n.helpers.native_prefill(wrapper,ids,mem[arm]);assert c.n.helpers.capture_equal(first,repeat)
            prior=load(nr/f'run/prefill_{arm}.pt')['first']['capture']
            assert torch.equal(first['capture']['hard'],prior['hard']) and torch.equal(first['capture']['scores'],prior['scores'])
            torch.save(dict(first=first,repeat=repeat),dest/f'prefill_{arm}.pt')
        if label=='baseline':
            old=load(nr/'run/evaluation/outputs.pt')['outputs'];old['canonical_only']=load(nr/'run/evaluation/canonical_only.pt')['result']
            for arm in out:equal(out[arm],old[arm])
        elif label=='reference':
            for arm in out:equal(out[arm],load(h.reference_scores(pool)/f'{arm}.pt')['first'])
        else:
            for arm in ['receiver_only','zero','canonical_only']:equal(out[arm],load(ROOT/'run'/f'baseline_{pool}'/f'{arm}.pt')['first'])
        gate=decision(out,info,bank.tokens);write(dest/'gate.json',gate);return gate
    finally:
        bridge.reader.canonical_only=False;wrapper.hook.remove()


def teacher_parity(model,bridge,bank,x,state,coord,theta,name):
    info=x['banks']['familiar'];ids=torch.tensor(info['ids'])[:8];labels=torch.tensor(info['targets']['original'])[:8];tails=bank.tokens[labels];seq=torch.cat([ids,tails[:,:1]],1)
    features=load(h.SOURCE/'run/features.pt')['projected'];ix=[x['training_ids'].index(b.IDS[int(t)]) for t in labels];matrix=h.weight(theta,coord)
    new={k:v.clone() for k,v in state.items()};new[h.NAME]=matrix.detach().clone();h.fixed(new,state);bridge.load_state_dict(new)
    native=b.Receiver(model,bridge)
    try:
        with torch.no_grad():a=native.forward(seq,load(h.native('familiar')/'run/memories.pt')['memories']['correct'][:8],2).clone()
    finally:native.hook.remove()
    seam=Boundary(model)
    try:
        with torch.no_grad():z=seam.forward(seq,h.render(features[ix],matrix,state),2).clone()
    finally:seam.close()
    error=float((a-z).abs().max());assert error<=1e-4 and torch.equal(a.argmax(-1),z.argmax(-1))
    bridge.load_state_dict(state);torch.save(dict(native=a,training=z,error=error,weight=matrix.detach()),ROOT/'run'/f'{name}.pt');return error


def preflight(model,bridge,bank,x,state,coord,saved):
    theta=saved['theta']
    errors=[teacher_parity(model,bridge,bank,x,state,coord,theta,'start_parity')]
    gen=torch.Generator().manual_seed(h.SEED);direction=torch.randn(theta.shape,generator=gen,dtype=torch.float64);direction/=direction.norm()
    errors.append(teacher_parity(model,bridge,bank,x,state,coord,theta+.01*direction,'perturbed_parity'))
    ids=torch.tensor(x['query_ids'])[:8];labels=torch.tensor([r['label'] for r in x['trainrows'][:8]]);t=bank.tokens[labels];seq=torch.cat([ids,t[:,:1]],1)
    features=load(h.SOURCE/'run/features.pt')['projected'];m=features[[r['feature'] for r in x['trainrows'][:8]]];seam=Boundary(model);parameter=theta.clone().requires_grad_()
    try:
        def loss(q):
            logits=seam.forward(seq,h.render(m,h.weight(q,coord),state),2);return torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),t.reshape(-1))
        value=loss(parameter);grad=torch.autograd.grad(value,parameter)[0];assert torch.isfinite(grad).all() and grad.norm()>0;d=grad/grad.norm()
        with torch.no_grad():plus=float(loss(theta+.01*d));minus=float(loss(theta-.01*d))
        analytic=float((grad*d).sum());finite=(plus-minus)/.02;relative=abs(finite-analytic)/abs(analytic);assert finite>0 and relative<=.1
        result=dict(parity_errors=errors,analytic=analytic,finite_difference=finite,relative_error=relative,epsilon=.01,loss=float(value.detach()),pass_gate=True)
        write(ROOT/'run/preflight.json',result);return result
    finally:seam.close();bridge.load_state_dict(state)


def fit(model,bank,x,state,coord,saved):
    features=load(h.SOURCE/'run/features.pt')['projected'];indices=[r['feature'] for r in x['trainrows']]
    assert [x['training_ids'][i] for i in indices]==[r['id'] for r in x['trainrows']]
    memory=features[indices];ids=torch.tensor(x['query_ids']);labels=torch.tensor([r['label'] for r in x['trainrows']]);targets=bank.tokens[labels]
    theta=torch.nn.Parameter(saved['theta'].clone());opt=torch.optim.Adam([theta],lr=h.LR);opt.load_state_dict(saved['optimizer'])
    assert int(opt.state[theta]['step'])==h.STEPS
    w=Boundary(model);history=[]
    try:
        for step in range(h.STEPS,h.TOTAL):
            opt.zero_grad(set_to_none=True);total=0.;correct=torch.zeros(2,dtype=torch.long)
            for start in range(0,224,8):
                q=ids[start:start+8];t=targets[start:start+8];seq=torch.cat([q,t[:,:1]],1);matrix=h.weight(theta,coord)
                logits=w.forward(seq,h.render(memory[start:start+8],matrix,state),2);loss=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),t.reshape(-1))/28
                loss.backward();total+=float(loss.detach());correct+=(logits.detach().argmax(-1)==t).sum(0)
            assert theta.grad is not None and torch.isfinite(theta.grad).all() and torch.isfinite(theta).all()
            assert all(p.grad is None and not p.requires_grad for p in model.parameters())
            history.append(dict(step_before_update=step,loss=total,gradient_norm=float(theta.grad.norm()),teacher_forced=correct.tolist()));opt.step()
            if (step+1)%5==0:b.event('convergence_fit',step=step+1,loss=total,teacher_forced=correct.tolist())
        matrix=h.weight(theta.detach(),coord);candidate={k:v.clone() for k,v in state.items()};candidate[h.NAME]=matrix.clone();h.fixed(candidate,state)
        torch.save(candidate,ROOT/'run/candidate.pt');torch.save(dict(theta=theta.detach(),optimizer=opt.state_dict(),history=history),ROOT/'run/fit.pt');return candidate
    finally:w.close()


def body(status):
    h.check();b.c.seed_all(h.SEED);x=read(ROOT/'inputs.json');saved=h.start();coord=saved['coordinates'];state=saved['candidate']
    status['start']=h.verify_start(saved);assert status['start']['start_weight_relative']==0.0
    bridge=c.bridge()
    for p in bridge.parameters():p.requires_grad_(False)
    model,tok=b.c.load('receiver');before=b.c.parameter_hash(model);assert before=='6a3e1782ebad58d1e9323d5eeaff191d112848b5f0058394a998d1ac54c492b3';status['before']=before
    bank=b.AnswerBank(tok);assert bank.tokens.tolist()==x['candidate_tokens']
    seq=torch.cat([torch.tensor(x['banks']['familiar']['ids'])[:8],bank.tokens[:8,:1]],1)
    with torch.no_grad():stock=b.native_forward(model,seq).clone()
    native=b.Receiver(model,bridge)
    try:
        with torch.no_grad():disabled=native.forward(seq,None).clone();zero=native.forward(seq,torch.zeros(8,8,64)).clone()
        assert torch.equal(stock,disabled) and torch.equal(stock,zero);torch.save(dict(stock=stock,disabled=disabled,zero=zero),ROOT/'run/stock.pt')
    finally:native.hook.remove()
    try:
        for pool in ['familiar','novel']:phase(model,bridge,bank,x,pool,'baseline')
        status['baseline_exact']=True
        # Correction from the mechanics-invalid predecessor: the sealed 822db91d
        # candidate replay requires the bridge to hold the sealed candidate state.
        bridge.load_state_dict(state)
        assert all(torch.equal(v,state[k]) for k,v in bridge.state_dict().items())
        for pool in ['familiar','novel']:reference(model,bridge,bank,x,pool)
        status['reference_exact']=True
        status['preflight']=preflight(model,bridge,bank,x,state,coord,saved);b.event('preflight_pass',**status['preflight'])
        candidate=fit(model,bank,x,state,coord,saved);status['fit_completed']=True
        assert all(torch.equal(v,candidate[k]) for k,v in load(ROOT/'run/candidate.pt').items());h.fixed(candidate,state)
        theta=load(ROOT/'run/fit.pt')['theta'];status['terminal_parity']=teacher_parity(model,bridge,bank,x,state,coord,theta,'terminal_parity');bridge.load_state_dict(candidate)
        status['familiar']=phase(model,bridge,bank,x,'familiar','trained');h.fixed(bridge.state_dict(),state)
        assert all(torch.equal(v,candidate[k]) for k,v in bridge.state_dict().items())
        if status['familiar']['pass_gate']:
            status['novel_evaluated']=True;status['novel']=phase(model,bridge,bank,x,'novel','trained')
            status['verdict']='NATIVE NOVEL GATE PASS' if status['novel']['pass_gate'] else 'CONVERGENCE NOVEL NO-GO'
        else:status['verdict']='CONVERGENCE FAMILIAR NO-GO; NOVEL NOT OPENED'
    finally:
        bridge.load_state_dict(state);assert all(torch.equal(v,state[k]) for k,v in bridge.state_dict().items());assert all(not m._forward_hooks for m in model.modules())
        with torch.no_grad():restored=b.native_forward(model,seq).clone()
        assert torch.equal(stock,restored);torch.save(restored,ROOT/'run/restored.pt')
        after=b.c.parameter_hash(model);assert after==before and all(p.grad is None and not p.requires_grad for p in model.parameters())
        status.update(after=after,restored_exact=True,no_hooks=True,no_receiver_gradients=True)
    h.check()


def run():
    (ROOT/'run').mkdir();start=time.monotonic();status=dict(completed=False,fit_completed=False,novel_evaluated=False,no_new_data=True)
    try:body(status);status['completed']=True
    except BaseException as exc:status.update(error_type=type(exc).__name__,error=str(exc));raise
    finally:write(ROOT/'run/status.json',{**status,'elapsed_s':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['freeze','check','run']);args=parser.parse_args()
    {'freeze':freeze,'check':h.check,'run':run}[args.mode]()
