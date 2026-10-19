"""One-shot novel whole-ID native gate; no optimization or later-view evaluation."""
import argparse
import copy
from pathlib import Path
import resource
import subprocess
import sys
import time
import types
import torch
import core as c
import provenance
import stats
ROOT,b=c.ROOT,c.b
CODE=['PREREGISTRATION.md','core.py','runtime.py','analysis.py','audit.py','test_core.py','job.py','provenance.py','report.py','close.py','stats.py','scalar.py','ANCHORS.json']
LADDER=ROOT.parent/'latent_binding_order_ladder_20260910'
def anchors():
    value=b.read(ROOT/'ANCHORS.json')
    return {Path(p):h for p,h in value.items()}

def imports():
    result={};pending=[c];seen=set()
    while pending:
        m=pending.pop()
        if id(m) in seen:continue
        seen.add(id(m));p=getattr(m,'__file__',None)
        if not p:continue
        p=Path(p).resolve()
        if not p.is_relative_to(ROOT.parent):continue
        if not p.is_relative_to(ROOT):result[str(p)]=b.sha(p)
        pending.extend(v for v in vars(m).values() if isinstance(v,types.ModuleType))
    return result
def identity():
    paths=[c.CANDIDATE,c.SOURCE/'inputs.json',c.SOURCE/'run/hybrid.pt',c.SOURCE/'run/first.pt',c.SOURCE/'run/repeat.pt',
        c.ISO/'run/isolated.pt',c.ISO/'run/bridge.pt',c.PARENT/'run/memories.pt',c.PARENT/'run/evaluation/outputs.pt',c.PARENT/'run/prefill_correct.pt',
        c.FIT/'train/inputs.pt',c.SOURCE/'data/examples.json',c.DATA/'data/cal/examples.json',c.DATA/'inputs.json',LADDER/'ARCHITECTURE_LOCK.json']
    return dict(code={n:b.sha(ROOT/n) for n in CODE},imports=imports(),inputs={str(p):b.sha(p) for p in paths},
        models={r:b.c.identities(r) for r in ['sender','receiver']},qualified_span=c.u.span_runtime.identity())
def check():
    for root,digest in anchors().items():
        assert b.sha(root/'seal.json')==digest;files=b.read(root/'seal.json')['files']
        assert set(files)=={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p!=root/'seal.json'}
        assert all(b.sha(root/n)==h for n,h in files.items())
    assert b.read(LADDER/'SUMMARY.json')['source_order_ladder_complete']
    provenance.check(c.u.span_runtime);assert identity()==b.read(ROOT/'FREEZE.json')['identity']
    assert b.read(c.PARENT/'analysis.json')['gate']['core_pass'] and b.read(c.SOURCE/'audit.json')['status']=='PASS'
    assert b.read(LADDER/'ARCHITECTURE_LOCK.json')['candidate_sha256']==c.CHECKPOINT
    assert b.read(c.SOURCE/'inputs.json')['seed']==191003
    assert b.read(ROOT.parent/'latent_binding_fresh_known_native_20260910/analysis.json')['gate']['native_view_pass']
    assert b.read(ROOT.parent/'latent_binding_fresh_known_text_20260910/receiver_audit.json')['actual_text_complete']
    assert not (c.DATA/'DEV_LOCK.json').exists() and not (c.DATA/'dev').exists()
def freeze():
    assert not (ROOT/'FREEZE.json').exists() and not (ROOT/'NOVEL_LOCK.json').exists();checks=[]
    for root,pattern in [(ROOT,'test_core.py'),(c.u.SPAN,'test_span.py')]:
        q=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(root),'-p',pattern,'-v'],capture_output=True,text=True,timeout=120)
        checks.append(dict(path=str(root),exit_code=q.returncode,stdout=q.stdout,stderr=q.stderr));assert q.returncode==0,checks
    value=identity();assert value['models']==b.read(c.FIT/'FREEZE.json')['identity']['models']
    assert value['models']==b.read(LADDER/'ARCHITECTURE_LOCK.json')['models']
    data,payload,source_join=c.load_data();bridge=c.bridge();initial={k:v.clone() for k,v in bridge.state_dict().items()}
    memories,donors=c.n.r.memories(bridge,data,payload);c.source_parity(memories,data);c.fixed(bridge,initial)
    targets=c.n.targets(data);assert len(targets['original'])==len(targets['changed'])==32
    del bridge,memories
    b.write(ROOT/'FREEZE.json',dict(identity=value,seed=67108,cap_s=2400,tests=checks,checkpoint=c.CHECKPOINT,
        cpu=Path('/proc/cpuinfo').read_text(),memory=Path('/proc/meminfo').read_text(),python=sys.version,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),git_status=subprocess.check_output(['git','status','--short'],text=True),
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),new_external_spend_usd=0))
    check();print('FROZEN',b.sha(ROOT/'FREEZE.json'))
def body(dest,status):
    check();b.c.seed_all(67108);assert torch.get_num_threads()==4 and torch.get_num_interop_threads()==1
    data,payload,source_join=c.load_data();rows=data['rows'];targets=c.n.targets(data)
    b.write(dest/'data.json',data)
    torch.save(dict(payload=payload,source_join=source_join),dest/'source.pt')
    training=c.load(c.FIT/'train/inputs.pt')
    b.write(dest/'text_status.json',dict(status='PENDING',new_order_text_measured=False,historical_original_order_not_substituted=True))
    bridge=c.bridge();initial={k:v.clone() for k,v in bridge.state_dict().items()}
    memories,donors=c.n.r.memories(bridge,data,payload)
    assert all(rows[i]['query']==rows[j]['query'] and i!=j and rows[i]['answer']!=rows[j]['answer'] for i,j in enumerate(donors))
    c.source_parity(memories,data)
    torch.save(dict(memories=memories,donors=donors),dest/'memories.pt')
    b.write(ROOT/'NOVEL_LOCK.json',dict(checkpoint=c.CHECKPOINT,freeze_sha256=b.sha(ROOT/'FREEZE.json'),data_sha256=b.sha(c.SOURCE/'data/examples.json'),seed=191003,burned=True,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    model,tok=b.c.load('receiver');before=b.c.parameter_hash(model);assert before==c.u.c.RECEIVER_HASH
    status.update(stage='mechanics',receiver_before=before);bank=b.AnswerBank(tok);ids=bank.prompts([r['query'] for r in rows])
    historical_ids=training['ids'][:32];history_rows=c.query_join(ids,historical_ids)
    assert torch.equal(ids,historical_ids[history_rows])
    assert bank.tokens.tolist()==b.read(c.DATA/'inputs.json')['candidate_tokens']
    labels=torch.tensor(targets['original']);seq=torch.cat([ids[:8],bank.tokens[labels[:8],:1]],dim=1)
    assert not model.model.layers[15]._forward_hooks;stock={}
    with torch.no_grad():
        for name,tokens in [('5',ids[:8]),('6',seq)]:stock[name]=dict(first=b.native_forward(model,tokens).clone(),repeat=b.native_forward(model,tokens).clone())
    torch.save(stock,dest/'stock.pt');oldstock=c.load(c.FIT/'train/stock.pt')
    assert all(torch.equal(v['first'],v['repeat']) for v in stock.values())
    historical_seq=torch.cat([historical_ids[:8],bank.tokens[training['labels'][:8],:1]],dim=1)
    historical_stock={}
    with torch.no_grad():
        for name,tokens in [('5',historical_ids[:8]),('6',historical_seq)]:historical_stock[name]=dict(first=b.native_forward(model,tokens).clone(),repeat=b.native_forward(model,tokens).clone())
    torch.save(historical_stock,dest/'historical_stock.pt')
    assert all(torch.equal(v['first'],v['repeat']) and torch.equal(v['first'],oldstock[k]['first']) for k,v in historical_stock.items())
    wrapper=b.Receiver(model,bridge)
    try:
        controls={}
        with torch.no_grad():
            for n,tokens in [('5',ids[:8]),('6',seq)]:controls[n]=dict(disabled=wrapper.forward(tokens,None).clone(),zero=wrapper.forward(tokens,torch.zeros(8,8,64)).clone())
            active=wrapper.forward(ids[:8],memories['correct'][:8]).clone();repeat=wrapper.forward(ids[:8],memories['correct'][:8]).clone()
            legacy=wrapper.forward(historical_ids[:8],c.load(c.NATIVE/'run/evaluation/outputs.pt')['memories']['correct'][:8]).clone()
            reference=c.load(c.NATIVE/'run/prefill_correct.pt')['first']['logits'][:8]
            torch.save(dict(raw=legacy,reference=reference,last=legacy[:,-1].clone()),dest/'legacy_parity.pt')
            c.last_logits(legacy,reference)
            fast=b.score(wrapper.forward,ids[:1],bank,memories['correct'][:1]);direct=b.direct_score(wrapper.forward,ids[:1],bank,memories['correct'][:1])
            reverse=copy.copy(bank);reverse.tokens=bank.tokens.flip(0);reverse.strings=bank.strings[::-1]
            reverse.indices={p:torch.where(reverse.tokens[:,0]==p)[0] for p in reverse.prefixes}
            reordered=b.score(wrapper.forward,ids[:1],reverse,memories['correct'][:1])
        torch.save(dict(controls=controls,legacy=legacy,active=active,repeat=repeat,fast=fast,direct=direct,reordered=reordered),dest/'mechanics.pt')
        assert all(torch.equal(v,stock[n]['first']) for n,row in controls.items() for v in row.values())
        assert torch.equal(active,repeat) and not torch.equal(active,stock['5']['first'])
        assert (fast['lp']-direct).abs().max()<=1e-4 and torch.equal(fast['prediction'],direct.argmax(-1))
        assert torch.equal(fast['lp'],reordered['lp'].flip(1));c.fixed(bridge,initial)
        captures={}
        for name in ['correct','wrong_binding']:
            first=c.n.helpers.native_prefill(wrapper,ids,memories[name]);rep=c.n.helpers.native_prefill(wrapper,ids,memories[name])
            torch.save(dict(first=first,repeat=rep),dest/f'prefill_{name}.pt');assert c.n.helpers.capture_equal(first,rep);captures[name]=first
        for name in ['correct','wrong_binding']:
            previous=c.load(c.NATIVE/f'run/prefill_{name}.pt')['first']['capture']
            assert torch.equal(captures[name]['capture']['hidden'],previous['hidden'][history_rows])
        b.write(dest/'manifest.json',dict(freeze_sha256=b.sha(ROOT/'FREEZE.json'),checkpoint=c.CHECKPOINT,example_ids=[r['id'] for r in rows],
            ids=ids.tolist(),candidate_tokens=bank.tokens.tolist(),targets=targets,donors=donors,seed=67108,training=False,
            view=0,order=[0,1,2,3],data_seed=191003,history_rows=history_rows.tolist(),dev_evaluated=True,novel_whole_id_evaluated=True,final_evaluated=False,text_measured=False,source_join=source_join))
        b.event('mechanics_pass');status['stage']='evaluation';phases={};metrics={}
        old=c.load(c.NATIVE/'run/evaluation/outputs.pt')['outputs'];old['canonical_only']=c.load(c.NATIVE/'run/evaluation/canonical_only.pt')['result']
        for phase in ['evaluation','repeat']:
            metric,out=c.evaluation.evaluate(wrapper,bridge,tok,data,payload,dest/phase);phases[phase]=out;metrics[phase]=metric
            assert all(torch.equal(out[k]['token_lp'],old[k]['token_lp'][history_rows]) for k in ['receiver_only','zero','canonical_only']);c.fixed(bridge,initial)
        assert all(torch.equal(v['token_lp'],phases['repeat'][k]['token_lp']) for k,v in phases['evaluation'].items())
        for name,key in [('correct','original'),('wrong_binding','changed')]:
            assert torch.equal(captures[name]['logits'].log_softmax(-1)[:,bank.tokens[:,0]],phases['evaluation'][name]['token_lp'][:,:,0])
            y=torch.tensor(targets[key]);six=torch.cat([ids,bank.tokens[y,:1]],dim=1)
            first=c.n.helpers.native_prefill(wrapper,six,memories[name]);rep=c.n.helpers.native_prefill(wrapper,six,memories[name])
            torch.save(dict(first=first,repeat=rep,ids=six,targets=y),dest/f'six_{name}.pt');assert c.n.helpers.capture_equal(first,rep)
            cap=first['capture'];assert torch.equal(cap['output'][:,5],cap['hidden'][:,5]) and not cap['used_delta'][:,5].any()
            lp=first['logits'].log_softmax(-1)
            for i,t in enumerate(y):
                ix=torch.where(bank.tokens[:,0]==bank.tokens[t,0])[0];assert torch.equal(lp[i,bank.tokens[ix,1]],phases['evaluation'][name]['token_lp'][i,ix,1])
        current,current_donors=c.n.r.memories(bridge,data,payload)
        assert current_donors==donors and all(torch.equal(v,memories[k]) for k,v in current.items());c.fixed(bridge,initial)
        torch.save(bridge.state_dict(),dest/'unchanged_state.pt')
        paired=stats.paired(phases['evaluation'],targets,rows);gate=c.gate(metrics['evaluation'],paired)
        b.write(dest/'gate.json',dict(**gate,paired=paired));status.update(stage='complete',completed=True,gate=gate,replays_exact=True,candidate_source_exact=True)
    finally:
        c.fixed(bridge,initial);assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        wrapper.hook.remove();assert not model.model.layers[15]._forward_hooks
        with torch.no_grad():restored=b.native_forward(model,seq).clone()
        torch.save(restored,dest/'restored_stock.pt');assert torch.equal(restored,stock['6']['first'])
        after=b.c.parameter_hash(model);assert after==before;status.update(receiver_after=after,bridge_unchanged=True,hooks_removed=True,stock_restored_exact=True)
    check();b.event('core_decision',**gate)
def run():
    dest=ROOT/'run';dest.mkdir();start=time.monotonic();status=dict(completed=False,stage='startup',new_external_spend_usd=0)
    try:body(dest,status)
    except BaseException as e:status.update(error_type=type(e).__name__,error=str(e)[:2000]);raise
    finally:b.write(dest/'status.json',dict(**status,elapsed_s=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','check','run']);{'freeze':freeze,'check':check,'run':run}[p.parse_args().command]()
