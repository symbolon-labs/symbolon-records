"""Fresh actual text run; identical inference, user-authorized longer runtime."""
import argparse
import importlib.metadata
from pathlib import Path
import resource
import subprocess
import sys
import time
import types
import torch
from transformers import AutoTokenizer
import core as c
ROOT,b,r=c.ROOT,c.b,c.r
CODE=['PREREGISTRATION.md','AUTHORIZATION.md','ANALYSIS_DEPENDENCIES.json','NATIVE_PARENT.json','core.py','runtime.py','audit.py','test_text.py','job.py','report.py','close.py']
def load(p):return torch.load(p,weights_only=True)
def imports():
    pending=[c];seen=set();result={}
    while pending:
        m=pending.pop()
        if id(m) in seen:continue
        seen.add(id(m));name=getattr(m,'__file__',None)
        if not name:continue
        p=Path(name).resolve()
        if not p.is_relative_to(ROOT.parent):continue
        if not p.is_relative_to(ROOT):result[str(p)]=b.sha(p)
        pending.extend(v for v in vars(m).values() if isinstance(v,types.ModuleType))
    return result
def identity():
    paths=[c.PARENT/'run/data.json',c.PARENT/'analysis.json',c.PARENT/'FREEZE.json',c.PARENT/'seal.json',c.PREVIOUS/'seal.json',c.PREVIOUS/'FREEZE.json',c.PREVIOUS/'inputs.json',c.PREVIOUS/'sender_audit.json']
    return dict(code={n:b.sha(ROOT/n) for n in CODE},imports=imports(),inputs={str(p):b.sha(p) for p in paths},data=b.sha(ROOT/'inputs.json'),
        models={role:b.c.identities(role) for role in c.CAPS},packages={n:importlib.metadata.version(n) for n in ['torch','transformers','tokenizers','numpy','safetensors']})
def check():
    assert b.sha(c.PREVIOUS/'seal.json')=='9a6365b8cb5a99be7748bc3043e41228675e0177d5bf2e99c00863b38f9265ee'
    prior=b.read(c.PREVIOUS/'seal.json')['files']
    assert all(b.sha(c.PREVIOUS/n)==h for n,h in prior.items())
    assert set(prior)=={str(p.relative_to(c.PREVIOUS)) for p in c.PREVIOUS.rglob('*') if p.is_file() and p!=c.PREVIOUS/'seal.json'}
    for name,h in b.read(ROOT/'ANALYSIS_DEPENDENCIES.json')['files'].items():assert b.sha(Path(name))==h
    assert b.sha(c.PARENT/'seal.json')==b.read(ROOT/'NATIVE_PARENT.json')['seal_sha256']
    files=b.read(c.PARENT/'seal.json')['files'];assert all(b.sha(c.PARENT/n)==h for n,h in files.items())
    assert set(files)=={str(p.relative_to(c.PARENT)) for p in c.PARENT.rglob('*') if p.is_file() and p!=c.PARENT/'seal.json'}
    assert identity()==b.read(ROOT/'FREEZE.json')['identity'] and b.read(c.PARENT/'audit.json')['status']=='PASS'
def freeze():
    assert not (ROOT/'FREEZE.json').exists()
    q=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(ROOT),'-p','test_text.py','-v'],capture_output=True,text=True,timeout=120);assert q.returncode==0,q.stderr
    data=b.read(c.PARENT/'run/data.json');rows=c.requests(data);models={role:b.c.identities(role) for role in c.CAPS}
    assert models==b.read(c.PARENT/'FREEZE.json')['identity']['models']
    tok={role:AutoTokenizer.from_pretrained(v['path'],local_files_only=True) for role,v in models.items()}
    prompts=[r.render(tok['sender'],c.sender_body(row)) for row in rows]
    bank=b.AnswerBank(tok['sender'],leading=False);ids=bank.prompts(prompts);assert ids.shape==(64,54)
    copy_prompts=[r.render(tok['receiver'],c.receiver_body(message)) for message in b.IDS]
    receiver_bank=b.AnswerBank(tok['receiver'],leading=False);copy_ids=receiver_bank.prompts(copy_prompts);assert copy_ids.shape==(128,52)
    b.write(ROOT/'inputs.json',dict(rows=rows,sender_prompts=prompts,sender_ids=ids.tolist(),sender_candidate_tokens=bank.tokens.tolist(),
        receiver_messages=b.IDS,receiver_prompts=copy_prompts,receiver_ids=copy_ids.tolist(),receiver_candidate_tokens=receiver_bank.tokens.tolist(),
        native_parent_sha256=b.sha(c.PARENT/'analysis.json'),view=0,order=[0,1,2,3],data_seed=191003,seed=68108))
    b.write(ROOT/'FREEZE.json',dict(identity=identity(),caps=c.CAPS,seed=68108,tests=dict(exit_code=q.returncode,stdout=q.stdout,stderr=q.stderr),
        cpu=Path('/proc/cpuinfo').read_text(),memory=Path('/proc/meminfo').read_text(),python=sys.version,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),git_status=subprocess.check_output(['git','status','--short'],text=True),
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),new_external_spend_usd=0));check();print('FROZEN',b.sha(ROOT/'FREEZE.json'),flush=True)
def successful(role):
    job=b.read(ROOT/f'job_{role}/completion.json');status=b.read(ROOT/role/'status.json');launch=b.read(ROOT/f'job_{role}/launch.json')
    assert job['exit_code']==0 and not job['timeout'] and status['completed']
    assert not any(Path(f'/proc/{launch[k]}').exists() for k in ['pid','wrapper_pid'])
def prepare_receiver():
    check();successful('sender');a=b.read(ROOT/'sender_audit.json');assert a['status']=='PASS' and a['outputs_sha256']==b.sha(ROOT/'sender/outputs.pt')
    x=b.read(ROOT/'inputs.json');sent=load(ROOT/'sender/outputs.pt');messages=c.message_ids(x['rows'],sent['prediction'].tolist());indices=[b.IDS.index(m) for m in messages]
    b.write(ROOT/'RECEIVER_LOCK.json',dict(sender_sha256=b.sha(ROOT/'sender/outputs.pt'),freeze_sha256=b.sha(ROOT/'FREEZE.json'),messages=messages,indices=indices,
        prompts=[x['receiver_prompts'][i] for i in indices],ids=[x['receiver_ids'][i] for i in indices]))
    print('RECEIVER LOCKED',b.sha(ROOT/'RECEIVER_LOCK.json'),flush=True)
def parity(model,ids,bank,dest):
    forward=r.native_model_forward(model);start=time.monotonic()
    fast=b.score(forward,ids[:1],bank);direct=b.direct_score(forward,ids[:1],bank);repeat=b.score(forward,ids[:1],bank)
    torch.save(dict(fast=fast,direct=direct,repeat=repeat),dest/'parity.pt')
    error=float((fast['lp']-direct).abs().max());assert error<=1e-4 and torch.equal(fast['prediction'],direct.argmax(-1))
    assert torch.equal(fast['token_lp'],repeat['token_lp'])
    b.event('native_scoring_parity',max_difference=error)
    return dict(max_difference=error,tolerance=1e-4,rerun_exact=True,prediction_exact=True,elapsed_s=time.monotonic()-start)
def body(role,dest,status):
    check();b.c.seed_all(68108);audit=b.read(ROOT/'input_audit.json');assert audit['status']=='PASS' and audit['freeze_sha256']==b.sha(ROOT/'FREEZE.json')
    x=b.read(ROOT/'inputs.json')
    if role=='receiver':
        successful('sender');lock=b.read(ROOT/'RECEIVER_LOCK.json');assert lock['sender_sha256']==b.sha(ROOT/'sender/outputs.pt') and lock['freeze_sha256']==b.sha(ROOT/'FREEZE.json')
        audited=b.read(ROOT/'sender_audit.json');assert audited['status']=='PASS' and audited['outputs_sha256']==lock['sender_sha256']
        messages=c.message_ids(x['rows'],load(ROOT/'sender/outputs.pt')['prediction'].tolist());indices=[b.IDS.index(m) for m in messages]
        assert lock['messages']==messages and lock['indices']==indices and lock['ids']==[x['receiver_ids'][i] for i in indices] and lock['prompts']==[x['receiver_prompts'][i] for i in indices]
    else:
        info=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines());available=int(info['MemAvailable'].split()[0]);assert available>=10*1024*1024;status['mem_available_kib_before_load']=available
    model,tok=b.c.load(role);before=b.c.parameter_hash(model);status['endpoint_before']=before;assert before==c.HASHES[role]
    assert all(not p.requires_grad and p.grad is None for p in model.parameters()) and all(not m._forward_hooks for m in model.modules())
    try:
        bank=b.AnswerBank(tok,leading=False);prompts=x['sender_prompts'] if role=='sender' else lock['prompts'];ids=bank.prompts(prompts)
        assert ids.tolist()==(x['sender_ids'] if role=='sender' else lock['ids']) and bank.tokens.tolist()==x[role+'_candidate_tokens']
        status['parity']=parity(model,ids,bank,dest);start=time.monotonic()
        if role=='sender':
            pieces=[];times=[]
            for offset in range(0,64,4):
                tick=time.monotonic();out=b.score(r.native_model_forward(model),ids[offset:offset+4],bank,batch=4)
                torch.save(out,dest/f'chunk_{offset:04d}.pt');pieces.append(out['token_lp']);times.append(time.monotonic()-tick);b.event('text_sender',completed=offset+4,total=64)
            token_lp=torch.cat(pieces);status['inference_s']=time.monotonic()-start
            repeated=b.score(r.native_model_forward(model),ids[:4],bank,batch=4)
            torch.save(dict(token_lp=token_lp,lp=token_lp.sum(-1),prediction=token_lp.sum(-1).argmax(-1),repeat_first4=repeated,candidate_tokens=bank.tokens,ids=ids),dest/'outputs.pt')
            assert torch.equal(token_lp[:4],repeated['token_lp']);status['chunk_times_s']=times
        else:
            output=b.score(r.native_model_forward(model),ids,bank);status['inference_s']=time.monotonic()-start
            repeated=b.score(r.native_model_forward(model),ids,bank);torch.save(dict(output=output,repeat=repeated,candidate_tokens=bank.tokens,ids=ids),dest/'outputs.pt')
            assert torch.equal(output['token_lp'],repeated['token_lp'])
            sent=load(ROOT/'sender/outputs.pt');messages=lock['messages']
            rows=[dict(**row,sender_prediction=b.IDS[pred],actual_prediction=b.IDS[int(output['prediction'][messages.index(b.IDS[pred])])],
                oracle_prediction=b.IDS[int(output['prediction'][messages.index(row['answer'])])]) for row,pred in zip(x['rows'],sent['prediction'].tolist(),strict=True)]
            b.write(dest/'rows.json',dict(rows=rows,messages=messages,prompts=prompts,text_bytes={m:len(m.encode('utf8')) for m in messages},
                text_token_counts={m:len(tok.encode(m,add_special_tokens=False)) for m in messages}))
        status.update(completed=True,repeat_exact=True,contexts=len(ids),prompt_tokens=ids.shape[1],outputs_sha256=b.sha(dest/'outputs.pt'))
    finally:
        after=b.c.parameter_hash(model);status['endpoint_after']=after
        assert after==before and all(not p.requires_grad and p.grad is None for p in model.parameters()) and all(not m._forward_hooks for m in model.modules())
        status.update(endpoint_unchanged=True,no_hooks=True)
    check()
def run(role):
    dest=ROOT/role;dest.mkdir();start=time.monotonic();status=dict(completed=False,role=role,new_external_spend_usd=0)
    try:body(role,dest,status)
    except BaseException as e:status.update(completed=False,error_type=type(e).__name__,error=str(e));raise
    finally:b.write(dest/'status.json',dict(**status,elapsed_s=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','check','prepare_receiver','sender','receiver']);q=p.parse_args()
    if q.command in c.CAPS:run(q.command)
    else:{'freeze':freeze,'check':check,'prepare_receiver':prepare_receiver}[q.command]()
