import sys,json,hashlib,importlib.util,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;R=ROOT.parent;REC=R/'latent_binding_training_fold_recovery_20260913';QUAL=R/'latent_binding_boundary_causal_rankmatched_20260914';PROBE=R/'latent_binding_matched_fold_probe_20260914'
sys.path.insert(0,str(REC))
spec=importlib.util.spec_from_file_location('native_causal_common',REC/'common.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
spec=importlib.util.spec_from_file_location('native_causal_seam',REC/'seam.py');seam=importlib.util.module_from_spec(spec);spec.loader.exec_module(seam)
b=h.b;c=h.c
read=h.read;load=h.load;write=h.write;sha=h.sha

def identity():
 h.check()
 assert sha(REC/'seal.json')=='6267a6d332f8e2fddc4ec8a0ae286a9e2801029100d89b09220f0c99c402d67d'
 assert h.inventory(REC)==read(REC/'seal.json')['files']
 assert sha(QUAL/'seal.json')=='12ec563e6b8ab7cf9afaac61a960f290bae333844a47ec8dd7efb265276158cc'
 assert h.inventory(QUAL)==read(QUAL/'seal.json')['files']
 assert sha(PROBE/'seal.json')=='c66c001afb069a126a36d2a4534ae3f7af113dbcdff8635a8c83772987690f1f'
 assert sha(PROBE/'RESULT.json')==read(PROBE/'seal.json')['files']['RESULT.json']
 return dict(dependencies=h.identity(),parent_seal=sha(QUAL/'seal.json'),recovery_seal=sha(REC/'seal.json'),probe_report=sha(PROBE/'RESULT.json'),code={f:sha(ROOT/f) for f in ['PLAN.md','runtime.py','analyze.py','test_mechanics.py','job.py','test_job.py']},cap_s=10800)
def check():assert json.loads(json.dumps(identity()))==read(ROOT/'FREEZE.json')
def forward(model,seq,payload):
 ss=seam.Boundary(model)
 try:return ss.forward(seq,payload,2).detach().clone()
 finally:ss.close()
def run():
 check();b.c.seed_all(h.SEED);assert torch.get_num_threads()==4 and torch.get_num_interop_threads()==1
 dest=ROOT/'run';dest.mkdir();status=dict(completed=False,model_training_steps=0,probe_fits=0,novel_evaluations=0,folds_completed=0);start=time.monotonic()
 model,tokenizer=b.c.load('receiver');model.eval();assert all(not p.requires_grad for p in model.parameters())
 before=b.c.parameter_hash(model);original=load(h.CANDIDATE);bridge=c.bridge();bridge.requires_grad_(False);bridge.load_state_dict(original)
 x=read(REC/'inputs.json');folds=read(REC/'folds.json')['folds'];bank=b.AnswerBank(tokenizer);assert bank.tokens.tolist()==x['candidate_tokens'];mem=load(REC/'memories.pt');q0=torch.tensor(x['query_ids'])[:8]
 with torch.no_grad():stock=b.native_forward(model,q0).clone()
 try:
  recv=b.Receiver(model,bridge)
  try:
   with torch.no_grad():disabled=recv.forward(q0,None).clone();zero=recv.forward(q0,torch.zeros(8,8,64)).clone()
   assert torch.equal(stock,disabled) and torch.equal(stock,zero)
  finally:recv.hook.remove()
  torch.save(dict(stock=stock,disabled=disabled,zero=zero),dest/'stock.pt')
  for f in folds:
   folder=dest/f"fold{f['fold']}";folder.mkdir();candidate=load(REC/f"run/fold{f['fold']}/step100/candidate.pt");h.fixed(candidate,original);bridge.load_state_dict(candidate)
   arrays=np.load(QUAL/f"run/f{f['fold']}.npz");assert arrays['indices'].tolist()==f['validation_rows'];old=load(REC/f"run/fold{f['fold']}/step100/contract.pt")
   collected={k:[] for k in ['baseline','native_repeat','seam','sham','treatment','treatment_repeat','control','control_repeat']}
   ixall=f['validation_rows'];targets=bank.tokens[[x['rows'][i]['label'] for i in ixall]];queries=torch.tensor(x['query_ids'])[ixall];seqs=torch.cat([queries,targets[:,:1]],1)
   for startrow in range(0,len(ixall),8):
    ix=ixall[startrow:startrow+8];seq=seqs[startrow:startrow+8];payloads={k:torch.from_numpy(arrays[k][startrow:startrow+8].copy()).float() for k in ['original','treatment','control']}
    recv=b.Receiver(model,bridge)
    try:
     with torch.no_grad():base=recv.forward(seq,mem['wire'][ix],2).clone();repeat=recv.forward(seq,mem['wire'][ix],2).clone()
     assert torch.equal(base,repeat) and torch.equal(base,old['native'][startrow:startrow+8])
     cap=next(r['capture'] for r in old['records'] if r['indices']==ix and r['tokens'].shape[1]==6)
     for key in ['hard','scores','memory','output']:assert torch.equal(bridge.reader.capture[key],cap[key])
     assert torch.equal(payloads['original'],cap['output'][:,3:5])
    finally:recv.hook.remove()
    with torch.no_grad():
     baseline=forward(model,seq,payloads['original']);sham=forward(model,seq,payloads['original']+torch.zeros_like(payloads['original']))
     assert torch.equal(base,baseline) and torch.equal(base,sham)
     treatment=forward(model,seq,payloads['treatment']);tr=forward(model,seq,payloads['treatment']);control=forward(model,seq,payloads['control']);cr=forward(model,seq,payloads['control'])
     assert torch.equal(treatment,tr) and torch.equal(control,cr)
    for key,value in zip(collected,[base,repeat,baseline,sham,treatment,tr,control,cr]):collected[key].append(value)
    print('batch',f['fold'],startrow,'exact controls PASS',flush=True)
   saved={k:torch.cat(v) for k,v in collected.items()};saved.update(indices=ixall,targets=targets,sequences=seqs)
   torch.save(saved,folder/'logits.pt');status['folds_completed']+=1
 finally:
  bridge.load_state_dict(original);assert all(torch.equal(v,original[k]) for k,v in bridge.state_dict().items())
  assert all(not module._forward_hooks for module in model.modules())
  with torch.no_grad():restored=b.native_forward(model,q0).clone()
  assert torch.equal(restored,stock);after=b.c.parameter_hash(model);assert before==after and all(p.grad is None and not p.requires_grad for p in model.parameters())
  torch.save(restored,dest/'restored.pt');status.update(completed=status['folds_completed']==4,restoration_exact=True,before=before,after=after,no_hooks=True,no_gradients=True,elapsed_s=time.monotonic()-start)
  write(dest/'status.json',status)
 check();assert status['folds_completed']==4
 write(dest/'complete.json',dict(completed=True,folds=4,restoration_exact=True,model_updates=0))
if __name__=='__main__':
 mode=sys.argv[1]
 if mode=='freeze':write(ROOT/'FREEZE.json',json.loads(json.dumps(identity())));print(sha(ROOT/'FREEZE.json'))
 elif mode=='check':check();print('identity PASS')
 elif mode=='run':run()
 else:raise ValueError(mode)
