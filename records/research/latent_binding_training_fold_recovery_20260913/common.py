"""Training-only dependencies; no novel-bank helper import or data access."""
import hashlib, importlib.util, importlib.metadata, json, platform, types
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent; R=ROOT.parent
spec=importlib.util.spec_from_file_location('fold_familiar_core',R/'latent_binding_shared_id_native_corrected_20260908/core.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c);b=c.b
SOURCE=R/'latent_binding_alignment_source_20260911'; FAMILIAR=R/'latent_binding_fresh_known_native_20260910'
CANDIDATE=c.PARENT/'run/candidate.pt'; NAME='reader.value.weight'; SEED=94117
STEPS=[0,50,100,200]; FOLD_CAP=9000; CAP=37800
PINS={SOURCE:'48d5c0842d864312e7cdbf59fdcbd5aa927c97648c1d321746d7c50a5305f3f4',FAMILIAR:'33d50d74532f21e37ff9dc99d4149f3b8b379e527370ba0ca92e15dfadf8a9ee'}
load=lambda p:torch.load(p,weights_only=True,map_location='cpu')
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def inventory(p):return {str(f.relative_to(p)):sha(f) for f in sorted(p.rglob('*')) if f.is_file() and f!=p/'seal.json'}
def imports():
 seen=set();result={};todo=[c]
 while todo:
  m=todo.pop()
  if id(m) in seen:continue
  seen.add(id(m));p=Path(getattr(m,'__file__',None) or '/nonexistent').resolve()
  if not p.is_relative_to(R):continue
  assert 'novel_whole' not in str(p),p
  if not p.is_relative_to(ROOT):result[str(p)]=sha(p)
  todo.extend(v for v in vars(m).values() if isinstance(v,types.ModuleType))
 return result

ORIGINAL=R/'latent_binding_training_fold_validation_20260912'
MATCHED=R/'latent_binding_training_fold_matched_20260913'
INCIDENT=R/'latent_binding_fold_interruption_20260913'
DIAGNOSTIC=R/'latent_binding_fold_parity_diagnostic_20260913'
PINS[DIAGNOSTIC]='baa0e964d7d287b34de1da4c66b75b6fa1aee07625ef65979c7ddf4ac21b6294'
def exact_tree(a,z):
 if isinstance(a,torch.Tensor):assert isinstance(z,torch.Tensor) and a.dtype==z.dtype and torch.equal(a,z)
 elif isinstance(a,dict):
  assert isinstance(z,dict) and a.keys()==z.keys()
  for k in a:exact_tree(a[k],z[k])
 elif isinstance(a,(list,tuple)):
  assert type(a)==type(z) and len(a)==len(z)
  for x,y in zip(a,z):exact_tree(x,y)
 else:assert type(a)==type(z) and a==z

def identity():
 assert inventory(MATCHED)==read(INCIDENT/'OBSERVATION.json')['files']
 assert inventory(ORIGINAL)==read(DIAGNOSTIC/'FREEZE.json')['identity']['parent_files']
 for n in ['inputs.json','features.pt','memories.pt','coordinates.pt','folds.json']:assert sha(ROOT/n)==sha(ORIGINAL/n)
 for p,d in PINS.items():assert sha(p/'seal.json')==d and inventory(p)==read(p/'seal.json')['files']
 assert sha(CANDIDATE)==c.CHECKPOINT
 models={role:b.c.identities(role) for role in ['sender','receiver']}
 assert models==read(FAMILIAR/'FREEZE.json')['identity']['models']
 return dict(code={n:sha(ROOT/n) for n in ['PLAN.md','common.py','runtime.py','seam.py','analysis.py','job.py','close.py','test_folds.py','test_job.py','DESIGN.md']},
             inputs={p.name:sha(p) for p in [ROOT/'inputs.json',ROOT/'features.pt',ROOT/'memories.pt',ROOT/'coordinates.pt',ROOT/'folds.json']},
             matched_inventory=inventory(MATCHED),original_inventory=read(DIAGNOSTIC/'FREEZE.json')['identity']['parent_files'],imports=imports(),parents={str(p):d for p,d in PINS.items()},candidate=sha(CANDIDATE),models=models,
             packages={n:importlib.metadata.version(n) for n in ['torch','numpy','transformers','tokenizers','safetensors']},platform=platform.platform())
def check():assert identity()==read(ROOT/'FREEZE.json')['identity']

def coordinates(state,fit_memory):
 a=(state['reader.gate'].double()*state['reader.norm_scale'].double())*state['reader.output.weight'].double()
 _,s,vh=torch.linalg.svd(a,full_matrices=False);assert int((s>960*torch.finfo(torch.float32).eps*s[0]).sum())==64
 x=fit_memory.double();n=len(x);_,si,vhi=torch.linalg.svd(x,full_matrices=False)
 rank=int((si>max(x.shape)*torch.finfo(torch.float64).eps*si[0]).sum());assert rank==n
 lam=.001*(x-x.mean(0)).square().sum()/n
 p=vhi.T/torch.sqrt(si.square()/n+lam);rms=(x@p).square().sum(-1).mean().sqrt();p=p/rms
 return dict(output_basis=vh.T/s,input_basis=p,input_singular=si,output_singular=s,input_lambda=float(lam),input_rms=float(rms),rank=rank,base=state[NAME].clone())
def weight(theta,co):return (co['base'].double()+co['output_basis']@theta@co['input_basis'].T).float()
def render(m,w,state):
 value=torch.nn.functional.linear(m.double(),w.double());mapped=torch.nn.functional.linear(value,state['reader.output.weight'].double()).float()
 return state['reader.query_center'][None]+((state['reader.gate']*mapped)*state['reader.norm_scale'])[:,None,:]
def fixed(state,base):
 assert set(state)==set(base) and len(state)==11
 assert all(v.dtype==torch.float32 and torch.isfinite(v).all() for v in state.values())
 assert all(torch.equal(v,base[k]) for k,v in state.items() if k!=NAME)
def counts(out,targets,tokens):
 groups=[torch.where(tokens[:,0]==p)[0] for p in dict.fromkeys(tokens[:,0].tolist())];reps=torch.tensor([int(g[0]) for g in groups]);pred=[]
 for part in out['token_lp']:
  g=groups[int(part[reps,0].argmax())];pred.append(int(g[part[g,1].argmax()]))
 return dict(joint=int((out['prediction']==torch.tensor(targets)).sum()),sequential=int((torch.tensor(pred)==torch.tensor(targets)).sum()),rows=len(targets))
def select(totals):
 chosen=min([50,100,200],key=lambda s:(-totals[s]['sequential'],s))
 qualifies=all(totals[chosen][k]>totals[0][k] for k in ['joint','sequential'])
 return dict(selected_step=chosen,qualifies=qualifies,rule='max sequential across positive grid, tie earlier; both metrics must exceed step0')
def batch_loss(logits,targets,total_rows):
 return torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets.reshape(-1),reduction='sum')/(2*total_rows)
