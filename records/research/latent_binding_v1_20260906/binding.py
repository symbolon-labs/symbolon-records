"""Frozen architecture, binding data and native full-ID scoring seam."""
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent/'latent_channel_smollm_20260906'
PRIOR_SEAL = json.loads((PRIOR/'seal.json').read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


assert sha(PRIOR/'channel_cf.py') == PRIOR_SEAL['files']['channel_cf.py']
spec=importlib.util.spec_from_file_location('sealed_binding_parent',PRIOR/'channel_cf.py')
cf=importlib.util.module_from_spec(spec); spec.loader.exec_module(cf)
c=cf.c
BridgeModel=cf.BridgeModel  # exact imported architecture; no subclass/redefinition
COLORS=['red','green','blue','yellow']
IDS=[a+str(d) for a in 'ABCDEFGHIJKLMNOP' for d in range(8)]
SPLITS={'tiny':(121001,8,'known'), 'train':(122001,256,'known'),
        'dev':(123001,64,'known'), 'final_known':(124001,64,'known'),
        'final_novel':(125001,128,'novel')}
CONDITIONS=['receiver_only','correct','shuffled','wrong_binding','random_71001','random_71002','zero']


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False); stream.write('\n')


def event(name,**kwargs):
    print(json.dumps(dict(event=name,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kwargs)),flush=True)


def pools():
    rng=random.Random(120501); letters=list('ABCDEFGHIJKLMNOP'); digits=list('01234567')
    rng.shuffle(letters); rng.shuffle(digits)
    known=sorted(a+d for i,a in enumerate(letters) for j,d in enumerate(digits) if (i+j)%2==0)
    novel=sorted(set(IDS)-set(known))
    assert len(known)==len(novel)==64
    return dict(known=known,novel=novel)


def parse_source(source):
    by_color={}
    for line in source.splitlines():
        identity,color=line.split(' = ')
        assert identity in IDS and color in COLORS and color not in by_color
        by_color[color]=identity
    assert len(set(by_color.values()))==len(by_color)==4
    return by_color


def canonical(bindings):
    return tuple(bindings[color] for color in COLORS)


def make_split(split,used):
    seed,count,poolname=SPLITS[split]; rng=random.Random(seed); pool=pools()[poolname]
    tuples=[]
    if split=='tiny':
        chosen=rng.sample(pool,32)
        tuples=[chosen[i:i+4] for i in range(0,32,4)]
    else:
        assert count%64==0
        for _ in range(count//64):
            permutations=[]
            for color in COLORS:
                for attempt in range(10000):
                    perm=rng.sample(pool,len(pool))
                    if all(perm[i] not in [p[i] for p in permutations] for i in range(64)):
                        break
                else: raise AssertionError('Balanced matching failed')
                permutations.append(perm)
            tuples.extend([[p[i] for p in permutations] for i in range(64)])
    records=[]
    for group,values in enumerate(tuples):
        bindings=dict(zip(COLORS,values)); key=canonical(bindings)
        assert key not in used, 'Repeated table; do not silently regenerate after inspection'
        order=rng.sample(COLORS,4); offset=rng.randrange(1,4)
        source='\n'.join(f'{bindings[color]} = {color}' for color in order)
        wrong='\n'.join(f'{bindings[color]} = {COLORS[(COLORS.index(color)+offset)%4]}' for color in order)
        wrong_bindings=parse_source(wrong)
        assert canonical(wrong_bindings) not in used
        used.update([key,canonical(wrong_bindings)])
        assert parse_source(source)==bindings
        records.append(dict(group=group,source=source,wrong_source=wrong,
                            bindings=bindings,wrong_bindings=wrong_bindings))
    rows=[]
    for rec in records:
        for color in COLORS:
            rows.append(dict(group=rec['group'],color=color,query=f'Which ID is {color}?',
                answer=rec['bindings'][color],wrong_answer=rec['wrong_bindings'][color]))
    rng.shuffle(rows)
    for i,row in enumerate(rows): row['id']=f'{split}-{i:06d}'
    return dict(split=split,pool=poolname,records=records,rows=rows,seed=seed)


def dataset(split):
    if split!='tiny':
        assert read(ROOT/'run_tiny/gates.json')['pass'], 'Tiny gate not passed'
    if split.startswith('final'):
        assert (ROOT/'run_full/FINAL_FREEZE.json').exists(), 'Final not frozen'
    used=set()
    for name in SPLITS:
        data=make_split(name,used)
        if name==split: return data
    raise ValueError(split)


def load_data(split):
    dest=ROOT/'data'/split
    data=read(dest/'examples.json'); manifest=read(dest/'manifest.json')
    assert sha(dest/'examples.json')==manifest['examples_sha256']
    assert sha(dest/'sender.pt')==manifest['capture_sha256']
    payload=torch.load(dest/'sender.pt',weights_only=True)
    return data,payload


class Receiver(cf.ReceiverWrapper):
    """Identical native hook; expose final two logits for answer NLL."""
    def forward(self,ids,memory=None,keep=1):
        self.enabled=memory is not None; self.memory=memory
        try:
            return self.model(input_ids=ids,use_cache=False,logits_to_keep=keep).logits
        finally:
            self.enabled=False; self.memory=None


class AnswerBank:
    def __init__(self,tok,leading=True):
        self.strings=[(' ' if leading else '')+x for x in IDS]
        self.tokens=torch.tensor([tok.encode(x,add_special_tokens=False) for x in self.strings])
        assert self.tokens.shape==(128,2)
        assert len(set(self.tokens[:,0].tolist()))==16
        self.prefixes=sorted(set(self.tokens[:,0].tolist()))
        self.indices={p:torch.where(self.tokens[:,0]==p)[0] for p in self.prefixes}
        self.tokenizer=tok

    def prompts(self,texts):
        token_lists=[self.tokenizer.encode(s,add_special_tokens=False) for s in texts]
        assert len(set(map(len,token_lists)))==1, 'Padding not permitted'
        for text,ids in zip(texts,token_lists):
            for label,tail in zip(self.strings,self.tokens.tolist()):
                assert self.tokenizer.encode(text+label,add_special_tokens=False)==ids+tail
        return torch.tensor(token_lists)


def native_forward(model,ids,memory=None,keep=1):
    assert memory is None
    return model(input_ids=ids,use_cache=False,logits_to_keep=keep).logits


def score(forward,ids,bank,memory=None,batch=8):
    """Prefix reuse without cache; returns both full-vocab token LPs."""
    all_parts=[]; diagnostics=[]
    with torch.no_grad():
        for start in range(0,len(ids),batch):
            q=ids[start:start+batch]; z=None if memory is None else memory[start:start+batch]
            first=forward(q,z,1)[:,-1].log_softmax(-1)
            parts=torch.empty(len(q),128,2)
            parts[:,:,0]=first[:,bank.tokens[:,0]]
            owner=getattr(forward,'__self__',None)
            if z is not None and isinstance(owner,Receiver):
                diagnostics.extend([{k:v[j] for k,v in owner.bridge.reader.diagnostics.items()} for j in range(len(q))])
            for prefix in bank.prefixes:
                seq=torch.cat([q,torch.full((len(q),1),prefix,dtype=torch.long)],dim=1)
                second=forward(seq,z,1)[:,-1].log_softmax(-1)
                ix=bank.indices[prefix]; parts[:,ix,1]=second[:,bank.tokens[ix,1]]
            all_parts.append(parts.cpu())
    parts=torch.cat(all_parts)
    return dict(token_lp=parts,lp=parts.sum(-1),prediction=parts.sum(-1).argmax(-1),diagnostics=diagnostics)


def direct_score(forward,ids,bank,memory=None):
    """Independent slow teacher-forced candidate loop for seam validation."""
    outputs=[]
    with torch.no_grad():
        for i,tail in enumerate(bank.tokens):
            seq=torch.cat([ids,tail[:1].expand(len(ids),1)],dim=1)
            lp=forward(seq,memory,2).log_softmax(-1)
            outputs.append(lp[:,0,tail[0]]+lp[:,1,tail[1]])
    return torch.stack(outputs,dim=1)


def train_logits(wrapper,bank,ids,labels,memory):
    tails=bank.tokens[labels]
    seq=torch.cat([ids,tails[:,:1]],dim=1)
    logits=wrapper.forward(seq,memory,2)
    losses=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),tails.reshape(-1),reduction='none')
    return losses.reshape(len(ids),2).sum(-1).mean()


def components():
    names=['PREREGISTRATION.md','TEXT_PROTOCOL.md','binding.py','run.py','job.py','test_binding.py']
    return {name:sha(ROOT/name) for name in names}
