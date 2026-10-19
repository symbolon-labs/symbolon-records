"""Convergence continuation of the sealed training-only alignment; no new data."""
import importlib.metadata
import importlib.util
import platform
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;R=ROOT.parent
ALIGN=R/'latent_binding_training_alignment_20260911'
spec=importlib.util.spec_from_file_location('convergence_parent_common',ALIGN/'common.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
b,c=p.b,p.c
read,write,sha,load,inventory=p.read,p.write,p.sha,p.load,p.inventory
native,fixed,counts,render,weight=p.native,p.fixed,p.counts,p.render,p.weight
NAME,SEED,LR,SOURCE=p.NAME,p.SEED,p.LR,p.SOURCE
STEPS=200;EXTRA=400;TOTAL=STEPS+EXTRA;CAP=14400
PARENT_SEAL='0c1f56b35383018eab212d5967fefcfc443ddad2b6ecab4fd670089d2070dc88'
PROGENITOR_ROOT=R/'latent_binding_convergence_20260911'
PROGENITOR_SEAL='e79399a83d705c53a3a233aa1d3dabb9dfa258ea60540019ccb5fb4a32420d17'
PARENT_PINS={'candidate':None,'fit':None,'coordinates':None}
CODE=['PLAN.md','CORRECTION.md','common.py','seam.py','stats.py','runtime.py','analysis.py','final_contract.py','per_id.py','job.py','close.py','test_convergence.py']


def identity():
    p.check()
    assert sha(ALIGN/'seal.json')==PARENT_SEAL and inventory(ALIGN)==read(ALIGN/'seal.json')['files']
    assert sha(PROGENITOR_ROOT/'seal.json')==PROGENITOR_SEAL
    parent=dict(candidate=sha(ALIGN/'run/candidate.pt'),fit=sha(ALIGN/'run/fit.pt'),coordinates=sha(ALIGN/'coordinates.pt'),seal=PARENT_SEAL)
    for key,digest in PARENT_PINS.items():
        if digest is not None:assert parent[key]==digest
    return dict(code={n:sha(ROOT/n) for n in CODE},imports=p.imports(),parent=parent,progenitor=PROGENITOR_SEAL,candidate=sha(ROOT/'start.pt'),
        inputs=sha(ROOT/'inputs.json'),packages={n:importlib.metadata.version(n) for n in ['torch','transformers','numpy','tokenizers','safetensors']},platform=platform.platform())


def check():assert identity()==read(ROOT/'FREEZE.json')['identity']


def start():
    saved=load(ROOT/'start.pt');assert set(saved)=={'theta','optimizer','coordinates','candidate'}
    return saved


def verify_start(saved):
    theta=saved['theta'];coord=saved['coordinates'];state=saved['candidate']
    sealed=load(ALIGN/'run/candidate.pt')
    assert all(torch.equal(v,sealed[k]) for k,v in state.items())
    assert theta.shape==(64,56) and theta.dtype==torch.float64
    assert torch.equal(weight(theta,coord),state[NAME])
    moments=saved['optimizer']['state'][0]
    assert set(moments)>= {'step','exp_avg','exp_avg_sq'} and int(moments['step'])==STEPS
    assert moments['exp_avg'].shape==theta.shape and moments['exp_avg_sq'].shape==theta.shape
    assert torch.isfinite(moments['exp_avg']).all() and torch.isfinite(moments['exp_avg_sq']).all() and (moments['exp_avg_sq']>0).all()
    return dict(start_step=int(moments['step']),start_weight_relative=float((weight(theta,coord).double()-sealed[NAME].double()).norm()/sealed[NAME].double().norm()))


def baseline_scores(pool):return ALIGN/'run'/f'baseline_{pool}'
def reference_scores(pool):return ALIGN/'run'/f'trained_{pool}'
