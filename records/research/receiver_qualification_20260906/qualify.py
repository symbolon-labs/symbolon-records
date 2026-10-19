"""Exhaustive fixed oracle task; native endpoint inference, no bridge or training."""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import resource
import sys
import time

import torch

ROOT=Path(__file__).resolve().parent
CF=ROOT.parent/'latent_channel_crossfamily_20260906'
V2=ROOT.parent/'latent_channel_v2_20260906'
MODELS={
    'qwen_control':('Qwen/Qwen3-0.6B','c1899de289a04d12100db370d81485cdf75e47ca'),
    'llama_diagnostic':('meta-llama/Llama-3.2-1B-Instruct','9213176726f574b556790deb65791e0c5aa438b6'),
    'smollm_candidate':('HuggingFaceTB/SmolLM2-360M-Instruct','a10cc1512eabd3dde888204e902eca88bddb4951'),
}
SMOL_WEIGHT='e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e'
TOL=1e-4


def read(p): return json.loads(Path(p).read_text())


def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def exclusive(p,v):
    with Path(p).open('x') as f:
        json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False); f.write('\n')


def verify_seal(root):
    files=read(root/'seal.json')['files']
    for name,digest in files.items(): assert sha(root/name)==digest,name
    return len(files)


def load_pinned(name,path,digest):
    assert sha(path)==digest
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def render(tok,body):
    return tok.apply_chat_template([{'role':'user','content':body}],tokenize=False,
        add_generation_prompt=True,enable_thinking=False,date_string='06 Sep 2026')


def compare(reference,other,tolerance=TOL):
    assert len(reference)==len(other)
    delta=max(abs(a-b) for r,s in zip(reference,other)
              for a,b in zip(r['answer_log_probabilities'],s['answer_log_probabilities']))
    return dict(max_abs_label_lp_difference=delta,predictions_match=all(r['prediction']==s['prediction'] for r,s in zip(reference,other)),
                pass_check=delta<=tolerance and all(r['prediction']==s['prediction'] for r,s in zip(reference,other)))


def main():
    p=argparse.ArgumentParser(); p.add_argument('model',choices=MODELS)
    p.add_argument('--attempt',choices=['initial','snapshot_path_fix'],default='initial'); args=p.parse_args()
    dest=ROOT/(args.model if args.attempt=='initial' else args.model+'_snapshot_path_fix')
    assert not dest.exists(), 'Preserve every attempted run'
    if args.model!='qwen_control':
        assert read(ROOT/'qwen_control/result.json')['qualified'], 'Positive-control harness must qualify first'
    sealed={str(x):verify_seal(x) for x in [V2,CF]}
    cf=load_pinned('qualification_cf_adapter',CF/'channel_cf.py',read(CF/'seal.json')['files']['channel_cf.py'])
    sys.modules['channel_v2']=cf
    text=load_pinned('qualification_sealed_text',V2/'text_baseline.py',read(V2/'seal.json')['files']['text_baseline.py'])
    c=cf.c; c.MODELS['receiver']=MODELS[args.model]; c.seed_all(93001)
    if args.attempt=='snapshot_path_fix':
        assert args.model=='smollm_candidate'
        # Hub1.30 otherwise demands unneeded ONNX variants from its cached tree.
        # Resolve precisely the already-downloaded native runtime files only.
        from huggingface_hub import snapshot_download
        def native_model_path(role):
            name,revision=c.MODELS[role]
            return Path(snapshot_download(name,revision=revision,local_files_only=True,
                allow_patterns=['config.json','generation_config.json','model.safetensors',
                    'tokenizer.json','tokenizer_config.json','special_tokens_map.json']))
        c.model_path=native_model_path
    dest.mkdir(); started=time.monotonic()
    exclusive(dest/'started.json',dict(command=sys.argv,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        preregistration_sha256=sha(ROOT/'PREREGISTRATION.md'),code_sha256=sha(__file__),historical_seals=sealed,
        model=MODELS[args.model],external_spend_usd=0))
    identity=c.identities('receiver')
    if args.model=='smollm_candidate':
        assert identity['files']['model.safetensors']['sha256']==SMOL_WEIGHT
    model,tok=c.load('receiver'); initial=c.parameter_hash(model)
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    prompts=[render(tok,text.RECEIVER_INSTRUCTION+'Signal: '+letter) for letter in text.LETTERS]
    missing=render(tok,text.RECEIVER_INSTRUCTION+'The Signal value was withheld.')
    assert text.RECEIVER_INSTRUCTION=='Copy the Signal value. Reply with exactly one letter: A, B, C, or D.\n\n'
    runtime=Path(inspect.getfile(type(model)))
    metadata=dict(identity=identity,model_config=model.config.to_dict(),runtime_file=str(runtime),runtime_sha256=sha(runtime),
        versions=dict(python=sys.version,torch=torch.__version__,transformers=c.transformers.__version__,numpy=c.np.__version__),
        frozen_parameter_sha256=initial,parameter_count=sum(p.numel() for p in model.parameters()),
        prompts=prompts,prompt_sha256=[hashlib.sha256(x.encode()).hexdigest() for x in prompts],missing_prompt=missing,
        tokenizer_class=type(tok).__name__,scorer_sha256=sha(V2/'text_baseline.py'),
        native_loader_sha256=sha(cf.BASE),seed=93001,threads=4,interop_threads=1,dtype='float32',quantization='none')
    exclusive(dest/'manifest.json',metadata)
    timings={}
    def score(name,ps,batch):
        t=time.monotonic(); result=text.score(model,tok,ps,batch=batch)
        timings[name]=time.monotonic()-t
        exclusive(dest/f'{name}.json',result)
        return result
    original=score('oracle',prompts,4)
    absent=score('missing',[missing],1)
    repeat=score('oracle_repeat',prompts,4)
    absent_repeat=score('missing_repeat',[missing],1)
    checks=dict(same_batch_exact=original==repeat,missing_repeat_exact=absent==absent_repeat)
    comparison={}
    for name,order in [('reverse',[3,2,1,0]),('permute',[2,0,3,1])]:
        reordered=score(name,[prompts[i] for i in order],4)
        restored=[reordered[order.index(i)] for i in range(4)]
        comparison[name]=compare(original,restored); checks[name]=comparison[name]['pass_check']
    singles=score('singletons',prompts,1)
    comparison['singletons']=compare(original,singles); checks['singletons']=comparison['singletons']['pass_check']
    donor_order=[1,2,3,0]
    donor=score('cyclic_donor',[prompts[i] for i in donor_order],4)
    comparison['donor']=compare([original[i] for i in donor_order],donor)
    checks['donor_probabilities']=comparison['donor']['pass_check']
    checks['donor_following']=all(row['prediction']==d for row,d in zip(donor,donor_order))
    checks['donor_wrong_for_original']=all(row['prediction']!=i for i,row in enumerate(donor))
    ids=torch.tensor([r['input_ids'] for r in original]); aids=original[0]['answer_ids']
    t=time.monotonic()
    with torch.no_grad():
        full=model(input_ids=ids,use_cache=False,logits_to_keep=0).logits[:,-1].clone()
        last=model(input_ids=ids,use_cache=False,logits_to_keep=1).logits[:,-1]
    delta=(full-last).abs().max().item()
    checks['full_vocab_scoring_seam']=delta<=TOL and torch.equal(full.argmax(-1),last.argmax(-1))
    checks['sealed_scorer_matches_last']=torch.equal(last.log_softmax(-1)[:,aids],torch.tensor([r['answer_log_probabilities'] for r in original]))
    timings['full_logit_seam']=time.monotonic()-t
    torch.save(dict(full_last_logits=full,last_only_logits=last),dest/'seam.pt')
    historical_exact=None
    if args.model in ['qwen_control','llama_diagnostic']:
        previous=V2/'text_final/receiver.json' if args.model=='qwen_control' else CF/'text_tiny/receiver.json'
        old=read(previous)
        historical_exact=(original==old['outputs'] and absent[0]==old['missing_output'] and prompts==old['prompts'] and initial==old['frozen_parameter_hash'])
        checks['historical_exact']=historical_exact
    margins=[row['answer_log_probabilities'][i]-max(p for j,p in enumerate(row['answer_log_probabilities']) if j!=i) for i,row in enumerate(original)]
    checks['oracle_all_four']=all(row['prediction']==i for i,row in enumerate(original))
    checks['each_correct_margin_positive']=all(x>0 for x in margins)
    checks['missing_exact_chance']=sum(absent[0]['prediction']==i for i in range(4))/4==.25
    final=c.parameter_hash(model); checks['endpoint_unchanged']=initial==final
    checks['no_endpoint_gradients']=all(not p.requires_grad and p.grad is None for p in model.parameters())
    checks['identity_unchanged']=identity==c.identities('receiver')
    checks['code_prereg_unchanged']=(sha(__file__)==read(dest/'started.json')['code_sha256'] and sha(ROOT/'PREREGISTRATION.md')==read(dest/'started.json')['preregistration_sha256'])
    result=dict(model=MODELS[args.model],qualified=all(checks.values()),checks=checks,
        oracle_accuracy=sum(r['prediction']==i for i,r in enumerate(original))/4,
        oracle_predictions=[r['prediction'] for r in original],margins=margins,
        donor_following=sum(r['prediction']==d for r,d in zip(donor,donor_order))/4,
        donor_original_target_accuracy=sum(r['prediction']==i for i,r in enumerate(donor))/4,
        missing_accuracy=.25,comparisons=comparison,full_logit_seam_max_abs=delta,
        historical_exact=historical_exact,parameter_hash_before=initial,parameter_hash_after=final,
        timings=timings,elapsed_s=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        external_spend_usd=0,unique_oracle_cases=4,
        scope='Exact finite-domain oracle receiver qualification only; no heldout bridge or general instruction claim.')
    exclusive(dest/'result.json',result)
    print(json.dumps(dict(model=args.model,qualified=result['qualified'],checks=checks)),flush=True)


if __name__=='__main__': main()
