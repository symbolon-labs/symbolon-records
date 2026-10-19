"""Independent saved-evidence audit; no model inference or result selection."""
import hashlib
import json
from pathlib import Path
import time

import torch
from huggingface_hub import HfApi
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parent
NAMES=['qwen_control','llama_diagnostic','smollm_candidate_snapshot_path_fix']
BODY='Copy the Signal value. Reply with exactly one letter: A, B, C, or D.\n\n'


def read(p): return json.loads(p.read_text())


def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    torch.set_num_threads(4)
    verified={}; total_s=0
    for name in NAMES:
        p=ROOT/name; m=read(p/'manifest.json'); a=read(p/'result.json'); start=read(p/'started.json')
        source=ROOT/('qualify.py' if name.endswith('snapshot_path_fix') else 'qualify_initial.py')
        assert sha(source)==start['code_sha256']
        assert start['preregistration_sha256']==sha(ROOT/'PREREGISTRATION.md')
        assert a['model']==start['model']==[m['identity']['repo'],m['identity']['revision']]
        for filename,ident in m['identity']['files'].items():
            f=Path(m['identity']['path'])/filename
            assert sha(f)==ident['sha256'] and f.stat().st_size==ident['bytes']
        assert sha(Path(m['runtime_file']))==m['runtime_sha256']
        tok=AutoTokenizer.from_pretrained(m['identity']['path'],local_files_only=True)
        expected=[tok.apply_chat_template([{'role':'user','content':BODY+'Signal: '+s}],tokenize=False,
            add_generation_prompt=True,enable_thinking=False,date_string='06 Sep 2026') for s in 'ABCD']
        assert expected==m['prompts']
        assert [hashlib.sha256(x.encode()).hexdigest() for x in expected]==m['prompt_sha256']
        oracle=read(p/'oracle.json'); missing=read(p/'missing.json')
        assert oracle==read(p/'oracle_repeat.json') and missing==read(p/'missing_repeat.json')
        aids=[]
        for s in 'ABCD':
            ids=tok.encode(s,add_special_tokens=False); assert len(ids)==1; aids.append(ids[0])
        for prompt,row in zip(expected,oracle):
            ids=tok.encode(prompt,add_special_tokens=False)
            assert row['input_ids']==ids and row['answer_ids']==aids
            for s,aid in zip('ABCD',aids):
                assert tok.encode(prompt+s,add_special_tokens=False)==ids+[aid]
        lp=torch.tensor([r['answer_log_probabilities'] for r in oracle])
        pred=lp.argmax(-1).tolist(); assert pred==a['oracle_predictions']
        seam=torch.load(p/'seam.pt',weights_only=True)
        assert torch.equal(seam['last_only_logits'].log_softmax(-1)[:,aids],lp)
        assert (seam['full_last_logits']-seam['last_only_logits']).abs().max().item()<=1e-4
        assert torch.equal(seam['full_last_logits'].argmax(-1),seam['last_only_logits'].argmax(-1))
        assert seam['last_only_logits'].argmax(-1).tolist()==[r['unconstrained_token'] for r in oracle]
        for condition,order in [('reverse',[3,2,1,0]),('permute',[2,0,3,1]),('singletons',[0,1,2,3]),('cyclic_donor',[1,2,3,0])]:
            rows=read(p/f'{condition}.json')
            alt=torch.tensor([r['answer_log_probabilities'] for r in rows])
            assert (alt-lp[order]).abs().max().item()<=1e-4
            assert alt.argmax(-1).tolist()==[pred[i] for i in order]
            assert [r['input_ids'] for r in rows]==[oracle[i]['input_ids'] for i in order]
        oracle_correct=sum(p==i for i,p in enumerate(pred))/4
        assert oracle_correct==a['oracle_accuracy']
        donor=read(p/'cyclic_donor.json')
        donor_follow=sum(r['prediction']==d for r,d in zip(donor,[1,2,3,0]))/4
        donor_original=sum(r['prediction']==i for i,r in enumerate(donor))/4
        assert donor_follow==a['donor_following'] and donor_original==a['donor_original_target_accuracy']
        assert sum(missing[0]['prediction']==i for i in range(4))/4==a['missing_accuracy']==.25
        margins=[float(lp[i,i]-max(lp[i,j] for j in range(4) if j!=i)) for i in range(4)]
        assert all(abs(x-y)<1e-6 for x,y in zip(margins,a['margins']))
        assert a['parameter_hash_before']==a['parameter_hash_after']==m['frozen_parameter_sha256']
        behavioral=oracle_correct==1 and donor_follow==1 and donor_original==0 and all(x>0 for x in margins)
        assert behavioral==a['qualified']
        verified[name]=dict(oracle_accuracy=oracle_correct,donor_following=donor_follow,
            donor_original_accuracy=donor_original,qualified=behavioral,
            result_sha256=sha(p/'result.json'),manifest_sha256=sha(p/'manifest.json'),
            margins=margins,parameter_count=m['parameter_count'])
        total_s+=a['elapsed_s']
    candidate=read(ROOT/'smollm_candidate_snapshot_path_fix/manifest.json')['identity']
    remote=HfApi().model_info(candidate['repo'],revision=candidate['revision'],files_metadata=True)
    assert remote.sha==candidate['revision']
    weight=next(f for f in remote.siblings if f.rfilename=='model.safetensors')
    assert weight.lfs.sha256==candidate['files']['model.safetensors']['sha256']
    historic={}
    for dirname in ['latent_channel_v2_20260906','latent_channel_crossfamily_20260906']:
        old=ROOT.parent/dirname; files=read(old/'seal.json')['files']
        for name,digest in files.items(): assert sha(old/name)==digest
        historic[dirname]=len(files)
    assert verified['qwen_control']['qualified'] and verified['smollm_candidate_snapshot_path_fix']['qualified']
    report=dict(verified=verified,upstream_candidate_sha256=weight.lfs.sha256,historical_seals=historic,
        status='RECEIVER QUALIFIED — NOT A LATENT COMMUNICATION RESULT',
        completed_native_stage_seconds=total_s,external_spend_usd=0,
        requirements=dict(exact_oracle_task=True,non_qwen_receiver=True,no_prompt_search=True,
            no_bridge_training=True,all_four_symbols=True,donor_counterfactual=True,
            baseline_chance=True,determinism_and_batching=True,full_logit_seam=True,
            frozen_identities=True,historical_integrity=True),
        code_sha256=sha(Path(__file__)),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    with (ROOT/'VERIFICATION.json').open('x') as f: json.dump(report,f,indent=2); f.write('\n')
    print(report['status'])


if __name__=='__main__': main()
