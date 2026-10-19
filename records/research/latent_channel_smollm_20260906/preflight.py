"""Read-only historical/model verification and exclusive new-run receipt."""
import inspect
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch
from transformers.models.llama import modeling_llama
from transformers.models.qwen3 import modeling_qwen3
import channel_cf as cf

ROOT,c=cf.ROOT,cf.c


def main():
    start=time.monotonic(); seals={}
    for name in ['latent_channel_v1_20260906','latent_channel_v2_20260906',
                 'latent_channel_crossfamily_20260906','receiver_qualification_20260906']:
        p=ROOT.parent/name; seal=json.loads((p/'seal.json').read_text())
        for filename,digest in seal['files'].items(): assert c.sha(p/filename)==digest,filename
        seals[name]=dict(files=len(seal['files']),seal_sha256=c.sha(p/'seal.json'))
    qualified=ROOT.parent/'receiver_qualification_20260906/smollm_candidate_snapshot_path_fix'
    result=json.loads((qualified/'result.json').read_text())
    manifest=json.loads((qualified/'manifest.json').read_text())
    assert result['qualified'] and all(result['checks'].values())
    receiver=c.identities('receiver'); sender=c.identities('sender')
    assert receiver==manifest['identity']
    old_sender=json.loads((cf.V2/'data/tiny/manifest.json').read_text())['model']
    assert sender==old_sender
    assert receiver['files']['model.safetensors']['sha256']=='e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e'
    assert c.sha(inspect.getfile(modeling_llama))==manifest['runtime_sha256']
    versions=dict(python=sys.version,torch=torch.__version__,transformers=c.transformers.__version__,numpy=c.np.__version__)
    assert versions==manifest['versions']
    bridge=cf.BridgeModel(); count=sum(p.numel() for p in bridge.parameters())
    assert count==569857 and count/manifest['parameter_count']<.01
    value=dict(receiver=receiver,sender=sender,qualification_result_sha256=c.sha(qualified/'result.json'),
        expected_receiver_parameter_hash=result['parameter_hash_after'],historical_seals=seals,
        versions=versions,platform=platform.platform(),cuda=torch.cuda.is_available(),
        runtime_hashes={name:c.sha(inspect.getfile(module)) for name,module in [('llama',modeling_llama),('qwen',modeling_qwen3)]},
        code_sha256=c.sha(__file__),extension_sha256=c.sha(ROOT/'channel_cf.py'),
        preregistration_sha256=c.sha(ROOT/'PREREGISTRATION.md'),trainable_parameters=count,
        receiver_parameters=manifest['parameter_count'],receiver_layer_zero_based=cf.RECEIVER_LAYER,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],text=True),
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),elapsed_s=time.monotonic()-start,external_spend_usd=0)
    with (ROOT/'acquisition.json').open('x') as f: json.dump(value,f,indent=2); f.write('\n')
    print('PASS: historical seals, qualified receiver, sender and runtime verified; no inference performed')


if __name__=='__main__': main()
