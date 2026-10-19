"""Record restoration, not an assertion that every old dependency is identical."""
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import sys
import time

import numpy
import torch
import transformers
from transformers.models.qwen3 import modeling_qwen3

ROOT=Path(__file__).resolve().parent


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    prior=json.loads((ROOT.parent/'latent_channel_v2_20260906/run_full/FINAL_FREEZE.json').read_text())
    actual=dict(torch=torch.__version__,numpy=numpy.__version__,transformers=transformers.__version__,python=sys.version)
    assert actual==prior['versions'], (actual,prior['versions'])
    runtime=inspect.getfile(modeling_qwen3)
    assert digest(runtime)==prior['components']['qwen_runtime']['sha256']
    acquisition=json.loads((ROOT/'acquisition.json').read_text())
    for name,identity in acquisition['files'].items():
        assert digest(Path(acquisition['path'])/name)==identity['sha256']
    # Recheck the complete historical seal, not only the imported source files.
    old=ROOT.parent/'latent_channel_v2_20260906'
    seal=json.loads((old/'seal.json').read_text())
    for name,expected in seal['files'].items():
        assert digest(old/name)==expected, name
    value=dict(versions=actual,qwen_runtime_sha256=digest(runtime),model_acquisition_rehashed=True,
        historical_files_verified=len(seal['files']),platform=platform.platform(),
        packages=sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions()),
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),external_spend_usd=0,
        caveat='All current packages recorded; only explicitly recorded old versions/runtime hashes asserted identical.')
    with (ROOT/'environment_restored.json').open('x') as f:
        json.dump(value,f,indent=2); f.write('\n')
    print('PASS: recorded versions, Qwen runtime, receiver acquisition, and historical seal verified')


if __name__=='__main__': main()
