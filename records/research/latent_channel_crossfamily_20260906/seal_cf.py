"""Seal completed negative experiment; never change historical artifacts."""
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parent


def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    result=json.loads((ROOT/'TINY_ANALYSIS.json').read_text())
    assert result['verdict']=='F — MECHANICS / TRAINING NO-GO'
    assert json.loads((ROOT/'tiny_job/state.json').read_text())['status']=='complete'
    assert not any((ROOT/p).exists() for p in ['data/train','data/dev','data/final','run_full','run_final'])
    old=ROOT.parent/'latent_channel_v2_20260906'
    historical=json.loads((old/'seal.json').read_text())['files']
    for name,digest in historical.items(): assert sha(old/name)==digest,name
    files={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*'))
           if p.is_file() and '__pycache__' not in p.parts and p.name!='seal.json'}
    value=dict(files=files,verdict=result['verdict'],prior_v2_sealed_files_verified=len(historical),
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),external_spend_usd=0,
        caveat='Local integrity seal; no external timestamp attestation. Final-stage code was not run.')
    with (ROOT/'seal.json').open('x') as f: json.dump(value,f,indent=2); f.write('\n')
    for name,digest in files.items(): assert sha(ROOT/name)==digest
    print(f'Sealed and reverified {len(files)} cross-family files; prior {len(historical)} unchanged')


if __name__=='__main__': main()
