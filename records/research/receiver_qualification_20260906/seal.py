"""Seal the independently verified receiver qualification, not a bridge PASS."""
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parent


def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    verification=json.loads((ROOT/'VERIFICATION.json').read_text())
    assert all(verification['requirements'].values())
    for name,result in verification['verified'].items():
        assert sha(ROOT/name/'result.json')==result['result_sha256']
        assert sha(ROOT/name/'manifest.json')==result['manifest_sha256']
    files={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.name!='seal.json'}
    with (ROOT/'seal.json').open('x') as f:
        json.dump(dict(files=files,status=verification['status'],external_spend_usd=0,
            utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),f,indent=2); f.write('\n')
    for name,digest in files.items(): assert sha(ROOT/name)==digest
    print(f'Sealed and reverified {len(files)} files')


if __name__=='__main__': main()
