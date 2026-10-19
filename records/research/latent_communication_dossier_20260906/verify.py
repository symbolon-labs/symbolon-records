"""Read-only evidence verification. No model loading, downloads or writes."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def verify(models=False):
    inventory = read(ROOT/'evidence/source_inventory.json')
    research = ROOT.parent
    for name,v in inventory['files'].items():
        path = research/name
        assert path.resolve().is_relative_to(research.resolve())
        assert path.stat().st_size == v['bytes'] and sha(path) == v['sha256'], name
    receipt = read(ROOT/'evidence/export_verification.json')
    assert sha(ROOT/'build_evidence.py') == receipt['generator_sha256']
    assert sha(ROOT/'evidence/source_inventory.json') == receipt['source_inventory_sha256']
    for name,digest in receipt['csvs'].items():
        assert sha(ROOT/'evidence'/name) == digest
    result = read(ROOT/'evidence/result.json')
    original = research/'latent_channel_smollm_20260906/run_final/analysis.json'
    assert result == read(original)
    assert result['gate4'] == 'PASS' and all(result['statistical_checks'].values()) and all(result['mechanical_checks'].values())
    identity = read(ROOT/'evidence/model_and_runtime_identities.json')
    checked = 0
    if models:
        for model in identity['models'].values():
            for name,v in model['files'].items():
                assert sha(Path(model['path'])/name) == v['sha256'], (model['repo'],name)
                checked += 1
        for name,v in identity['components'].items():
            assert sha(Path(v['path'])) == v['sha256'], name
    seal = ROOT/'documentation_seal.json'
    if seal.exists():
        for name,digest in read(seal)['files'].items():
            assert sha(ROOT/name) == digest, name
    return {'status':'PASS', 'source_files':len(inventory['files']), 'model_files_rehashed':checked,
            'documentation_seal_checked':seal.exists(), 'inference':False, 'writes':False}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--models',action='store_true')
    args=parser.parse_args(); print(json.dumps(verify(args.models)))
