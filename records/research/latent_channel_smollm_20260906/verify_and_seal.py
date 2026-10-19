"""Post-primary integrity closure; never selects or changes an experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def verify():
    final = read(ROOT/'run_final/analysis.json')
    independent = read(ROOT/'independent_core_audit.json')
    inputs = read(ROOT/'postfreeze_input_audit.json')
    assert inputs['status'] == 'PASS'
    for condition, value in independent['accuracy'].items():
        assert value == final['accuracy'][condition]['mean']
    for condition, bounds in independent['correct_minus_control_ci99'].items():
        assert bounds == final['correct_minus'][condition]['accuracy_ci99']
    assert independent['shuffled_donor_following'] == final['shuffled_prediction_matches_donor']
    freeze = read(ROOT/'run_full/FINAL_FREEZE.json')
    for component in freeze['components'].values():
        assert sha(Path(component['path'])) == component['sha256']
    for model in freeze['models'].values():
        for name, entry in model['files'].items():
            assert sha(Path(model['path'])/name) == entry['sha256']
    assert sha(ROOT/'run_full/best.pt') == freeze['checkpoint_sha256']
    acquisition = read(ROOT/'acquisition.json')
    assert freeze['receiver_parameter_sha256'] == acquisition['expected_receiver_parameter_hash']
    histories = {}
    for name, receipt in acquisition['historical_seals'].items():
        directory = ROOT.parent/name
        assert sha(directory/'seal.json') == receipt['seal_sha256']
        previous = read(directory/'seal.json')
        for filename, digest in previous['files'].items():
            assert sha(directory/filename) == digest, (name, filename)
        histories[name] = len(previous['files'])
    jobs = {phase: read(ROOT/f'{phase}_job/state.json')
            for phase in ['tiny', 'development', 'confirmation']}
    assert all(job['status'] == 'complete' for job in jobs.values())
    assert all(stage['status'] == 'complete' and stage['exit_code'] == 0
               for job in jobs.values() for stage in job['stages'])
    for job in jobs.values():
        for pid in [job['pid'], *[s['pid'] for s in job['stages']]]:
            proc = Path(f'/proc/{pid}/cmdline')
            assert not proc.exists() or str(ROOT).encode() not in proc.read_bytes(), 'Research process still active'
    assert all(job['external_spend_usd'] == 0 for job in jobs.values())
    value = dict(status='PASS', purpose='Integrity and independent arithmetic verification, not a new scientific decision',
        scientific_verdict=final['verdict'], independent_statistics_exact=True,
        frozen_models_tokenizers_components_checkpoint_rehashed=True,
        qualified_receiver_parameter_hash_matches=True, historical_sealed_files_reverified=histories,
        all_jobs_complete=True, external_spend_usd=0,
        sequential_job_wall_seconds=sum(j['elapsed_s'] for j in jobs.values()),
        stages={phase: {s['name']: s['elapsed_s'] for s in j['stages']} for phase,j in jobs.items()},
        cumulative_child_peak_rss_kib=max(s['peak_child_rss_kib'] for j in jobs.values() for s in j['stages']),
        source_sha256={name:sha(ROOT/name) for name in ['run_final/analysis.json',
            'independent_core_audit.json','postfreeze_input_audit.json','run_full/FINAL_FREEZE.json']},
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    write(ROOT/'VERIFICATION.json',value)
    print(json.dumps(value))


def seal():
    assert read(ROOT/'VERIFICATION.json')['status'] == 'PASS'
    report = (ROOT/'REPORT.md').read_text()
    assert 'PENDING' not in report and 'IN PROGRESS' not in report
    files = {str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*'))
             if p.is_file() and '__pycache__' not in p.parts and p.name != 'seal.json'}
    value = dict(files=files, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                 note='Local integrity seal, not an external timestamp attestation. Do not modify this tree.')
    write(ROOT/'seal.json',value)
    assert all(sha(ROOT/name) == digest for name,digest in read(ROOT/'seal.json')['files'].items())
    print(f'Sealed and reverified {len(files)} files')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['verify','seal'])
    args=parser.parse_args(); {'verify':verify,'seal':seal}[args.command]()
