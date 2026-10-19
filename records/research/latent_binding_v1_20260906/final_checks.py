"""Post-job verification and generated provenance; no model inference."""
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent


def main():
    state = json.loads((ROOT / 'tiny_job/state.json').read_text())
    assert state['status'] == 'complete'
    assert all(s['exit_code'] == 0 for s in state['stages'])
    assert not state['tiny_pass']
    assert sorted(p.name for p in (ROOT / 'data').iterdir()) == ['tiny']
    assert not (ROOT / 'run_full').exists()
    def run(argv):
        result = subprocess.run(argv, check=True, capture_output=True, text=True)
        return result.stdout
    historical = json.loads(run([sys.executable,
        str(ROOT.parent / 'latent_communication_dossier_20260906/verify.py'), '--models']))
    assert historical['status'] == 'PASS'
    audit = json.loads(run([sys.executable, str(ROOT / 'audit.py'), '--save', str(ROOT / 'audit.json')]))
    assert audit['status'] == 'PASS' and not audit['tiny_pass_independently_rederived']
    tests = run([sys.executable, '-m', 'pytest', str(ROOT / 'test_binding.py'),
                 str(ROOT / 'test_audit.py'), '-q', '--junitxml=' + str(ROOT / 'final_tests.xml')])
    svg = ET.parse(ROOT / 'training_curves.svg').getroot()
    assert len(svg.findall('.//{http://www.w3.org/2000/svg}circle')) == 64
    acquisition = json.loads((ROOT / 'acquisition.json').read_text())
    commit = run(['git', 'rev-parse', 'HEAD']).strip()
    status = run(['git', 'status', '--short'])
    assert commit == acquisition['git_commit']
    cpu = next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
               if line.startswith('model name'))
    memory = next(line for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemTotal:'))
    result = dict(status='PASS', scope='Integrity, scoring audit and tests; scientific verdict remains NO-GO',
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        historical_verification=historical, independent_audit=audit['status'],
        tests=tests, svg_data_points=64, git_commit=commit,
        git_status_unchanged=status == acquisition['git_status'], git_status=status,
        versions={name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'numpy', 'tokenizers', 'safetensors']},
        python=sys.version, platform=platform.platform(), cpu=cpu, memory=memory,
        job_elapsed_s=state['elapsed_s'], external_spend_usd=state['external_spend_usd'],
        local_energy_and_assistant_service_cost='not metered',
        only_tiny_data_exists=True, endpoint_models_rehashed=True)
    with (ROOT / 'final_checks.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
