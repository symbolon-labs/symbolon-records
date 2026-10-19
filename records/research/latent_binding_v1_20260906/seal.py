"""Seal or read-only verify this completed local research tree."""
import argparse
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inventory():
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted(ROOT.rglob('*'))
            if path.is_file() and path.name != 'seal.json'
            and not {'__pycache__', '.pytest_cache'} & set(path.relative_to(ROOT).parts)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['create', 'verify'])
    args = parser.parse_args()
    path = ROOT / 'seal.json'
    if args.command == 'create':
        state = json.loads((ROOT / 'tiny_job/state.json').read_text())
        assert state['status'] in {'complete', 'oracle_no_go', 'failed'}
        assert all(stage['status'] != 'running' for stage in state['stages'])
        assert 'IN PROGRESS' not in (ROOT / 'REPORT.md').read_text()
        audit = json.loads((ROOT / 'audit.json').read_text())
        assert audit['status'] == 'PASS'
        result = dict(schema='local.binding.research.seal.v1',
                      utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                      files=inventory(),
                      note='Integrity inventory; not a scientific PASS. Read REPORT.md.')
        with path.open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    saved = json.loads(path.read_text())
    assert inventory() == saved['files'], 'Changed, missing, or added payload files'
    print(json.dumps(dict(status='PASS', scope='Integrity only',
                          files=len(saved['files']), seal_sha256=sha(path))))


if __name__ == '__main__':
    main()
