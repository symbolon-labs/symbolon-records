"""DRAFT (EXPLORATORY code, not frozen). Download the TriviaQA files of the confirmatory uncertainty design.
Run this only after [researcher]'s explicit yes (APPROVAL_REQUEST.md, item D1; item D2 only for the second train shard).

    ../../.venv/bin/python -I -B fetch_data.py --researcher-said-yes "<[researcher]'s words, date>" --items rcwiki
    ../../.venv/bin/python -I -B fetch_data.py --researcher-said-yes "<[researcher]'s words, date>" --items train1   (item D2 only)

The script downloads only the files below, from one repository at one pinned revision, into ./data/trivia_qa (a new
folder). It checks the byte count and the sha256 of each file against the Hub file list (read on 2026-10-21) and stops
on any difference. The files are untrusted data: uconfirm.py reads them as data only (pyarrow), with python -I.
data/FETCH_LOG.json records each download and the approval text. Nothing in this file uses the network on import.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
REPO = 'mandarjoshi/trivia_qa'
REVISION = '0f7faf33a3908546c6fd5b73a660e0f8ff173c2f'   # main on 2026-10-21 (design section 5)
LICENSE = 'unknown (Hub card tag); check before any public release of derived data (design section 16.2)'
# key: (path in the repository, bytes, LFS sha256). Source: https://huggingface.co/api/datasets/mandarjoshi/trivia_qa/
# tree/0f7faf33a3908546c6fd5b73a660e0f8ff173c2f/rc.wikipedia, read on 2026-10-21.
FILES = {
    'validation': ('rc.wikipedia/validation-00000-of-00001.parquet', 234_778_394,
                   '59dcbdde6c1b67b4ef0b7e2591ce682200d32b523cd311e83721716489c104c8'),
    'train0': ('rc.wikipedia/train-00000-of-00007.parquet', 240_036_659,
               'd4f952bc93e9b8ff8306d712a6dbc11a60ac90ad17807c34320164b1dc060cb2'),
    'train1': ('rc.wikipedia/train-00001-of-00007.parquet', 260_772_194,
               '55cd82902467361c86f350f95c408e741bef13d23ad3ecbc07cb6eca2d9def59'),
}
# rcwiki: the two files of the design (474,815,053 bytes). train1: only if shard 0 has fewer than 7,000 usable rows.
ITEMS = {'rcwiki': ['validation', 'train0'], 'train1': ['train1']}


def local_path(key, data_dir=DATA):
    return Path(data_dir) / 'trivia_qa' / FILES[key][0]


def sha256_file(path, block=1 << 23):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(block), b''):
            h.update(b)
    return h.hexdigest()


def verify(path, size, sha256):
    """Stop if the file does not have the expected byte count and sha256. Returns the sha256."""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f'{path}: missing. Run fetch_data.py first (it needs [researcher]\'s yes).')
    n = path.stat().st_size
    if n != size:
        raise SystemExit(f'{path}: {n} bytes, expected {size}')
    got = sha256_file(path)
    if got != sha256:
        raise SystemExit(f'{path}: sha256 {got}, expected {sha256}')
    return got


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--researcher-said-yes', required=True, help="[researcher]'s approval text and date, copied into FETCH_LOG.json")
    ap.add_argument('--items', required=True, help=f'comma list of {sorted(ITEMS)}')
    a = ap.parse_args(argv)
    items = a.items.split(',')
    bad = [i for i in items if i not in ITEMS]
    if bad or 'yes' not in a.researcher_said_yes.lower():
        ap.error(f'unknown items {bad}, or the approval text does not contain "yes"')
    from huggingface_hub import hf_hub_download
    root = DATA / 'trivia_qa'
    log = []
    for name in items:
        for key in ITEMS[name]:
            remote, size, sha = FILES[key]
            dest = local_path(key)
            if not dest.exists():
                p = Path(hf_hub_download(REPO, remote, repo_type='dataset', revision=REVISION, local_dir=root))
                if p.resolve() != dest.resolve():
                    raise SystemExit(f'unexpected download path {p}')
            got = verify(dest, size, sha)
            log.append(dict(item=name, key=key, repo=REPO, revision=REVISION, remote=remote, local=str(dest.relative_to(HERE)),
                            bytes=size, sha256=got, license=LICENSE))
            print(f'ok {key}: {remote} ({size} bytes, sha256 {got[:12]})', flush=True)
    lp = DATA / 'FETCH_LOG.json'
    old = json.loads(lp.read_text()) if lp.exists() else []
    old.append(dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), researcher_said_yes=a.researcher_said_yes, files=log))
    lp.write_text(json.dumps(old, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
