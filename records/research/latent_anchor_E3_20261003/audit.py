"""Independent audit of the E3 decision. Imports neither e3.py nor lab.py.

Recomputes, from outputs/<split>.pt and data/<split>.json only: ensemble predictions, the four
H3 criteria on FINAL-novel-pure for the primary arm, and the same for the secondary arm. Also
re-hashes every file in FREEZE.json and the data/capture manifests. Writes AUDIT.json.
"""
import hashlib
import json
import math
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
IDS = [a + str(d) for a in 'ABCDEFGHIJKLMNOP' for d in range(8)]
COLORS = ['red', 'green', 'blue', 'yellow']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out = {}
    fz = json.loads((HERE / 'FREEZE.json').read_text())
    out['frozen_files_unchanged'] = all(sha(HERE.parent / k) == v for k, v in fz['files'].items())
    for d in ['data', 'captures']:
        man = json.loads((HERE / d / 'MANIFEST.json').read_text())
        ext = '.json' if d == 'data' else '.pt'
        out[f'{d}_manifest_ok'] = all(sha(HERE / d / f'{k}{ext}') == v['sha256'] for k, v in man.items())

    data = json.loads((HERE / 'data/final_novel_pure.json').read_text())
    saved = torch.load(HERE / 'outputs/final_novel_pure.pt', weights_only=True)
    rows = data['rows']
    labels = torch.tensor([IDS.index(r['answer']) for r in rows])
    wrong = torch.tensor([IDS.index(r['wrong_answer']) for r in rows])
    assert torch.equal(labels, saved['labels']) and torch.equal(wrong, saved['wrong_labels'])
    assert all(r['novel_answer'] for r in rows)
    pool = set(fz['final_novel56'])
    assert {i for rec in data['records'] for i in rec['bindings'].values()} == pool

    # independent donor reconstruction: numpy-free derangement check via saved index
    donor = saved['donor_index']
    assert all(rows[int(donor[i])]['group'] != rows[i]['group'] and rows[int(donor[i])]['color'] == rows[i]['color']
               for i in range(len(rows)))
    same = labels[donor] == labels

    for arm in ['primary', 'secondary']:
        members = saved[arm]
        def ens(c):
            lp = torch.stack([m[c][:, :, 0] + m[c][:, :, 1] for m in members])
            return (torch.logsumexp(lp, 0) - math.log(len(members))).argmax(-1)
        pc, pw, ps = ens('correct'), ens('wrong_binding'), ens('shuffled')
        ro = (saved['receiver_only'][:, :, 0] + saved['receiver_only'][:, :, 1]).argmax(-1)
        r = dict(joint=(pc == labels).float().mean().item(),
                 changed_following=(pw == wrong).float().mean().item(),
                 shuffled_excl_same=(ps == labels)[~same].float().mean().item(),
                 zero=(ens('zero') == labels).float().mean().item(),
                 random_71001=(ens('random_71001') == labels).float().mean().item(),
                 random_71002=(ens('random_71002') == labels).float().mean().item(),
                 receiver_only=(ro == labels).float().mean().item(),
                 members_joint=[((m['correct'][:, :, 0] + m['correct'][:, :, 1]).argmax(-1) == labels).float().mean().item()
                                for m in members])
        r['pass'] = (r['joint'] >= 0.75 and r['changed_following'] >= 0.75 and r['shuffled_excl_same'] <= 0.10
                     and all(r[c] <= 0.10 for c in ['zero', 'random_71001', 'random_71002', 'receiver_only']))
        out[arm] = r
    res = json.loads((HERE / 'RESULT.json').read_text())
    e = res['splits']['final_novel_pure']['primary']['ensemble']
    out['matches_e3_result'] = (abs(e['joint'] - out['primary']['joint']) < 1e-9
                                and abs(e['changed_following'] - out['primary']['changed_following']) < 1e-9
                                and res['PRIMARY_H3'] == ('PASS' if out['primary']['pass'] else 'FAIL'))
    out['AUDIT_PRIMARY_H3'] = 'PASS' if out['primary']['pass'] else 'FAIL'
    (HERE / 'AUDIT.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
