"""Stage 0.5: LOO qualification of the fitted additive composition.

Training-ID-only (the sealed 224 rows); no novel label is read. No candidate fit.
Modes: offline (build + payload-level checks), run (receiver scoring + decision).
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
R = ROOT.parent
ALIGN = R / 'latent_binding_training_alignment_20260911'
SOURCE = R / 'latent_binding_alignment_source_20260911'
ORACLE = R / 'latent_binding_exact_component_oracle_20260911'
NAME = 'reader.value.weight'
SEED = 94117
CAP_S = 3600
ARMS = ['sham', 'loo', 'null']


def imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


h822 = imp('stage05_align', ALIGN / 'common.py')
seam = imp('stage05_seam', ORACLE / 'seam.py')
hora = imp('stage05_oracle', ORACLE / 'common.py')
b = hora.b
load, read, write = hora.load, hora.read, hora.write


def labels_matrix(names):
    return torch.tensor(np.stack([hora.labels(n) for n in names]), dtype=torch.float64)


def build():
    x = read(ALIGN / 'inputs.json')
    train_ids, rows, qids = x['training_ids'], x['trainrows'], x['query_ids']
    assert len(train_ids) == 56 and len(rows) == 224 and len(qids) == 224
    features = load(SOURCE / 'run/features.pt')['projected'].double()
    assert features.shape == (56, 960)
    for r in rows:
        assert b.IDS[r['label']] == r['id'] and train_ids[r['feature']] == r['id']
    Y = labels_matrix(train_ids)
    coord = load(ALIGN / 'coordinates.pt')
    theta = load(ALIGN / 'run/fit.pt')['theta'].double()
    W = load(ALIGN / 'run/candidate.pt')[NAME].double()
    B, H = coord['input_basis'].double(), coord['output_basis'].double() @ theta
    A = []
    for i in range(56):
        keep = [j for j in range(56) if j != i]
        A.append(torch.linalg.lstsq(Y[keep], features[keep] @ B).solution)
    perm = [(i + 1) % 56 for i in range(56)]
    qnat = features @ B
    q_loo = torch.stack([labels_matrix([train_ids[i]])[0] @ A[i] for i in range(56)])
    q_null = torch.stack([labels_matrix([train_ids[perm[i]]])[0] @ A[i] for i in range(56)])
    index = torch.tensor([train_ids.index(r['id']) for r in rows])
    m = features[index]
    vexport = m @ W.T
    values = {'sham': vexport, 'loo': vexport + (q_loo[index] - qnat[index]) @ H.T,
              'null': vexport + (q_null[index] - qnat[index]) @ H.T}
    d0 = load(hora.c.CANDIDATE)
    arms = {name: dict(payload=hora.render_value(d0, value)) for name, value in values.items()}
    case = hora.case('familiar', 'original')
    lookup = {name: i for i, name in enumerate(train_ids)}
    assert all(name in lookup for name in case['expected'])
    err = max(float((case['memory'][i] - features[lookup[case['expected'][i]]]).abs().max())
              for i in range(32))
    checks = {'familiar_memory_eq_training_features': err}
    for name in ARMS:
        assert arms[name]['payload'].shape == (224, 2, 960) and arms[name]['payload'].dtype == torch.float32
    return dict(arms=arms, ids=torch.tensor(qids), targets=torch.tensor([r['label'] for r in rows]),
                names=[r['id'] for r in rows], tokens=torch.tensor(x['candidate_tokens']),
                checks=checks, qdiff={n: float((values[n] - vexport).norm()) for n in ARMS})


def score(forward, ids, bank, payload):
    first = b.score(forward, ids, bank, payload)
    repeat = b.score(forward, ids, bank, payload)
    for key in ['token_lp', 'lp', 'prediction']:
        assert torch.equal(first[key], repeat[key]), key
    return first


def readout(out, names, targets, tokens):
    preds = out['prediction']
    rows = []
    for i in range(len(targets)):
        p, t = int(preds[i]), int(targets[i])
        rows.append(dict(id=names[i], prediction=b.IDS[p], target=b.IDS[t],
                         letter=bool(tokens[p, 0] == tokens[t, 0]), digit=bool(tokens[p, 1] == tokens[t, 1]),
                         joint=bool(p == t)))
    per_id = {}
    for name in dict.fromkeys(names):
        sel = [r for r in rows if r['id'] == name]
        per_id[name] = dict(letters=int(sum(r['letter'] for r in sel)), digits=int(sum(r['digit'] for r in sel)),
                            joint=int(sum(r['joint'] for r in sel)),
                            predictions=sorted({r['prediction'] for r in sel}))
    return dict(joint=int((preds == targets).sum()), letters=int(sum(r['letter'] for r in rows)),
                digits=int(sum(r['digit'] for r in rows)), per_id=per_id)


def run(data):
    b.c.seed_all(SEED)
    model, tok = b.c.load('receiver')
    before = b.c.parameter_hash(model)
    assert before == '6a3e1782ebad58d1e9323d5eeaff191d112848b5f0058394a998d1ac54c492b3', before
    bank = b.AnswerBank(tok)
    assert bank.tokens.tolist() == data['tokens'].tolist()
    boundary = seam.Boundary(model)
    result = dict(checks=dict(data['checks']), qdiff=data['qdiff'], arms={}, cap_s=CAP_S)
    try:
        for name in ARMS:
            out = score(boundary.forward, data['ids'], bank, data['arms'][name]['payload'])
            torch.save(dict(payload=data['arms'][name]['payload'],
                            out={k: out[k] for k in ['token_lp', 'lp', 'prediction']}), ROOT / 'run' / f'{name}.pt')
            result['arms'][name] = readout(out, data['names'], data['targets'], data['tokens'])
            b.event('stage05_arm', arm=name, joint=result['arms'][name]['joint'],
                    letters=result['arms'][name]['letters'])
    finally:
        boundary.close()
        assert not any(module._forward_hooks for module in model.modules())
        after = b.c.parameter_hash(model)
        assert after == before and all(p.grad is None and not p.requires_grad for p in model.parameters())
        result['receiver_hash'] = dict(before=before, after=after, no_hooks=True)
    sham, loo, null = (result['arms'][n] for n in ARMS)
    result['decision'] = dict(
        letters_ratio=loo['letters'] / sham['letters'],
        joint_vs_null=(loo['joint'] / null['joint']) if null['joint'] else None,
        null_letters_ratio=null['letters'] / sham['letters'],
        pass_gate=bool(loo['letters'] >= 0.90 * sham['letters'] and loo['joint'] > 3 * null['joint']
                       and null['letters'] <= 0.10 * sham['letters']))
    return result


def main(mode):
    start = time.monotonic()
    data = build()
    if mode == 'offline':
        print(json.dumps(dict(checks=data['checks'], qdiff=data['qdiff']), indent=2))
        return
    (ROOT / 'run').mkdir(exist_ok=True)
    result = run(data)
    result['elapsed_s'] = time.monotonic() - start
    write(ROOT / 'run/RESULT05.json', result)
    print(json.dumps({n: {k: result['arms'][n][k] for k in ['joint', 'letters', 'digits']} for n in ARMS}))
    print(json.dumps(result['decision'], indent=2))
    print(json.dumps({n: result['arms'][n]['per_id'] for n in ['loo']}, sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['offline', 'run'])
    main(parser.parse_args().mode)
