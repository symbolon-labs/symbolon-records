"""Stage 0: reconstruct the composition diagnostic from sealed artifacts.

Read-only over sealed trees. No fitting, no candidate, no gates, no new data.
Modes:
  offline  build every payload and run the payload-level checks (no receiver)
  run      one receiver-only scoring pass per arm (3600s cap), then readout
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
COMP = R / 'latent_binding_component_alignment_20260911'
SUB = R / 'latent_binding_component_subspace_20260911'
PROBE = R / 'latent_binding_training_feature_probe_20260911'
SOURCE = R / 'latent_binding_alignment_source_20260911'
ORACLE = R / 'latent_binding_exact_component_oracle_20260911'
NOVEL = R / 'latent_binding_novel_whole_native_20260910'
NAME = 'reader.value.weight'
SEED = 94117
CAP_S = 3600
PROBE_NOVEL = {'A0': 'H3', 'A2': 'M3', 'C4': 'C4', 'I2': 'I2',
               'K0': 'K0', 'K3': 'K3', 'M0': 'M0', 'M3': 'M3'}
ARMS = ['f61_sham', 'f61_R3', 'f61_R1', 'f61_R2',
        'd822_sham', 'd822_R1', 'd822_R2']


def imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


h822 = imp('stage0_align', ALIGN / 'common.py')
seam = imp('stage0_seam', ORACLE / 'seam.py')
hora = imp('stage0_oracle', ORACLE / 'common.py')
b = hora.b
load, read, write = hora.load, hora.read, hora.write


def labels_matrix(names):
    return torch.tensor(np.stack([hora.labels(n) for n in names]), dtype=torch.float64)


def build():
    case = hora.case('novel', 'original')
    info = hora.bank('novel')
    expected = case['expected']
    features = load(SOURCE / 'run/features.pt')['projected'].double()
    train_ids = read(SOURCE / 'inputs.json')['ids']
    assert len(train_ids) == 56
    Y = labels_matrix(train_ids)
    Ye = labels_matrix(expected)
    probe_src = {r['id']: r['prediction'] for r in read(PROBE / 'RESULT.json')['novel']['rows']}
    assert probe_src == PROBE_NOVEL
    Yp = labels_matrix([PROBE_NOVEL[e] for e in expected])
    m = case['memory'].double()
    d0 = load(hora.c.CANDIDATE)

    coord822 = load(ALIGN / 'coordinates.pt')
    theta822 = load(ALIGN / 'run/fit.pt')['theta'].double()
    W822 = load(ALIGN / 'run/candidate.pt')[NAME].double()
    B822 = coord822['input_basis'].double()
    H822 = coord822['output_basis'].double() @ theta822
    A822 = torch.linalg.lstsq(Y, features @ B822).solution

    z = np.load(SUB / 'arrays.npz', allow_pickle=False)
    B61 = torch.from_numpy(z['input_basis'].copy()).double()
    Ro61 = torch.from_numpy(z['output_basis'].copy()).double()
    P61 = torch.from_numpy(z['component_conditioner'].copy()).double()
    theta61 = load(COMP / 'run/fit.pt')['theta'].double()
    W61 = load(COMP / 'run/candidate.pt')[NAME].double()
    H61 = Ro61 @ theta61
    A61 = torch.linalg.lstsq(Y, features @ B61).solution

    def payload(value):
        return hora.render_value(d0, value)

    def composed(W, B, H, qhat):
        vexport = m @ W.T
        qnat = m @ B
        return payload(vexport + (qhat - qnat) @ H.T), vexport, qnat

    arms = {}
    arms['f61_sham'] = dict(payload=payload(m @ W61.T))
    arms['f61_R3'] = dict(payload=composed(W61, B61, H61, Ye @ P61)[0])
    arms['f61_R1'] = dict(payload=composed(W61, B61, H61, Ye @ A61)[0])
    arms['f61_R2'] = dict(payload=composed(W61, B61, H61, Yp @ P61)[0])
    arms['d822_sham'] = dict(payload=payload(m @ W822.T))
    arms['d822_R1'] = dict(payload=composed(W822, B822, H822, Ye @ A822)[0])
    arms['d822_R2'] = dict(payload=composed(W822, B822, H822, Yp @ A822)[0])

    capture61 = hora.capture('novel', 'original')['output'][:, 3:5]
    capture822 = load(ALIGN / 'run/trained_novel/prefill_correct.pt')['first']['capture']['output'][:, 3:5]
    state822 = load(ALIGN / 'run/candidate.pt')
    state61 = load(COMP / 'run/candidate.pt')
    base = load(hora.c.CANDIDATE)[NAME]
    checks = {
        'f61_sham_eq_capture': float((arms['f61_sham']['payload'] - capture61).abs().max()),
        'd822_sham_eq_capture': float((arms['d822_sham']['payload'] - capture822).abs().max()),
        'd822_weight_reconstruction': float((h822.weight(theta822, coord822) - state822[NAME]).abs().max()),
        'f61_weight_reconstruction': float((h822.weight(theta61, dict(base=base.clone(), input_basis=B61, output_basis=Ro61)) - state61[NAME]).abs().max()),
    }
    for arm in ARMS:
        assert arms[arm]['payload'].shape == (32, 2, 960) and arms[arm]['payload'].dtype == torch.float32, arm
    return dict(case=case, info=info, expected=expected, features=features, train_ids=train_ids,
                Y=Y, Ye=Ye, Yp=Yp, m=m, d0=d0, A822=A822, A61=A61, W822=W822, W61=W61,
                B822=B822, B61=B61, H822=H822, H61=H61, P61=P61,
                arms=arms, checks=checks, capture61=capture61, capture822=capture822)


def score(forward, ids, bank, payload):
    first = b.score(forward, ids, bank, payload)
    repeat = b.score(forward, ids, bank, payload)
    for key in ['token_lp', 'lp', 'prediction']:
        assert torch.equal(first[key], repeat[key]), key
    return first


def readout(out, expected, targets, tokens):
    preds = out['prediction']
    rows = []
    for i, name in enumerate(expected):
        p, t = int(preds[i]), int(targets[i])
        rows.append(dict(row=i, id=name, prediction=b.IDS[p], target=b.IDS[t],
                         first_token=int(tokens[p, 0]), target_first=int(tokens[t, 0]),
                         first_token_correct=bool(tokens[p, 0] == tokens[t, 0]), joint=bool(p == t)))
    per_id = {}
    for name in sorted(set(expected)):
        sel = [r for r in rows if r['id'] == name]
        per_id[name] = dict(predictions=[r['prediction'] for r in sel],
                            first_token_correct=int(sum(r['first_token_correct'] for r in sel)),
                            joint=int(sum(r['joint'] for r in sel)))
    first_argmax = out['token_lp'][:, :, 0].argmax(-1)
    first_argmax_correct = int(sum(int(tokens[int(first_argmax[i]), 0]) == int(tokens[int(targets[i]), 0])
                                   for i in range(len(expected))))
    return dict(rows=rows, per_id=per_id, joint=int((preds == targets).sum()),
                first_token_any=int(sum(r['first_token_correct'] for r in rows)),
                first_argmax_correct=first_argmax_correct)


def run(data):
    b.c.seed_all(SEED)
    case, info = data['case'], data['info']
    ids = torch.tensor(info['ids'])
    targets = torch.tensor(info['targets']['original'])
    tokens = torch.tensor(hora.inputs()['candidate_tokens'])
    model, tok = b.c.load('receiver')
    before = b.c.parameter_hash(model)
    assert before == '6a3e1782ebad58d1e9323d5eeaff191d112848b5f0058394a998d1ac54c492b3', before
    bank = b.AnswerBank(tok)
    assert bank.tokens.tolist() == hora.inputs()['candidate_tokens']
    boundary = seam.Boundary(model)
    result = dict(checks=dict(data['checks']), arms={}, cap_s=CAP_S, arms_order=ARMS)
    try:
        sealed61 = load(COMP / 'run/trained_novel/correct.pt')['first']
        sealed822 = load(ALIGN / 'run/trained_novel/correct.pt')['first']
        for arm in ['f61_sham', 'd822_sham']:
            out = score(boundary.forward, ids, bank, data['arms'][arm]['payload'])
            sealed = sealed61 if arm == 'f61_sham' else sealed822
            result['checks'][f'{arm}_sealed_replay'] = {
                k: float((out[k] - sealed[k]).abs().max()) for k in ['token_lp', 'lp']}
            result['checks'][f'{arm}_prediction_replay'] = bool(torch.equal(out['prediction'], sealed['prediction']))
        payload0 = data['arms']['f61_sham']['payload'][:1]
        fast = b.score(boundary.forward, ids[:1], bank, payload0)
        direct = b.direct_score(boundary.forward, ids[:1], bank, payload0)
        error = float((fast['lp'] - direct).abs().max())
        assert error <= 1e-4 and torch.equal(fast['prediction'], direct.argmax(-1))
        result['checks']['direct_parity_error'] = error
        for arm in ARMS:
            out = score(boundary.forward, ids, bank, data['arms'][arm]['payload'])
            torch.save(dict(payload=data['arms'][arm]['payload'], out={k: out[k] for k in ['token_lp', 'lp', 'prediction']}),
                       ROOT / 'run' / f'{arm}.pt')
            result['arms'][arm] = readout(out, data['expected'], targets, tokens)
            b.event('stage0_arm', arm=arm, joint=result['arms'][arm]['joint'],
                    first_any=result['arms'][arm]['first_token_any'])
    finally:
        boundary.close()
        assert not any(module._forward_hooks for module in model.modules())
        after = b.c.parameter_hash(model)
        assert after == before
        assert all(p.grad is None and not p.requires_grad for p in model.parameters())
        result['receiver_hash'] = dict(before=before, after=after, restored_exact=True, no_hooks=True)
    return result


def main(mode):
    start = time.monotonic()
    data = build()
    if mode == 'offline':
        print(json.dumps(dict(checks=data['checks'],
                              A822_norm=float(data['A822'].norm()), A61_norm=float(data['A61'].norm()),
                              payload_norms={a: float(data['arms'][a]['payload'].norm()) for a in ARMS}), indent=2))
        return
    (ROOT / 'run').mkdir(exist_ok=True)
    result = run(data)
    result['elapsed_s'] = time.monotonic() - start
    write(ROOT / 'run/RESULT.json', result)
    print(json.dumps({k: result[k] for k in ['checks']}, indent=2))
    for arm in ARMS:
        print(arm, json.dumps(result['arms'][arm]['per_id'], sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['offline', 'run'])
    main(parser.parse_args().mode)
