"""E3 confirmatory run. Stages: freeze | build | capture | score | text | analyze.

    python -B e3.py freeze     # hash code, checkpoints, dev data; refuses if FINAL data exists
    python -B e3.py build      # FINAL splits (only after FREEZE, with every frozen hash unchanged)
    python -B e3.py capture    # Qwen3-1.7B block-13 states for FINAL tables
    python -B e3.py score      # all arms, all members, all FINAL splits -> outputs/<split>.pt
    python -B e3.py text       # text baselines on FINAL splits -> outputs/text.json
    python -B e3.py analyze    # decision + reports -> RESULT.json
See PREREGISTRATION.md. audit.py recomputes the decision independently.
"""
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PROG = HERE.parent / 'latent_anchor_program_20260930'
sys.path.insert(0, str(PROG))
sys.path.insert(0, str(PROG / 'E0_data'))
import lab  # noqa: E402
import make_splits as ms  # noqa: E402

DATA, CAPS, OUT = HERE / 'data', HERE / 'captures', HERE / 'outputs'
FREEZE = HERE / 'FREEZE.json'
ARMS = {
    'primary': dict(kind='factored', files=[f'E3R_factored_ridge_k4w64_lr0.003_s{s}_g1_best.pt' for s in (1, 2, 3)]),
    'secondary': dict(kind='lineslots', files=[f'E3C_lineslots_k4w64_lr0.001_s{s}_g1_best.pt' for s in (1, 2, 3)]),
}
CONTROLS = ['correct', 'shuffled', 'wrong_binding', 'zero', 'random_71001', 'random_71002']
FINAL = ['final_novel_pure', 'final_mixed', 'final_known']
CODE = [PROG / 'lab.py', PROG / 'sweep.py', PROG / 'E0_data/make_splits.py', PROG / 'E0_data/capture.py',
        PROG / 'E0_data/text_arms.py', HERE / 'e3.py', HERE / 'audit.py', HERE / 'PREREGISTRATION.md']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def frozen_inputs():
    files = {str(p.relative_to(HERE.parent)): sha(p) for p in CODE}
    for arm in ARMS.values():
        for f in arm['files']:
            p = PROG / 'runs' / f
            files[str(p.relative_to(HERE.parent))] = sha(p)
    for name in ms.ORDER:
        for p in [PROG / f'E0_data/splits/{name}.json', PROG / f'E0_data/captures/{name}.pt']:
            files[str(p.relative_to(HERE.parent))] = sha(p)
    return files


def freeze():
    assert not DATA.exists() and not CAPS.exists() and not OUT.exists(), 'FINAL data already exists'
    assert not FREEZE.exists(), 'already frozen'
    FREEZE.write_text(json.dumps(dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                      files=frozen_inputs(), sender_hash=lab.SENDER_HASH,
                                      receiver_hash=lab.RECEIVER_HASH, final_novel56=ms.FINAL_NOVEL56,
                                      arms=ARMS), indent=1) + '\n')
    print('frozen', len(json.loads(FREEZE.read_text())['files']), 'files')


def check_frozen():
    fz = json.loads(FREEZE.read_text())
    now = frozen_inputs()
    bad = [k for k in fz['files'] if fz['files'][k] != now.get(k)]
    assert not bad, f'frozen inputs changed: {bad}'
    return fz


def build():
    check_frozen()
    assert not DATA.exists()
    used = set()
    for name in ms.ORDER:                      # regenerate dev splits; must be byte-identical
        data = ms.build(name, used)
        blob = (json.dumps(data, indent=1) + '\n').encode()
        assert hashlib.sha256(blob).hexdigest() == sha(PROG / f'E0_data/splits/{name}.json'), name
    ms.BALANCED['final_novel_pure'] = (125001, 1, ms.FINAL_NOVEL56)
    ms.BALANCED['final_known'] = (124001, 1, ms.TRAINED56)
    ms.MIXED['final_mixed'] = (125002, ms.FINAL_NOVEL56, ms.TRAINED56)
    ms.UNSEEN.update(final_novel_pure=ms.FINAL_NOVEL56, final_mixed=ms.FINAL_NOVEL56, final_known=[])
    DATA.mkdir()
    manifest = {}
    dev_ids = set(ms.OOB8) | set(ms.BURNED)
    for name in FINAL:
        data = ms.build(name, used)
        ids = {i for r in data['records'] for i in r['bindings'].values()}
        assert not ids & dev_ids, 'development-only ID in a FINAL split'
        for rec in data['records']:
            assert all(rec['bindings'][c] != rec['wrong_bindings'][c] for c in ms.COLORS)
        if name == 'final_novel_pure':
            assert ids == set(ms.FINAL_NOVEL56)
        if name == 'final_known':
            assert ids == set(ms.TRAINED56)
        if name == 'final_mixed':
            assert all(r['unseen_in_table'] == 1 for r in data['records'])
            assert len({(r['answer'], r['color']) for r in data['rows'] if r['novel_answer']}) == 224
        counts = {}
        if data['kind'] == 'balanced':
            for r in data['rows']:
                counts[(r['answer'], r['color'])] = counts.get((r['answer'], r['color']), 0) + 1
            assert set(counts.values()) == {1}
        p = DATA / f'{name}.json'
        p.write_text(json.dumps(data, indent=1) + '\n')
        manifest[name] = dict(sha256=sha(p), tables=len(data['records']), rows=len(data['rows']),
                              novel_answer_rows=sum(r['novel_answer'] for r in data['rows']))
        print(name, manifest[name])
    (DATA / 'MANIFEST.json').write_text(json.dumps(manifest, indent=1) + '\n')


def capture():
    check_frozen()
    assert DATA.exists() and not CAPS.exists()
    dev = lab.device()
    model, tok, digest = lab.load('sender', dev)
    CAPS.mkdir()
    grabbed = []
    handle = model.model.layers[lab.SENDER_LAYER].register_forward_hook(
        lambda m, a, o: grabbed.append(o.detach().float().cpu().clone()))
    manifest = {}
    try:
        for name in FINAL:
            recs = json.loads((DATA / f'{name}.json').read_text())['records']
            ids = lab.tokens(tok, [r['source'] for r in recs] + [r['wrong_source'] for r in recs])
            assert ids.shape == (2 * len(recs), lab.SOURCE_TOKENS)
            grabbed.clear()
            with torch.no_grad():
                for i in range(0, len(ids), 16):
                    model.model(input_ids=ids[i:i + 16].to(dev), use_cache=False)
            h = torch.cat(grabbed)
            n = len(recs)
            assert h.shape == (2 * n, lab.SOURCE_TOKENS, lab.SENDER_WIDTH) and torch.isfinite(h).all()
            torch.save(dict(hidden=h[:n], wrong_hidden=h[n:], source_ids=ids), CAPS / f'{name}.pt')
            manifest[name] = dict(sha256=sha(CAPS / f'{name}.pt'), tables=n)
            print('captured', name, n)
    finally:
        handle.remove()
    assert lab.parameter_hash(model.cpu()) == digest == lab.SENDER_HASH
    (CAPS / 'MANIFEST.json').write_text(json.dumps(manifest, indent=1) + '\n')


def load_final(name):
    lab.SPLITS_DIR, lab.CAPTURE_DIR = DATA, CAPS
    return lab.load_split(name)


def bridges(kind, files, dev):
    out = []
    for f in files:
        b = lab.Bridge(4, 64, factored=kind == 'factored', lineslots=kind == 'lineslots', gate_init=1.0)
        b.load_state_dict(torch.load(PROG / 'runs' / f, weights_only=True), strict=True)
        b.to(dev).eval()
        if b.factored is not None:
            b.factored.hard = True
        out.append(b)
    return out


def score():
    check_frozen()
    assert CAPS.exists() and not OUT.exists()
    OUT.mkdir()
    dev = lab.device()
    model, tok, rhash = lab.load('receiver', dev)
    bank = lab.AnswerBank(tok)
    rec = None
    for name in FINAL:
        s = load_final(name)
        ids = bank.prompts([r['query'] for r in s['data']['rows']])
        saved = dict(labels=s['labels'], wrong_labels=s['wrong_labels'], groups=s['groups'],
                     novel=s['novel'], novel_wrong=s['novel_wrong'], candidate_tokens=bank.tokens)
        for arm, spec in ARMS.items():
            saved[arm] = []
            for b in bridges(spec['kind'], spec['files'], dev):
                if rec is None:
                    rec = lab.Receiver(model, b)
                rec.bridge = b
                values, meta = lab.memories(b, s, dev, CONTROLS)
                saved['donor_index'], saved['positional'] = meta['donor_index'], meta['positional']
                saved[arm].append({c: lab.score_cached(rec, ids, bank, values[c], dev=dev)['token_lp'] for c in CONTROLS})
                print(name, arm, 'member done', flush=True)
        saved['receiver_only'] = lab.score_cached(rec, ids, bank, None, dev=dev)['token_lp']
        torch.save(saved, OUT / f'{name}.pt')
    assert lab.parameter_hash(model.cpu()) == rhash == lab.RECEIVER_HASH
    print('scored')


def text():
    check_frozen()
    import text_arms as ta
    dev = lab.device()
    data = {name: json.loads((DATA / f'{name}.json').read_text()) for name in FINAL}
    sent, _ = ta.sender_stage(data, dev)
    copy, _, _, _ = ta.receiver_stage(data, sent, dev)
    out = {}
    for name, d in data.items():
        rows = d['rows']
        o = {}
        for kind, key in [('original', 'answer'), ('changed', 'wrong_answer')]:
            preds = sent[name][kind]['prediction']
            gold = [r[key] for r in rows]
            o[kind] = dict(sender=sum(p == g for p, g in zip(preds, gold)) / len(rows),
                           actual_text=sum(copy[p]['prediction'] == g for p, g in zip(preds, gold)) / len(rows),
                           oracle_text=sum(copy[g]['prediction'] == g for g in gold) / len(rows))
        nv = [i for i, r in enumerate(rows) if r['novel_answer']]
        if nv:
            preds = sent[name]['original']['prediction']
            o['novel_rows_actual_text'] = sum(copy[preds[i]]['prediction'] == rows[i]['answer'] for i in nv) / len(nv)
        out[name] = o
        print(name, o)
    (OUT / 'text.json').write_text(json.dumps(out, indent=1) + '\n')


def collision(data):
    recs = data['records']
    res = []
    for r in data['rows']:
        others = [i for i in recs[r['group']]['bindings'].values() if i != r['answer']]
        res.append(any(i[0] == r['answer'][0] or i[1] == r['answer'][1] for i in others))
    return torch.tensor(res)


def ensemble(members, control):
    lp = torch.stack([m[control].sum(-1) for m in members])
    return torch.logsumexp(lp, 0) - math.log(len(members))


def bootstrap(correct_by_row, groups, seed=125099, n=10000):
    g = groups.tolist()
    tables = sorted(set(g))
    per = {t: [] for t in tables}
    for v, t in zip(correct_by_row.tolist(), g):
        per[t].append(v)
    rng = random.Random(seed)
    stats = []
    for _ in range(n):
        pick = [rng.choice(tables) for _ in tables]
        vals = [v for t in pick for v in per[t]]
        stats.append(sum(vals) / len(vals))
    stats.sort()
    return [stats[int(0.025 * n)], stats[int(0.975 * n) - 1]]


def metrics(saved, lp_by_control, data, mask):
    labels, wrong, donor = saved['labels'], saved['wrong_labels'], saved['donor_index']
    same = labels[donor] == labels
    pred = {c: lp.argmax(-1) for c, lp in lp_by_control.items()}
    coll = collision(data)
    m = {}
    m['joint'] = (pred['correct'] == labels)[mask].float().mean().item()
    m['joint_ci95'] = bootstrap((pred['correct'] == labels)[mask].float(), saved['groups'][mask])
    wmask = saved['novel_wrong'] if not bool(mask.all()) else mask
    m['changed_following'] = (pred['wrong_binding'] == wrong)[wmask].float().mean().item()
    m['changed_ci95'] = bootstrap((pred['wrong_binding'] == wrong)[wmask].float(), saved['groups'][wmask])
    m['donor_following'] = (pred['shuffled'] == labels[donor])[mask & ~same].float().mean().item()
    m['shuffled'] = (pred['shuffled'] == labels)[mask & ~same].float().mean().item()
    for c in ['zero', 'random_71001', 'random_71002', 'receiver_only']:
        m[c] = (pred[c] == labels)[mask].float().mean().item()
    m['joint_collision'] = (pred['correct'] == labels)[mask & coll].float().mean().item() if (mask & coll).any() else None
    m['joint_no_collision'] = (pred['correct'] == labels)[mask & ~coll].float().mean().item() if (mask & ~coll).any() else None
    m['collision_rows'] = int((mask & coll).sum())
    recs = data['records']
    m['in_table'] = sum(lab.IDS[int(p)] in recs[int(g)]['bindings'].values()
                        for p, g in zip(pred['correct'][mask], saved['groups'][mask])) / int(mask.sum())
    m['rows'] = int(mask.sum())
    return m


def decide(m):
    return (m['joint'] >= 0.75 and m['changed_following'] >= 0.75 and m['shuffled'] <= 0.10
            and all(m[c] <= 0.10 for c in ['zero', 'random_71001', 'random_71002', 'receiver_only']))


def analyze():
    check_frozen()
    result = dict(splits={})
    for name in FINAL:
        saved = torch.load(OUT / f'{name}.pt', weights_only=True)
        data = json.loads((DATA / f'{name}.json').read_text())
        mask = saved['novel'] if 'mixed' in name else torch.ones(len(saved['labels']), dtype=torch.bool)
        res = {}
        for arm in ARMS:
            members = saved[arm]
            ens = {c: ensemble(members, c) for c in CONTROLS}
            ens['receiver_only'] = saved['receiver_only'].sum(-1)
            res[arm] = dict(ensemble=metrics(saved, ens, data, mask))
            res[arm]['members'] = []
            for mbr in members:
                lp = {c: mbr[c].sum(-1) for c in CONTROLS}
                lp['receiver_only'] = saved['receiver_only'].sum(-1)
                res[arm]['members'].append(metrics(saved, lp, data, mask))
            res[arm]['decision_rule_pass'] = decide(res[arm]['ensemble'])
        result['splits'][name] = res
    p = result['splits'][FINAL[0]]
    result['PRIMARY_H3'] = 'PASS' if p['primary']['decision_rule_pass'] else 'FAIL'
    result['SECONDARY_freeform_descriptive'] = 'PASS' if p['secondary']['decision_rule_pass'] else 'FAIL'
    if (OUT / 'text.json').exists():
        result['text'] = json.loads((OUT / 'text.json').read_text())
    (HERE / 'RESULT.json').write_text(json.dumps(result, indent=1) + '\n')
    for name, res in result['splits'].items():
        for arm, r in res.items():
            e = r['ensemble']
            print(f"{name:17s} {arm:9s} joint {e['joint']:.3f} {e['joint_ci95']} changed {e['changed_following']:.3f} "
                  f"shuffled {e['shuffled']:.3f} zero {e['zero']:.3f} rand {e['random_71001']:.3f}/{e['random_71002']:.3f} "
                  f"recv {e['receiver_only']:.3f} coll {e['joint_collision']} nocoll {e['joint_no_collision']} "
                  f"members {[round(m['joint'], 3) for m in r['members']]}")
    print('PRIMARY H3:', result['PRIMARY_H3'], '| secondary (descriptive):', result['SECONDARY_freeform_descriptive'])


if __name__ == '__main__':
    {'freeze': freeze, 'build': build, 'capture': capture, 'score': score, 'text': text, 'analyze': analyze}[sys.argv[1]]()
