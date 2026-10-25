#!/usr/bin/env python
"""Harness of the format-gap test: a finer text format of the probe values against soft vectors (closed-book
TriviaQA). Design: ../../../reports/format_gap_design_20261023.md (below: "the design"). The fixed choices G1 to G40 that
the comments cite are in PREREGISTRATION.md.

The sealed confirmatory run ../latent_uncertainty_confirm_20261021/ is read-only. This file imports its uconfirm.py as
a library and changes no file there. Run every mode with python -I -B, so that no .pyc file goes into the sealed folder.

Modes (one per call):
  test       the unit tests: synthetic data and tiny random models on the CPU; no model load, no data, no network
  prefreeze  FREEZE_D.json: the unit tests, the hashes of the frozen files, the seal check of the sealed inputs (CPU)
  dev        Phase D' (scratch mode), stage gate: receiver passes on the 3,000 dev items for F3 and the six candidate
             formats, readouts, the gates G-F2 and G-F1, the selection -> dev/dev_selection.json (GPU, then CPU)
  confirm    Phase C' (confirmatory mode). Every stage needs the decision 'go' in dev/dev_selection.json:
               fetch     train shard 1 into ./data (network; [researcher]'s words go in --researcher-said-yes)
               items     fresh/fresh_items.json and fresh/fresh_manifest.json (CPU)
               freeze    FREEZE.json (CPU)
               sender    the sender on one pool of fresh items, chunks of 250 that resume (GPU; --pool, --chunks)
               seeds     probes, text passes, L-probe and L for the given seeds (GPU; --seeds, --t0c)
               analysis  readouts, metrics, two-level bootstrap, reading rule -> confirm/results.json (CPU; the only
                         stage that computes a fresh-item label)

Run the model stages only through run_ugap.sh (machine checks, caffeinate, power guard, time limit).
"""
import argparse
import json
import math
import os
import platform
import sys
import time
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SEALED = HERE.parent / 'latent_uncertainty_confirm_20261021'
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(SEALED))
import uconfirm as U  # noqa: E402  (sealed library; it does not run anything on import)

fetch_data = U.fetch_data
STATUS = 'PREREGISTERED (format-gap test, 2026-10-23)'
DEVICE = torch.device('mps')

# ---------------------------------------------------------------- fixed values (design sections 3, 4, 7, 9, 10)
DEV_SEED = U.DEV_SEED                                    # 2026102113, the Phase D seed (design 7, step 3)
TEXT_SEEDS = (20261101, 20261102, 20261103)
LP_SEEDS = TEXT_SEEDS + (20261104, 20261105, 20261106)  # G12: six seeds for L-probe and L-probe-R
BOOT_SEED, N_BOOT, BOOT_CHUNK = 20261023, 10_000, 250
S1_PERM_SEED = 2026102320                                # design 4, rule 2
POOLS = ('V', 'S0', 'S1')
POOL_SPLIT_INDEX = dict(V=4, S0=5, S1=6)                 # design 5: sample stream [2026102112, k, chunk]
N_S1 = 4000                                              # Plan M (design 4, rule 4)
EXPECTED_KEPT = dict(V=1991, S0=1838, S1=N_S1)           # G7: design 4 counts; a difference stops the item build
CHUNK = U.CHUNK                                          # 250
F3_DEV_TEST_REF, F2_TOL = 0.40904, 0.001                 # G-F2 (design 7)
GAIN_MIN, TIE = 0.005, 0.001                             # G-F1 and the tie rule (design 7)
TAG_MAX_TOKENS = 11                                      # design 3, length rule (G4)
CLOSE_MEAN, CLOSE_HI = 0.005, 0.010                      # design 10.1
REF = 'F3'
CANDIDATES = ('D2', 'D3', 'LOs', 'ICD7', 'WD2', 'ICD7D2')
FORMATS = (REF,) + CANDIDATES
FAMILY = dict(D2='digits', D3='digits', LOs='digits', ICD7='words', WD2='words', ICD7D2='words')   # design 7
ICD7_LEVELS = [(0.05, 'almost no chance'), (0.20, 'very unlikely'), (0.45, 'unlikely'), (0.55, 'roughly even chance'),
               (0.80, 'likely'), (0.95, 'very likely'), (math.inf, 'almost certain')]
C_STAR = 'L-probe-R'
L_SPEC, L_EPOCHS = dict(kind='L', rank=64, lr=3e-4), 2              # sealed dev/selection.json
LP_SPEC, LP_EPOCHS = dict(kind='L-probe', rank=None, lr=1e-3), 3
RECV_PARAM_SHA = 'fe5f214635a19e4d8052924e712fb52e517279e8161d212e8d1e60d0ee9628f4'
FROZEN_FILES = ['PREREGISTRATION.md', 'ugap.py', 'test_ugap.py', 'run_ugap.sh', 'guard.sh', 'chain.sh']
SEALED_INPUTS = (['uconfirm.py', 'fetch_data.py', 'items/items.json', 'items/manifest.json', 'dev/selection.json',
                  'dev/probes.npz', 'dev/text.json', 'dev/text_probs_eval.npz', 'dev/recv/T4_F3_dev_train.npz',
                  'dev/recv/T4_F3_dev_test.npz', 'confirm/sender_train.json', 'confirm/sender_train.npz',
                  'confirm/hashes.jsonl', 'data/trivia_qa/rc.wikipedia/train-00000-of-00007.parquet',
                  'data/trivia_qa/rc.wikipedia/validation-00000-of-00001.parquet']
                 + [f'dev/sender_{s}.{e}' for s in ('dev_train', 'dev_test') for e in ('json', 'npz')]
                 + [f'confirm/seed{s}/{f}' for s in TEXT_SEEDS for f in ('probes.npz', 'latent.json',
                                                                           'latent_L-probe.npz')])
_GUARD = dict(fresh_labels=False)   # only the analysis stage, after the freeze check, sets this (G23)


def log(msg):
    U.log(msg)


def sha(path):
    return fetch_data.sha256_file(Path(path))


def code_sha():
    return sha(Path(__file__))


def save_npz(path, **arrays):
    """Write to a temporary name, then rename: a stop during the write leaves no broken file under the final name."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.part')
    with open(tmp, 'wb') as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)


def empty_cache():
    if DEVICE.type == 'mps':
        torch.mps.empty_cache()


# ================================================================ text formats (design section 3; G1 to G4)

def level(p, levels):
    return next(i for i, (t, _) in enumerate(levels) if p < t)


def logit_text(p):
    q = min(max(float(p), U.P_CLIP), 1 - U.P_CLIP)
    return f'{math.log(q / (1 - q)):.1f}'


def fmt_value(p, fmt):
    """The text of v for the probe probability p (design 3)."""
    p = float(p)
    if fmt == 'F3':
        return U.fmt_value(p, 'F3')
    if fmt == 'D2':
        return f'{p:.2f}'
    if fmt == 'D3':
        return f'{p:.3f}'
    if fmt == 'LOs':
        return f'logit {logit_text(p)}'
    if fmt == 'ICD7':
        return ICD7_LEVELS[level(p, ICD7_LEVELS)][1]
    if fmt == 'WD2':
        return f"{U.fmt_value(p, 'F3')}, {p:.2f}"
    if fmt == 'ICD7D2':
        return f"{fmt_value(p, 'ICD7')}, {p:.2f}"
    raise ValueError(fmt)


def fmt_feature(p, fmt):
    """The information that the text keeps, as features for c_round (design 2.1; G29): the number that the text shows,
    a one-hot of the word level, or both."""
    p = float(p)
    if fmt == 'F3':
        return U.fmt_feature(p, 'F3')
    if fmt in ('D2', 'D3'):
        return [float(fmt_value(p, fmt))]
    if fmt == 'LOs':
        return [1 / (1 + math.exp(-float(logit_text(p))))]
    if fmt == 'ICD7':
        k = level(p, ICD7_LEVELS)
        return [float(i == k) for i in range(len(ICD7_LEVELS))]
    if fmt == 'WD2':
        return fmt_feature(p, 'F3') + fmt_feature(p, 'D2')
    if fmt == 'ICD7D2':
        return fmt_feature(p, 'ICD7') + fmt_feature(p, 'D2')
    raise ValueError(fmt)


def tag(p, fmt):
    return f' ({fmt_value(p, fmt)})'


def message(q, cands, ps, fmt):
    """The T4 layout (design 3): the text around the candidate tags does not change; only v changes."""
    return U.join_tags(U.cand_segments(q, cands), [tag(p, fmt) for p in ps])


def messages(fmt, items, sj, P=None):
    """All receiver texts of one format for one split. 'T0c': the candidate strings without values."""
    if fmt == 'T0c':
        return [U.msg_T0c(x['question'], r['cands']) for x, r in zip(items, sj)]
    return [message(x['question'], r['cands'], [float(v) for v in P[i, :len(r['cands'])]], fmt)
            for i, (x, r) in enumerate(zip(items, sj))]


def token_counts(tok, fmt, items, sj, P):
    """G4: the receiver tokens of each tag ' (v)' (tokenized alone, no special tokens) over all present slots, and of
    each full message."""
    ids = lambda t: len(tok(t, add_special_tokens=False).input_ids)
    tags = [] if fmt == 'T0c' else [ids(tag(float(P[i, j]), fmt)) for i, r in enumerate(sj) for j in range(len(r['cands']))]
    msgs = [ids(t) for t in messages(fmt, items, sj, P)]
    return dict(tag_mean=float(np.mean(tags)) if tags else 0.0, tag_max=int(max(tags)) if tags else 0,
                message_mean=float(np.mean(msgs)), message_max=int(max(msgs)), slots=len(tags), messages=len(msgs))


# ================================================================ selection and the dev gate (design 7; G5, G6)

def select_format(pooled, tag_mean, tag_max, pool=CANDIDATES):
    """The lowest pooled dev NLL among the eligible formats of pool. Tie: all formats within TIE of the best; of
    these, the fewest mean tag tokens, then the lower pooled NLL, then the order of CANDIDATES."""
    elig = [f for f in pool if tag_max[f] <= TAG_MAX_TOKENS]
    if not elig:
        return None, []
    best = min(pooled[f] for f in elig)
    tied = [f for f in elig if pooled[f] <= best + TIE]
    return min(tied, key=lambda f: (tag_mean[f], pooled[f], CANDIDATES.index(f))), tied


def gate_decision(res, tokens):
    """G-F2, then G-F1, then the decision. res: format -> dict(dev_test_nll, pooled_nll)."""
    pooled = {f: res[f]['pooled_nll'] for f in FORMATS}
    tm = {f: tokens[f]['tag_mean'] for f in FORMATS}
    tx = {f: tokens[f]['tag_max'] for f in FORMATS}
    fstar, tied = select_format(pooled, tm, tx)
    fam = FAMILY[fstar] if fstar else None
    fo, _ = select_format(pooled, tm, tx, [f for f in CANDIDATES if FAMILY[f] != fam]) if fstar else (None, [])
    f3 = res[REF]['dev_test_nll']
    g2 = dict(value=f3, reference=F3_DEV_TEST_REF, tolerance=F2_TOL, pass_=bool(abs(f3 - F3_DEV_TEST_REF) <= F2_TOL),
              rule='dev-test NLL of F3 = 0.40904 +- 0.001 (Phase D value)')
    gain = pooled[REF] - pooled[fstar] if fstar else float('nan')
    g1 = dict(value=gain, threshold=GAIN_MIN, pass_=bool(fstar is not None and gain >= GAIN_MIN),
              rule='pooled dev NLL(F3) - pooled dev NLL(F*) >= 0.005')
    if not g2['pass_']:
        decision, outcome = 'stop', 'G-F2 failed: the pipeline does not repeat (design 7)'
    elif not g1['pass_']:
        decision, outcome = 'stop', 'outcome G: G-F1 failed; no fresh item is used (design 7, 12)'
    else:
        decision, outcome = 'go', 'G-F2 and G-F1 pass: Plan M runs ([researcher], 2026-10-23, about 20:30)'
    rn = lambda d: {('pass' if k == 'pass_' else k): v for k, v in d.items()}
    return dict(F_star=fstar, F_other=fo, family_star=fam, tied=tied, eligible=[f for f in CANDIDATES if tx[f] <= TAG_MAX_TOKENS],
                pooled_nll=pooled, gain_star=gain, gates={'G-F2': rn(g2), 'G-F1': rn(g1)}, decision=decision,
                outcome=outcome)


# ================================================================ hashes, freeze and seal checks (G24 to G27)

def weight_roots(sender=True):
    return ([U.SENDERS['main']['path']] if sender else []) + [U.RECV]


def record_hashes(out, when, stage, roots):
    rec = dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), when=when, stage=stage,
               files={str(f): sha(f) for root in roots for f in sorted(Path(root).glob('*.safetensors'))})
    Path(out).mkdir(parents=True, exist_ok=True)
    with open(Path(out) / 'hashes.jsonl', 'a') as fh:
        fh.write(json.dumps(rec) + '\n')
    log(f'weight hashes ({when}): {len(rec["files"])} files')
    return rec


def sealed_weight_hashes():
    out = {}
    for line in (SEALED / 'confirm' / 'hashes.jsonl').read_text().splitlines():
        if line.strip():
            for f, h in json.loads(line)['files'].items():
                out.setdefault(f, set()).add(h)
    return {f: next(iter(h)) for f, h in out.items() if len(h) == 1}


def weights_ok(out, need_files):
    """V2, file part: at least two records; each record holds need_files files; each hash equals the confirmatory one."""
    p = Path(out) / 'hashes.jsonl'
    recs = [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
    ref = sealed_weight_hashes()
    bad = sorted({f for r in recs for f, h in r['files'].items() if ref.get(f) != h})
    short = [r['time'] for r in recs if len(r['files']) != need_files]
    ok = bool(len(recs) >= 2 and not bad and not short)
    return ok, dict(records=len(recs), changed_or_unknown=bad, records_with_missing_files=short)


def seal_check(files=None):
    """The sha256 of each sealed input file against SEAL.json (design 8, step 2)."""
    seal = json.loads((SEALED / 'SEAL.json').read_text())['sha256']
    files = SEALED_INPUTS if files is None else files
    bad = {f: dict(sealed=seal.get(f), now=sha(SEALED / f) if (SEALED / f).exists() else None) for f in files
           if seal.get(f) is None or not (SEALED / f).exists() or sha(SEALED / f) != seal[f]}
    return dict(ok=not bad, files=len(files), bad=bad)


def run_tests():
    import test_ugap
    res = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(test_ugap))
    return res, dict(run=res.testsRun, failures=len(res.failures), errors=len(res.errors))


def stage_prefreeze(a=None):
    fz = HERE / 'FREEZE_D.json'
    if fz.exists():
        raise SystemExit('FREEZE_D.json exists. A new freeze needs a new, dated section and [researcher]\'s word.')
    missing = [f for f in FROZEN_FILES if not (HERE / f).exists()]
    if missing:
        raise SystemExit(f'prefreeze: missing {missing}')
    res, info = run_tests()
    if not res.wasSuccessful():
        raise SystemExit('prefreeze: the unit tests fail')
    sc = seal_check()
    if not sc['ok']:
        raise SystemExit(f'prefreeze: sealed inputs differ from SEAL.json: {sc["bad"]}')
    U.write_json(fz, dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), sha256={f: sha(HERE / f) for f in FROZEN_FILES},
                          tests=info, seal_check=sc, sealed_root_sha256=json.loads((SEALED / 'SEAL.json').read_text())['root_sha256'],
                          rule='Every stage checks these hashes before it starts and stops on a difference (G24).'))
    log(f'prefreeze: FREEZE_D.json written ({info["run"]} tests pass; {sc["files"]} sealed inputs equal SEAL.json)')


def check_freeze_d():
    fz = U.read_json(HERE / 'FREEZE_D.json')
    if fz is None:
        raise SystemExit('FREEZE_D.json is missing: run the prefreeze mode first')
    bad = [f for f in FROZEN_FILES if fz['sha256'].get(f) != sha(HERE / f)]
    if bad:
        raise SystemExit(f'frozen files changed after FREEZE_D.json: {bad}')
    return fz


def freeze_hashes():
    s = {f: sha(HERE / f) for f in FROZEN_FILES}
    for f in ('fresh/fresh_items.json', 'fresh/fresh_manifest.json', 'dev/dev_selection.json', 'dev/dev_result.json',
              'data/trivia_qa/' + fetch_data.FILES['train1'][0]):
        s[f] = sha(HERE / f)
    return s


def check_freeze():
    check_freeze_d()
    fz = U.read_json(HERE / 'FREEZE.json')
    if fz is None:
        raise SystemExit('FREEZE.json is missing: run the freeze stage (job F) first')
    now = freeze_hashes()
    bad = [k for k, v in now.items() if fz['sha256'].get(k) != v]
    if bad:
        raise SystemExit(f'frozen files changed after FREEZE.json: {bad}')
    return fz


def require_go():
    sel = U.read_json(HERE / 'dev' / 'dev_selection.json')
    if sel is None or sel.get('decision') != 'go':
        raise SystemExit('Phase C\' needs the decision "go" in dev/dev_selection.json (gates G-F2 and G-F1)')
    return sel


# ================================================================ sealed inputs

def sealed_items():
    return json.loads((SEALED / 'items' / 'items.json').read_text())


def sealed_n_test():
    return int(json.loads((SEALED / 'dev' / 'selection.json').read_text())['n_test'])


def train_data():
    """The train split of the confirmatory run (4,000 items) with its cached sender outputs (design 5)."""
    its = sealed_items()['train']
    sz, sj = U.load_sender_out(SEALED / 'confirm', 'train')
    assert len(its) == len(sj) == len(sz['n_cand'])
    return dict(items=its, sj=sj, sz=sz, y=U.cand_labels(its, sj, 'train'), m=U.cand_mask(sz),
                S=U.block_states(sz, 'main'))


def fresh_items():
    return json.loads((HERE / 'fresh' / 'fresh_items.json').read_text())['items']


def fresh_data():
    its = fresh_items()
    sz, sj = U.load_sender_out(HERE / 'confirm', 'fresh')
    assert len(its) == len(sj) == len(sz['n_cand'])
    return dict(items=its, sj=sj, sz=sz, m=U.cand_mask(sz), S=U.block_states(sz, 'main'),
                pool=np.array([x['pool'] for x in its]))


def fresh_labels(items, sj):
    """y_j for each candidate of the fresh items (n, 3); -1 when absent. Only the analysis stage may call this (G23)."""
    if not _GUARD['fresh_labels']:
        raise RuntimeError('fresh-item labels are read only by the analysis stage, after the freeze check (G23)')
    y = np.full((len(items), 3), -1, dtype=np.int64)
    for i, (x, r) in enumerate(zip(items, sj)):
        for j, c in enumerate(r['cands']):
            y[i, j] = U.label_alias(c, x['aliases'])
    return y


# ================================================================ Phase D' (design 7)

def dev_splits():
    items = sealed_items()
    P = np.load(SEALED / 'dev' / 'probes.npz')
    sp = {}
    for split, key in (('dev_train', 'logistic_fit'), ('dev_test', 'logistic_eval')):
        sz, sj = U.load_sender_out(SEALED / 'dev', split)
        its = items[split]
        assert len(its) == len(sj) == len(sz['n_cand']) == len(P[key])
        sp[split] = dict(items=its, sj=sj, sz=sz, P=P[key], y=U.cand_labels(its, sj, split)[:, 0])
    return sp


def dev_readouts(sp, out):
    """Design 7, steps 3 and 4. Returns format -> values, and format -> pooled item NLL (3,000 items: the out-of-fold
    dev-train values, then the dev-test values of the full dev-train fit)."""
    res, nl = {}, {}
    ytr, yte = sp['dev_train']['y'], sp['dev_test']['y']
    for fmt in FORMATS:
        A = {s: np.load(Path(out) / 'recv' / f'{fmt}_{s}.npz') for s in sp}
        oof, pte, info = U.cv_logistic(A['dev_train']['h'].astype(np.float64), ytr, A['dev_test']['h'].astype(np.float64),
                                       DEV_SEED)
        a, b = U.item_nll(oof, ytr), U.item_nll(pte, yte)
        nl[fmt] = np.concatenate([a, b])
        res[fmt] = dict(dev_test_nll=float(b.mean()), dev_train_oof_nll=float(a.mean()), pooled_nll=float(nl[fmt].mean()),
                        readout=info, dev_test_metrics=U.metrics(pte, yte))
        log(f'dev {fmt}: dev-test NLL {b.mean():.5f}; pooled NLL {nl[fmt].mean():.5f}')
    return res, nl


def stage_gate(a):
    check_freeze_d()
    out = HERE / 'dev'
    if (out / 'dev_selection.json').exists():
        raise SystemExit("dev/dev_selection.json exists: Phase D' ran already")
    t0 = time.time()
    sp = dev_splits()
    tok, R, yn = U.load_receiver(DEVICE)
    record_hashes(out, 'start', 'gate', weight_roots(sender=False))
    h0 = U.param_hash(R)
    times = {}
    for fmt in FORMATS:
        for split, d in sp.items():
            path = out / 'recv' / f'{fmt}_{split}.npz'
            if path.exists():
                continue
            texts = messages(fmt, d['items'], d['sj'], d['P'])
            t1 = time.time()
            H, Z = U.receiver_text(tok, R, texts, yn, tok.pad_token_id, DEVICE, a.recv_bs)
            save_npz(path, h=H.astype(np.float16), z=Z)
            times[f'{fmt}_{split}_s_per_pass'] = (time.time() - t1) / len(texts)
            log(f'dev {fmt} {split}: {len(texts)} passes ({time.time() - t1:.0f} s)')
    tokens = {}
    for fmt in FORMATS + ('T0c',):
        c = [token_counts(tok, fmt, d['items'], d['sj'], d['P']) for d in sp.values()]
        tokens[fmt] = dict(tag_mean=float(np.average([x['tag_mean'] for x in c], weights=[max(x['slots'], 1) for x in c])),
                           tag_max=max(x['tag_max'] for x in c), message_max=max(x['message_max'] for x in c),
                           message_mean=float(np.average([x['message_mean'] for x in c], weights=[x['messages'] for x in c])))
    d = sp['dev_test']
    examples = {fmt: messages(fmt, d['items'][:3], d['sj'][:3], d['P'][:3]) for fmt in FORMATS}
    h1 = U.param_hash(R)
    del R
    empty_cache()
    record_hashes(out, 'end', 'gate', weight_roots(sender=False))
    t_pass = time.time() - t0
    res, nl = dev_readouts(sp, out)
    dec = gate_decision(res, tokens)
    diffs = {f'F3 minus {f}': nl[REF] - nl[f] for f in CANDIDATES}
    ci = U.bootstrap_ci(diffs, N_BOOT, BOOT_SEED)
    repro = {}
    for split in sp:
        new = np.load(out / 'recv' / f'F3_{split}.npz')['h']
        old = np.load(SEALED / 'dev' / 'recv' / f'T4_F3_{split}.npz')['h']
        repro[split] = dict(max_abs_h_diff=float(np.abs(new.astype(np.float32) - old.astype(np.float32)).max()),
                            share_equal=float((new == old).mean()))
    ok_w, w_info = weights_ok(out, need_files=1)
    np.savez(out / 'dev_item_nll.npz', **nl)
    result = dict(status="DEVELOPMENT (Phase D', scratch mode). Not evidence (design 7).", time=time.strftime('%Y-%m-%d %H:%M:%S'),
                  formats=res, tokens=tokens, gain_ci95_item_bootstrap={k: dict(mean=float(v.mean()), ci95=list(ci[k]))
                                                                        for k, v in diffs.items()},
                  reproduction_of_phase_d_F3=repro, receiver_param_sha256=[h0, h1],
                  receiver_param_unchanged=bool(h0 == h1 == RECV_PARAM_SHA), weight_files=dict(ok=ok_w, **w_info),
                  examples=examples, seconds_passes=t_pass, seconds_total=time.time() - t0, s_per_pass=times,
                  code_sha256=code_sha(), dev_seed=DEV_SEED)
    U.write_json(out / 'dev_result.json', result)
    sel = dict(status="DEVELOPMENT (Phase D' selection). Not evidence.", time=time.strftime('%Y-%m-%d %H:%M:%S'),
               **dec, tag_tokens={f: tokens[f]['tag_mean'] for f in FORMATS}, code_sha256=code_sha(),
               prereg_sha256=sha(HERE / 'PREREGISTRATION.md'), dev_result_sha256=sha(out / 'dev_result.json'))
    U.write_json(out / 'dev_selection.json', sel)
    log(f"gate: G-F2 {dec['gates']['G-F2']['pass']} ({dec['gates']['G-F2']['value']:.5f}); F* {dec['F_star']}; "
        f"G-F1 {dec['gates']['G-F1']['pass']} (gain {dec['gain_star']:+.5f}); decision {dec['decision']}: {dec['outcome']}")


# ================================================================ fresh items (design 4; G7 to G9)

def spent_questions(items, n_test):
    """Design 4, rule 1: the normalized questions of the dev, train and confirmatory test items."""
    xs = items['dev_train'] + items['dev_test'] + items['train'] + items['test_pool'][:n_test]
    return {U.normalize_answer(x['question']) for x in xs}


def shard0_split(rows, need):
    """The frozen pick of the item build (uconfirm.build_items with C1): returns the picked row indices (in pick order)
    and the indices after the last pick, in the order of the seed-2026102110 permutation."""
    perm = np.random.default_rng(U.PERM_SEED_TRAIN).permutation(len(rows))
    seen, picked = set(), []
    for n, i in enumerate(perm):
        nq = U.normalize_answer(rows[i]['question'])
        if nq in seen:
            continue
        seen.add(nq)
        picked.append(int(i))
        if len(picked) == need:
            return picked, [int(j) for j in perm[n + 1:]]
    raise SystemExit(f'shard 0: only {len(picked)} picks, {need} needed')


def fresh_record(row, pool):
    return dict(pool=pool, qid=row['question_id'], question=' '.join(row['question'].split()),
                aliases=sorted(U.alias_set(row['answer'])), value=row['answer'].get('value'))


def build_fresh(spent, cands, limits):
    """Design 4, rules 2 to 4. cands: pool -> records in pool order (the pools in the order V, S0, S1). A record is
    removed if its normalized question is spent or equal to an earlier fresh question. limits: pool -> maximum kept."""
    seen, out, report = set(), [], {}
    for pool, recs in cands.items():
        kept, rm_spent, rm_dup, scanned = 0, 0, 0, 0
        for r in recs:
            if pool in limits and kept >= limits[pool]:
                break
            scanned += 1
            nq = U.normalize_answer(r['question'])
            if nq in spent:
                rm_spent += 1
                continue
            if nq in seen:
                rm_dup += 1
                continue
            seen.add(nq)
            out.append(dict(r, pool=pool))
            kept += 1
        report[pool] = dict(rows=len(recs), scanned=scanned, kept=kept, removed_spent=rm_spent, removed_duplicate=rm_dup)
    return out, report


def stage_fetch(a):
    check_freeze_d()
    require_go()
    if not a.researcher_said_yes or 'yes' not in a.researcher_said_yes.lower():
        raise SystemExit('fetch: --researcher-said-yes must hold [researcher]\'s words with "yes"')
    remote, size, want = fetch_data.FILES['train1']
    root = HERE / 'data' / 'trivia_qa'
    dest = root / remote
    if not dest.exists():
        from huggingface_hub import hf_hub_download
        p = Path(hf_hub_download(fetch_data.REPO, remote, repo_type='dataset', revision=fetch_data.REVISION,
                                 local_dir=root))
        if p.resolve() != dest.resolve():
            raise SystemExit(f'unexpected download path {p}')
    got = fetch_data.verify(dest, size, want)
    lp = HERE / 'data' / 'FETCH_LOG.json'
    old = U.read_json(lp, [])
    old.append(dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), researcher_said_yes=a.researcher_said_yes, repo=fetch_data.REPO,
                    revision=fetch_data.REVISION, remote=remote, local=str(dest.relative_to(HERE)), bytes=size,
                    sha256=got, license=fetch_data.LICENSE))
    U.write_json(lp, old)
    log(f'fetch: {remote} ({size} bytes, sha256 {got})')


def stage_items(a):
    check_freeze_d()
    require_go()
    fi = HERE / 'fresh' / 'fresh_items.json'
    if fi.exists():
        raise SystemExit('fresh/fresh_items.json exists')
    t0 = time.time()
    data_sha = {}
    for key, root in (('validation', SEALED / 'data'), ('train0', SEALED / 'data'), ('train1', HERE / 'data')):
        data_sha[key] = fetch_data.verify(fetch_data.local_path(key, root), *fetch_data.FILES[key][1:])
    items, n_test = sealed_items(), sealed_n_test()
    need = sum(U.SIZES.values())
    rows0 = [r for r in U.read_rows(fetch_data.local_path('train0', SEALED / 'data')) if U.usable(r)]
    picked, rest = shard0_split(rows0, need)
    want = [x['qid'] for k in ('dev_train', 'dev_test', 'train') for x in items[k]]
    if [rows0[i]['question_id'] for i in picked] != want:
        raise SystemExit('shard 0: the replicated pick is not equal to the frozen dev and train items')
    rows1 = [r for r in U.read_rows(fetch_data.local_path('train1', HERE / 'data')) if U.usable(r)]
    perm1 = np.random.default_rng(S1_PERM_SEED).permutation(len(rows1))
    cands = dict(V=[dict(pool='V', qid=x['qid'], question=x['question'], aliases=x['aliases'], value=x['value'])
                    for x in items['test_pool'][n_test:]],
                 S0=[fresh_record(rows0[i], 'S0') for i in rest],
                 S1=[fresh_record(rows1[i], 'S1') for i in perm1])
    fresh, report = build_fresh(spent_questions(items, n_test), cands, dict(S1=N_S1))
    counts = {k: v['kept'] for k, v in report.items()}
    man = dict(status=STATUS, time=time.strftime('%Y-%m-%d %H:%M:%S'), counts=counts, n_fresh=len(fresh), pools=report,
               expected=EXPECTED_KEPT, shard0_usable_rows=len(rows0), shard1_usable_rows=len(rows1),
               shard0_rest_rows=len(rest), s1_perm_seed=S1_PERM_SEED, n_spent=len(spent_questions(items, n_test)),
               data_sha256=data_sha, sealed_items_sha256=sha(SEALED / 'items' / 'items.json'), code_sha256=code_sha(),
               seconds=time.time() - t0)
    if counts != EXPECTED_KEPT:
        U.write_json(HERE / 'fresh' / 'fresh_manifest_STOP.json', man)
        raise SystemExit(f'stop (G7): the fresh pool counts {counts} are not the design counts {EXPECTED_KEPT}')
    U.write_json(fi, dict(status=STATUS, items=fresh))
    man['fresh_items_sha256'] = sha(fi)
    U.write_json(HERE / 'fresh' / 'fresh_manifest.json', man)
    log(f'items: {counts} ({len(fresh)} fresh items); shard 1 usable rows {len(rows1)}')


def stage_freeze(a):
    fz = HERE / 'FREEZE.json'
    if fz.exists():
        raise SystemExit('FREEZE.json exists. A new freeze needs a new, dated section and [researcher]\'s word.')
    check_freeze_d()
    require_go()
    res, info = run_tests()
    if not res.wasSuccessful():
        raise SystemExit('freeze: the unit tests fail')
    sc = seal_check()
    if not sc['ok']:
        raise SystemExit(f'freeze: sealed inputs differ from SEAL.json: {sc["bad"]}')
    U.write_json(fz, dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), sha256=freeze_hashes(), tests=info, seal_check=sc,
                          rule='Every Phase C\' stage checks these hashes and stops on a difference (G25).'))
    log(f'freeze: FREEZE.json written ({info["run"]} tests pass)')


# ================================================================ Phase C' model stages (design 8)

def chunk_range(spec, n_chunks):
    if not spec:
        return range(n_chunks)
    lo, hi = (int(x) for x in spec.split(':'))
    return range(max(0, lo), min(hi, n_chunks))


def stage_sender(a):
    check_freeze()
    fresh = fresh_items()
    cdir = HERE / 'confirm' / 'sender_chunks'
    cdir.mkdir(parents=True, exist_ok=True)
    tok = m = W = None
    for pool in ([a.pool] if a.pool else list(POOLS)):
        its = [x for x in fresh if x['pool'] == pool]
        n_chunks = (len(its) + CHUNK - 1) // CHUNK
        for c in chunk_range(a.chunks, n_chunks):
            base = cdir / f'{pool}_{c:03d}'
            if base.with_suffix('.npz').exists():
                continue
            if m is None:
                record_hashes(HERE / 'confirm', 'start', 'sender', weight_roots())
                tok, m = U.load_sender('main', DEVICE)
                W = m.lm_head.weight.detach().float()
            rows, arrays, info = U.sender_chunk(tok, m, W, its[c * CHUNK:(c + 1) * CHUNK], U.SENDERS['main']['blocks'],
                                                [U.SAMPLE_SEED, POOL_SPLIT_INDEX[pool], c], DEVICE, a.sender_bs,
                                                a.sample_items, a.sender_bs)
            U.write_json(base.with_suffix('.json'), dict(rows=rows, info=info))
            save_npz(base.with_suffix('.npz'), **arrays)            # the .npz file marks the chunk as done
            fails = sum(r['a1_flags']['failed'] for r in rows)
            log(f'sender {pool} chunk {c + 1}/{n_chunks}: {info["n"]} items, {info["seconds"]:.0f} s '
                f'({info["seconds"] / info["n"]:.2f} s/item), parser failures {fails}')
    if m is not None:
        del m, W
        empty_cache()
        record_hashes(HERE / 'confirm', 'end', 'sender', weight_roots())
    merge_sender(fresh)


def merge_sender(fresh):
    """After the last chunk: sender_fresh.npz and sender_fresh.json in the order of fresh_items.json (V, S0, S1)."""
    out = HERE / 'confirm'
    if (out / 'sender_fresh.npz').exists():
        return True
    bases = []
    for pool in POOLS:
        n = sum(x['pool'] == pool for x in fresh)
        bases += [out / 'sender_chunks' / f'{pool}_{c:03d}' for c in range((n + CHUNK - 1) // CHUNK)]
    if not all(b.with_suffix('.npz').exists() for b in bases):
        log(f'sender: {sum(b.with_suffix(".npz").exists() for b in bases)} of {len(bases)} chunks exist; no merge yet')
        return False
    assert [x['pool'] for x in fresh] == sorted((x['pool'] for x in fresh), key=POOLS.index)
    rows, arrays, secs = [], [], 0.0
    for b in bases:
        js = U.read_json(b.with_suffix('.json'))
        rows += js['rows']
        secs += js['info']['seconds']
        arrays.append(np.load(b.with_suffix('.npz')))
    merged = {k: np.concatenate([z[k] for z in arrays]) for k in arrays[0].files}
    assert len(rows) == len(fresh) == len(merged['n_cand'])
    U.write_json(out / 'sender_fresh.json', rows)
    U.write_json(out / 'sender_times.json', dict(s_per_item=secs / len(fresh), items=len(fresh)))
    save_npz(out / 'sender_fresh.npz', **merged)           # the .npz file marks the merge as done: written last
    log(f'sender: merged {len(fresh)} fresh items ({secs / len(fresh):.2f} s/item)')
    return True


def seed_list(spec):
    seeds = [int(s) for s in spec.split(',')] if spec else list(LP_SEEDS)
    bad = [s for s in seeds if s not in LP_SEEDS]
    if bad:
        raise SystemExit(f'seeds {bad} are not in {LP_SEEDS}')
    return seeds


def stage_seeds(a):
    check_freeze()
    out = HERE / 'confirm'
    tr, fr = train_data(), fresh_data()
    seeds = seed_list(a.seeds)
    st = dict(tok=None)

    def receiver():
        if st['tok'] is None:
            record_hashes(out, 'start', 'seeds', weight_roots())
            st['tok'], st['R'], st['yn'] = U.load_receiver(DEVICE)
            st['lays'] = dict(train=U.make_layouts(st['tok'], tr['items'], tr['sj']),
                              fresh=U.make_layouts(st['tok'], fr['items'], fr['sj']))
        return st['tok'], st['R'], st['yn']

    def text_pass(fmt, path, items, sj, P):
        if path.exists():
            return
        tok, R, yn = receiver()
        t1 = time.time()
        H, Z = U.receiver_text(tok, R, messages(fmt, items, sj, P), yn, tok.pad_token_id, DEVICE, a.recv_bs)
        save_npz(path, h=H.astype(np.float16), z=Z)
        log(f'{path.parent.name} {path.stem}: {len(items)} passes ({time.time() - t1:.0f} s)')

    if a.t0c:
        for split, d in (('train', tr), ('fresh', fr)):
            text_pass('T0c', out / 'recv' / f'T0c_{split}.npz', d['items'], d['sj'], None)
    X = {}
    for s in seeds:
        d = out / f'seed{s}'
        d.mkdir(parents=True, exist_ok=True)
        t = time.time()
        if not (d / 'probes.npz').exists():
            P_fit, P_eval, info = U.crossfit_probe('logistic', tr['S'], tr['m'], tr['y'], fr['S'], fr['m'], s,
                                                   U.MLP['max_epochs'])
            save_npz(d / 'probes.npz', fit=P_fit, eval=P_eval)
            U.write_json(d / 'probes.json', info)
            log(f'seed {s}: probes (L2 weight {info["lam"]:g})')
        P = np.load(d / 'probes.npz')
        if s in TEXT_SEEDS:
            for fmt in FORMATS:
                for split, dd, Pv in (('train', tr, P['fit']), ('fresh', fr, P['eval'])):
                    text_pass(fmt, d / f'recv_{fmt}_{split}.npz', dd['items'], dd['sj'], Pv)
            if not (d / 'tokens.json').exists():
                tok, _, _ = receiver()
                U.write_json(d / 'tokens.json', {fmt: token_counts(tok, fmt, fr['items'], fr['sj'], P['eval'])
                                                 for fmt in FORMATS + ('T0c',)})
        jobs = [('L-probe', LP_SPEC, LP_EPOCHS, 'z')] + ([('L', L_SPEC, L_EPOCHS, 'x')] if s in TEXT_SEEDS else [])
        for arm, spec, E, kind in jobs:
            part = d / f'latent_{arm}.npz'
            if part.exists():
                continue
            if kind == 'z':
                X_tr, X_fr = U.probe_z(P['fit'], tr['m'], P['eval'], fr['m'])
            else:
                if not X:
                    X['train'], X['fresh'] = U.slot_standardize(tr['S'], tr['m'], fr['S'], fr['m'])
                X_tr, X_fr = X['train'], X['fresh']
            tok, R, yn = receiver()
            pad = tok.pad_token_id
            h0 = U.param_hash(R)
            t1 = time.time()
            ch, hist, _ = U.train_channel(R, st['lays']['train'], X_tr, tr['y'][:, 0], spec, s, E, pad, DEVICE)
            log(f'seed {s} {arm}: {E} epochs, last train loss {hist[-1]["train_loss"]:.4f} ({hist[-1]["seconds"]:.0f} s)')
            res = {}
            for split, Xs in (('fresh', X_fr), ('fit', X_tr)):
                lays = st['lays']['fresh' if split == 'fresh' else 'train']
                res[f'p_{arm}_{split}'], res[f'h_{arm}_{split}'], res[f'z_{arm}_{split}'] = U.predict_channel(
                    R, ch, lays, Xs, yn, pad, DEVICE)
            U.write_json(d / f'latent_{arm}.json', dict(
                history=hist, finite=all(math.isfinite(r['train_loss']) for r in hist), spec=spec, epochs=E, seed=s,
                receiver_param_sha256_before=h0, receiver_param_sha256_after=U.param_hash(R),
                seconds=time.time() - t1))
            # the .npz file marks the arm as done, so it is written last
            save_npz(part, **{k: (v.astype(np.float16) if k.startswith('h_') else v) for k, v in res.items()})
            del ch
            empty_cache()
        log(f'seed {s}: done ({time.time() - t:.0f} s)')
    if st['tok'] is not None:
        record_hashes(out, 'end', 'seeds', weight_roots())


# ================================================================ statistics and the reading rule (design 9, 10)

def two_level_ci(nl, pairs, n_boot=N_BOOT, seed=BOOT_SEED, chunk=BOOT_CHUNK):
    """Design 9. nl: arm -> (n_seeds, n_items) item NLL. pairs: (X, C) -> gap(X, C) = the mean over items of
    NLL_i(X) - NLL_i(C), with NLL_i the mean over the seeds of the arm. Each draw resamples the items (the same items for
    every arm) and, for each arm, that arm's seeds, independently (G16). The same draws serve every pair. Draw order in
    a chunk: the item indices (chunk, n), then the seed indices (chunk, S) of each arm in sorted order of the names.
    Returns (X, C) -> dict(mean, ci95)."""
    arms = sorted({a for p in pairs for a in p})
    T = {a: np.asarray(nl[a], np.float64).T for a in arms}            # (n, S)
    n = next(iter(T.values())).shape[0]
    assert all(v.shape[0] == n for v in T.values())
    rng = np.random.default_rng(seed)
    stats = {p: [] for p in pairs}
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(b, n))
        cnt = np.bincount((idx + n * np.arange(b)[:, None]).ravel(), minlength=b * n).reshape(b, n).astype(np.float64)
        val = {}
        for arm in arms:
            S = T[arm].shape[1]
            sd = rng.integers(0, S, size=(b, S))
            w = np.bincount((sd + S * np.arange(b)[:, None]).ravel(), minlength=b * S).reshape(b, S) / S
            val[arm] = ((cnt @ T[arm]) * w).sum(1) / n
        for x, c in pairs:
            stats[(x, c)].append(val[x] - val[c])
        done += b
    out = {}
    for x, c in pairs:
        v = np.concatenate(stats[(x, c)])
        out[(x, c)] = dict(mean=float(T[x].mean(1).mean() - T[c].mean(1).mean()),
                           ci95=[float(q) for q in np.percentile(v, [2.5, 97.5])])
    return out


def labels(row):
    """Design 10.1: 'closes' and 'a gap remains' for one gap row."""
    closes = bool(row['mean'] <= CLOSE_MEAN and row['ci95'][1] < CLOSE_HI)
    return dict(closes=closes, remains=bool(not closes and row['ci95'][0] > 0))


def item_ci(nl, pairs, n_boot=N_BOOT, seed=BOOT_SEED):
    """The item-only paired bootstrap of the confirmatory run (descriptive here): on the seed means."""
    d = {f'{x} minus {c}': nl[x].mean(0) - nl[c].mean(0) for x, c in pairs}
    ci = U.bootstrap_ci(d, n_boot, seed)
    return {(x, c): list(ci[f'{x} minus {c}']) for x, c in pairs}


def beats(nl, a, b, ci):
    """The 'beats' rule of the confirmatory run (sealed reading rule): a beats b if the mean of NLL_i(b) - NLL_i(a) is
    0.02 or more, the CI lower bound is above 0 and every per-seed mean is above 0 (G21)."""
    mean = float((nl[b].mean(0) - nl[a].mean(0)).mean())
    per_seed = (nl[b] - nl[a]).mean(1)
    return dict(a=a, b=b, mean_b_minus_a=mean, ci95_item=ci, per_seed=per_seed.tolist(),
                beats=bool(mean >= U.MARGIN and ci[0] > 0 and (per_seed > 0).all()))


CASES = {'0': 'invalid', 'R': 'the gap does not replicate on fresh items',
         'A1': 'a finer text format closes the gap; F* gains over F3',
         'A2': 'a finer text format closes the gap; the gain over F3 is not resolved',
         'B': 'the format narrows the gap; a gap remains', 'C': 'the gap is not in the number format',
         'D': 'unresolved'}


def rule_pairs(fstar):
    prim = [(fstar, C_STAR), (REF, C_STAR), (REF, fstar)]
    sec = [(fstar, 'L-probe'), (REF, 'L-probe'), ('L-probe', C_STAR), (fstar, 'T0c'), (REF, 'T0c')]
    sec += [p for f in CANDIDATES if f != fstar for p in ((f, C_STAR), (REF, f))]
    sec += [(arm, 'probe-direct') for arm in (REF, fstar, C_STAR, 'L-probe', 'L')]
    return prim, list(dict.fromkeys(sec))


def reading_rule(nl, fstar, fo, validity, pool=None, n_boot=N_BOOT):
    """Design 10.2, steps 0 to 5, then the secondary rows of 10.3 (no case)."""
    prim, sec = rule_pairs(fstar)
    ci = two_level_ci(nl, prim + sec, n_boot)
    g_star, g_ref, gain = ci[(fstar, C_STAR)], ci[(REF, C_STAR)], ci[(REF, fstar)]
    lab_star, lab_ref = labels(g_star), labels(g_ref)
    gains = bool(gain['ci95'][0] > 0)
    if not all(validity.values()):
        case = '0'
    elif lab_ref['closes']:
        case = 'R'
    elif lab_star['closes']:
        case = 'A1' if gains else 'A2'
    elif lab_star['remains'] and gains:
        case = 'B'
    elif lab_star['remains']:
        case = 'C'
    else:
        case = 'D'
    share = gain['mean'] / g_ref['mean'] if g_ref['mean'] != 0 else float('nan')
    ic = item_ci(nl, prim, n_boot)
    row = lambda p: dict(x=p[0], c=p[1], mean=ci[p]['mean'], ci95=ci[p]['ci95'], ci95_item_only=ic.get(p),
                         per_seed_x=nl[p[0]].mean(1).tolist(), per_seed_c=nl[p[1]].mean(1).tolist())
    primary = dict(gap_star=dict(row(prim[0]), **lab_star), gap_ref=dict(row(prim[1]), **lab_ref),
                   gain_star=dict(row(prim[2]), gains=gains), share_of_F3_gap_removed=share)
    secondary = {f'gap({x}, {c})': dict(mean=ci[(x, c)]['mean'], ci95=ci[(x, c)]['ci95']) for x, c in sec}
    for x in (fstar, REF):
        secondary[f'gap({x}, L-probe)'].update(labels(ci[(x, 'L-probe')]))
    lci = item_ci(nl, [(REF, 'L'), (fstar, 'L')], n_boot)
    secondary['beats'] = [beats(nl, t, 'L', [-lci[(t, 'L')][1], -lci[(t, 'L')][0]]) for t in (REF, fstar)]
    secondary['beats'] += [beats(nl, 'L', t, lci[(t, 'L')]) for t in (REF, fstar)]
    per_pool = {}
    if pool is not None:
        for pl in POOLS:
            k = pool == pl
            if k.any():
                cp = two_level_ci({a: v[:, k] for a, v in nl.items()}, prim, n_boot)
                per_pool[pl] = dict(n=int(k.sum()), **{f'gap({x}, {c})': cp[(x, c)] for x, c in prim})
    return dict(case=case, case_name=CASES[case], validity=validity, F_star=fstar, F_other=fo, C_star=C_STAR,
                primary=primary, secondary=secondary, per_pool=per_pool, means={a: float(v.mean()) for a, v in nl.items()},
                close_mean=CLOSE_MEAN, close_hi=CLOSE_HI, n_boot=n_boot, boot_seed=BOOT_SEED)


def c_round_format(P_fit, m_fit, y_fit, P_eval, m_eval, y_eval, fmt, seed):
    """The c_round recipe (design 2.1, sealed P15) for any format: the NLL of a logistic fit on the formatted values of
    all slots (with slot-present flags) minus the NLL of a fit on the raw values. Returns (raw cost, info)."""
    def feats(P, m, formatted):
        cols = []
        for j in range(3):
            v = np.where(m[:, j], np.nan_to_num(P[:, j], nan=0.0), 0.0)
            f = np.array([fmt_feature(x, fmt) if formatted else [x] for x in v]) * m[:, j:j + 1]
            cols += [f, m[:, j:j + 1].astype(np.float64)]
        return np.hstack(cols)

    _, p_fmt, _ = U.cv_logistic(feats(P_fit, m_fit, True), y_fit, feats(P_eval, m_eval, True), seed)
    _, p_raw, _ = U.cv_logistic(feats(P_fit, m_fit, False), y_fit, feats(P_eval, m_eval, False), seed)
    a, b = U.mean_nll(p_fmt, y_eval), U.mean_nll(p_raw, y_eval)
    return a - b, dict(nll_formatted=a, nll_raw=b, clipped=max(0.0, a - b))


# ================================================================ analysis (job A)

def validity_checks(tr, fr):
    out = HERE / 'confirm'
    v1 = {}
    for s in TEXT_SEEDS:
        new = np.load(out / f'seed{s}' / 'probes.npz')['fit']
        old = np.load(SEALED / 'confirm' / f'seed{s}' / 'probes.npz')['fit']
        same_mask = bool((np.isnan(new) == np.isnan(old)).all())
        diff = float(np.nanmax(np.abs(new - old))) if same_mask else float('inf')
        v1[s] = dict(same_nan_mask=same_mask, max_abs_diff=diff)
    V1 = all(v['same_nan_mask'] and v['max_abs_diff'] <= 1e-6 for v in v1.values())
    ok_w, w_info = weights_ok(out, need_files=3)
    lat = {f'{s}_{arm}': U.read_json(out / f'seed{s}' / f'latent_{arm}.json')
           for s in LP_SEEDS for arm in (['L-probe'] + (['L'] if s in TEXT_SEEDS else []))}
    mem = sorted({h for v in lat.values() for h in (v['receiver_param_sha256_before'], v['receiver_param_sha256_after'])})
    V2 = bool(ok_w and mem == [RECV_PARAM_SHA])
    fin_fit = {k: v['finite'] for k, v in lat.items()}
    fin_p = {}
    for k in lat:
        s, arm = k.split('_', 1)
        z = np.load(out / f'seed{s}' / f'latent_{arm}.npz')
        fin_p[k] = bool(np.isfinite(z[f'p_{arm}_fresh']).all())
    V3 = all(fin_fit.values()) and all(fin_p.values())
    items, n_test = sealed_items(), sealed_n_test()
    spent = spent_questions(items, n_test)
    hit = sum(U.normalize_answer(x['question']) in spent for x in fr['items'])
    man = U.read_json(HERE / 'fresh' / 'fresh_manifest.json')
    counts = {pl: int((fr['pool'] == pl).sum()) for pl in POOLS}
    V4 = bool(hit == 0 and counts == man['counts'] == EXPECTED_KEPT)
    validity = dict(V1=bool(V1), V2=V2, V3=bool(V3), V4=V4)
    detail = dict(V1=v1, V2=dict(weight_files=w_info, receiver_in_memory_hashes=mem), V3=dict(fits=fin_fit, fresh_p=fin_p),
                  V4=dict(fresh_equal_to_spent=hit, counts=counts, manifest_counts=man['counts']))
    return validity, detail


def stage_analysis(a):
    fz = check_freeze()
    _GUARD['fresh_labels'] = True
    try:
        analysis_body(fz)
    finally:
        _GUARD['fresh_labels'] = False


def analysis_body(fz):
    t0 = time.time()
    sel = require_go()
    fstar, fo = sel['F_star'], sel['F_other']
    out = HERE / 'confirm'
    tr, fr = train_data(), fresh_data()
    y_tr = tr['y'][:, 0]
    y_fr = fresh_labels(fr['items'], fr['sj'])[:, 0]
    p, info = defaultdict(list), {}
    for s in TEXT_SEEDS:
        d = out / f'seed{s}'
        for arm in FORMATS + ('T0c',):
            A = {k: np.load(out / 'recv' / f'T0c_{k}.npz' if arm == 'T0c' else d / f'recv_{arm}_{k}.npz')
                 for k in ('train', 'fresh')}
            _, pe, inf = U.cv_logistic(A['train']['h'].astype(np.float64), y_tr, A['fresh']['h'].astype(np.float64), s)
            p[arm].append(pe)
            info[f'{arm}_seed{s}'] = inf
        p['L'].append(np.load(d / 'latent_L.npz')['p_L_fresh'])
        p['probe-direct'].append(np.load(d / 'probes.npz')['eval'][:, 0])
        log(f'analysis seed {s}: text readouts done')
    for s in LP_SEEDS:
        z = np.load(out / f'seed{s}' / 'latent_L-probe.npz')
        p['L-probe'].append(z['p_L-probe_fresh'])
        _, pe, inf = U.cv_logistic(z['h_L-probe_fit'].astype(np.float64), y_tr, z['h_L-probe_fresh'].astype(np.float64), s)
        p[C_STAR].append(pe)
        info[f'{C_STAR}_seed{s}'] = inf
    log('analysis: L-probe-R readouts done')
    nl = {arm: np.stack([U.item_nll(v, y_fr) for v in vs]) for arm, vs in p.items()}
    validity, vdetail = validity_checks(tr, fr)
    rule = reading_rule(nl, fstar, fo, validity, fr['pool'])
    cr = {}
    for fmt in FORMATS:
        vals = []
        for s in TEXT_SEEDS:
            P = np.load(out / f'seed{s}' / 'probes.npz')
            vals.append(c_round_format(P['fit'], tr['m'], y_tr, P['eval'], fr['m'], y_fr, fmt, s))
        cr[fmt] = dict(mean=float(np.mean([v for v, _ in vals])), per_seed=[v for v, _ in vals],
                       clipped_mean=float(np.mean([i['clipped'] for _, i in vals])))
    per_arm = {}
    for arm, vs in p.items():
        ms = [U.metrics(v, y_fr) for v in vs]
        per_arm[arm] = dict({k: float(np.nanmean([m[k] for m in ms])) for k in ms[0]}, seeds=len(vs),
                            nll_per_seed=[m['nll'] for m in ms])
    train_side = {}
    for s in TEXT_SEEDS:
        new = np.load(out / f'seed{s}' / 'latent_L-probe.npz')['p_L-probe_fit']
        old = np.load(SEALED / 'confirm' / f'seed{s}' / 'latent_L-probe.npz')['p_L-probe_fit']
        train_side[s] = dict(max_abs_p_diff=float(np.abs(new - old).max()), train_nll_new=U.mean_nll(new, y_tr),
                             train_nll_sealed=U.mean_nll(old, y_tr))
    tokens = {s: U.read_json(out / f'seed{s}' / 'tokens.json') for s in TEXT_SEEDS}
    save_npz(out / 'fresh_probs.npz', y=y_fr, pool=fr['pool'],
             **{f'p_{arm}_seed{s}': v for arm, vs in p.items()
                for s, v in zip(LP_SEEDS if arm in (C_STAR, 'L-probe') else TEXT_SEEDS, vs)})
    res = dict(status='CONFIRMATORY (frozen run; the independent audit is still to do)', freeze=fz,
               sizes=dict(train=len(y_tr), fresh=len(y_fr), per_pool={pl: int((fr['pool'] == pl).sum()) for pl in POOLS}),
               base_rate_train=float(y_tr.mean()), base_rate_fresh=float(y_fr.mean()), rule=rule, arms=per_arm,
               c_round_fresh=cr, validity_detail=vdetail, train_side_L_probe=train_side, tokens_fresh=tokens,
               readout_info=info, selection=sel, seeds=dict(text=list(TEXT_SEEDS), soft=list(LP_SEEDS)),
               code_sha256=code_sha(), env=dict(torch=torch.__version__, python=platform.python_version()),
               seconds=time.time() - t0)
    U.write_json(out / 'results.json', res)
    (out / 'table.md').write_text(table(res) + '\n')
    print(table(res))
    log(f'analysis: case {rule["case"]} ({rule["case_name"]})')


def table(res):
    r = res['rule']
    lines = [f"Status: {res['status']}", '', f"Case {r['case']}: {r['case_name']}. F* = {r['F_star']}; F° = "
             f"{r['F_other']}; C* = {r['C_star']}. Validity: {r['validity']}.", '',
             '| arm | seeds | NLL | AUROC | Brier | ECE |', '|---|---|---|---|---|---|']
    for arm, v in res['arms'].items():
        lines.append(f"| {arm} | {v['seeds']} | {v['nll']:.4f} | {v['auroc']:.3f} | {v['brier']:.4f} | {v['ece']:.4f} |")
    pr = r['primary']
    lines += ['', '| primary row | mean | 95% CI (two-level) | 95% CI (item only) | label |', '|---|---|---|---|---|']
    for k in ('gap_star', 'gap_ref', 'gain_star'):
        v = pr[k]
        lab = ('closes' if v.get('closes') else 'gap remains' if v.get('remains') else 'gains' if v.get('gains')
               else 'none')
        lines.append(f"| {k}: NLL({v['x']}) - NLL({v['c']}) | {v['mean']:+.4f} | [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] "
                     f"| [{v['ci95_item_only'][0]:+.4f}, {v['ci95_item_only'][1]:+.4f}] | {lab} |")
    lines += ['', f"Share of the F3 gap that F* removes: {pr['share_of_F3_gap_removed']:.3f}", '',
              '| secondary row | mean | 95% CI (two-level) |', '|---|---|---|']
    for k, v in r['secondary'].items():
        if k != 'beats':
            lines.append(f"| {k} | {v['mean']:+.4f} | [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] |")
    lines += ['', '| A beats B? (confirmatory rule) | mean NLL(B) - NLL(A) | 95% CI (item) | per seed | beats |',
              '|---|---|---|---|---|']
    for c in r['secondary']['beats']:
        lines.append(f"| {c['a']} vs {c['b']} | {c['mean_b_minus_a']:+.4f} | [{c['ci95_item'][0]:+.4f}, "
                     f"{c['ci95_item'][1]:+.4f}] | {', '.join(f'{v:+.3f}' for v in c['per_seed'])} | {c['beats']} |")
    lines += ['', '| pool | n | gap(F*, C*) | gap(F3, C*) | gain(F*) |', '|---|---|---|---|---|']
    for pl, v in r['per_pool'].items():
        g = [v[f'gap({x}, {c})'] for x, c in rule_pairs(r['F_star'])[0]]
        lines.append(f"| {pl} | {v['n']} | " + ' | '.join(f"{x['mean']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]"
                                                         for x in g) + ' |')
    lines += ['', '| format | c_round on fresh items (mean of 3 seeds) |', '|---|---|']
    for f, v in res['c_round_fresh'].items():
        lines.append(f"| {f} | {v['mean']:+.4f} |")
    return '\n'.join(lines)


# ================================================================ main

STAGES = dict(gate=stage_gate, fetch=stage_fetch, items=stage_items, freeze=stage_freeze, sender=stage_sender,
              seeds=stage_seeds, analysis=stage_analysis)
MODES = dict(dev=['gate'], confirm=['fetch', 'items', 'freeze', 'sender', 'seeds', 'analysis'])
MODEL_STAGES = {'gate', 'sender', 'seeds'}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['test', 'prefreeze', 'dev', 'confirm'])
    ap.add_argument('--stages', default=None, help='comma list (dev: gate; confirm: one or more stages)')
    ap.add_argument('--pool', choices=list(POOLS), default=None, help='sender: one pool only')
    ap.add_argument('--chunks', default=None, help='sender: lo:hi chunk indices of the pool')
    ap.add_argument('--seeds', default=None, help='seeds: comma list of seeds')
    ap.add_argument('--t0c', action='store_true', help='seeds: also the T0c passes (job Q1)')
    ap.add_argument('--researcher-said-yes', dest='researcher_said_yes', default=None, help="fetch: [researcher]'s approval text")
    ap.add_argument('--sender_bs', type=int, default=16)
    ap.add_argument('--sample_items', type=int, default=4)
    ap.add_argument('--recv_bs', type=int, default=64)
    a = ap.parse_args(argv)
    if a.mode == 'test':
        res, _ = run_tests()
        sys.exit(0 if res.wasSuccessful() else 1)
    if a.mode == 'prefreeze':
        stage_prefreeze(a)
        return
    stages = a.stages.split(',') if a.stages else MODES[a.mode]
    bad = [s for s in stages if s not in MODES[a.mode]]
    if bad:
        ap.error(f'stages {bad} are not part of mode {a.mode}: {MODES[a.mode]}')
    if MODEL_STAGES & set(stages) and not torch.backends.mps.is_available():
        raise SystemExit('MPS is not available')
    log(f'{STATUS} ugap.py {a.mode} stages {stages} code {code_sha()[:12]}')
    for s in [s for s in STAGES if s in stages]:
        t = time.time()
        log(f'== stage {s}')
        STAGES[s](a)
        log(f'== stage {s} done ({time.time() - t:.0f} s)')
    log('done')


if __name__ == '__main__':
    main()
