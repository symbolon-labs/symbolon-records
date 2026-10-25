#!/usr/bin/env python
"""DRAFT harness (EXPLORATORY code, not frozen, not run on a model or on real data) for the confirmatory test
"probe text against a latent message for uncertainty transfer" on closed-book TriviaQA.

Design: ../../../reports/uncertainty_confirmatory_design_20261021.md. Choices that the design leaves open, and the
section numbers that the comments cite: PREREG_DRAFT.md. Approval items and the run order: APPROVAL_REQUEST.md.
Reused code: ../latent_uncertainty_followup_20261021/followup.py (logistic fits, cross-fit, readouts, map init,
bootstrap, run wrapper). That folder does not change.

Modes (one per call):
  test     the unit tests: synthetic data and tiny random models on the CPU; no model load, no data, no network
  smoke    Phase 0: every stage of dev and confirm on 48 dev-train and 16 dev-test items; parser view; times
  dev      Phase D (scratch mode): dev-train 2,000 and dev-test 1,000 items; gates G-D1 to G-D4; selection.json
  confirm  Phase C (confirmatory mode): train 4,000 and test n_test items; every stage needs FREEZE.json

Stages (--stages, comma list, run in this order; default: all stages of the mode):
  items     build items/items.json from the two parquet files (CPU; once, for all modes)
  sender    answer, 10 samples, candidates, P_seq, block states, verbal digit (GPU; chunks of 250 items resume)
  gates     G-D1 base rate, G-D4 label-rule agreement, parser failure rate (CPU)                      [smoke, dev]
  probes    both probe families with cross-fit; probe family selection; G-D2 (CPU)                     [smoke, dev]
  text      every format of every text arm; readouts; format selection; G-D3 (GPU)                     [smoke, dev]
  latent    L and L-probe grids with the grid-end rule; the SD for the n rule (GPU; configs resume)      [smoke, dev]
  select    selection.json and selection.md (CPU)                                                      [smoke, dev]
  freeze    FREEZE.json: hashes of PREREGISTRATION.md, the code, the items and the selection (CPU)    [confirm]
  ctext     the seed-free text arms with the selected formats (GPU)                                    [smoke, confirm]
  seeds     for each seed: probes, T3 and T4, L, L-shuf, L-testshuf and L-probe (GPU)                  [smoke, confirm]
  analysis  readouts, metrics, bootstrap, reading rule -> results.json (CPU; the only reader of test labels)

Run smoke, dev and confirm only through run_uconfirm.sh, and only after [researcher]'s yes for that run.
"""
import argparse
import copy
import hashlib
import json
import math
import platform
import re
import string
import sys
import time
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fetch_data  # noqa: E402  (the file table and the hash check; no network use on import)

STATUS = 'DRAFT'
HUB = Path.home() / '.cache/huggingface/hub'
SENDERS = {
    'main': dict(repo='Qwen/Qwen3-1.7B', blocks=(21, 28),
                 path=HUB / 'models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'),
    # fixed here (P13): 3/4 depth and the last block of 36, as 21 and 28 of 28 for the main sender
    'fallback': dict(repo='Qwen/Qwen3-4B-Instruct-2507', blocks=(27, 36),
                     path=HUB / 'models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554'),
}
RECV = HUB / 'models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/a10cc1512eabd3dde888204e902eca88bddb4951'

# design sections 5, 6, 10 and 11
PERM_SEED_TRAIN, PERM_SEED_VAL, SAMPLE_SEED = 2026102110, 2026102111, 2026102112
DEV_SEED = 2026102113                                  # fixed here (P20): the Phase D seed
CONFIRM_SEEDS = (20261101, 20261102, 20261103)
BOOT_SEED, N_BOOT = 20261021, 10_000
SIZES = dict(dev_train=2000, dev_test=1000, train=4000)
N_TEST, SD_LIMIT = 6000, 0.60
DEDUPE_QUESTIONS = True                                # open: [researcher] decides (C1)
ANSWER_USER = 'Answer the question with a short answer only (a name, a number or a few words).\nQuestion: {question}'
VERBAL_USER = ('How sure are you that your answer is right? Reply with one digit from 0 (not sure at all) '
               'to 9 (completely sure).')
MAX_NEW, K_SAMPLES, CHUNK = 16, 10, 250
R_PREFIX, R_SUFFIX = 'Message: ', '\nIs the proposed answer correct? Answer:'
NO_ANSWER = '(no answer)'                              # fixed here (P8): an empty sample answer in the T6 list
SOURCE_WORDS = 120
F3_LEVELS = [(0.2, 'very unlikely'), (0.4, 'unlikely'), (0.6, 'uncertain'), (0.8, 'likely'), (math.inf, 'very likely')]
FORMATS = dict(T2=('F1', 'F2', 'F3'), T3=('F1', 'F2', 'F3'), T4=('F1', 'F2', 'F3'), T5=('F1', 'F2', 'F3'),
               T6=('list', 'counts'))
T5_RAW = True                                          # open: [researcher] decides (C4): v_j = P_seq(c_j), not renormalized
T6_COUNT_TOP = 'samples'                               # open: [researcher] decides (C2): x1, x2 = top two sample answers
SEED_FREE_ARMS = ['T0', 'T0c', 'T1', 'T2', 'T5', 'T6', 'Source', 'Q-only']
SEEDED_TEXT_ARMS = ['T3', 'T4']
LATENT_ARMS = ['L', 'L-shuf', 'L-testshuf', 'L-probe']
S_ARMS = ['T0', 'T0c', 'T1', 'T2', 'T5', 'T6']         # S* candidates (design 12.1)

# fits (design section 8; follow-up code)
LAMS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
MLP_WDS = [1e-4, 1e-3, 1e-2]
MLP = dict(hidden=256, lr=1e-3, bs=256, max_epochs=200, patience=10, inner=0.2)   # fixed here (P10)
N_FOLDS, MARGIN, N_BINS, P_CLIP = 5, 0.02, 15, 1e-6
FULL_LRS, LOW_LRS, RANKS, LPROBE_LRS = [1e-5, 3e-5, 1e-4], [3e-4, 1e-3, 3e-3], [16, 64], [1e-4, 1e-3, 1e-2]
LADDER = [1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]  # fixed here (P17)
LAT = dict(bs=32, wd=0.01, clip=1.0, inner=0.2, max_lr_rounds=2)
LATENT_READOUT_INPUT = 'raw'                           # open: [researcher] decides (C3): 'raw' final state, as in the pilot

MODES = {
    'smoke': dict(fit='dev_train', eval='dev_test', n_fit=48, n_eval=16, max_epochs=2, ext_epochs=3, patience=1,
                  mlp_max_epochs=5, grids=dict(full=[3e-5, 1e-4], r16=[1e-3, 3e-3], lprobe=[1e-3, 1e-2]),
                  stages=['items', 'sender', 'gates', 'probes', 'text', 'latent', 'select', 'ctext', 'seeds',
                          'analysis']),
    'dev': dict(fit='dev_train', eval='dev_test', n_fit=SIZES['dev_train'], n_eval=SIZES['dev_test'], max_epochs=15,
                ext_epochs=25, patience=3, mlp_max_epochs=MLP['max_epochs'],
                grids=dict(full=FULL_LRS, r16=LOW_LRS, r64=LOW_LRS, lprobe=LPROBE_LRS),
                stages=['items', 'sender', 'gates', 'probes', 'text', 'latent', 'select']),
    'confirm': dict(fit='train', eval='test', n_fit=SIZES['train'], n_eval=None, mlp_max_epochs=MLP['max_epochs'],
                    stages=['freeze', 'sender', 'ctext', 'seeds', 'analysis']),
}
GRID_KEYS = dict(full=('L', None), r16=('L', 16), r64=('L', 64), lprobe=('L-probe', None))
FROZEN_FILES = ['PREREGISTRATION.md', 'uconfirm.py', 'test_uconfirm.py', 'run_uconfirm.sh', 'fetch_data.py']
_GUARD = dict(test_labels=False)   # only the confirm analysis stage, after the freeze check, sets this


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def code_sha():
    return fetch_data.sha256_file(Path(__file__))


# ================================================================ text rules (design 3.1, 6.1, 6.2, 6.4, 9.2)

_PUNCT = {ord(c): ' ' for c in string.punctuation + '‘’´`'}
_ARTICLES = re.compile(r'\b(a|an|the)\b')


def normalize_answer(s):
    """The TriviaQA normalization (triviaqa_evaluation.py): '_' to space, lower case, punctuation to space, no articles,
    single spaces."""
    s = (s or '').replace('_', ' ').lower().translate(_PUNCT)
    return ' '.join(_ARTICLES.sub(' ', s).split())


def alias_set(answer):
    """Fixed here (P1): the normalized forms of 'normalized_aliases' and 'aliases', without empty strings."""
    raw = list(answer.get('normalized_aliases') or []) + list(answer.get('aliases') or [])
    return {k for k in (normalize_answer(x) for x in raw) if k}


def label_alias(ans, aliases):
    """Design 3.1: 1 if the answer matches a gold alias after the normalization."""
    return int(normalize_answer(ans) in set(aliases))


def token_f1(pred, gold):
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    same = sum((Counter(p) & Counter(g)).values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def label_f1(ans, aliases, thr=0.8):
    """G-D4 second rule: token F1 of at least 0.8 with any alias."""
    return int(max((token_f1(ans, a) for a in aliases), default=0.0) >= thr)


def parse_answer(text, ended):
    """Design 6.1: stop at a newline or the end token, remove one final period. Fixed here (P2): spaces are stripped;
    'failed' = empty, or no newline and no end token in 16 new tokens."""
    head, nl, _ = text.partition('\n')
    a = head.strip()
    if a.endswith('.'):
        a = a[:-1].rstrip()
    flags = dict(empty=not a, truncated=not (ended or nl), prefixed=a.lower().startswith('answer'))
    flags['failed'] = flags['empty'] or flags['truncated']
    return a, flags


def select_candidates(a1, samples):
    """Design 6.2: c1 = a1; c2 and c3 are the two most frequent normalized sample answers not equal to a1 (ties: first
    in draw order). Fixed here (P3): empty answers are not candidates; the text of a candidate is its first sample."""
    n1 = normalize_answer(a1)
    count, first = Counter(), {}
    for i, s in enumerate(samples):
        k = normalize_answer(s)
        if not k or k == n1:
            continue
        count[k] += 1
        first.setdefault(k, (i, s))
    order = sorted(count, key=lambda k: (-count[k], first[k][0]))
    return [a1] + [first[k][1] for k in order[:2]]


def sample_entropy(samples):
    """Design 6.4: the entropy (nats) of the normalized answers. Fixed here (P4): empty answers are one class."""
    c = np.array(list(Counter(normalize_answer(s) for s in samples).values()), np.float64)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def se_labels(se_fit, se_eval):
    """y_SE = 1 if SE is above the median SE of the train split (design 6.4)."""
    med = float(np.median(se_fit))
    return (np.asarray(se_fit) > med).astype(int), (np.asarray(se_eval) > med).astype(int), med


# ================================================================ messages (design 7.1 to 7.3)

def fmt_value(p, fmt):
    if fmt == 'F1':
        return f'{int(math.floor(9 * p + 0.5))} out of 9'
    if fmt == 'F2':
        return f'{int(math.floor(100 * p + 0.5))}%'
    if fmt == 'F3':
        return next(w for t, w in F3_LEVELS if p < t)
    raise ValueError(fmt)


def fmt_feature(p, fmt):
    """The information that a formatted value keeps, as features (c_round, design 12.1). Fixed here (P15)."""
    if fmt == 'F1':
        return [math.floor(9 * p + 0.5) / 9]
    if fmt == 'F2':
        return [math.floor(100 * p + 0.5) / 100]
    if fmt == 'F3':
        k = next(i for i, (t, _) in enumerate(F3_LEVELS) if p < t)
        return [float(i == k) for i in range(len(F3_LEVELS))]
    raise ValueError(fmt)


def cand_segments(q, cands):
    """The text around the candidate tags. Message = seg[0] + tag[0] + seg[1] + ... + tag[n-1] + seg[n]. A text arm puts
    ' (v)' or '' in a tag; L puts one soft vector in each tag (design 7.2; P6: no space token next to a slot)."""
    segs = [f'{R_PREFIX}Question: {q}\nProposed answer: {cands[0]}']
    if len(cands) > 1:
        segs.append(f'. Other answers: {cands[1]}')
    if len(cands) > 2:
        segs.append(f'; {cands[2]}')
    return segs + ['.' + R_SUFFIX]


def join_tags(segs, tags):
    assert len(tags) == len(segs) - 1, (len(segs), len(tags))
    return ''.join(s + t for s, t in zip(segs, list(tags) + ['']))


def msg_T0(q, a1):
    return join_tags(cand_segments(q, [a1]), [''])


def msg_T0c(q, cands):
    return join_tags(cand_segments(q, cands), [''] * len(cands))


def msg_T1(q, a1, digit):
    return join_tags(cand_segments(q, [a1]), [f' ({int(digit)} out of 9)'])


def msg_scalar(q, a1, p, fmt):
    """T2 (p = P_seq(c1)) and T3 (p = probe value of c1)."""
    return join_tags(cand_segments(q, [a1]), [f' ({fmt_value(p, fmt)})'])


def msg_dist(q, cands, ps, fmt):
    """T4 (probe values) and T5 (P_seq values) for all candidates."""
    return join_tags(cand_segments(q, cands), [f' ({fmt_value(p, fmt)})' for p in ps])


def t6_list(samples):
    return '; '.join(s if s else NO_ANSWER for s in samples)


def t6_counts(samples, named=None):
    """C2 default (named None): x1 and x2 are the two most frequent non-empty sample answers (ties: first in draw
    order). C2 alternative (named = [a1, c2]): the counts of a1 and c2. The rest, with empty answers, is 'other'; a part
    with count 0 is not written."""
    n = len(samples)
    if named is not None:
        parts, used = [], 0
        for x in named:
            k = normalize_answer(x)
            c = sum(normalize_answer(s) == k for s in samples) if k else 0
            if c:
                parts.append(f'{c} of {n} say {x}')
                used += c
        if n - used:
            parts.append(f'{n - used} of {n} other')
        return '; '.join(parts)
    count, first = Counter(), {}
    for i, s in enumerate(samples):
        k = normalize_answer(s)
        if k:
            count[k] += 1
            first.setdefault(k, (i, s))
    top = sorted(count, key=lambda k: (-count[k], first[k][0]))[:2]
    parts = [f'{count[k]} of {n} say {first[k][1]}' for k in top]
    rest = n - sum(count[k] for k in top)
    if rest:
        parts.append(f'{rest} of {n} other')
    return '; '.join(parts)


def msg_T6(q, a1, samples, fmt, named=None):
    body = t6_list(samples) if fmt == 'list' else t6_counts(samples, named)
    return f'{R_PREFIX}Question: {q}\nProposed answer: {a1}. Ten sampled answers: {body}.{R_SUFFIX}'


def msg_source(q, a1, window):
    return f'{R_PREFIX}Question: {q}\nProposed answer: {a1}.\nEvidence: {window}{R_SUFFIX}'


def msg_qonly(q):
    return f'{R_PREFIX}Question: {q}{R_SUFFIX}'


def text_messages(arm, fmt, items, sj, sz, probe_vals=None):
    """All receiver texts of one arm variant for one split. sj: sender json rows; sz: sender arrays."""
    out = []
    for i, x in enumerate(items):
        q, cands = x['question'], sj[i]['cands']
        a1, n = cands[0], len(cands)
        if arm == 'T0':
            t = msg_T0(q, a1)
        elif arm == 'T0c':
            t = msg_T0c(q, cands)
        elif arm == 'T1':
            t = msg_T1(q, a1, int(np.argmax(sz['digit_logits'][i])))
        elif arm == 'T2':
            t = msg_scalar(q, a1, math.exp(sz['logp'][i, 0]), fmt)
        elif arm == 'T3':
            t = msg_scalar(q, a1, float(probe_vals[i, 0]), fmt)
        elif arm == 'T4':
            t = msg_dist(q, cands, [float(v) for v in probe_vals[i, :n]], fmt)
        elif arm == 'T5':
            ps = np.exp(sz['logp'][i, :n])
            t = msg_dist(q, cands, [float(v) for v in (ps if T5_RAW else ps / ps.sum())], fmt)
        elif arm == 'T6':
            t = msg_T6(q, a1, sj[i]['samples'], fmt, cands[:2] if T6_COUNT_TOP == 'candidates' else None)
        elif arm == 'Source':
            t = msg_source(q, a1, x['window'])
        elif arm == 'Q-only':
            t = msg_qonly(q)
        else:
            raise ValueError(arm)
        out.append(t)
    return out


# ================================================================ data (design section 5; source window 7.2)

def page_texts(row):
    ep = row.get('entity_pages') or {}
    return [t for t in (ep.get('wiki_context') or []) if t and t.strip()]


def usable(row):
    """Fixed here (P5): a question, at least one non-empty normalized alias and one non-empty evidence page."""
    return bool((row.get('question') or '').strip() and alias_set(row.get('answer') or {}) and page_texts(row))


def source_window(pages, aliases, n=SOURCE_WORDS):
    """Design 7.2: n words centered on the first match of a normalized gold alias. Fixed here (P7): the window comes
    from the first page (list order) with a match; the first match is the earliest position of any alias. With no
    match in any page: the first n words of the first page. Returns (window, matched)."""
    keys = sorted(set(aliases))
    for page in pages:
        norm = ' ' + normalize_answer(page) + ' '
        best = None
        for a in keys:
            j = norm.find(' ' + a + ' ')
            if j >= 0 and (best is None or j < best[0] or (j == best[0] and len(a) > len(best[1]))):
                best = (j, a)
        if best is None:
            continue
        j, a = best
        k0 = norm[:j + 1].count(' ') - 1          # index of the first matched normalized token
        k1 = k0 + len(a.split()) - 1
        words = page.split()
        cum, start, end = 0, None, None
        for w_i, w in enumerate(words):
            c = len(normalize_answer(w).split())
            if start is None and cum + c > k0:
                start = w_i
            if cum + c > k1:
                end = w_i
                break
            cum += c
        if start is None or end is None:
            start = end = 0
        center = (start + end) // 2
        lo = max(0, min(center - n // 2, len(words) - n))
        return ' '.join(words[lo:lo + n]), True
    return ' '.join(pages[0].split()[:n]), False


def read_rows(path):
    import pyarrow.parquet as pq
    return pq.read_table(str(path), columns=['question', 'question_id', 'answer', 'entity_pages']).to_pylist()


def item_record(row, split):
    aliases = alias_set(row['answer'])
    window, matched = source_window(page_texts(row), aliases)
    return dict(split=split, qid=row['question_id'], question=' '.join(row['question'].split()),
                aliases=sorted(aliases), value=row['answer'].get('value'), window=window, window_match=matched)


def build_items(data_dir, items_dir, files=None, sizes=None, dedupe=DEDUPE_QUESTIONS):
    """Design section 5. One permutation (seed 2026102110) of the usable shard-0 rows gives dev-train, dev-test and
    train, in that order; one permutation (seed 2026102111) of the usable validation rows gives the test pool, without
    the questions of the 7,000 dev and train items. files: key -> (path, bytes, sha256), for the tests."""
    t0 = time.time()
    sizes = sizes or SIZES
    need = sizes['dev_train'] + sizes['dev_test'] + sizes['train']
    if files is None:
        files = {k: (fetch_data.local_path(k, data_dir),) + fetch_data.FILES[k][1:] for k in fetch_data.FILES}
    hashes = {k: fetch_data.verify(*files[k]) for k in ('validation', 'train0')}
    rows, report = [], {}
    for shard in ('train0', 'train1'):
        if shard == 'train1':
            if not Path(files['train1'][0]).exists():
                raise SystemExit(f'shard 0 has {distinct} distinct usable questions, fewer than {need}. The design then '
                                 'also uses shard 00001-of-00007 (260.8 MB): approval item D2, then fetch_data.py.')
            hashes['train1'] = fetch_data.verify(*files['train1'])
        got = [r for r in read_rows(files[shard][0]) if usable(r)]
        report[f'{shard}_usable_rows'] = len(got)
        rows += got
        distinct = len({normalize_answer(r['question']) for r in rows}) if dedupe else len(rows)
        if distinct >= need:
            break
    perm = np.random.default_rng(PERM_SEED_TRAIN).permutation(len(rows))
    picked, seen, dup = [], set(), 0
    for i in perm:
        nq = normalize_answer(rows[i]['question'])
        if dedupe and nq in seen:
            dup += 1
            continue
        seen.add(nq)
        picked.append(int(i))
        if len(picked) == need:
            break
    if len(picked) < need:
        raise SystemExit(f'only {len(picked)} usable train rows, {need} needed')
    a, b = sizes['dev_train'], sizes['dev_train'] + sizes['dev_test']
    parts = dict(dev_train=picked[:a], dev_test=picked[a:b], train=picked[b:need])
    used = {normalize_answer(rows[i]['question']) for i in picked}
    val = [r for r in read_rows(files['validation'][0]) if usable(r)]
    pv = np.random.default_rng(PERM_SEED_VAL).permutation(len(val))
    test, seen_v, overlap, dup_v = [], set(), 0, 0
    for i in pv:
        nq = normalize_answer(val[i]['question'])
        if nq in used:
            overlap += 1
            continue
        if dedupe and nq in seen_v:
            dup_v += 1
            continue
        seen_v.add(nq)
        test.append(int(i))
    items = {k: [item_record(rows[i], k) for i in v] for k, v in parts.items()}
    items['test_pool'] = [item_record(val[i], 'test') for i in test]
    agree, total = 0, 0
    for r in [rows[i] for i in picked] + [val[i] for i in test]:
        na = set(r['answer'].get('normalized_aliases') or [])
        for x in r['answer'].get('aliases') or []:
            total += 1
            agree += normalize_answer(x) in na
    report.update(validation_usable_rows=len(val), train_duplicates_skipped=dup, test_removed_overlap=overlap,
                  test_removed_duplicates=dup_v, test_pool=len(test), dedupe_questions=dedupe,
                  alias_normalization_agreement=agree / max(total, 1),
                  window_match_share={k: float(np.mean([x['window_match'] for x in v])) for k, v in items.items()},
                  sizes={k: len(v) for k, v in items.items()})
    items_dir = Path(items_dir)
    write_json(items_dir / 'items.json', items)
    write_json(items_dir / 'manifest.json', dict(status=STATUS, time=time.strftime('%Y-%m-%d %H:%M:%S'),
                                                 data_sha256=hashes, seeds=[PERM_SEED_TRAIN, PERM_SEED_VAL],
                                                 code_sha256=code_sha(), seconds=time.time() - t0, **report))
    log(f'items: {report["sizes"]}; alias normalization agreement {report["alias_normalization_agreement"]:.4f}; '
        f'window match {report["window_match_share"]}')
    return items, report


# ================================================================ fits on the CPU (follow-up code; design 8.1, 8.2)

def standardize(Xtr, *Xs):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return [(X - mu) / sd for X in (Xtr,) + Xs]


def fit_logistic(X, y, lam):
    """Binary L2 logistic regression (960 -> 1 with bias for the readouts), LBFGS in float64, as in the follow-up."""
    X = torch.as_tensor(X, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=1000, history_size=20, tolerance_grad=1e-8,
                            tolerance_change=1e-12, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(X @ w + b, y) + lam * w.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def predict_logistic(wb, X):
    return torch.sigmoid(torch.as_tensor(X, dtype=torch.float64) @ wb[0] + wb[1]).numpy()


def item_nll(p, y):
    p = np.clip(np.asarray(p, np.float64), P_CLIP, 1 - P_CLIP)
    y = np.asarray(y, np.float64)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def mean_nll(p, y):
    return float(item_nll(p, y).mean())


def item_folds(n_items, seed, k=N_FOLDS):
    return np.random.default_rng(seed).permutation(n_items) % k


def cv_logistic(Xtr, ytr, Xte, seed, groups=None, lams=LAMS):
    """L2 weight by 5-fold CV NLL on train (folds over items: rows of one item share a fold), then a fit on all train.
    Returns out-of-fold train probabilities, test probabilities and info."""
    Xtr, Xte = standardize(np.asarray(Xtr, np.float64), np.asarray(Xte, np.float64))
    ytr = np.asarray(ytr, np.float64)
    g = np.arange(len(ytr)) if groups is None else np.asarray(groups)
    fold = item_folds(int(g.max()) + 1, seed)[g]
    best, grid = None, {}
    for lam in lams:
        oof = np.zeros(len(ytr))
        for f in range(N_FOLDS):
            tr, va = fold != f, fold == f
            if va.any():
                oof[va] = predict_logistic(fit_logistic(Xtr[tr], ytr[tr], lam), Xtr[va])
        nll = mean_nll(oof, ytr)
        grid[str(lam)] = nll
        if best is None or nll < best[0]:
            best = (nll, lam, oof)
    nll, lam, oof = best
    pte = predict_logistic(fit_logistic(Xtr, ytr, lam), Xte)
    return oof, pte, dict(family='logistic', lam=lam, cv_nll=nll, cv_grid=grid, at_grid_end=lam in (lams[0], lams[-1]))


def fit_mlp(X, y, groups, wd, seed, max_epochs):
    """Probe MLP (design 8.1): d -> 256 (ReLU) -> 1, weight decay wd, early stop on an inner fold of 20% of the items.
    Fixed here (P10): AdamW, lr 1e-3, batch 256, patience 10, float32 on the CPU."""
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    yt = torch.as_tensor(np.asarray(y), dtype=torch.float32)
    groups = np.asarray(groups)
    items = np.unique(groups)
    va_items = np.random.default_rng(seed).permutation(items)[:max(1, int(round(len(items) * MLP['inner'])))]
    va_mask = np.isin(groups, va_items)
    tr, va = np.flatnonzero(~va_mask), np.flatnonzero(va_mask)
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], MLP['hidden']), torch.nn.ReLU(), torch.nn.Linear(MLP['hidden'], 1))
    opt = torch.optim.AdamW(net.parameters(), lr=MLP['lr'], weight_decay=wd)
    gen = torch.Generator().manual_seed(seed)
    best = (math.inf, 0, copy.deepcopy(net.state_dict()))
    for ep in range(1, max_epochs + 1):
        net.train()
        perm = tr[torch.randperm(len(tr), generator=gen).numpy()]
        for i in range(0, len(perm), MLP['bs']):
            j = torch.as_tensor(perm[i:i + MLP['bs']])
            loss = F.binary_cross_entropy_with_logits(net(X[j]).squeeze(-1), yt[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = F.binary_cross_entropy_with_logits(net(X[va]).squeeze(-1), yt[va]).item()
        if v < best[0]:
            best = (v, ep, copy.deepcopy(net.state_dict()))
        if ep - best[1] >= MLP['patience']:
            break
    net.load_state_dict(best[2])
    net.eval()
    return net, dict(best_epoch=best[1], inner_val_nll=best[0])


def predict_mlp(net, X):
    with torch.no_grad():
        return torch.sigmoid(net(torch.as_tensor(np.asarray(X), dtype=torch.float32)).squeeze(-1)).double().numpy()


def cv_mlp(Xtr, ytr, Xte, seed, groups, max_epochs=MLP['max_epochs']):
    Xtr, Xte = standardize(np.asarray(Xtr, np.float64), np.asarray(Xte, np.float64))
    ytr, groups = np.asarray(ytr, np.float64), np.asarray(groups)
    fold = item_folds(int(groups.max()) + 1, seed)[groups]
    best, grid = None, {}
    for wd in MLP_WDS:
        oof = np.zeros(len(ytr))
        for f in range(N_FOLDS):
            tr, va = fold != f, fold == f
            if va.any():
                net, _ = fit_mlp(Xtr[tr], ytr[tr], groups[tr], wd, seed, max_epochs)
                oof[va] = predict_mlp(net, Xtr[va])
        nll = mean_nll(oof, ytr)
        grid[str(wd)] = nll
        if best is None or nll < best[0]:
            best = (nll, wd, oof)
    nll, wd, oof = best
    net, info = fit_mlp(Xtr, ytr, groups, wd, seed, max_epochs)
    return oof, predict_mlp(net, Xte), dict(family='mlp', wd=wd, cv_nll=nll, cv_grid=grid, final=info,
                                            at_grid_end=wd in (MLP_WDS[0], MLP_WDS[-1]))


def crossfit_probe(family, S_fit, m_fit, y_fit, S_eval, m_eval, seed, mlp_max_epochs):
    """Design 8.1: one probe for all candidate slots, input the standardized state (train statistics over all present
    slots, P11), target y_j. Train values are out of fold (5 folds over items); eval values come from the fit on all
    train. Returns P_fit (n, 3) and P_eval (n, 3), NaN for an absent slot, and info."""
    gi, sj = np.nonzero(m_fit)
    Xtr, ytr = S_fit[gi, sj].astype(np.float64), y_fit[gi, sj]
    ge, se = np.nonzero(m_eval)
    Xte = S_eval[ge, se].astype(np.float64)
    if family == 'logistic':
        oof, pte, info = cv_logistic(Xtr, ytr, Xte, seed, groups=gi)
    elif family == 'mlp':
        oof, pte, info = cv_mlp(Xtr, ytr, Xte, seed, gi, mlp_max_epochs)
    else:
        raise ValueError(family)
    P_fit, P_eval = np.full(m_fit.shape, np.nan), np.full(m_eval.shape, np.nan)
    P_fit[gi, sj], P_eval[ge, se] = oof, pte
    return P_fit, P_eval, info


def logit(p):
    p = np.clip(np.asarray(p, np.float64), P_CLIP, 1 - P_CLIP)
    return np.log(p) - np.log1p(-p)


def probe_z(P_fit, m_fit, P_eval, m_eval):
    """L-probe input (design 7.2): the standardized probe logit, train statistics over present slots; 0 when absent."""
    L_fit, L_eval = logit(np.nan_to_num(P_fit, nan=0.5)), logit(np.nan_to_num(P_eval, nan=0.5))
    mu, sd = L_fit[m_fit].mean(), max(L_fit[m_fit].std(), 1e-6)
    z_fit, z_eval = (L_fit - mu) / sd, (L_eval - mu) / sd
    z_fit[~m_fit], z_eval[~m_eval] = 0.0, 0.0
    return z_fit.astype(np.float32), z_eval.astype(np.float32)


def slot_standardize(S_fit, m_fit, S_eval, m_eval):
    rows = S_fit[m_fit].astype(np.float64)
    mu, sd = rows.mean(0), rows.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)

    def f(S, m):
        Z = ((S.astype(np.float64) - mu) / sd).astype(np.float32)
        Z[~m] = 0.0
        return Z

    return f(S_fit, m_fit), f(S_eval, m_eval)


def slot_permutation(mask, seed):
    """L-shuf and L-testshuf (design 7.2). Fixed here (P12): for each slot j, a permutation among the items that have
    slot j. src[i, j] = the item whose state fills slot j of item i (-1 if item i has no slot j)."""
    rng = np.random.default_rng(seed)
    src = np.full(mask.shape, -1, dtype=np.int64)
    for j in range(mask.shape[1]):
        have = np.flatnonzero(mask[:, j])
        src[have, j] = have[rng.permutation(len(have))]
    return src


def apply_permutation(X, src):
    out = np.zeros_like(X)
    for j in range(src.shape[1]):
        have = src[:, j] >= 0
        out[have, j] = X[src[have, j], j]
    return out


# ================================================================ metrics, bootstrap, reading rule (design 11, 12)

def auroc(score, y):
    score, y = np.asarray(score, np.float64), np.asarray(y).astype(int)
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    order = np.argsort(score, kind='mergesort')
    s = score[order]
    first = np.r_[True, s[1:] != s[:-1]]
    starts = np.flatnonzero(first)
    ends = np.r_[starts[1:], len(s)]
    ranks = np.empty(len(s))
    for a, b in zip(starts, ends):
        ranks[order[a:b]] = 0.5 * (a + 1 + b)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece(p, y, bins=N_BINS):
    """Fixed here (P16): 15 equal bins of p(y = 1) on [0, 1]."""
    p, y = np.asarray(p, np.float64), np.asarray(y, np.float64)
    b = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(y[b == k].mean() - p[b == k].mean()) * (b == k).sum() for k in range(bins) if (b == k).any())
                 / len(p))


def metrics(p, y):
    p = np.clip(np.asarray(p, np.float64), P_CLIP, 1 - P_CLIP)
    y = np.asarray(y, np.float64)
    return dict(nll=mean_nll(p, y), auroc=auroc(p, y), brier=float(((p - y) ** 2).mean()), ece=ece(p, y),
                acc=float(((p >= 0.5) == (y == 1)).mean()))


def bootstrap_ci(diffs, n_boot=N_BOOT, seed=BOOT_SEED, chunk=500):
    """Paired bootstrap over items (design 11): the same draws for every vector. Returns name -> (2.5, 97.5) percentiles
    of the mean."""
    names = list(diffs)
    n = len(diffs[names[0]])
    rng = np.random.default_rng(seed)
    means = {k: [] for k in names}
    done = 0
    while done < n_boot:
        idx = rng.integers(0, n, size=(min(chunk, n_boot - done), n))
        for k in names:
            means[k].append(np.asarray(diffs[k], np.float64)[idx].mean(1))
        done += len(idx)
    return {k: tuple(float(v) for v in np.percentile(np.concatenate(means[k]), [2.5, 97.5])) for k in names}


def reading_rule(nl, c_round, v3_ok, v4_ok, n_boot=N_BOOT, boot_seed=BOOT_SEED):
    """Design 12. nl: arm -> (n_seeds, n_items) item NLL; NLL_i is the mean over seeds. 'probe-direct' is the probe
    value of c1 read as a probability (12.6)."""
    m = {a: float(np.mean(v)) for a, v in nl.items()}
    s_star = min(S_ARMS, key=lambda a: m[a])
    pairs = [('L', s_star), ('L', 'T4'), ('L', 'L-probe'), ('T4', s_star), ('T4', 'T3'), ('T4', 'T0c'), ('T5', 'T2'),
             ('T6', 'T5'), ('L', 'T6'), ('L', 'T3'), ('L', 'L-testshuf'), ('L-probe', 'T4')]
    route = [a for a in ('T3', 'T4', 'L', 'L-probe')]
    pairs = list(dict.fromkeys(pairs))
    d = {f'{b} minus {a}': nl[b].mean(0) - nl[a].mean(0) for a, b in pairs}
    d.update({f'{a} minus probe-direct': nl[a].mean(0) - nl['probe-direct'].mean(0) for a in route})
    ci = bootstrap_ci(d, n_boot, boot_seed)

    def cmp(a, b):
        """a beats b: mean of NLL_i(b) - NLL_i(a) >= 0.02, CI lower bound > 0, and every per-seed mean > 0."""
        k = f'{b} minus {a}'
        if k in d:
            mean, lo, hi = float(d[k].mean()), ci[k][0], ci[k][1]
        else:
            k2 = f'{a} minus {b}'
            mean, lo, hi = -float(d[k2].mean()), -ci[k2][1], -ci[k2][0]
        per_seed = (nl[b] - nl[a]).mean(1)
        beats = bool(mean >= MARGIN and lo > 0 and (per_seed > 0).all())
        return dict(a=a, b=b, mean_b_minus_a=mean, ci95=[lo, hi], per_seed=per_seed.tolist(), beats=beats)

    # Amendment A1 (2026-10-21, [researcher]): Source is descriptive. V2 is reported but does not enter the validity check.
    V = dict(V1=bool(m['L-shuf'] >= m['T0c'] - MARGIN), V3=bool(v3_ok), V4=bool(v4_ok))
    V2_descriptive = bool(m['Source'] <= m['T0'])
    L_S, L_T4, T4_L, L_LP = cmp('L', s_star), cmp('L', 'T4'), cmp('T4', 'L'), cmp('L', 'L-probe')
    if not all(V.values()):
        case = '0'
    elif not L_S['beats']:
        case = '3'
    elif L_T4['beats']:
        case = '1' if (L_LP['beats'] and L_T4['ci95'][0] > c_round) else '1m'
    else:
        case = '2a' if T4_L['beats'] else '2b'
    names = {'0': 'invalid', '1': 'latent carrier', '1m': 'medium effect, not a latent carrier',
             '2a': 'probe text carries it (T4 beats L)', '2b': 'probe text carries it (neither beats the other)',
             '3': "no state advantage over the sender's own text"}
    secondary = [cmp(a, b) for a, b in [('T4', s_star), ('T4', 'T3'), ('T4', 'T0c'), ('T5', 'T2'), ('T6', 'T5'),
                                         ('L', 'T6'), ('L', 'T3'), ('L', 'L-testshuf'), ('L-probe', 'T4')]]
    secondary += [cmp(r['b'], r['a']) for r in list(secondary)]
    route_loss = {a: dict(mean=float(d[f'{a} minus probe-direct'].mean()), ci95=list(ci[f'{a} minus probe-direct']))
                  for a in route}
    return dict(case=case, case_name=names[case], validity=V, V2_descriptive=V2_descriptive, s_star=s_star, means=m, c_round=c_round,
                L_beats_S_star=L_S, L_beats_T4=L_T4, T4_beats_L=T4_L, L_beats_L_probe=L_LP,
                T4_minus_L_ci_lower_above_c_round=bool(L_T4['ci95'][0] > c_round), secondary=secondary,
                route_loss=route_loss, margin=MARGIN, n_boot=n_boot, boot_seed=boot_seed)


def c_round_cost(P_fit, m_fit, y_fit, P_eval, m_eval, y_eval, fmt, seed):
    """Design 12.1: the test NLL of a logistic fit on the formatted T4 values of all slots (with slot-present flags)
    minus the test NLL of a fit on the unformatted values; 0 if negative. Fixed here (P15): the formatted value is
    the number that the text shows (F1, F2) or a one-hot of the five words (F3)."""
    def feats(P, m, formatted):
        cols = []
        for j in range(3):
            v = np.where(m[:, j], np.nan_to_num(P[:, j], nan=0.0), 0.0)
            f = np.array([fmt_feature(x, fmt) if formatted else [x] for x in v]) * m[:, j:j + 1]
            cols += [f, m[:, j:j + 1].astype(np.float64)]
        return np.hstack(cols)

    _, p_fmt, _ = cv_logistic(feats(P_fit, m_fit, True), y_fit, feats(P_eval, m_eval, True), seed)
    _, p_raw, _ = cv_logistic(feats(P_fit, m_fit, False), y_fit, feats(P_eval, m_eval, False), seed)
    raw_cost = mean_nll(p_fmt, y_eval) - mean_nll(p_raw, y_eval)
    return max(0.0, raw_cost), dict(nll_formatted=mean_nll(p_fmt, y_eval), nll_raw=mean_nll(p_raw, y_eval))


# ================================================================ batching helpers

def pad_right(seqs, pad_id):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    am = torch.zeros((len(seqs), L), dtype=torch.long)
    for r, s in enumerate(seqs):
        ids[r, :len(s)] = torch.as_tensor(s, dtype=torch.long)
        am[r, :len(s)] = 1
    return ids, am


def pad_left(seqs, pad_id):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    am = torch.zeros((len(seqs), L), dtype=torch.long)
    for r, s in enumerate(seqs):
        ids[r, L - len(s):] = torch.as_tensor(s, dtype=torch.long)
        am[r, L - len(s):] = 1
    return ids, am


def final_state(model, am, **inputs):
    """The output of the final norm at the last real token (right padding)."""
    h = model.model(attention_mask=am, use_cache=False, **inputs).last_hidden_state
    last = am.sum(1) - 1
    return h[torch.arange(len(last), device=h.device), last]


# ================================================================ sender (design section 6)

def load_sender(key, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = SENDERS[key]['path']
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    m = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.bfloat16).to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return tok, m


def answer_prompt(tok, q):
    turn = [dict(role='user', content=ANSWER_USER.format(question=q))]
    return tok.apply_chat_template(turn, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def verbal_prompt(tok, q, a1):
    turns = [dict(role='user', content=ANSWER_USER.format(question=q)), dict(role='assistant', content=a1),
             dict(role='user', content=VERBAL_USER)]
    return tok.apply_chat_template(turns, tokenize=False, add_generation_prompt=True, enable_thinking=False)


@torch.no_grad()
def decode(m, prompts, max_new, end_ids, pad_id, device, rng=None):
    """Batched decoding with a KV cache and left padding. rng None: greedy (argmax). Else (P9): one draw from the full
    softmax (temperature 1.0, top-p 1.0) by the inverse CDF, with uniform numbers from the numpy Generator rng (float64,
    CPU). Returns the new ids of each prompt before its first end token, and whether an end token came."""
    ids, am = pad_left(prompts, pad_id)
    ids, am = ids.to(device), am.to(device)
    pos = (am.cumsum(1) - 1).clamp(min=0)
    out = m(input_ids=ids, attention_mask=am, position_ids=pos, use_cache=True, logits_to_keep=1)
    past, logits = out.past_key_values, out.logits[:, -1]
    nxt_pos = pos[:, -1:] + 1
    B = len(prompts)
    new, ended = [[] for _ in range(B)], np.zeros(B, bool)
    for t in range(max_new):
        if rng is None:
            tok_ids = logits.float().argmax(-1).cpu().numpy()
        else:
            c = np.cumsum(torch.softmax(logits.float().cpu().double(), -1).numpy(), 1)   # MPS has no float64
            u = rng.random(B) * c[:, -1]
            tok_ids = np.minimum((c < u[:, None]).sum(1), c.shape[1] - 1)
        for r in range(B):
            if not ended[r]:
                if int(tok_ids[r]) in end_ids:
                    ended[r] = True
                else:
                    new[r].append(int(tok_ids[r]))
        if ended.all() or t == max_new - 1:
            break
        am = torch.cat([am, torch.ones((B, 1), dtype=am.dtype, device=device)], 1)
        out = m(input_ids=torch.as_tensor(tok_ids, dtype=torch.long, device=device)[:, None], attention_mask=am,
                position_ids=nxt_pos, past_key_values=past, use_cache=True, logits_to_keep=1)
        past, logits = out.past_key_values, out.logits[:, -1]
        nxt_pos = nxt_pos + 1
    return new, ended.tolist()


@torch.no_grad()
def teacher_forced(m, seqs, blocks, pad_id, device, bs, W):
    """seqs: (prompt ids, candidate ids, end id). Design 6.3: log P_seq = the sum of the log probabilities of the
    candidate tokens and the end token; the state after each block at the last candidate token (P2: the last prompt
    token for an empty candidate). W: the output weights in float32 (P14)."""
    caught = {}
    hooks = [m.model.layers[k - 1].register_forward_hook(
        lambda mod, inp, o, k=k: caught.__setitem__(k, o[0] if isinstance(o, tuple) else o)) for k in blocks]
    logp = np.zeros(len(seqs))
    states = {k: np.zeros((len(seqs), m.config.hidden_size), np.float32) for k in blocks}
    try:
        for i in range(0, len(seqs), bs):
            chunk = seqs[i:i + bs]
            ids, am = pad_right([p + c + [e] for p, c, e in chunk], pad_id)
            h = m.model(input_ids=ids.to(device), attention_mask=am.to(device), use_cache=False).last_hidden_state
            for r, (p, c, e) in enumerate(chunk):
                tgt = torch.as_tensor(c + [e], dtype=torch.long, device=device)
                lp = torch.log_softmax(h[r, len(p) - 1:len(p) - 1 + len(tgt)].float() @ W.T, -1)
                logp[i + r] = float(lp[torch.arange(len(tgt), device=device), tgt].sum())
                for k in blocks:
                    states[k][i + r] = caught[k][r, len(p) + len(c) - 1].float().cpu().numpy()
    finally:
        for hk in hooks:
            hk.remove()
    return logp, states


@torch.no_grad()
def last_token_logits(m, prompts, token_ids, pad_id, device, bs, W):
    """T1 (design 6.1): the logits of the given tokens at the first assistant position (float32 weights)."""
    Wt = W[torch.as_tensor(token_ids, device=W.device)]
    out = np.zeros((len(prompts), len(token_ids)), np.float32)
    for i in range(0, len(prompts), bs):
        ids, am = pad_right(prompts[i:i + bs], pad_id)
        h = final_state(m, am.to(device), input_ids=ids.to(device))
        out[i:i + len(h)] = (h.float() @ Wt.T).cpu().numpy()
    return out


def end_ids_of(tok, m):
    e = m.generation_config.eos_token_id
    return set(e if isinstance(e, (list, tuple)) else [e]) | {tok.convert_tokens_to_ids('<|im_end|>')}


def sender_chunk(tok, m, W, items, blocks, seed_key, device, bs_gen=16, n_sample_items=4, bs_tf=16):
    """One chunk of the sender stage. seed_key: [SAMPLE_SEED, split index, chunk index] for the sample stream (P9)."""
    t0 = time.time()
    pad, im_end = tok.pad_token_id, tok.convert_tokens_to_ids('<|im_end|>')
    end = end_ids_of(tok, m)
    digit_ids = [tok.encode(str(d), add_special_tokens=False) for d in range(10)]
    assert all(len(d) == 1 for d in digit_ids), digit_ids
    digit_ids = [d[0] for d in digit_ids]
    prompts = [tok(answer_prompt(tok, x['question']), add_special_tokens=False).input_ids for x in items]
    rows = []
    for i in range(0, len(prompts), bs_gen):
        new, ended = decode(m, prompts[i:i + bs_gen], MAX_NEW, end, pad, device)
        for ids, e in zip(new, ended):
            raw = tok.decode(ids)
            a1, flags = parse_answer(raw, e)
            rows.append(dict(a1_raw=raw, a1=a1, a1_flags=flags))
    rng = np.random.default_rng(seed_key)
    for i in range(0, len(prompts), n_sample_items):
        ps = [p for p in prompts[i:i + n_sample_items] for _ in range(K_SAMPLES)]
        new, ended = decode(m, ps, MAX_NEW, end, pad, device, rng=rng)
        for r in range(len(ps) // K_SAMPLES):
            raws = [tok.decode(new[r * K_SAMPLES + k]) for k in range(K_SAMPLES)]
            parsed = [parse_answer(x, ended[r * K_SAMPLES + k]) for k, x in enumerate(raws)]
            rows[i + r].update(samples_raw=raws, samples=[a for a, _ in parsed],
                               samples_failed=sum(f['failed'] for _, f in parsed))
    t_gen = time.time() - t0
    for r in rows:
        r['cands'] = select_candidates(r['a1'], r['samples'])
        r['se'] = sample_entropy(r['samples'])
    seqs, where = [], []
    for i, r in enumerate(rows):
        for j, c in enumerate(r['cands']):
            seqs.append((prompts[i], tok(c, add_special_tokens=False).input_ids if c else [], im_end))
            where.append((i, j))
    lp, st = teacher_forced(m, seqs, blocks, pad, device, bs_tf, W)
    n, d = len(items), m.config.hidden_size
    logp = np.full((n, 3), np.nan)
    S = {k: np.zeros((n, 3, d), np.float32) for k in blocks}
    for q, (i, j) in enumerate(where):
        logp[i, j] = lp[q]
        for k in blocks:
            S[k][i, j] = st[k][q]
    for k in blocks:
        mx = float(np.abs(S[k]).max())
        if not math.isfinite(mx) or mx > 65000:
            raise SystemExit(f'block {k} state max abs {mx}: fp16 storage is not safe (P14)')
    vp = [tok(verbal_prompt(tok, x['question'], r['a1']), add_special_tokens=False).input_ids for x, r in zip(items, rows)]
    digit_logits = last_token_logits(m, vp, digit_ids, pad, device, bs_tf, W)
    arrays = dict(logp=logp, n_cand=np.array([len(r['cands']) for r in rows]), digit_logits=digit_logits,
                  se=np.array([r['se'] for r in rows]), **{f's{k}': S[k].astype(np.float16) for k in blocks})
    return rows, arrays, dict(seconds=time.time() - t0, seconds_generation=t_gen, n=n)


def load_sender_out(out, name):
    z = np.load(Path(out) / f'sender_{name}.npz')
    return {k: z[k] for k in z.files}, read_json(Path(out) / f'sender_{name}.json')


def cand_mask(sz):
    return np.arange(3)[None, :] < sz['n_cand'][:, None]


def cand_labels(items, sj, split_name):
    """y_j for each candidate (n, 3); -1 when absent. The confirm test split is read only by the analysis stage."""
    if split_name == 'test' and not _GUARD['test_labels']:
        raise RuntimeError('test labels are read only by the analysis stage, after the freeze check (design 10)')
    y = np.full((len(items), 3), -1, dtype=np.int64)
    for i, (x, r) in enumerate(zip(items, sj)):
        for j, c in enumerate(r['cands']):
            y[i, j] = label_alias(c, x['aliases'])
    return y


# ================================================================ receiver and latent arms (design 7.1, 8.2 to 8.4)

def load_receiver(device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(RECV, local_files_only=True)
    R = AutoModelForCausalLM.from_pretrained(RECV, local_files_only=True, dtype=torch.float32).to(device).eval()
    for p in R.parameters():
        p.requires_grad_(False)
    yn = [tok.encode(w, add_special_tokens=False) for w in (' Yes', ' No')]
    assert all(len(t) == 1 for t in yn), yn
    return tok, R, [t[0] for t in yn]


@torch.no_grad()
def receiver_text(tok, R, texts, yn_ids, pad_id, device, bs=64):
    """Final states (n, 960) and the zero-shot ' Yes' and ' No' logits (n, 2) of the token route. Batches by length."""
    ids_all = [tok(t, add_special_tokens=False).input_ids for t in texts]
    order = np.argsort([len(s) for s in ids_all], kind='stable')
    H = np.zeros((len(texts), R.config.hidden_size), np.float32)
    Z = np.zeros((len(texts), 2), np.float32)
    for i in range(0, len(texts), bs):
        idx = order[i:i + bs]
        ids, am = pad_right([ids_all[k] for k in idx], pad_id)
        h = final_state(R, am.to(device), input_ids=ids.to(device))
        H[idx] = h.float().cpu().numpy()
        Z[idx] = R.lm_head(h)[:, yn_ids].float().cpu().numpy()
    return H, Z


@torch.no_grad()
def route_identity(tok, R, text, pad_id, device):
    """The embedding route must give the same final state as the token route (follow-up check)."""
    ids, am = pad_right([tok(text, add_special_tokens=False).input_ids], pad_id)
    ids, am = ids.to(device), am.to(device)
    a = final_state(R, am, input_ids=ids)
    b = final_state(R, am, inputs_embeds=R.model.embed_tokens(ids))
    return float((a - b).abs().max())


def make_layouts(tok, items, sj):
    """L, L-shuf, L-testshuf and L-probe (design 7.2): token ids of the T0c text, with one slot (-1) after each
    candidate. Returns a list of (ids, slot positions)."""
    lays = []
    for x, r in zip(items, sj):
        segs = cand_segments(x['question'], r['cands'])
        ids, slots = [], []
        for j, s in enumerate(segs):
            ids += tok(s, add_special_tokens=False).input_ids
            if j < len(segs) - 1:
                slots.append(len(ids))
                ids.append(-1)
        lays.append((ids, slots))
    return lays


def embed_batch(emb, lays, idx, V, pad_id, device):
    """Input embeddings of items idx: token embeddings, with V[r, j] (B, 3, d) at slot j of item r. Right padding."""
    B, L = len(idx), max(len(lays[i][0]) for i in idx)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    am = torch.zeros((B, L), dtype=torch.long)
    M = torch.zeros((B, L, 3))
    for r, i in enumerate(idx):
        t, slots = lays[i]
        tt = torch.as_tensor(t, dtype=torch.long)
        ids[r, :len(t)] = torch.where(tt < 0, torch.full_like(tt, pad_id), tt)
        am[r, :len(t)] = 1
        for j, pos in enumerate(slots):
            M[r, pos, j] = 1.0
    ids, am, M = ids.to(device), am.to(device), M.to(device=device, dtype=V.dtype)
    x = emb(ids)
    x = x * (1 - M.sum(-1, keepdim=True)) + torch.einsum('blk,bkd->bld', M, V)
    return x, am


class StateMap(torch.nn.Module):
    """L (design 8.3): one map for all slots. rank None: Linear(s_dim, e_dim); rank r: Linear(s_dim, r, no bias), then
    Linear(r, e_dim). Init as in the follow-up: output bias = mean token embedding; output scale = the pilot scale."""

    def __init__(self, s_dim, emb_w, rank=None):
        super().__init__()
        e_dim, e_std = emb_w.shape[1], float(emb_w.std())
        if rank is None:
            self.map = torch.nn.Linear(s_dim, e_dim)
            torch.nn.init.normal_(self.map.weight, std=e_std / math.sqrt(s_dim))
            bias = self.map.bias
        else:
            down, up = torch.nn.Linear(s_dim, rank, bias=False), torch.nn.Linear(rank, e_dim)
            torch.nn.init.normal_(down.weight, std=1 / math.sqrt(s_dim))
            torch.nn.init.normal_(up.weight, std=e_std / math.sqrt(rank))
            self.map = torch.nn.Sequential(down, up)
            bias = up.bias
        with torch.no_grad():
            bias.copy_(emb_w.mean(0))

    def forward(self, s):
        return self.map(s)


class ProbeMap(torch.nn.Module):
    """L-probe (design 8.4): s_j = b + w * z_j; b starts at the mean token embedding, w with std = embedding std."""

    def __init__(self, emb_w):
        super().__init__()
        self.b = torch.nn.Parameter(emb_w.mean(0).clone())
        self.w = torch.nn.Parameter(torch.randn(emb_w.shape[1]) * float(emb_w.std()))

    def forward(self, z):
        return self.b + z[..., None] * self.w


class Channel(torch.nn.Module):
    """Slot map and readout. C3: the readout reads (h - mu) / sd; mu = 0 and sd = 1 ('raw', the pilot) unless norm
    gives fixed statistics ('standardized': the T0c final states of the train split)."""

    def __init__(self, slot_map, e_dim, norm=None):
        super().__init__()
        self.slot_map = slot_map
        self.readout = torch.nn.Linear(e_dim, 1)
        mu, sd = (np.zeros(e_dim), np.ones(e_dim)) if norm is None else norm
        self.register_buffer('mu', torch.as_tensor(np.asarray(mu), dtype=torch.float32))
        self.register_buffer('sd', torch.as_tensor(np.where(np.asarray(sd) < 1e-6, 1.0, sd), dtype=torch.float32))


def make_channel(spec, x_dim, emb_w, norm=None):
    sm = ProbeMap(emb_w) if spec['kind'] == 'L-probe' else StateMap(x_dim, emb_w, spec.get('rank'))
    return Channel(sm, emb_w.shape[1], norm)


def readout_norm(H_fit):
    """C3 alternative: fixed statistics for the latent readout input, or None for the default 'raw'."""
    if LATENT_READOUT_INPUT == 'raw':
        return None
    H = np.asarray(H_fit, np.float64)
    return H.mean(0), H.std(0)


def channel_logits(R, ch, lays, X, idx, pad_id, device):
    V = ch.slot_map(torch.as_tensor(X[idx], device=device))
    x, am = embed_batch(R.model.embed_tokens, lays, list(idx), V, pad_id, device)
    h = final_state(R, am, inputs_embeds=x)
    return ch.readout((h - ch.mu) / ch.sd).squeeze(-1), h


@torch.no_grad()
def predict_channel(R, ch, lays, X, yn_ids, pad_id, device, bs=64):
    """Readout probabilities (n,), final states (n, d) and zero-shot ' Yes' and ' No' logits (n, 2)."""
    ch.eval()
    P, H, Z = [], [], []
    for i in range(0, len(lays), bs):
        idx = np.arange(i, min(i + bs, len(lays)))
        lg, h = channel_logits(R, ch, lays, X, idx, pad_id, device)
        P.append(torch.sigmoid(lg.float()).cpu().double())
        H.append(h.float().cpu())
        Z.append(R.lm_head(h)[:, yn_ids].float().cpu())
    return torch.cat(P).numpy(), torch.cat(H).numpy(), torch.cat(Z).numpy()


def train_channel(R, lays, X, y, spec, seed, max_epochs, pad_id, device, val=None, patience=3, bs=LAT['bs'],
                  norm=None):
    """Map and readout train together (design 8.2): AdamW (weight decay 0.01), batch 32, gradient norm clip 1.0,
    binary cross-entropy. With val = (lays, X, y): early stop (patience) and the best-epoch state. The seed sets the
    init and the batch order (design 10). norm: C3 (readout_norm)."""
    emb_w = R.model.embed_tokens.weight.detach()
    torch.manual_seed(seed)
    ch = make_channel(spec, X.shape[-1], emb_w, norm).to(device)
    opt = torch.optim.AdamW(ch.parameters(), lr=spec['lr'], weight_decay=LAT['wd'])
    gen = torch.Generator().manual_seed(seed)
    yt = torch.as_tensor(np.asarray(y), dtype=torch.float32, device=device)
    hist, best, t0 = [], dict(val_nll=math.inf, epoch=0, state=None), time.time()
    for ep in range(1, max_epochs + 1):
        ch.train()
        perm = torch.randperm(len(y), generator=gen).numpy()
        tot = 0.0
        for i in range(0, len(perm), bs):
            j = perm[i:i + bs]
            lg, _ = channel_logits(R, ch, lays, X, j, pad_id, device)
            loss = F.binary_cross_entropy_with_logits(lg, yt[torch.as_tensor(j, device=device)])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ch.parameters(), LAT['clip'])
            opt.step()
            tot += loss.item() * len(j)
        row = dict(epoch=ep, train_loss=tot / len(y), seconds=time.time() - t0)
        if val is not None:
            p, _, _ = predict_channel(R, ch, val[0], val[1], [0, 1], pad_id, device)
            row['val_nll'] = mean_nll(p, val[2])
            if row['val_nll'] < best['val_nll']:
                best = dict(val_nll=row['val_nll'], epoch=ep,
                            state={k: v.detach().cpu().clone() for k, v in ch.state_dict().items()})
        hist.append(row)
        if not math.isfinite(row['train_loss']):
            break
        if val is not None and ep - best['epoch'] >= patience:
            break
    if val is not None and best['state'] is not None:
        ch.load_state_dict(best['state'])
    return ch, hist, best


def spec_name(spec):
    fam = 'full' if spec.get('rank') is None else f"r{spec['rank']}"
    head = 'Lprobe' if spec['kind'] == 'L-probe' else f'L_{fam}'
    return f"{head}_lr{spec['lr']:g}_ep{spec['max_epochs']}"


def ladder_step(lr, direction):
    k = next(i for i, v in enumerate(LADDER) if math.isclose(v, lr, rel_tol=1e-9))
    return LADDER[k + direction]


def grid_search(fit, initial, max_epochs, ext_epochs, max_rounds=LAT['max_lr_rounds']):
    """Design 9.1 with the grid-end rule. fit(spec) -> dict(best_val_nll, best_epoch, ...). If the selected learning
    rate is at an end of the learning rates tried for its family and rank, add the next ladder value beyond that end
    and select again (at most two times). Then, if the selected epoch count is max_epochs, train that configuration
    again with the limit ext_epochs (one time); that result replaces the first one (P17)."""
    res = {}
    for spec in initial:
        r = fit(dict(spec, max_epochs=max_epochs))
        res[r['name']] = r
    log_ = []
    group = lambda s: (s['kind'], s.get('rank'))
    for _ in range(max_rounds):
        best = min(res.values(), key=lambda r: r['best_val_nll'])
        tried = sorted({r['spec']['lr'] for r in res.values() if group(r['spec']) == group(best['spec'])})
        lr = best['spec']['lr']
        step = -1 if lr == tried[0] else 1 if lr == tried[-1] else 0
        if step == 0:
            break
        r = fit(dict(best['spec'], lr=ladder_step(lr, step)))
        res[r['name']] = r
        log_.append(dict(rule='lr at grid end', from_lr=lr, added=r['spec']['lr'], group=list(group(best['spec']))))
    best = min(res.values(), key=lambda r: r['best_val_nll'])
    if best['best_epoch'] >= max_epochs:
        r = fit(dict(best['spec'], max_epochs=ext_epochs))
        res.pop(best['name'])
        res[r['name']] = r
        log_.append(dict(rule='epoch limit raised', config=best['name'], to=ext_epochs))
        best = min(res.values(), key=lambda r: r['best_val_nll'])
    return best, res, log_


def param_hash(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# ================================================================ run context and stages

class Ctx:
    def __init__(self, a):
        self.a, self.mode, self.cfg = a, a.mode, MODES[a.mode]
        default_out = a.mode if not (a.mode == 'dev' and a.sender == 'fallback') else 'dev_fallback'
        self.out = Path(a.out) if a.out else HERE / default_out
        self.items_dir, self.data = Path(a.items), Path(a.data)
        self.dev_dir = Path(a.dev_dir)
        self.device = torch.device('mps')
        self._items = None
        self.sel_path = (self.dev_dir if self.mode == 'confirm' else self.out) / 'selection.json'
        self.sender_key = self.selection()['sender'] if self.mode == 'confirm' else a.sender

    def selection(self):
        sel = read_json(self.sel_path)
        if sel is None:
            raise SystemExit(f'{self.sel_path} is missing (the select stage of Phase D writes it)')
        return sel

    def split_name(self, split):
        return self.cfg[split]

    def all_items(self):
        if self._items is None:
            p = self.items_dir / 'items.json'
            if not p.exists():
                raise SystemExit(f'{p} is missing: run the items stage first')
            self._items = json.loads(p.read_text())
        return self._items

    def items(self, split):
        allx = self.all_items()
        if self.mode == 'confirm' and split == 'eval':
            n = self.selection()['n_test']
            return allx['test_pool'][:n]
        return allx[self.split_name(split)][:self.cfg['n_fit' if split == 'fit' else 'n_eval']]

    def add_time(self, key, value):
        p = self.out / 'times.json'
        t = read_json(p, {})
        t[key] = value
        write_json(p, t)


def record_hashes(ctx, when):
    """V3 (design 12.2): sha256 of the weight files of the sender and the receiver, at the start and end of a job."""
    rec = dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), when=when, stages=ctx.a.stages, files={})
    for root in (SENDERS[ctx.sender_key]['path'], RECV):
        for f in sorted(Path(root).glob('*.safetensors')):
            rec['files'][str(f)] = fetch_data.sha256_file(f)
    with open(ctx.out / 'hashes.jsonl', 'a') as fh:
        fh.write(json.dumps(rec) + '\n')
    log(f'weight hashes ({when}): {len(rec["files"])} files')


def hashes_ok(out):
    p = Path(out) / 'hashes.jsonl'
    if not p.exists():
        return False, dict(records=0, files=0, changed=[], note='no hashes.jsonl')
    recs = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    seen = {}
    for r in recs:
        for f, h in r['files'].items():
            seen.setdefault(f, set()).add(h)
    bad = [f for f, hs in seen.items() if len(hs) != 1]
    return (not bad and len(recs) >= 2), dict(records=len(recs), files=len(seen), changed=bad)


def require_gates(ctx, names):
    if ctx.mode == 'smoke':
        return
    g = read_json(ctx.out / 'gates.json', {})
    bad = [n for n in names if not g.get(n, {}).get('pass')]
    if bad:
        raise SystemExit(f'stop: gate(s) {bad} not passed or not computed (design 9.2); see {ctx.out / "gates.json"}')


def update_gates(ctx, **gates):
    p = ctx.out / 'gates.json'
    g = read_json(p, {})
    g.update(gates)
    write_json(p, g)
    for k, v in gates.items():
        log(f'gate {k}: {("pass" if v["pass"] else "FAIL") if "pass" in v else "(record)"} {v}')


def freeze_hashes(ctx):
    sha = {f: fetch_data.sha256_file(HERE / f) for f in FROZEN_FILES}
    sha['selection.json'] = fetch_data.sha256_file(ctx.sel_path)
    sha['items.json'] = fetch_data.sha256_file(ctx.items_dir / 'items.json')
    return sha


def check_freeze(ctx):
    if ctx.mode != 'confirm':
        return None
    fz = read_json(HERE / 'FREEZE.json')
    if fz is None:
        raise SystemExit('FREEZE.json is missing: run the freeze stage after [researcher] approves PREREGISTRATION.md')
    now = freeze_hashes(ctx)
    bad = [k for k, v in now.items() if fz['sha256'].get(k) != v]
    if bad:
        raise SystemExit(f'frozen files changed after the freeze: {bad}')
    return fz


def stage_items(ctx):
    if (ctx.items_dir / 'items.json').exists():
        log('items: items/items.json exists')
        return
    t = time.time()
    build_items(ctx.data, ctx.items_dir)
    ctx.add_time('items_seconds', time.time() - t)


def stage_sender(ctx):
    check_freeze(ctx)   # fixed here (P18): in confirm mode the freeze comes before C1
    splits = [ctx.a.split] if ctx.a.split else ['fit', 'eval']
    key = ctx.sender_key
    blocks = SENDERS[key]['blocks']
    tok = m = W = None
    for split in splits:
        name = ctx.split_name(split)
        final = ctx.out / f'sender_{name}.npz'
        if final.exists():
            log(f'sender {name}: exists')
            continue
        items = ctx.items(split)
        cdir = ctx.out / 'sender_chunks'
        cdir.mkdir(parents=True, exist_ok=True)
        split_idx = ['dev_train', 'dev_test', 'train', 'test'].index(name)
        n_chunks = (len(items) + CHUNK - 1) // CHUNK
        for c in range(n_chunks):
            base = cdir / f'{name}_{c:03d}'
            if base.with_suffix('.npz').exists():
                continue
            if m is None:
                tok, m = load_sender(key, ctx.device)
                W = m.lm_head.weight.detach().float()
            rows, arrays, info = sender_chunk(tok, m, W, items[c * CHUNK:(c + 1) * CHUNK], blocks,
                                              [SAMPLE_SEED, split_idx, c], ctx.device, ctx.a.sender_bs,
                                              ctx.a.sample_items, ctx.a.sender_bs)
            write_json(base.with_suffix('.json'), dict(rows=rows, info=info))
            np.savez(base.with_suffix('.npz'), **arrays)
            fails = sum(r['a1_flags']['failed'] for r in rows)
            log(f'sender {name} chunk {c + 1}/{n_chunks}: {info["n"]} items, {info["seconds"]:.0f} s '
                f'({info["seconds"] / info["n"]:.2f} s/item), parser failures {fails}')
        rows, arrays, secs = [], [], 0.0
        for c in range(n_chunks):
            base = cdir / f'{name}_{c:03d}'
            js = read_json(base.with_suffix('.json'))
            rows += js['rows']
            secs += js['info']['seconds']
            arrays.append(np.load(base.with_suffix('.npz')))
        merged = {k: np.concatenate([z[k] for z in arrays]) for k in arrays[0].files}
        assert len(rows) == len(items)
        np.savez(final, **merged)
        write_json(ctx.out / f'sender_{name}.json', rows)
        ctx.add_time(f'sender_{name}_s_per_item', secs / len(items))
        log(f'sender {name}: merged {len(items)} items ({secs / len(items):.2f} s/item)')
        if ctx.mode == 'smoke' and split == 'fit':
            write_json(ctx.out / 'smoke_sender_checks.json', smoke_sender_checks(tok, m, W, items[:8], blocks, ctx.device))
    if m is not None:
        del m, W
        torch.mps.empty_cache()


def smoke_sender_checks(tok, m, W, items, blocks, device):
    """Smoke only: batched against one-at-a-time decoding and teacher-forced passes (bf16 on MPS can differ a little)."""
    if m is None:
        return dict(skipped='sender outputs existed')
    pad, im_end, end = tok.pad_token_id, tok.convert_tokens_to_ids('<|im_end|>'), end_ids_of(tok, m)
    prompts = [tok(answer_prompt(tok, x['question']), add_special_tokens=False).input_ids for x in items]
    batched, _ = decode(m, prompts, MAX_NEW, end, pad, device)
    single = [decode(m, [p], MAX_NEW, end, pad, device)[0][0] for p in prompts]
    seqs = [(p, b, im_end) for p, b in zip(prompts, batched)]
    lb, sb = teacher_forced(m, seqs, blocks, pad, device, 8, W)
    one = [teacher_forced(m, [s], blocks, pad, device, 1, W) for s in seqs]
    return dict(greedy_batched_equals_single=float(np.mean([b == s for b, s in zip(batched, single)])),
                logp_max_abs_diff=float(max(abs(lb[i] - one[i][0][0]) for i in range(len(seqs)))),
                state_max_abs_diff={k: float(max(np.abs(sb[k][i] - one[i][1][k][0]).max() for i in range(len(seqs))))
                                    for k in blocks},
                examples=[dict(question=x['question'], answer=tok.decode(b)) for x, b in zip(items, batched)])


def stage_gates(ctx):
    sz, sj, ys, its = {}, {}, {}, {}
    for split in ('fit', 'eval'):
        name = ctx.split_name(split)
        sz[split], sj[split] = load_sender_out(ctx.out, name)
        its[split] = ctx.items(split)
        ys[split] = cand_labels(its[split], sj[split], name)
    rows = sj['fit'] + sj['eval']
    items = its['fit'] + its['eval']
    y1 = np.concatenate([ys['fit'][:, 0], ys['eval'][:, 0]])
    acc = float(y1.mean())
    f1 = np.array([label_f1(r['a1'], x['aliases']) for r, x in zip(rows, items)])
    agree = float((f1 == y1).mean())
    dis = [dict(question=x['question'], answer=r['a1'], alias_rule=int(a), f1_rule=int(b), aliases=x['aliases'][:10])
           for x, r, a, b in zip(items, rows, y1, f1) if a != b][:50]
    with open(ctx.out / 'gd4_disagreements.jsonl', 'w') as fh:
        fh.writelines(json.dumps(d, ensure_ascii=False) + '\n' for d in dis)
    fail = float(np.mean([r['a1_flags']['failed'] for r in rows]))
    view = [dict(question=x['question'], raw=r['a1_raw'], parsed=r['a1'], flags=r['a1_flags'], label=int(y),
                 aliases=x['aliases'][:5], samples=r['samples'], cands=r['cands'])
            for x, r, y in zip(its['fit'][:20], sj['fit'][:20], ys['fit'][:20, 0])]
    write_json(ctx.out / 'parser_view.json', view)
    update_gates(ctx, **{
        'G-D1': {'value': acc, 'rule': '0.20 <= dev accuracy <= 0.80', 'pass': bool(0.20 <= acc <= 0.80)},
        'G-D4': {'value': agree, 'rule': 'alias and token-F1 rules agree on >= 90% of dev items',
                 'pass': bool(agree >= 0.90), 'disagreements_file': 'gd4_disagreements.jsonl'},
        'parser': {'failure_rate': fail, 'change_needed': bool(fail > 0.05), 'view_file': 'parser_view.json',
                   'rule': 'design 6.1: more than 5% failures -> a dated parser change in the preregistration'}})
    if not 0.20 <= acc <= 0.80 and ctx.mode == 'dev':
        log('G-D1 failed: the design says run Phase D again with --sender fallback (a new GPU run: [researcher]\'s yes first)')


def block_states(sz, key):
    return sz[f's{SENDERS[key]["blocks"][0]}']


def stage_probes(ctx):
    require_gates(ctx, ['G-D1', 'G-D4'])
    t = time.time()
    sz, sj, ys, its = {}, {}, {}, {}
    for split in ('fit', 'eval'):
        name = ctx.split_name(split)
        sz[split], sj[split] = load_sender_out(ctx.out, name)
        its[split] = ctx.items(split)
        ys[split] = cand_labels(its[split], sj[split], name)
    m_fit, m_eval = cand_mask(sz['fit']), cand_mask(sz['eval'])
    S_fit, S_eval = block_states(sz['fit'], ctx.sender_key), block_states(sz['eval'], ctx.sender_key)
    res, arrays = {}, {}
    for fam in ('logistic', 'mlp'):
        P_fit, P_eval, info = crossfit_probe(fam, S_fit, m_fit, ys['fit'], S_eval, m_eval, DEV_SEED,
                                             ctx.cfg['mlp_max_epochs'])
        arrays[f'{fam}_fit'], arrays[f'{fam}_eval'] = P_fit, P_eval
        res[fam] = dict(info=info, direct_eval=metrics(P_eval[:, 0], ys['eval'][:, 0]))
        log(f'probe {fam}: dev-test NLL {res[fam]["direct_eval"]["nll"]:.4f} AUROC {res[fam]["direct_eval"]["auroc"]:.3f}')
    family = min(res, key=lambda f: res[f]['direct_eval']['nll'])
    np.savez(ctx.out / 'probes.npz', **arrays)
    write_json(ctx.out / 'probes.json', dict(selected=family, families=res, seed=DEV_SEED))
    au = res[family]['direct_eval']['auroc']
    update_gates(ctx, **{'G-D2': {'value': au, 'rule': 'probe direct on c1: dev-test AUROC >= 0.65', 'family': family,
                                  'pass': bool(au >= 0.65)}})
    ctx.add_time('probes_seconds', time.time() - t)


def stage_text(ctx):
    require_gates(ctx, ['G-D1', 'G-D4', 'G-D2'])
    t = time.time()
    pr = read_json(ctx.out / 'probes.json')
    P = np.load(ctx.out / 'probes.npz')
    fam = pr['selected']
    tok, R, yn = load_receiver(ctx.device)
    pad = tok.pad_token_id
    sz, sj, its, ys = {}, {}, {}, {}
    variants = [('T0', None), ('T0c', None), ('T1', None), ('Source', None), ('Q-only', None)]
    variants += [(arm, f) for arm, fs in FORMATS.items() for f in fs]
    n_pass = 0
    for split in ('fit', 'eval'):
        name = ctx.split_name(split)
        sz[split], sj[split] = load_sender_out(ctx.out, name)
        its[split] = ctx.items(split)
        ys[split] = cand_labels(its[split], sj[split], name)
        pv = P[f'{fam}_{split}']
        for arm, f in variants:
            vn = arm if f is None else f'{arm}_{f}'
            path = ctx.out / 'recv' / f'{vn}_{name}.npz'
            if path.exists():
                continue
            texts = text_messages(arm, f, its[split], sj[split], sz[split], pv)
            t1 = time.time()
            H, Z = receiver_text(tok, R, texts, yn, pad, ctx.device, ctx.a.recv_bs)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, h=H.astype(np.float16), z=Z)
            n_pass += len(texts)
            ctx.add_time(f'recv_{vn}_{name}_s_per_pass', (time.time() - t1) / len(texts))
        if split == 'eval':
            write_json(ctx.out / 'messages_eval_examples.json',
                       {(a if f is None else f'{a}_{f}'): text_messages(a, f, its[split][:3], sj[split][:3],
                                                                        {k: v[:3] for k, v in sz[split].items()},
                                                                        pv[:3]) for a, f in variants})
    idc = route_identity(tok, R, text_messages('T0', None, its['eval'][:1], sj['eval'][:1], sz['eval'], None)[0],
                         pad, ctx.device)
    del R
    torch.mps.empty_cache()
    res, probs = {}, {}
    for arm, f in variants:
        vn = arm if f is None else f'{arm}_{f}'
        A = {s: np.load(ctx.out / 'recv' / f'{vn}_{ctx.split_name(s)}.npz') for s in ('fit', 'eval')}
        _, p, info = cv_logistic(A['fit']['h'].astype(np.float64), ys['fit'][:, 0], A['eval']['h'].astype(np.float64),
                                 DEV_SEED)
        probs[vn] = p
        res[vn] = dict(readout=info, eval=metrics(p, ys['eval'][:, 0]))
        log(f'text {vn}: dev-test NLL {res[vn]["eval"]["nll"]:.4f}')
    formats = {arm: min(fs, key=lambda f: res[f'{arm}_{f}']['eval']['nll']) for arm, fs in FORMATS.items()}
    np.savez(ctx.out / 'text_probs_eval.npz', **probs)
    write_json(ctx.out / 'text.json', dict(variants=res, formats=formats, probe_family=fam, route_identity_max_abs=idc,
                                           seed=DEV_SEED))
    src, t0 = res['Source']['eval']['nll'], res['T0']['eval']['nll']
    update_gates(ctx, **{'G-D3': {'value': [src, t0], 'rule': 'dev-test NLL(Source) < NLL(T0)', 'pass': bool(src < t0),
                                  'descriptive': True, 'amendment': 'A1 2026-10-21: Source is descriptive; G-D3 does not stop'}})
    ctx.add_time('text_seconds', time.time() - t)
    ctx.add_time('text_passes', n_pass)


def stage_latent(ctx):
    require_gates(ctx, ['G-D1', 'G-D4', 'G-D2'])   # G-D3 descriptive (amendment A1)
    t = time.time()
    pr, P = read_json(ctx.out / 'probes.json'), np.load(ctx.out / 'probes.npz')
    txt = read_json(ctx.out / 'text.json')
    fam = pr['selected']
    tok, R, yn = load_receiver(ctx.device)
    pad = tok.pad_token_id
    sz, sj, its, ys, lays, mk = {}, {}, {}, {}, {}, {}
    for split in ('fit', 'eval'):
        name = ctx.split_name(split)
        sz[split], sj[split] = load_sender_out(ctx.out, name)
        its[split] = ctx.items(split)
        ys[split] = cand_labels(its[split], sj[split], name)[:, 0]
        lays[split] = make_layouts(tok, its[split], sj[split])
        mk[split] = cand_mask(sz[split])
    X_fit, X_eval = slot_standardize(block_states(sz['fit'], ctx.sender_key), mk['fit'],
                                     block_states(sz['eval'], ctx.sender_key), mk['eval'])
    z_fit, z_eval = probe_z(P[f'{fam}_fit'], mk['fit'], P[f'{fam}_eval'], mk['eval'])
    inner = np.random.default_rng(DEV_SEED).permutation(len(ys['fit']))
    n_va = max(1, int(round(len(inner) * LAT['inner'])))
    va, tr = np.sort(inner[:n_va]), np.sort(inner[n_va:])
    sub = lambda L, idx: [L[i] for i in idx]
    gdir = ctx.out / 'latent_grid'
    gdir.mkdir(parents=True, exist_ok=True)
    norm = None if LATENT_READOUT_INPUT == 'raw' else readout_norm(
        np.load(ctx.out / 'recv' / f'T0c_{ctx.split_name("fit")}.npz')['h'])
    h0 = param_hash(R)

    def fitter(Xa):
        def fit(spec):
            name = spec_name(spec)
            jp = gdir / f'{name}.json'
            if jp.exists():
                return read_json(jp)
            ch, hist, best = train_channel(R, sub(lays['fit'], tr), Xa[tr], ys['fit'][tr], spec, DEV_SEED,
                                           spec['max_epochs'], pad, ctx.device,
                                           val=(sub(lays['fit'], va), Xa[va], ys['fit'][va]), patience=ctx.cfg['patience'],
                                           norm=norm)
            torch.save(best['state'], gdir / f'{name}.pt')
            r = dict(name=name, spec=spec, best_val_nll=best['val_nll'], best_epoch=best['epoch'], epochs_run=len(hist),
                     history=hist, finite=all(math.isfinite(h['train_loss']) for h in hist))
            write_json(jp, r)
            log(f'latent {name}: best inner-val NLL {r["best_val_nll"]:.4f} at epoch {r["best_epoch"]} of {len(hist)} '
                f'({hist[-1]["seconds"]:.0f} s)')
            return r
        return fit

    grids = ctx.cfg['grids']
    L_init = [dict(kind='L', rank=GRID_KEYS[g][1], lr=lr) for g in grids if g != 'lprobe' for lr in grids[g]]
    LP_init = [dict(kind='L-probe', rank=None, lr=lr) for lr in grids['lprobe']]
    me, ee = ctx.cfg['max_epochs'], ctx.cfg['ext_epochs']
    bestL, resL, logL = grid_search(fitter(X_fit), L_init, me, ee)
    bestP, resP, logP = grid_search(fitter(z_fit), LP_init, me, ee)
    ch = make_channel(bestL['spec'], X_fit.shape[-1], R.model.embed_tokens.weight.detach(), norm).to(ctx.device)
    ch.load_state_dict(torch.load(gdir / f'{bestL["name"]}.pt'))
    pL, _, _ = predict_channel(R, ch, lays['eval'], X_eval, yn, pad, ctx.device)
    h1 = param_hash(R)
    pT4 = np.load(ctx.out / 'text_probs_eval.npz')[f'T4_{txt["formats"]["T4"]}']
    diff = item_nll(pT4, ys['eval']) - item_nll(pL, ys['eval'])
    sd = float(diff.std(ddof=1))
    n_pool = len(ctx.all_items()['test_pool'])
    n_test = n_pool if sd > SD_LIMIT else min(N_TEST, n_pool)
    np.savez(ctx.out / 'latent_eval.npz', p_L=pL, p_T4=pT4, y=ys['eval'])
    write_json(ctx.out / 'latent.json', dict(
        selected_L=bestL, selected_L_probe=bestP, rounds_L=logL, rounds_L_probe=logP,
        grid_L={k: {kk: v[kk] for kk in ('spec', 'best_val_nll', 'best_epoch', 'epochs_run', 'finite')} for k, v in resL.items()},
        grid_L_probe={k: {kk: v[kk] for kk in ('spec', 'best_val_nll', 'best_epoch', 'epochs_run', 'finite')}
                      for k, v in resP.items()},
        sd_binary_T4_minus_L=sd, mean_binary_T4_minus_L=float(diff.mean()), n_test=n_test, n_test_pool=n_pool,
        n_rule=f'SD above {SD_LIMIT} -> all {n_pool} pool items; else {N_TEST}', inner_train=len(tr), inner_val=len(va),
        receiver_param_sha256_before=h0, receiver_param_sha256_after=h1,
        dev_test_nll=dict(L=mean_nll(pL, ys['eval']), T4=mean_nll(pT4, ys['eval'])),
        s_per_step=s_per_step(list(resL.values()) + list(resP.values()), len(tr))))
    log(f'latent: L* {bestL["name"]} ({bestL["best_val_nll"]:.4f}); L-probe* {bestP["name"]} ({bestP["best_val_nll"]:.4f}); '
        f'SD {sd:.3f} -> n_test {n_test}')
    ctx.add_time('latent_seconds', time.time() - t)


def s_per_step(results, n_train):
    """Seconds for each train step (with the inner-validation pass of each epoch), the median over configurations."""
    steps = math.ceil(n_train / LAT['bs'])
    rates = [r['history'][-1]['seconds'] / (len(r['history']) * steps) for r in results if r['history']]
    return float(np.median(rates)) if rates else float('nan')


def stage_select(ctx):
    require_gates(ctx, ['G-D1', 'G-D2', 'G-D4'])   # G-D3 descriptive (amendment A1)
    pr, txt, lat = read_json(ctx.out / 'probes.json'), read_json(ctx.out / 'text.json'), read_json(ctx.out / 'latent.json')
    g, times = read_json(ctx.out / 'gates.json'), read_json(ctx.out / 'times.json', {})
    sL, sP = lat['selected_L'], lat['selected_L_probe']
    sel = dict(
        status='DEVELOPMENT (Phase D). Not evidence for or against the claim (design 9.3).' if ctx.mode == 'dev'
        else 'SMOKE (code check only)', time=time.strftime('%Y-%m-%d %H:%M:%S'), mode=ctx.mode,
        sender=ctx.sender_key, sender_repo=SENDERS[ctx.sender_key]['repo'], blocks=list(SENDERS[ctx.sender_key]['blocks']),
        probe_family=pr['selected'], probe_info=pr['families'][pr['selected']]['info'], formats=txt['formats'],
        L=dict(rank=sL['spec']['rank'], lr=sL['spec']['lr'], epochs=sL['best_epoch'], name=sL['name'],
               inner_val_nll=sL['best_val_nll']),
        L_probe=dict(lr=sP['spec']['lr'], epochs=sP['best_epoch'], name=sP['name'], inner_val_nll=sP['best_val_nll']),
        n_test=lat['n_test'], sd_binary_T4_minus_L=lat['sd_binary_T4_minus_L'], gates=g, times=times,
        code_sha256=code_sha(), latent_readout_input=LATENT_READOUT_INPUT, dedupe_questions=DEDUPE_QUESTIONS,
        t5_raw=T5_RAW, t6_count_top=T6_COUNT_TOP,
        dev_test_nll={k: v['eval']['nll'] for k, v in txt['variants'].items()} | {'L': lat['dev_test_nll']['L']},
        projection_hours=projection(times, lat, ctx.mode))
    write_json(ctx.out / 'selection.json', sel)
    lines = ['# Phase D selection (copy into PREREGISTRATION.md; development values, not evidence)', '',
             f"- Sender: {sel['sender_repo']} (blocks {sel['blocks']}).", f"- Probe family: {sel['probe_family']}.",
             f"- Formats: {sel['formats']}.",
             f"- L: rank {sel['L']['rank']}, lr {sel['L']['lr']:g}, {sel['L']['epochs']} epochs.",
             f"- L-probe: lr {sel['L_probe']['lr']:g}, {sel['L_probe']['epochs']} epochs.",
             f"- n_test: {sel['n_test']} (dev-test SD of binary NLL(T4) - NLL(L): {sel['sd_binary_T4_minus_L']:.3f}).",
             f"- Gates: " + ', '.join(f"{k} {'pass' if v.get('pass') else 'FAIL'}" for k, v in g.items() if k.startswith('G-')),
             f"- Code sha256: {sel['code_sha256']}."]
    (ctx.out / 'selection.md').write_text('\n'.join(lines) + '\n')
    log('select: ' + json.dumps({k: sel[k] for k in ('probe_family', 'formats', 'L', 'L_probe', 'n_test')}))


def projection(times, lat, mode):
    """Hours for the later runs, from the measured times (smoke and dev); a planning aid only."""
    s_item = np.mean([v for k, v in times.items() if k.startswith('sender_') and k.endswith('_s_per_item')] or [np.nan])
    s_pass = np.mean([v for k, v in times.items() if k.startswith('recv_')] or [np.nan])
    s_step = lat.get('s_per_step', np.nan) if lat else np.nan
    h = lambda s: float(s / 3600)
    return dict(s_per_item=s_item, s_per_pass=s_pass, s_per_step=s_step,
                D1_3000_items=h(3000 * s_item), D2a_57000_passes=h(57000 * s_pass), C1_4000_items=h(4000 * s_item),
                C2_6000_items=h(6000 * s_item), C3_80000_passes=h(80000 * s_pass),
                per_seed_20000_passes_plus_3_fits_of_6_epochs=h(20000 * s_pass + 3 * 6 * 125 * s_step))


def stage_freeze(ctx):
    fz = HERE / 'FREEZE.json'
    if fz.exists():
        raise SystemExit('FREEZE.json exists. A new freeze needs a new, dated section and [researcher]\'s word.')
    missing = [f for f in FROZEN_FILES if not (HERE / f).exists()]
    if missing:
        raise SystemExit(f'freeze: missing {missing} ([researcher] writes or approves PREREGISTRATION.md from selection.md)')
    sel = ctx.selection()
    if not sel.get('status', '').startswith('DEVELOPMENT'):
        raise SystemExit('freeze: the selection is not a Phase D (dev mode) selection')
    import test_uconfirm
    res = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(test_uconfirm))
    if not res.wasSuccessful():
        raise SystemExit('freeze: the unit tests fail')
    sha = freeze_hashes(ctx)
    write_json(fz, dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), sha256=sha, selection_path=str(ctx.sel_path),
                        tests=dict(run=res.testsRun, failures=len(res.failures), errors=len(res.errors)),
                        rule='Every confirm stage checks these hashes and stops on a difference.'))
    log(f'freeze: FREEZE.json written ({res.testsRun} tests pass)')


def stage_ctext(ctx):
    check_freeze(ctx)
    sel = ctx.selection()
    t = time.time()
    tok, R, yn = load_receiver(ctx.device)
    for split in ('fit', 'eval'):
        name = ctx.split_name(split)
        sz, sj = load_sender_out(ctx.out, name)
        items = ctx.items(split)
        for arm in SEED_FREE_ARMS:
            path = ctx.out / 'recv' / f'{arm}_{name}.npz'
            if path.exists():
                continue
            texts = text_messages(arm, sel['formats'].get(arm), items, sj, sz)
            H, Z = receiver_text(tok, R, texts, yn, tok.pad_token_id, ctx.device, ctx.a.recv_bs)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, h=H.astype(np.float16), z=Z)
            log(f'ctext {arm} {name}: {len(texts)} passes')
    write_json(ctx.out / 'ctext.json', dict(route_identity_max_abs=route_identity(
        tok, R, msg_T0(ctx.items('eval')[0]['question'], 'x'), tok.pad_token_id, ctx.device)))
    ctx.add_time('ctext_seconds', time.time() - t)


def seed_list(ctx):
    return [int(s) for s in ctx.a.seeds.split(',')] if ctx.a.seeds else list(CONFIRM_SEEDS)


def stage_seeds(ctx):
    check_freeze(ctx)
    sel = ctx.selection()
    fam, fmts = sel['probe_family'], sel['formats']
    fit_name, eval_name = ctx.split_name('fit'), ctx.split_name('eval')
    sz, sj = load_sender_out(ctx.out, fit_name)
    zz, zj = load_sender_out(ctx.out, eval_name)
    it_fit, it_eval = ctx.items('fit'), ctx.items('eval')
    y_fit = cand_labels(it_fit, sj, fit_name)
    m_fit, m_eval = cand_mask(sz), cand_mask(zz)
    S_fit, S_eval = block_states(sz, ctx.sender_key), block_states(zz, ctx.sender_key)
    X_fit, X_eval = slot_standardize(S_fit, m_fit, S_eval, m_eval)
    norm = None if LATENT_READOUT_INPUT == 'raw' else readout_norm(
        np.load(ctx.out / 'recv' / f'T0c_{fit_name}.npz')['h'])
    tok = R = None
    for s in seed_list(ctx):
        assert s in CONFIRM_SEEDS, s
        d = ctx.out / f'seed{s}'
        d.mkdir(parents=True, exist_ok=True)
        t = time.time()
        if not (d / 'probes.npz').exists():
            P_fit, P_eval, info = crossfit_probe(fam, S_fit, m_fit, y_fit, S_eval, m_eval, s, ctx.cfg['mlp_max_epochs'])
            np.savez(d / 'probes.npz', fit=P_fit, eval=P_eval)
            write_json(d / 'probes.json', info)
        P = np.load(d / 'probes.npz')
        if tok is None:
            tok, R, yn = load_receiver(ctx.device)
            pad = tok.pad_token_id
            lays_fit, lays_eval = make_layouts(tok, it_fit, sj), make_layouts(tok, it_eval, zj)
        for arm in SEEDED_TEXT_ARMS:
            for split, items, js, zs, name in (('fit', it_fit, sj, sz, fit_name), ('eval', it_eval, zj, zz, eval_name)):
                path = d / f'recv_{arm}_{name}.npz'
                if path.exists():
                    continue
                texts = text_messages(arm, fmts[arm], items, js, zs, P[split])
                H, Z = receiver_text(tok, R, texts, yn, pad, ctx.device, ctx.a.recv_bs)
                np.savez(path, h=H.astype(np.float16), z=Z)
        if not (d / 'latent.npz').exists():
            z_fit, z_eval = probe_z(P['fit'], m_fit, P['eval'], m_eval)
            src_tr, src_te = slot_permutation(m_fit, [s, 1]), slot_permutation(m_eval, [s, 2])
            X_fit_p, X_eval_p = apply_permutation(X_fit, src_tr), apply_permutation(X_eval, src_te)
            specL = dict(kind='L', rank=sel['L']['rank'], lr=sel['L']['lr'])
            specP = dict(kind='L-probe', rank=None, lr=sel['L_probe']['lr'])
            jobs = (('L', specL, X_fit, sel['L']['epochs'], [('L', X_eval), ('L-testshuf', X_eval_p)]),
                    ('L-shuf', specL, X_fit_p, sel['L']['epochs'], [('L-shuf', X_eval_p)]),
                    ('L-probe', specP, z_fit, sel['L_probe']['epochs'], [('L-probe', z_eval)]))
            for arm, spec, X, E, evals in jobs:      # one file for each arm: a stopped job continues at the next arm
                part = d / f'latent_{arm}.npz'
                if part.exists():
                    continue
                h0 = param_hash(R)
                ch, hist, _ = train_channel(R, lays_fit, X, y_fit[:, 0], spec, s, E, pad, ctx.device, norm=norm)
                log(f'seed {s} {arm}: {E} epochs, last train loss {hist[-1]["train_loss"]:.4f} ({hist[-1]["seconds"]:.0f} s)')
                out = {}
                for name, Xe in evals:
                    out[f'p_{name}_eval'], out[f'h_{name}_eval'], out[f'z_{name}_eval'] = predict_channel(
                        R, ch, lays_eval, Xe, yn, pad, ctx.device)
                out[f'p_{arm}_fit'], out[f'h_{arm}_fit'], out[f'z_{arm}_fit'] = predict_channel(
                    R, ch, lays_fit, X, yn, pad, ctx.device)
                np.savez(part, **{k: (v.astype(np.float16) if k.startswith('h_') else v) for k, v in out.items()})
                write_json(d / f'latent_{arm}.json', dict(
                    history=hist, finite=all(math.isfinite(r['train_loss']) for r in hist), spec=spec, epochs=E,
                    receiver_param_sha256_before=h0, receiver_param_sha256_after=param_hash(R)))
                del ch
            out, info = {}, {}
            for arm, *_ in jobs:
                z = np.load(d / f'latent_{arm}.npz')
                out.update({k: z[k] for k in z.files})
                info[arm] = read_json(d / f'latent_{arm}.json')
            for k in ('p', 'h', 'z'):
                out[f'{k}_L-testshuf_fit'] = out[f'{k}_L_fit']
            np.savez(d / 'latent.npz', **out)
            write_json(d / 'latent.json', dict(
                history={a: v['history'] for a, v in info.items()}, finite={a: v['finite'] for a, v in info.items()},
                specs=dict(L=specL, L_probe=specP), epochs=dict(L=sel['L']['epochs'], L_probe=sel['L_probe']['epochs']),
                receiver_param_sha256={a: [v['receiver_param_sha256_before'], v['receiver_param_sha256_after']]
                                       for a, v in info.items()}))
        ctx.add_time(f'seed{s}_seconds', time.time() - t)
        log(f'seed {s}: done ({time.time() - t:.0f} s)')


def stage_analysis(ctx):
    fz = check_freeze(ctx)
    _GUARD['test_labels'] = True
    try:
        analysis_body(ctx, fz)
    finally:
        _GUARD['test_labels'] = False


def analysis_body(ctx, fz):
    sel = ctx.selection()
    seeds = list(CONFIRM_SEEDS)
    fit_name, eval_name = ctx.split_name('fit'), ctx.split_name('eval')
    sz, sj = load_sender_out(ctx.out, fit_name)
    zz, zj = load_sender_out(ctx.out, eval_name)
    it_fit, it_eval = ctx.items('fit'), ctx.items('eval')
    y_fit, y_eval = cand_labels(it_fit, sj, fit_name)[:, 0], cand_labels(it_eval, zj, eval_name)[:, 0]
    yse_fit, yse_eval, se_median = se_labels(sz['se'], zz['se'])
    m_fit, m_eval = cand_mask(sz), cand_mask(zz)
    arms = SEED_FREE_ARMS + SEEDED_TEXT_ARMS + LATENT_ARMS
    p = {a: [] for a in arms + ['probe-direct', 'sender-direct', 'base-rate']}
    pse = {a: [] for a in arms}
    zs = {a: [] for a in arms}
    cr, readout_info = [], {}
    for s in seeds:
        d = ctx.out / f'seed{s}'
        Lz = np.load(d / 'latent.npz')
        Pp = np.load(d / 'probes.npz')
        for a in arms:
            if a in SEED_FREE_ARMS:
                A = {k: np.load(ctx.out / 'recv' / f'{a}_{n}.npz') for k, n in (('fit', fit_name), ('eval', eval_name))}
            elif a in SEEDED_TEXT_ARMS:
                A = {k: np.load(d / f'recv_{a}_{n}.npz') for k, n in (('fit', fit_name), ('eval', eval_name))}
            else:
                A = {k: dict(h=Lz[f'h_{a}_{k}'], z=Lz[f'z_{a}_{k}']) for k in ('fit', 'eval')}
            hf, he = A['fit']['h'].astype(np.float64), A['eval']['h'].astype(np.float64)
            if a in LATENT_ARMS:
                p[a].append(Lz[f'p_{a}_eval'])
            else:
                _, pe, info = cv_logistic(hf, y_fit, he, s)
                p[a].append(pe)
                readout_info[f'{a}_seed{s}'] = info
            pse[a].append(cv_logistic(hf, yse_fit, he, s)[1])
            zs[a].append(torch.softmax(torch.as_tensor(A['eval']['z'], dtype=torch.float64), 1)[:, 0].numpy())
        p['probe-direct'].append(Pp['eval'][:, 0])
        p['sender-direct'].append(cv_logistic(logit(np.exp(sz['logp'][:, :1])), y_fit, logit(np.exp(zz['logp'][:, :1])), s)[1])
        p['base-rate'].append(np.full(len(y_eval), y_fit.mean()))
        cr.append(c_round_cost(Pp['fit'], m_fit, y_fit, Pp['eval'], m_eval, y_eval, sel['formats']['T4'], s))
        log(f'analysis seed {s}: readouts done')
    nl = {a: np.stack([item_nll(v, y_eval) for v in p[a]]) for a in p}
    v3, v3_info = hashes_ok(ctx.out)
    lat = {s: read_json(ctx.out / f'seed{s}' / 'latent.json') for s in seeds}
    mem = {h for s in seeds for pair in lat[s]['receiver_param_sha256'].values() for h in pair}
    v3_info['receiver_in_memory_hashes'] = len(mem)
    v3 = bool(v3 and len(mem) == 1)   # P19: the frozen receiver is the same before and after every latent fit
    fin = {s: lat[s]['finite'] for s in seeds}
    v4 = all(all(f.values()) for f in fin.values())
    c_round = float(np.mean([c for c, _ in cr]))
    rule = reading_rule(nl, c_round, v3, v4)
    per_arm = {}
    for a in p:
        ms = [metrics(v, y_eval) for v in p[a]]
        per_arm[a] = {k: float(np.nanmean([m[k] for m in ms])) for k in ms[0]}
        if a in pse:
            mse = [metrics(v, yse_eval) for v in pse[a]]
            per_arm[a]['y_SE'] = {k: float(np.nanmean([m[k] for m in mse])) for k in mse[0]}
            per_arm[a]['auroc_y_SE'] = per_arm[a]['y_SE']['auroc']
            per_arm[a]['zero_shot'] = metrics(np.mean(zs[a], 0), y_eval)
    paper_view = dict(T6_auroc_y_SE=per_arm['T6']['auroc_y_SE'], L_auroc_y_SE=per_arm['L']['auroc_y_SE'],
                      sentence=("the paper's label is carried by its own samples as text"
                                if per_arm['T6']['auroc_y_SE'] >= per_arm['L']['auroc_y_SE'] else None),
                      paper_table2_auroc=dict(latent=0.874, textual=0.785, scalar=0.667, answer_only=0.618))
    np.savez(ctx.out / 'test_probs.npz', y=y_eval, y_SE=yse_eval,
             **{f'p_{a}_seed{s}': v for a in p for s, v in zip(seeds, p[a])},
             **{f'pse_{a}_seed{s}': v for a in pse for s, v in zip(seeds, pse[a])})
    status = ('CONFIRMATORY (frozen run; the independent audit is still to do)' if ctx.mode == 'confirm'
              else 'SMOKE (code check only; not evidence)')
    res = dict(status=status, freeze=fz, sizes=dict(train=len(y_fit), test=len(y_eval)), seeds=seeds,
               base_rate_train=float(y_fit.mean()), se_median_train=se_median, rule=rule, arms=per_arm,
               paper_view=paper_view, c_round_per_seed=cr, validity_detail=dict(V3=v3_info, V4=fin),
               readout_info=readout_info, selection=sel, code_sha256=code_sha(),
               env=dict(torch=torch.__version__, python=platform.python_version()))
    write_json(ctx.out / 'results.json', res)
    (ctx.out / 'table.md').write_text(table(res) + '\n')
    print(table(res))
    log(f'analysis: case {rule["case"]} ({rule["case_name"]}); status {status}')


def table(res):
    lines = [f"Status: {res['status']}", '', '| arm | NLL | AUROC | Brier | ECE | acc | AUROC y_SE | zero-shot NLL |',
             '|---|---|---|---|---|---|---|---|']
    for a, v in res['arms'].items():
        zs = v.get('zero_shot', {}).get('nll')
        lines.append(f"| {a} | {v['nll']:.4f} | {v['auroc']:.3f} | {v['brier']:.4f} | {v['ece']:.4f} | {v['acc']:.3f} | "
                     f"{v.get('auroc_y_SE', float('nan')):.3f} | {'' if zs is None else f'{zs:.4f}'} |")
    r = res['rule']
    lines += ['', f"Case {r['case']}: {r['case_name']}. S* = {r['s_star']}. Validity: {r['validity']}. "
              f"c_round = {r['c_round']:.4f}.", '', '| A beats B? | mean NLL(B) - NLL(A) | 95% CI | per seed | beats |',
              '|---|---|---|---|---|']
    for c in [r['L_beats_S_star'], r['L_beats_T4'], r['T4_beats_L'], r['L_beats_L_probe']] + r['secondary']:
        lines.append(f"| {c['a']} vs {c['b']} | {c['mean_b_minus_a']:+.4f} | [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] | "
                     f"{', '.join(f'{v:+.3f}' for v in c['per_seed'])} | {c['beats']} |")
    lines += ['', '| route loss: NLL(arm) - NLL(probe direct) | mean | 95% CI |', '|---|---|---|']
    for a, v in r['route_loss'].items():
        lines.append(f"| {a} | {v['mean']:+.4f} | [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] |")
    return '\n'.join(lines)


STAGES = dict(items=stage_items, sender=stage_sender, gates=stage_gates, probes=stage_probes, text=stage_text,
              latent=stage_latent, select=stage_select, freeze=stage_freeze, ctext=stage_ctext, seeds=stage_seeds,
              analysis=stage_analysis)
MODEL_STAGES = {'sender', 'text', 'latent', 'ctext', 'seeds'}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['test', 'smoke', 'dev', 'confirm'])
    ap.add_argument('--stages', default=None, help='comma list; default: all stages of the mode')
    ap.add_argument('--out', default=None, help='default: ./<mode> (./dev_fallback for dev with --sender fallback)')
    ap.add_argument('--data', default=str(HERE / 'data'))
    ap.add_argument('--items', default=str(HERE / 'items'))
    ap.add_argument('--sender', choices=list(SENDERS), default='main', help='smoke and dev; confirm reads the selection')
    ap.add_argument('--dev_dir', default=str(HERE / 'dev'), help='confirm: the Phase D folder with selection.json')
    ap.add_argument('--split', choices=['fit', 'eval'], default=None, help='sender stage: one split only')
    ap.add_argument('--seeds', default=None, help='seeds stage: comma list of confirm seeds (default: all three)')
    ap.add_argument('--sender_bs', type=int, default=16)
    ap.add_argument('--sample_items', type=int, default=4, help='items per sampling batch (x 10 samples); P9')
    ap.add_argument('--recv_bs', type=int, default=64)
    a = ap.parse_args(argv)
    if a.mode == 'test':
        import test_uconfirm
        res = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_uconfirm))
        sys.exit(0 if res.wasSuccessful() else 1)
    if a.mode == 'confirm' and a.sender != 'main':
        ap.error('confirm reads the sender from the Phase D selection')
    stages = a.stages.split(',') if a.stages else MODES[a.mode]['stages']
    bad = [s for s in stages if s not in MODES[a.mode]['stages'] and s != 'items']
    if bad:
        ap.error(f'stages {bad} are not part of mode {a.mode}: {MODES[a.mode]["stages"]}')
    stages = [s for s in list(STAGES) if s in stages]
    if not torch.backends.mps.is_available():
        raise SystemExit('MPS is not available')
    ctx = Ctx(a)
    ctx.out.mkdir(parents=True, exist_ok=True)
    log(f'{STATUS} uconfirm.py {a.mode} stages {stages} out {ctx.out} sender {ctx.sender_key} code {code_sha()[:12]}')
    uses_models = bool(MODEL_STAGES & set(stages))
    if uses_models:
        record_hashes(ctx, 'start')
    for s in stages:
        t = time.time()
        log(f'== stage {s}')
        STAGES[s](ctx)
        log(f'== stage {s} done ({time.time() - t:.0f} s)')
    if uses_models:
        record_hashes(ctx, 'end')
    log('done')


if __name__ == '__main__':
    main()
