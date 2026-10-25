"""Unit tests for ugap.py: synthetic data, tiny random models and temporary folders only. No real model load, no
real data file, no network; CPU only. The sealed folder is not read and not changed by these tests.

    ../../.venv/bin/python -I -B test_ugap.py -v      (or: ../../.venv/bin/python -I -B ugap.py test)
"""
import argparse
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import ugap as G  # noqa: E402

U = G.U
CPU = torch.device('cpu')


def tiny_llama(seed=0):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=1, max_position_embeddings=1024, tie_word_embeddings=True)
    m = LlamaForCausalLM(cfg).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def tiny_qwen3(seed=1):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    m = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=3, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, max_position_embeddings=1024, tie_word_embeddings=True, eos_token_id=1,
        pad_token_id=0, bos_token_id=2)).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


class FakeChatTok:
    """A reversible character tokenizer with the methods that the sender and receiver stages use (copy of the sealed
    test helper)."""
    ALPH = 'abcdefghijklmnopqrstuvwxyz0123456789 .,?:\n()%;-ABCDEFGHIJKL'   # ids 3 to 61; id 63 for any other character
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [self.ALPH.index(c) + 3 if c in self.ALPH else 63 for c in text]

    def __call__(self, text, add_special_tokens=False):
        return type('Enc', (), {'input_ids': self.encode(text)})()

    def decode(self, ids):
        return ''.join(self.ALPH[i - 3] if 3 <= i < 3 + len(self.ALPH) else '?' for i in ids)

    def convert_tokens_to_ids(self, t):
        return {'<|im_end|>': 1, '<|endoftext|>': 2}[t]

    def apply_chat_template(self, turns, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        assert not tokenize and add_generation_prompt and not enable_thinking
        return ''.join(f"{t['role'][0]}: {t['content'][-40:]}\n" for t in turns) + 'a:'


def synth_split(rng, n, prefix):
    """Items, sender rows and arrays (blocks 1 and 3, width 16) and probe values for n synthetic items."""
    items, sj = [], []
    for i in range(n):
        k = 1 + i % 3
        cands = [f'{prefix}ans{i % 5}'] + [f'alt{j}' for j in range(1, k)]
        items.append(dict(qid=f'{prefix}{i}', question=f'what is {prefix} {i}?',
                          aliases=[f'{prefix}ans{i % 5}'] if i % 2 == 0 else ['zzz'], value='v'))
        sj.append(dict(a1=cands[0], cands=cands, samples=[cands[0]] * 10, a1_flags=dict(failed=False)))
    nc = np.array([len(r['cands']) for r in sj])
    m = np.arange(3)[None, :] < nc[:, None]
    y = np.array([[1.0 if i % 2 == 0 and j == 0 else 0.0 for j in range(3)] for i in range(n)])
    S = rng.normal(size=(n, 3, 16)).astype(np.float32)
    S[:, :, 0] += 2.0 * y.astype(np.float32)          # a weak signal for the probes
    S = (S * m[..., None]).astype(np.float16)
    sz = dict(n_cand=nc, logp=np.where(m, -1.0, np.nan), s1=S, s3=S, digit_logits=np.zeros((n, 10), np.float32),
              se=np.zeros(n))
    P = np.where(m, rng.uniform(0, 1, size=(n, 3)), np.nan)
    return items, sj, sz, P


# ---------------------------------------------------------------- formats and messages

class TestFormats(unittest.TestCase):
    def test_design_examples(self):
        want = dict(F3='unlikely', D2='0.27', D3='0.270', LOs='logit -1.0', ICD7='unlikely', WD2='unlikely, 0.27',
                    ICD7D2='unlikely, 0.27')
        self.assertEqual({f: G.fmt_value(0.27, f) for f in G.FORMATS}, want)

    def test_boundaries(self):
        icd = {0.0: 'almost no chance', 0.0499: 'almost no chance', 0.05: 'very unlikely', 0.1999: 'very unlikely',
               0.2: 'unlikely', 0.45: 'roughly even chance', 0.5499: 'roughly even chance', 0.55: 'likely',
               0.8: 'very likely', 0.9499: 'very likely', 0.95: 'almost certain', 1.0: 'almost certain'}
        for p, w in icd.items():
            self.assertEqual(G.fmt_value(p, 'ICD7'), w, p)
        for p in (0.0, 0.1, 0.2, 0.39, 0.4, 0.6, 0.8, 1.0):
            self.assertEqual(G.fmt_value(p, 'F3'), U.fmt_value(p, 'F3'))
        self.assertEqual(G.fmt_value(0.0, 'LOs'), 'logit -13.8')
        self.assertEqual(G.fmt_value(1.0, 'LOs'), 'logit 13.8')
        self.assertEqual(G.fmt_value(0.5, 'LOs'), 'logit 0.0')
        self.assertEqual(G.fmt_value(0.999, 'D2'), '1.00')
        self.assertEqual(G.fmt_value(0.0004, 'D3'), '0.000')
        self.assertEqual(G.fmt_value(0.91, 'ICD7D2'), 'very likely, 0.91')
        with self.assertRaises(ValueError):
            G.fmt_value(0.5, 'F2')

    def test_features(self):
        for p in (0.03, 0.27, 0.5, 0.91):
            self.assertEqual(G.fmt_feature(p, 'F3'), U.fmt_feature(p, 'F3'))
            self.assertEqual(G.fmt_feature(p, 'D2'), [float(f'{p:.2f}')])
            self.assertAlmostEqual(G.fmt_feature(p, 'LOs')[0], 1 / (1 + math.exp(-round(math.log(p / (1 - p)), 1))))
            self.assertEqual(sum(G.fmt_feature(p, 'ICD7')), 1.0)
            self.assertEqual(len(G.fmt_feature(p, 'WD2')), 6)
            self.assertEqual(len(G.fmt_feature(p, 'ICD7D2')), 8)

    def test_messages(self):
        rng = np.random.default_rng(0)
        items, sj, sz, P = synth_split(rng, 9, 'a')
        self.assertEqual(G.messages('F3', items, sj, P), U.text_messages('T4', 'F3', items, sj, sz, P))
        self.assertEqual(G.messages('T0c', items, sj), [U.msg_T0c(x['question'], r['cands']) for x, r in zip(items, sj)])
        for fmt in G.CANDIDATES:
            for x, r, t, i in zip(items, sj, G.messages(fmt, items, sj, P), range(9)):
                tags = [G.tag(P[i, j], fmt) for j in range(len(r['cands']))]
                self.assertEqual(t, U.join_tags(U.cand_segments(x['question'], r['cands']), tags))
                stripped = t
                for tg in tags:
                    stripped = stripped.replace(tg, '', 1)
                self.assertEqual(stripped, U.msg_T0c(x['question'], r['cands']))

    def test_token_counts(self):
        rng = np.random.default_rng(0)
        items, sj, sz, P = synth_split(rng, 6, 'a')
        tc = G.token_counts(FakeChatTok(), 'D2', items, sj, P)
        self.assertEqual(tc['slots'], int(sz['n_cand'].sum()))
        self.assertEqual((tc['tag_mean'], tc['tag_max']), (7.0, 7))     # ' (0.27)': seven characters
        t0 = G.token_counts(FakeChatTok(), 'T0c', items, sj, None)
        self.assertEqual((t0['slots'], t0['tag_max']), (0, 0))

    def test_c_round_F3_equals_sealed(self):
        rng = np.random.default_rng(3)
        n = 300
        m = np.arange(3)[None, :] < rng.integers(1, 4, n)[:, None]
        P_fit = np.where(m, rng.uniform(0, 1, (n, 3)), np.nan)
        P_eval = np.where(m[:100], rng.uniform(0, 1, (100, 3)), np.nan)
        y_fit = (rng.uniform(0, 1, n) < np.nan_to_num(P_fit[:, 0])).astype(int)
        y_eval = (rng.uniform(0, 1, 100) < np.nan_to_num(P_eval[:, 0])).astype(int)
        mine, info = G.c_round_format(P_fit, m, y_fit, P_eval, m[:100], y_eval, 'F3', 5)
        want, winfo = U.c_round_cost(P_fit, m, y_fit, P_eval, m[:100], y_eval, 'F3', 5)
        self.assertAlmostEqual(info['clipped'], want, places=12)
        self.assertAlmostEqual(info['nll_formatted'], winfo['nll_formatted'], places=12)
        d2, _ = G.c_round_format(P_fit, m, y_fit, P_eval, m[:100], y_eval, 'D2', 5)
        self.assertLess(abs(d2), 0.01)


# ---------------------------------------------------------------- selection and the gate

class TestGate(unittest.TestCase):
    def setUp(self):
        self.tm = dict(F3=4.0, D2=6.0, D3=7.0, LOs=8.0, ICD7=4.4, WD2=10.0, ICD7D2=10.4)
        self.tx = {f: 10 for f in G.FORMATS}

    def res(self, pooled, f3_test=0.40904):
        r = {f: dict(pooled_nll=v, dev_test_nll=0.5) for f, v in pooled.items()}
        r['F3']['dev_test_nll'] = f3_test
        return r, {f: dict(tag_mean=self.tm[f], tag_max=self.tx[f]) for f in G.FORMATS}

    def test_select(self):
        pooled = dict(F3=0.40, D2=0.41, D3=0.42, LOs=0.43, ICD7=0.390, WD2=0.3895, ICD7D2=0.395)
        f, tied = G.select_format(pooled, self.tm, self.tx)
        self.assertEqual((f, sorted(tied)), ('ICD7', ['ICD7', 'WD2']))   # tie within 0.001: fewer tag tokens
        pooled['WD2'] = 0.3885
        self.assertEqual(G.select_format(pooled, self.tm, self.tx)[0], 'WD2')
        tx = dict(self.tx, WD2=12)                                       # length rule: WD2 not eligible
        self.assertEqual(G.select_format(pooled, self.tm, tx)[0], 'ICD7')
        fo, _ = G.select_format(pooled, self.tm, self.tx, ['D2', 'D3', 'LOs'])
        self.assertEqual(fo, 'D2')
        self.assertEqual(G.select_format(pooled, self.tm, {f: 99 for f in G.FORMATS}), (None, []))

    def test_decision(self):
        base = dict(F3=0.40, D2=0.41, D3=0.42, LOs=0.43, ICD7=0.3949, WD2=0.399, ICD7D2=0.398)
        d = G.gate_decision(*self.res(base))
        self.assertEqual((d['F_star'], d['F_other'], d['decision']), ('ICD7', 'D2', 'go'))
        self.assertTrue(d['gates']['G-F2']['pass'] and d['gates']['G-F1']['pass'])
        d = G.gate_decision(*self.res(dict(base, ICD7=0.3951)))
        self.assertEqual(d['decision'], 'stop')
        self.assertIn('outcome G', d['outcome'])
        d = G.gate_decision(*self.res(base, f3_test=0.40904 + 0.0011))
        self.assertEqual(d['decision'], 'stop')
        self.assertIn('G-F2', d['outcome'])
        d = G.gate_decision(*self.res(base, f3_test=0.40904 - 0.0009))
        self.assertEqual(d['decision'], 'go')


# ---------------------------------------------------------------- fresh items

def make_row(i, q, aliases, pages):
    k = len(pages)
    return dict(question=q, question_id=f'id{i}', question_source='src',
                entity_pages=dict(doc_source=['W'] * k, filename=['f'] * k, title=['t'] * k, wiki_context=pages),
                answer=dict(aliases=aliases, normalized_aliases=[U.normalize_answer(a) for a in aliases],
                            matched_wiki_entity_name='', normalized_matched_wiki_entity_name='',
                            normalized_value=U.normalize_answer(aliases[0]) if aliases else '', type='W',
                            value=aliases[0] if aliases else ''))


def write_parquet(path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), str(path))
    return path, path.stat().st_size, G.sha(path)


def synth_shards():
    page = lambda a: f'Some text about {a} and more words here.'
    t0 = [make_row(i, f'Train question {i}?', [f'Answer {i}'], [page(f'Answer {i}')]) for i in range(60)]
    t0[30]['question'] = 'train question 0'            # duplicate of row 0
    t0[33] = make_row(33, 'No page?', ['x'], [])         # not usable
    val = [make_row(100 + i, f'Validation question {i}?', [f'V {i}'], [page('nothing')]) for i in range(30)]
    val[3]['question'] = 'Train question 4?'             # equal to a shard-0 question
    t1 = [make_row(200 + i, f'Shard one question {i}?', [f'S {i}'], [page('s')]) for i in range(40)]
    t1[5]['question'] = 'Validation question 20?'        # equal to a validation question
    t1[6]['question'] = 'shard one question 7'           # duplicate inside shard 1
    t1[8]['question'] = 'Train question 2?'              # equal to a shard-0 question
    return t0, val, t1


class TestFreshItems(unittest.TestCase):
    def test_shard0_split_matches_build_items(self):
        t0, val, _ = synth_shards()
        sizes = dict(dev_train=10, dev_test=5, train=15)
        with tempfile.TemporaryDirectory() as d:
            files = {'train0': write_parquet(Path(d) / 't0.parquet', t0), 'validation': write_parquet(Path(d) / 'v.parquet', val),
                     'train1': (Path(d) / 'absent.parquet', 0, '')}
            items, _ = U.build_items(d, Path(d) / 'items', files=files, sizes=sizes)
        usable = [r for r in t0 if U.usable(r)]
        picked, rest = G.shard0_split(usable, 30)
        self.assertEqual([usable[i]['question_id'] for i in picked],
                         [x['qid'] for k in ('dev_train', 'dev_test', 'train') for x in items[k]])
        perm = np.random.default_rng(U.PERM_SEED_TRAIN).permutation(len(usable))
        k = list(perm).index(picked[-1])
        self.assertEqual(rest, [int(i) for i in perm[k + 1:]])
        self.assertFalse(set(rest) & set(picked))
        with self.assertRaises(SystemExit):
            G.shard0_split(usable, 100)

    def test_build_fresh(self):
        rec = lambda q, pool: dict(pool=pool, qid=q, question=q, aliases=['a'], value='a')
        spent = {'x', 'y'}
        cands = dict(V=[rec('a', 'V'), rec('b', 'V')], S0=[rec('x', 'S0'), rec('c', 'S0'), rec('A', 'S0')],
                     S1=[rec('d', 'S1'), rec('b', 'S1'), rec('e', 'S1'), rec('f', 'S1'), rec('g', 'S1')])
        out, rep = G.build_fresh(spent, cands, dict(S1=2))
        self.assertEqual([x['qid'] for x in out], ['a', 'b', 'c', 'd', 'e'])
        self.assertEqual([x['pool'] for x in out], ['V', 'V', 'S0', 'S1', 'S1'])
        self.assertEqual(rep['S0'], dict(rows=3, scanned=3, kept=1, removed_spent=1, removed_duplicate=1))
        self.assertEqual(rep['S1'], dict(rows=5, scanned=3, kept=2, removed_spent=0, removed_duplicate=1))

    def test_label_guard(self):
        with self.assertRaises(RuntimeError):
            G.fresh_labels([dict(aliases=['bob'])], [dict(cands=['bob'])])
        G._GUARD['fresh_labels'] = True
        try:
            y = G.fresh_labels([dict(aliases=['bob'])], [dict(cands=['Bob.', 'b'])])
        finally:
            G._GUARD['fresh_labels'] = False
        self.assertEqual(y.tolist(), [[1, 0, -1]])


# ---------------------------------------------------------------- statistics and the reading rule

def slow_two_level(nl, pairs, n_boot, seed):
    arms = sorted({a for p in pairs for a in p})
    n = nl[arms[0]].shape[1]
    rng = np.random.default_rng(seed)
    st = {p: [] for p in pairs}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=(1, n))[0]
        val = {}
        for a in arms:
            S = nl[a].shape[0]
            sd = rng.integers(0, S, size=(1, S))[0]
            val[a] = np.mean([nl[a][s, idx].mean() for s in sd])
        for x, c in pairs:
            st[(x, c)].append(val[x] - val[c])
    return {p: [float(q) for q in np.percentile(v, [2.5, 97.5])] for p, v in st.items()}


class TestStats(unittest.TestCase):
    def test_two_level_equals_slow_version(self):
        rng = np.random.default_rng(0)
        nl = dict(A=rng.normal(0.4, 0.3, (3, 50)), B=rng.normal(0.38, 0.3, (6, 50)), C=rng.normal(0.4, 0.3, (1, 50)))
        pairs = [('A', 'B'), ('C', 'B'), ('A', 'C')]
        fast = G.two_level_ci(nl, pairs, n_boot=300, seed=7, chunk=1)
        slow = slow_two_level(nl, pairs, 300, 7)
        for p in pairs:
            np.testing.assert_allclose(fast[p]['ci95'], slow[p], atol=1e-12)
            self.assertAlmostEqual(fast[p]['mean'], nl[p[0]].mean() - nl[p[1]].mean(), places=12)
        again = G.two_level_ci(nl, pairs, n_boot=300, seed=7, chunk=64)
        self.assertEqual(again, G.two_level_ci(nl, pairs, n_boot=300, seed=7, chunk=64))

    def test_seed_part(self):
        n = 200
        X = np.repeat(np.array([[0.0], [0.01], [0.02]]), n, 1)          # no item variance; seed variance only
        nl = dict(X=X, C=np.zeros((6, n)))
        r = G.two_level_ci(nl, [('X', 'C')], n_boot=2000, seed=1)[('X', 'C')]
        self.assertAlmostEqual(r['mean'], 0.01)
        self.assertGreater(r['ci95'][1] - r['ci95'][0], 0.005)
        self.assertTrue(-1e-12 <= r['ci95'][0] < 0.01 < r['ci95'][1] <= 0.02 + 1e-12, r['ci95'])
        r1 = G.two_level_ci(dict(X=X[:1] + 0.01, C=np.zeros((1, n))), [('X', 'C')], n_boot=500, seed=1)[('X', 'C')]
        np.testing.assert_allclose(r1['ci95'], [0.01, 0.01])

    def test_labels_and_beats(self):
        self.assertEqual(G.labels(dict(mean=0.005, ci95=[-0.001, 0.0099])), dict(closes=True, remains=False))
        self.assertEqual(G.labels(dict(mean=0.005, ci95=[0.001, 0.0100])), dict(closes=False, remains=True))
        self.assertEqual(G.labels(dict(mean=0.0051, ci95=[-0.001, 0.008])), dict(closes=False, remains=False))
        nl = dict(T=np.full((3, 10), 0.30), L=np.full((3, 10), 0.33))
        self.assertTrue(G.beats(nl, 'T', 'L', [0.01, 0.05])['beats'])
        self.assertFalse(G.beats(nl, 'T', 'L', [-0.01, 0.05])['beats'])
        nl['L'][2] = 0.29
        self.assertFalse(G.beats(nl, 'T', 'L', [0.01, 0.05])['beats'])


def rule_arms(rng, n, g_ref, g_star, sd_star=0.05, sd=0.05):
    """Item NLL arrays for every arm of the rule: C* = base + noise; F3 = C* + g_ref; F* (ICD7) = C* + g_star. The noise
    of each seed row has mean 0, so each arm mean is exact."""
    base = rng.uniform(0.1, 0.8, n)

    def noise(S, s=sd):
        e = rng.normal(0, s, (S, n))
        return base + e - e.mean(1, keepdims=True)

    nl = {f: noise(3) + 0.03 for f in G.FORMATS}
    nl['F3'] = noise(3) + g_ref
    nl['ICD7'] = noise(3, sd_star) + g_star
    nl.update({'T0c': noise(3) + 0.05, 'L': noise(3) + 0.04, 'probe-direct': noise(3) - 0.02, 'L-probe': noise(6),
               'L-probe-R': noise(6)})
    return nl


class TestReadingRule(unittest.TestCase):
    V = dict(V1=True, V2=True, V3=True, V4=True)

    def case(self, g_ref, g_star, sd_star=0.05, validity=None, n=2000, seed=0):
        nl = rule_arms(np.random.default_rng(seed), n, g_ref, g_star, sd_star)
        pool = np.array(['V', 'S0', 'S1'] * (n // 3) + ['V'] * (n % 3))
        return G.reading_rule(nl, 'ICD7', 'D2', validity or self.V, pool, n_boot=2000)

    def test_cases(self):
        self.assertEqual(self.case(0.0, 0.0)['case'], 'R')
        self.assertEqual(self.case(0.03, 0.0)['case'], 'A1')
        self.assertEqual(self.case(0.0055, 0.0045)['case'], 'A2')
        self.assertEqual(self.case(0.03, 0.02)['case'], 'B')
        self.assertEqual(self.case(0.03, 0.03)['case'], 'C')
        self.assertEqual(self.case(0.03, 0.006, sd_star=0.5)['case'], 'D')
        self.assertEqual(self.case(0.03, 0.0, validity=dict(self.V, V3=False))['case'], '0')

    def test_fields(self):
        r = self.case(0.03, 0.02)
        p = r['primary']
        self.assertTrue(p['gap_star']['remains'] and p['gain_star']['gains'])
        self.assertAlmostEqual(p['share_of_F3_gap_removed'], p['gain_star']['mean'] / p['gap_ref']['mean'])
        self.assertEqual(len(p['gap_star']['per_seed_c']), 6)
        self.assertEqual(set(r['per_pool']), {'V', 'S0', 'S1'})
        self.assertIn('gap(D2, L-probe-R)', r['secondary'])
        self.assertIn('closes', r['secondary']['gap(ICD7, L-probe)'])
        self.assertEqual([(b['a'], b['b']) for b in r['secondary']['beats']],
                         [('F3', 'L'), ('ICD7', 'L'), ('L', 'F3'), ('L', 'ICD7')])
        self.assertEqual(r['secondary']['beats'][0]['beats'], False)
        self.assertIn('ICD7', G.table(dict(status='x', rule=r, arms={}, c_round_fresh={})))


# ---------------------------------------------------------------- every stage on synthetic data, tiny models, CPU

class TestPipelineTiny(unittest.TestCase):
    def test_stages(self):
        rng = np.random.default_rng(0)
        saved = dict(SEALED=G.SEALED, HERE=G.HERE, DEVICE=G.DEVICE, run_tests=G.run_tests, weight_roots=G.weight_roots,
                     EXPECTED_KEPT=G.EXPECTED_KEPT, N_S1=G.N_S1, F2_TOL=G.F2_TOL, RECV_PARAM_SHA=G.RECV_PARAM_SHA,
                     cv=U.cv_logistic, load_sender=U.load_sender, load_receiver=U.load_receiver,
                     sizes=dict(U.SIZES), sender=dict(U.SENDERS['main']), files=dict(G.fetch_data.FILES))
        receiver, sender = tiny_llama(2), tiny_qwen3(1)
        cv = U.cv_logistic
        U.cv_logistic = lambda *x, **k: cv(*x, **dict(k, lams=[1e-2, 1.0]))   # fewer fits: this test checks the glue
        U.load_sender = lambda key, device: (FakeChatTok(), sender)
        U.load_receiver = lambda device: (FakeChatTok(), receiver, [5, 6])
        U.SENDERS['main'].update(blocks=(1, 3))
        U.SIZES.clear()
        U.SIZES.update(dev_train=10, dev_test=5, train=15)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                sealed, here, wdir = tmp / 'sealed', tmp / 'here', tmp / 'weights'
                G.SEALED, G.HERE, G.DEVICE = sealed, here, CPU
                G.run_tests = lambda: (type('R', (), {'wasSuccessful': lambda self: True})(), dict(run=0))
                for sub, k in (('sender', 2), ('recv', 1)):
                    for j in range(k):
                        (wdir / sub).mkdir(parents=True, exist_ok=True)
                        (wdir / sub / f'w{j}.safetensors').write_text(f'{sub}{j}')
                G.weight_roots = lambda sender=True: ([wdir / 'sender'] if sender else []) + [wdir / 'recv']
                self.build_sealed(rng, sealed, wdir, receiver)
                here.mkdir()
                for f in G.FROZEN_FILES:
                    (here / f).write_text(f'frozen {f}')
                a = argparse.Namespace(recv_bs=64, sender_bs=16, sample_items=4, pool=None, chunks=None, seeds=None,
                                       t0c=False, researcher_said_yes=None)
                with self.assertRaises(SystemExit):
                    G.stage_gate(a)                                     # no FREEZE_D.json
                U.write_json(here / 'FREEZE_D.json', dict(sha256={f: G.sha(here / f) for f in G.FROZEN_FILES}))
                G.F2_TOL = 10.0                                         # the tiny receiver cannot give 0.40904
                G.stage_gate(a)
                sel = U.read_json(here / 'dev' / 'dev_selection.json')
                res = U.read_json(here / 'dev' / 'dev_result.json')
                self.assertEqual(set(res['formats']), set(G.FORMATS))
                self.assertEqual(sel['tag_tokens']['D2'], 7.0)
                for split in ('dev_train', 'dev_test'):
                    self.assertEqual(np.load(here / 'dev' / 'recv' / f'ICD7_{split}.npz')['h'].shape[1], 16)
                with self.assertRaises(SystemExit):
                    G.stage_gate(a)                                     # Phase D' runs one time only
                if sel['decision'] != 'go':
                    self.assertIn('outcome G', sel['outcome'])
                    with self.assertRaises(SystemExit):
                        G.stage_items(a)
                    sel.update(decision='go', F_star='ICD7', F_other='D2')   # the test continues with a forced 'go'
                    U.write_json(here / 'dev' / 'dev_selection.json', sel)
                fstar = sel['F_star']
                # fresh items: shard 1 in ./data of this folder
                _, _, t1 = synth_shards()
                p1 = write_parquet(here / 'data' / 'trivia_qa' / G.fetch_data.FILES['train1'][0], t1)
                G.fetch_data.FILES['train1'] = (G.fetch_data.FILES['train1'][0],) + p1[1:]
                G.N_S1 = 25
                G.EXPECTED_KEPT = dict(V=1, S0=1, S1=1)
                with self.assertRaises(SystemExit):
                    G.stage_items(a)                                     # counts differ from the expected counts
                stop = U.read_json(here / 'fresh' / 'fresh_manifest_STOP.json')
                want = self.brute_force_counts(sealed, t1)
                self.assertEqual(stop['counts'], want)
                G.EXPECTED_KEPT = want
                G.stage_items(a)
                fresh = G.fresh_items()
                self.assertEqual([x['pool'] for x in fresh], ['V'] * want['V'] + ['S0'] * want['S0'] + ['S1'] * want['S1'])
                with self.assertRaises(SystemExit):
                    G.stage_sender(a)                                    # no FREEZE.json
                G.stage_freeze(a)
                G.stage_sender(argparse.Namespace(**dict(vars(a), pool='V')))
                self.assertFalse((here / 'confirm' / 'sender_fresh.npz').exists())
                G.stage_sender(a)
                sj = U.read_json(here / 'confirm' / 'sender_fresh.json')
                self.assertEqual(len(sj), len(fresh))
                for i, (x, r) in enumerate(zip(fresh, sj)):              # mixed labels for the tiny run
                    r['cands'] = [f'f{i % 4}'] + [f'alt{j}' for j in range(1, len(r['cands']))]
                    r['a1'] = r['cands'][0]
                    x['aliases'] = [f'f{i % 4}'] if i % 2 == 0 else ['zzz']
                U.write_json(here / 'confirm' / 'sender_fresh.json', sj)
                fi = U.read_json(here / 'fresh' / 'fresh_items.json')
                fi['items'] = fresh
                U.write_json(here / 'fresh' / 'fresh_items.json', fi)
                with self.assertRaises(SystemExit):
                    G.stage_seeds(a)                                     # fresh_items.json changed after FREEZE.json
                man = U.read_json(here / 'fresh' / 'fresh_manifest.json')
                man['fresh_items_sha256'] = G.sha(here / 'fresh' / 'fresh_items.json')
                U.write_json(here / 'fresh' / 'fresh_manifest.json', man)
                (here / 'FREEZE.json').unlink()
                G.stage_freeze(a)
                G.stage_seeds(argparse.Namespace(**dict(vars(a), t0c=True, seeds='20261101')))
                G.stage_seeds(a)
                for s in G.LP_SEEDS:
                    d = here / 'confirm' / f'seed{s}'
                    self.assertTrue((d / 'latent_L-probe.npz').exists())
                    self.assertEqual((d / 'latent_L.npz').exists(), s in G.TEXT_SEEDS)
                    self.assertEqual((d / f'recv_{fstar}_fresh.npz').exists(), s in G.TEXT_SEEDS)
                G.RECV_PARAM_SHA = U.param_hash(receiver)
                G.stage_analysis(a)
                self.assertFalse(G._GUARD['fresh_labels'])
                r = U.read_json(here / 'confirm' / 'results.json')
                self.assertTrue(r['status'].startswith('CONFIRMATORY'))
                self.assertEqual(r['rule']['validity'], dict(V1=True, V2=True, V3=True, V4=True))
                self.assertIn(r['rule']['case'], G.CASES)
                self.assertEqual(r['sizes']['fresh'], len(fresh))
                self.assertEqual(set(r['arms']), set(G.FORMATS) | {'T0c', 'L', 'probe-direct', 'L-probe', 'L-probe-R'})
                self.assertEqual(r['arms']['L-probe-R']['seeds'], 6)
                fp = np.load(here / 'confirm' / 'fresh_probs.npz')
                self.assertEqual(fp['p_L-probe-R_seed20261106'].shape, (len(fresh),))
                self.assertTrue((here / 'confirm' / 'table.md').exists())
                (here / 'ugap.py').write_text('changed')
                with self.assertRaises(SystemExit):
                    G.stage_analysis(a)
        finally:
            for k in ('SEALED', 'HERE', 'DEVICE', 'run_tests', 'weight_roots', 'EXPECTED_KEPT', 'N_S1', 'F2_TOL',
                      'RECV_PARAM_SHA'):
                setattr(G, k, saved[k])
            U.cv_logistic, U.load_sender, U.load_receiver = saved['cv'], saved['load_sender'], saved['load_receiver']
            U.SIZES.clear()
            U.SIZES.update(saved['sizes'])
            U.SENDERS['main'] = saved['sender']
            G.fetch_data.FILES.clear()
            G.fetch_data.FILES.update(saved['files'])

    def brute_force_counts(self, sealed, t1):
        """An independent count of the fresh pools for the synthetic shards."""
        items = json.loads((sealed / 'items' / 'items.json').read_text())
        n_test = json.loads((sealed / 'dev' / 'selection.json').read_text())['n_test']
        spent = {U.normalize_answer(x['question']) for k in ('dev_train', 'dev_test', 'train') for x in items[k]}
        spent |= {U.normalize_answer(x['question']) for x in items['test_pool'][:n_test]}
        V = [U.normalize_answer(x['question']) for x in items['test_pool'][n_test:]]
        picked = {x['qid'] for k in ('dev_train', 'dev_test', 'train') for x in items[k]}
        t0, _, _ = synth_shards()
        usable0 = [r for r in t0 if U.usable(r)]
        perm = np.random.default_rng(U.PERM_SEED_TRAIN).permutation(len(usable0))
        last = max(k for k, i in enumerate(perm) if usable0[i]['question_id'] in picked)
        seen = set(V)
        s0 = 0
        for i in perm[last + 1:]:
            q = U.normalize_answer(usable0[i]['question'])
            if q not in spent and q not in seen:
                seen.add(q)
                s0 += 1
        usable1 = [r for r in t1 if U.usable(r)]
        s1 = 0
        for i in np.random.default_rng(G.S1_PERM_SEED).permutation(len(usable1)):
            q = U.normalize_answer(usable1[i]['question'])
            if s1 < G.N_S1 and q not in spent and q not in seen:
                seen.add(q)
                s1 += 1
        return dict(V=len(V), S0=s0, S1=s1)

    def build_sealed(self, rng, sealed, wdir, receiver):
        """A synthetic sealed folder: items, dev and train sender outputs, probes, data files, hashes and SEAL.json."""
        t0, val, _ = synth_shards()
        p0 = write_parquet(sealed / 'data' / 'trivia_qa' / G.fetch_data.FILES['train0'][0], t0)
        pv = write_parquet(sealed / 'data' / 'trivia_qa' / G.fetch_data.FILES['validation'][0], val)
        G.fetch_data.FILES['train0'] = (G.fetch_data.FILES['train0'][0],) + p0[1:]
        G.fetch_data.FILES['validation'] = (G.fetch_data.FILES['validation'][0],) + pv[1:]
        files = {k: G.fetch_data.FILES[k] for k in ('train0', 'validation')}
        files = {k: (G.fetch_data.local_path(k, sealed / 'data'),) + v[1:] for k, v in files.items()}
        files['train1'] = (sealed / 'absent.parquet', 0, '')
        items, _ = U.build_items(sealed / 'data', sealed / 'items', files=files, sizes=dict(U.SIZES))
        for sub in ('dev/recv', 'confirm'):
            (sealed / sub).mkdir(parents=True)
        U.write_json(sealed / 'dev' / 'selection.json', dict(n_test=20))
        probes = {}
        for split, prefix, key in (('dev_train', 'd', 'logistic_fit'), ('dev_test', 'e', 'logistic_eval')):
            its = items[split]
            _, sj, sz, probes[key] = synth_split(rng, len(its), prefix)
            for x, r in zip(its, sj):
                x['aliases'] = [r['cands'][0]] if int(x['qid'][2:]) % 2 == 0 else ['zzz']
            np.savez(sealed / 'dev' / f'sender_{split}.npz', **sz)
            U.write_json(sealed / 'dev' / f'sender_{split}.json', sj)
            np.savez(sealed / 'dev' / 'recv' / f'T4_F3_{split}.npz', h=rng.normal(size=(len(its), 16)).astype(np.float16))
        np.savez(sealed / 'dev' / 'probes.npz', **probes)
        its = items['train']
        _, sj, sz, _ = synth_split(rng, len(its), 't')
        for x, r in zip(its, sj):
            x['aliases'] = [r['cands'][0]] if int(x['qid'][2:]) % 2 == 0 else ['zzz']
        U.write_json(sealed / 'items' / 'items.json', items)
        np.savez(sealed / 'confirm' / 'sender_train.npz', **sz)
        U.write_json(sealed / 'confirm' / 'sender_train.json', sj)
        y = U.cand_labels(its, sj, 'train')
        m = U.cand_mask(sz)
        for s in G.TEXT_SEEDS:
            d = sealed / 'confirm' / f'seed{s}'
            d.mkdir()
            P_fit, _, _ = U.crossfit_probe('logistic', sz['s1'], m, y, sz['s1'][:2], m[:2], s, 10)
            np.savez(d / 'probes.npz', fit=P_fit, eval=np.zeros((2, 3)))
            np.savez(d / 'latent_L-probe.npz', **{'p_L-probe_fit': rng.uniform(0, 1, len(its))})
            U.write_json(d / 'latent.json', {})
        hashes = {str(f): G.sha(f) for f in sorted(wdir.rglob('*.safetensors'))}
        (sealed / 'confirm' / 'hashes.jsonl').write_text(json.dumps(dict(files=hashes)) + '\n')
        for f in ('uconfirm.py', 'fetch_data.py', 'dev/text.json', 'dev/text_probs_eval.npz', 'items/manifest.json'):
            if not (sealed / f).exists():
                (sealed / f).write_text('x')
        U.write_json(sealed / 'SEAL.json', dict(root_sha256='r', sha256={f: G.sha(sealed / f) for f in G.SEALED_INPUTS}))
        self.assertTrue(G.seal_check()['ok'])
        (sealed / 'dev' / 'text.json').write_text('y')
        self.assertFalse(G.seal_check()['ok'])
        (sealed / 'dev' / 'text.json').write_text('x')

    def test_main_refuses_wrong_stages(self):
        with self.assertRaises(SystemExit):
            G.main(['dev', '--stages', 'sender'])
        with self.assertRaises(SystemExit):
            G.main(['confirm', '--stages', 'gate'])
        with self.assertRaises(SystemExit):
            G.seed_list('20261107')


if __name__ == '__main__':
    unittest.main()
