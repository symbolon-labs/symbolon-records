"""DRAFT (EXPLORATORY code). Unit tests for uconfirm.py and fetch_data.py: synthetic data and tiny random tensors only.
No model load, no tokenizer load, no data file, no network; CPU only.

    ../../.venv/bin/python -I -B test_uconfirm.py -v      (or: ../../.venv/bin/python -I -B uconfirm.py test)
"""
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

import fetch_data  # noqa: E402
import uconfirm as U  # noqa: E402

CPU = torch.device('cpu')
SUFFIX = '\nIs the proposed answer correct? Answer:'


class FakeTok:
    """Character tokenizer: one id for each character (3 to 52); pad id 0. No merges, so segment ids add up."""
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return type('Enc', (), {'input_ids': [ord(c) % 50 + 3 for c in text]})()


def tiny_llama(seed=0):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=1, max_position_embeddings=512, tie_word_embeddings=True)
    m = LlamaForCausalLM(cfg).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def tiny_qwen3(seed=0):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=3, num_attention_heads=2,
                      num_key_value_heads=1, head_dim=8, max_position_embeddings=512, tie_word_embeddings=True)
    m = Qwen3ForCausalLM(cfg).double().eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


# ---------------------------------------------------------------- answer parser, alias match, label rule

class TestTextRules(unittest.TestCase):
    def test_normalize(self):
        cases = {'The Beatles': 'beatles', 'U.S.A.': 'u s', 'U.S.': 'u s', 'rock_and_roll': 'rock and roll',
                 '  Paris,  France ': 'paris france', '‘Tis': 'tis', 'An apple a day': 'apple day', 'Theatre': 'theatre',
                 '': '', None: ''}
        for s, want in cases.items():
            self.assertEqual(U.normalize_answer(s), want, s)

    def test_alias_match(self):
        al = U.alias_set(dict(aliases=['The Beatles', 'Fab Four', ''], normalized_aliases=['beatles', 'fab four']))
        self.assertEqual(al, {'beatles', 'fab four'})
        self.assertEqual(U.label_alias('the Beatles.', al), 1)
        self.assertEqual(U.label_alias('FAB-FOUR', al), 1)
        self.assertEqual(U.label_alias('Beatle', al), 0)
        self.assertEqual(U.label_alias('', al), 0)

    def test_token_f1_rule(self):
        self.assertEqual(U.token_f1('Kennedy John F', 'john f. kennedy'), 1.0)
        self.assertAlmostEqual(U.token_f1('John Fitzgerald Kennedy', 'john f kennedy'), 2 / 3)
        self.assertEqual(U.label_f1('John Fitzgerald Kennedy', ['john f kennedy']), 0)
        self.assertEqual(U.label_f1('kennedy john f', ['jfk', 'john f kennedy']), 1)
        self.assertEqual(U.label_f1('', ['x']), 0)

    def test_parser(self):
        cases = [(('Paris', True), 'Paris', False), (('Paris.\nThe capital', False), 'Paris', False),
                 ((' Paris. ', True), 'Paris', False), (('U.S.', True), 'U.S', False), (('Paris..', True), 'Paris.', False),
                 (('', True), '', True), (('a b c d e f g h', False), 'a b c d e f g h', True)]
        for (text, ended), want, failed in cases:
            a, f = U.parse_answer(text, ended)
            self.assertEqual(a, want, text)
            self.assertEqual(f['failed'], failed, text)
        self.assertTrue(U.parse_answer('Answer: Paris', True)[1]['prefixed'])
        self.assertTrue(U.parse_answer('x y z', False)[1]['truncated'])
        self.assertFalse(U.parse_answer('x y\n', False)[1]['truncated'])

    def test_candidates(self):
        s = ['Paris', 'Lyon', 'paris.', 'Nice', 'Lyon', '', 'Nice', 'Marseille', 'Lyon', 'Nice']
        self.assertEqual(U.select_candidates('Paris', s), ['Paris', 'Lyon', 'Nice'])
        self.assertEqual(U.select_candidates('Paris', ['Nice', 'Lyon', 'Paris']), ['Paris', 'Nice', 'Lyon'])
        self.assertEqual(U.select_candidates('Paris', ['the Lyon', 'Lyon', 'Lyon']), ['Paris', 'the Lyon'])
        self.assertEqual(U.select_candidates('Paris', ['Paris'] * 9 + ['']), ['Paris'])
        self.assertEqual(U.select_candidates('', ['', 'x']), ['', 'x'])

    def test_entropy_and_se_label(self):
        self.assertEqual(U.sample_entropy(['a'] * 10), 0.0)
        self.assertAlmostEqual(U.sample_entropy(['a'] * 5 + ['the A'] * 0 + ['b'] * 5), math.log(2))
        self.assertAlmostEqual(U.sample_entropy(['a', 'A.', 'b', 'b']), math.log(2))
        yf, ye, med = U.se_labels([0.0, 0.5, 1.0, 1.5], [0.5, 0.76, 2.0])
        self.assertEqual(med, 0.75)
        self.assertEqual(list(yf), [0, 0, 1, 1])
        self.assertEqual(list(ye), [0, 1, 1])

    def test_label_guard(self):
        items, sj = [dict(aliases=['paris'])], [dict(cands=['Paris', 'Lyon'])]
        np.testing.assert_array_equal(U.cand_labels(items, sj, 'dev_test'), [[1, 0, -1]])
        with self.assertRaises(RuntimeError):
            U.cand_labels(items, sj, 'test')


# ---------------------------------------------------------------- message formats of every arm

class TestMessages(unittest.TestCase):
    q, c = 'Who wrote Hamlet?', ['William Shakespeare', 'Marlowe', 'Ben Jonson']
    head = 'Message: Question: Who wrote Hamlet?\nProposed answer: William Shakespeare'

    def test_formats(self):
        self.assertEqual([U.fmt_value(p, 'F1') for p in (0, 0.05, 0.5, 0.73, 1)],
                         ['0 out of 9', '0 out of 9', '5 out of 9', '7 out of 9', '9 out of 9'])
        self.assertEqual([U.fmt_value(p, 'F2') for p in (0, 0.004, 0.736, 1)], ['0%', '0%', '74%', '100%'])
        self.assertEqual([U.fmt_value(p, 'F3') for p in (0, 0.1999, 0.2, 0.4, 0.6, 0.7999, 0.8, 1.0)],
                         ['very unlikely', 'very unlikely', 'unlikely', 'uncertain', 'likely', 'likely', 'very likely',
                          'very likely'])
        self.assertEqual(U.fmt_feature(0.73, 'F1'), [7 / 9])
        self.assertEqual(U.fmt_feature(0.5, 'F3'), [0, 0, 1, 0, 0])

    def test_every_arm(self):
        q, c, h = self.q, self.c, self.head
        self.assertEqual(U.msg_T0(q, c[0]), h + '.' + SUFFIX)
        self.assertEqual(U.msg_T0c(q, c), h + '. Other answers: Marlowe; Ben Jonson.' + SUFFIX)
        self.assertEqual(U.msg_T0c(q, c[:2]), h + '. Other answers: Marlowe.' + SUFFIX)
        self.assertEqual(U.msg_T0c(q, c[:1]), U.msg_T0(q, c[0]))
        self.assertEqual(U.msg_T1(q, c[0], 7), h + ' (7 out of 9).' + SUFFIX)
        self.assertEqual(U.msg_scalar(q, c[0], 0.736, 'F2'), h + ' (74%).' + SUFFIX)
        self.assertEqual(U.msg_scalar(q, c[0], 0.9, 'F3'), h + ' (very likely).' + SUFFIX)
        self.assertEqual(U.msg_dist(q, c, [0.736, 0.2, 0.05], 'F2'),
                         h + ' (74%). Other answers: Marlowe (20%); Ben Jonson (5%).' + SUFFIX)
        self.assertEqual(U.msg_dist(q, c[:2], [0.5, 0.1], 'F1'), h + ' (5 out of 9). Other answers: Marlowe (1 out of 9).'
                         + SUFFIX)
        smp = ['Shakespeare', 'Marlowe', '', 'shakespeare.', 'Bacon', 'Shakespeare', 'Marlowe', 'X', 'Shakespeare', 'Y']
        self.assertEqual(U.msg_T6(q, c[0], smp, 'list'), h + '. Ten sampled answers: Shakespeare; Marlowe; (no answer); '
                         'shakespeare.; Bacon; Shakespeare; Marlowe; X; Shakespeare; Y.' + SUFFIX)
        self.assertEqual(U.msg_T6(q, c[0], smp, 'counts'), h + '. Ten sampled answers: 4 of 10 say Shakespeare; '
                         '2 of 10 say Marlowe; 4 of 10 other.' + SUFFIX)
        self.assertEqual(U.t6_counts(['x'] * 10), '10 of 10 say x')
        self.assertEqual(U.t6_counts(smp, ['William Shakespeare', 'Marlowe']), '2 of 10 say Marlowe; 8 of 10 other')
        self.assertEqual(U.t6_counts(smp, ['Shakespeare', 'Bacon']), '4 of 10 say Shakespeare; 1 of 10 say Bacon; '
                         '5 of 10 other')
        self.assertEqual(U.t6_counts(['A'] * 10), '10 of 10 other')   # 'a' is an article: it normalizes to ''
        self.assertEqual(U.msg_source(q, c[0], 'w1 w2'), h + '.\nEvidence: w1 w2' + SUFFIX)
        self.assertEqual(U.msg_qonly(q), 'Message: Question: Who wrote Hamlet?' + SUFFIX)

    def test_dispatch(self):
        items = [dict(question=self.q, window='ev')]
        sj = [dict(cands=self.c, samples=['x'] * 10)]
        sz = dict(digit_logits=np.eye(10)[[3]], logp=np.log([[0.6, 0.2, 0.1]]))
        pv = np.array([[0.9, 0.3, 0.1]])
        h = self.head
        want = dict(T0=U.msg_T0(self.q, self.c[0]), T0c=U.msg_T0c(self.q, self.c), T1=h + ' (3 out of 9).' + SUFFIX,
                    T2=h + ' (60%).' + SUFFIX, T3=h + ' (90%).' + SUFFIX,
                    T4=U.msg_dist(self.q, self.c, [0.9, 0.3, 0.1], 'F2'), T5=U.msg_dist(self.q, self.c, [0.6, 0.2, 0.1], 'F2'),
                    T6=U.msg_T6(self.q, self.c[0], ['x'] * 10, 'F2'), Source=U.msg_source(self.q, self.c[0], 'ev'))
        want['Q-only'] = U.msg_qonly(self.q)
        for arm, t in want.items():
            self.assertEqual(U.text_messages(arm, 'F2', items, sj, sz, pv)[0], t, arm)
        self.assertIn('Ten sampled answers: 10 of 10 say x.', U.text_messages('T6', 'counts', items, sj, sz)[0])


# ---------------------------------------------------------------- latent slot layout and the receiver route

class TestReceiverTiny(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.R, cls.tok = tiny_llama(), FakeTok()

    def test_token_route_batched_equals_single(self):
        texts = ['Message: abc', 'Message: a longer text here', 'x']
        H, Z = U.receiver_text(self.tok, self.R, texts, [3, 4], 0, CPU, bs=2)
        for i, t in enumerate(texts):
            H1, Z1 = U.receiver_text(self.tok, self.R, [t], [3, 4], 0, CPU, bs=1)
            np.testing.assert_allclose(H[i], H1[0], atol=1e-5)
            np.testing.assert_allclose(Z[i], Z1[0], atol=1e-5)
        self.assertLess(U.route_identity(self.tok, self.R, 'Message: hello', 0, CPU), 1e-5)

    def test_slot_layout(self):
        items = [dict(question='q1'), dict(question='q2'), dict(question='q3')]
        sj = [dict(cands=['A', 'BB', 'C']), dict(cands=['D']), dict(cands=['E', 'F'])]
        lays = U.make_layouts(self.tok, items, sj)
        for (ids, slots), x, r in zip(lays, items, sj):
            self.assertEqual(len(slots), len(r['cands']))
            self.assertEqual([t for t in ids if t >= 0], self.tok(U.msg_T0c(x['question'], r['cands'])).input_ids)
            segs = U.cand_segments(x['question'], r['cands'])
            pos = 0
            for j, s in enumerate(segs[:-1]):
                pos += len(s) + (1 if j else 0)
                self.assertEqual(slots[j], pos)
                self.assertEqual(ids[slots[j]], -1)
            self.assertEqual(len(ids), sum(map(len, segs)) + len(slots))
        V = torch.randn(3, 3, 16)
        x, am = U.embed_batch(self.R.model.embed_tokens, lays, [0, 1, 2], V, 0, CPU)
        emb = self.R.model.embed_tokens.weight
        h_b = U.final_state(self.R, am, inputs_embeds=x)
        for r, (ids, slots) in enumerate(lays):
            self.assertEqual(int(am[r].sum()), len(ids))
            seq = torch.stack([V[r, slots.index(p)] if t < 0 else emb[t] for p, t in enumerate(ids)])
            torch.testing.assert_close(x[r, :len(ids)], seq)
            h1 = U.final_state(self.R, torch.ones(1, len(ids), dtype=torch.long), inputs_embeds=seq[None])
            torch.testing.assert_close(h_b[r], h1[0], atol=1e-5, rtol=1e-5)

    def test_gradient_reaches_present_slots_only(self):
        lays = U.make_layouts(self.tok, [dict(question='q')], [dict(cands=['A', 'B'])])
        V = torch.randn(1, 3, 16, requires_grad=True)
        x, am = U.embed_batch(self.R.model.embed_tokens, lays, [0], V, 0, CPU)
        U.final_state(self.R, am, inputs_embeds=x).sum().backward()
        self.assertGreater(float(V.grad[0, :2].abs().sum()), 0)
        self.assertEqual(float(V.grad[0, 2].abs().sum()), 0.0)

    def test_maps(self):
        emb = self.R.model.embed_tokens.weight.detach()
        for rank in (None, 4):
            sm = U.StateMap(8, emb, rank)
            out = sm(torch.zeros(2, 3, 8))
            self.assertEqual(tuple(out.shape), (2, 3, 16))
            torch.testing.assert_close(out[0, 0], emb.mean(0))
        pm = U.ProbeMap(emb)
        z = torch.tensor([[0.0, 1.0, -2.0]])
        out = pm(z)
        torch.testing.assert_close(out[0, 0], emb.mean(0))
        torch.testing.assert_close(out[0, 2], emb.mean(0) - 2 * pm.w)

    def test_train_channel(self):
        rng = np.random.default_rng(0)
        n = 24
        sj = [dict(cands=['A', 'B', 'C'][:1 + i % 3]) for i in range(n)]
        lays = U.make_layouts(self.tok, [dict(question=f'q{i}') for i in range(n)], sj)
        mask = np.array([[j < len(r['cands']) for j in range(3)] for r in sj])
        X = rng.normal(size=(n, 3, 8)).astype(np.float32) * mask[..., None]
        y = rng.integers(0, 2, n)
        h0 = U.param_hash(self.R)
        args = (self.R, lays[:16], X[:16], y[:16], dict(kind='L', rank=4, lr=1e-2), 7, 3, 0, CPU)
        ch, hist, best = U.train_channel(*args, val=(lays[16:], X[16:], y[16:]), patience=2, bs=8)
        self.assertEqual(U.param_hash(self.R), h0)
        self.assertTrue(all(math.isfinite(h['train_loss']) for h in hist))
        self.assertTrue(1 <= best['epoch'] <= len(hist))
        p, h, z = U.predict_channel(self.R, ch, lays[16:], X[16:], [3, 4], 0, CPU)
        self.assertAlmostEqual(U.mean_nll(p, y[16:]), best['val_nll'], places=6)
        self.assertEqual(h.shape, (8, 16))
        _, hist2, _ = U.train_channel(*args, val=(lays[16:], X[16:], y[16:]), patience=2, bs=8)
        self.assertEqual([r['train_loss'] for r in hist], [r['train_loss'] for r in hist2])
        H = rng.normal(2.0, 3.0, size=(40, 16))
        saved = U.LATENT_READOUT_INPUT
        self.assertIsNone(U.readout_norm(H))
        U.LATENT_READOUT_INPUT = 'standardized'
        try:
            norm = U.readout_norm(H)
        finally:
            U.LATENT_READOUT_INPUT = saved
        ch4, _, _ = U.train_channel(*args, bs=8, norm=norm)
        np.testing.assert_allclose(ch4.mu.numpy(), H.mean(0), rtol=1e-5)
        ch5 = U.make_channel(args[4], 8, self.R.model.embed_tokens.weight.detach(), norm)
        ch5.load_state_dict(ch4.state_dict())
        np.testing.assert_allclose(U.predict_channel(self.R, ch5, lays[16:], X[16:], [3, 4], 0, CPU)[0],
                                   U.predict_channel(self.R, ch4, lays[16:], X[16:], [3, 4], 0, CPU)[0])
        zp = rng.normal(size=(16, 3)).astype(np.float32) * mask[:16]
        ch3, hist3, _ = U.train_channel(self.R, lays[:16], zp, y[:16], dict(kind='L-probe', lr=1e-2), 7, 2, 0, CPU, bs=8)
        self.assertEqual(len(hist3), 2)


# ---------------------------------------------------------------- sender helpers on a tiny random Qwen3

class TestSenderTiny(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = tiny_qwen3()
        rng = np.random.default_rng(0)
        cls.prompts = [[int(t) for t in rng.integers(3, 60, size=n)] for n in (5, 9, 7, 12)]

    def greedy_nocache(self, p, max_new, end):
        ids, out = list(p), []
        for _ in range(max_new):
            with torch.no_grad():
                t = int(self.m(input_ids=torch.tensor([ids])).logits[0, -1].argmax())
            if t in end:
                return out, True
            out.append(t)
            ids.append(t)
        return out, False

    def test_greedy(self):
        for end in ({1, 2}, set(range(3, 64, 2))):
            new, ended = U.decode(self.m, self.prompts, 8, end, 0, CPU)
            for p, n, e in zip(self.prompts, new, ended):
                n1, e1 = U.decode(self.m, [p], 8, end, 0, CPU)
                self.assertEqual((n, e), (n1[0], e1[0]))
                self.assertEqual((n, e), self.greedy_nocache(p, 8, end))

    def test_sampling(self):
        ps = self.prompts * 3
        a = U.decode(self.m, ps, 6, {1}, 0, CPU, rng=np.random.default_rng([1, 2, 3]))
        b = U.decode(self.m, ps, 6, {1}, 0, CPU, rng=np.random.default_rng([1, 2, 3]))
        c = U.decode(self.m, ps, 6, {1}, 0, CPU, rng=np.random.default_rng([1, 2, 4]))
        self.assertEqual(a, b)
        self.assertNotEqual(a[0], c[0])
        n = 3000
        new, _ = U.decode(self.m, [self.prompts[0]] * n, 1, set(), 0, CPU, rng=np.random.default_rng(5))
        with torch.no_grad():
            p = torch.softmax(self.m(input_ids=torch.tensor([self.prompts[0]])).logits[0, -1], -1).numpy()
        freq = np.bincount([x[0] for x in new], minlength=len(p)) / n
        self.assertLess(np.abs(freq - p).max(), 0.02)

    def test_teacher_forced_and_digits(self):
        W = self.m.lm_head.weight.detach().float()
        seqs = [(p, [5, 6, 7][:k], 1) for p, k in zip(self.prompts, (3, 1, 0, 2))]
        lp, st = U.teacher_forced(self.m, seqs, (1, 3), 0, CPU, 3, W)
        for i, (p, c, e) in enumerate(seqs):
            caught = {}
            hooks = [self.m.model.layers[k - 1].register_forward_hook(
                lambda mod, inp, o, k=k: caught.__setitem__(k, o[0] if isinstance(o, tuple) else o)) for k in (1, 3)]
            with torch.no_grad():
                lsm = torch.log_softmax(self.m(input_ids=torch.tensor([p + c + [e]])).logits[0], -1)
            for hk in hooks:
                hk.remove()
            want = sum(float(lsm[len(p) - 1 + t, tok]) for t, tok in enumerate(c + [e]))
            self.assertAlmostEqual(lp[i], want, places=4)
            for k in (1, 3):
                np.testing.assert_allclose(st[k][i], caught[k][0, len(p) + len(c) - 1].float().numpy(), atol=1e-5)
        dl = U.last_token_logits(self.m, self.prompts, [10, 11, 12], 0, CPU, 3, W)
        for i, p in enumerate(self.prompts):
            with torch.no_grad():
                want = self.m(input_ids=torch.tensor([p])).logits[0, -1, [10, 11, 12]].float().numpy()
            np.testing.assert_allclose(dl[i], want, atol=1e-5)


# ---------------------------------------------------------------- items, splits and the source window

def make_row(i, q, aliases, pages):
    k = len(pages)
    return dict(question=q, question_id=f'id{i}', question_source='src',
                entity_pages=dict(doc_source=['W'] * k, filename=['f'] * k, title=['t'] * k, wiki_context=pages),
                answer=dict(aliases=aliases, normalized_aliases=[U.normalize_answer(a) for a in aliases],
                            matched_wiki_entity_name='', normalized_matched_wiki_entity_name='',
                            normalized_value=U.normalize_answer(aliases[0]) if aliases else '', type='W',
                            value=aliases[0] if aliases else ''))


class TestData(unittest.TestCase):
    def test_source_window(self):
        words = [f'w{i}' for i in range(300)]
        page = ' '.join(words[:200] + ['the', 'Blue-Whale,'] + words[202:])
        win, ok = U.source_window(['no match here', page], {'blue whale'})
        self.assertTrue(ok)
        ws = win.split()
        self.assertEqual(len(ws), 120)
        self.assertIn('Blue-Whale,', ws)
        self.assertEqual(ws.index('Blue-Whale,'), 60)
        page = ' '.join(['--', '(', ').'] * 20 + words[:140] + ['Blue', 'whale'] + words[140:])   # 60 words with no token
        win, ok = U.source_window([page], {'blue whale'})
        self.assertTrue(ok)
        self.assertEqual(win.split()[60:62], ['Blue', 'whale'])
        win, ok = U.source_window(['x -- y blue whale ' + ' '.join(words)], {'blue whale'})
        self.assertTrue(ok and win.split()[:5] == ['x', '--', 'y', 'blue', 'whale'])
        win, ok = U.source_window([' '.join(words) + ' blue whale'], {'blue whale'})
        self.assertTrue(ok and win.split()[-2:] == ['blue', 'whale'] and len(win.split()) == 120)
        win, ok = U.source_window([' '.join(words), 'other'], {'orca'})
        self.assertFalse(ok)
        self.assertEqual(win.split(), words[:120])
        win, ok = U.source_window(['a b c d e'], {'c', 'b'})
        self.assertTrue(ok and win == 'a b c d e')

    def write(self, d, name, rows):
        import pyarrow as pa
        import pyarrow.parquet as pq
        p = Path(d) / name
        pq.write_table(pa.Table.from_pylist(rows), p)
        return (p, p.stat().st_size, fetch_data.sha256_file(p))

    def test_build_items(self):
        page = lambda a: f'Some text about {a} and more words here.'
        train = [make_row(i, f'Train question {i}?', [f'Answer {i}'], [page(f'Answer {i}')]) for i in range(40)]
        train[30]['question'] = 'train question 0'           # duplicate of row 0 after normalization
        train[31]['question'] = 'Train   question 1?'         # duplicate of row 1
        train[32] = make_row(32, 'No page?', ['x'], [])
        train[33] = make_row(33, 'No alias?', [], [page('y')])
        val = [make_row(100 + i, f'Validation question {i}?', [f'V {i}'], [page('nothing')]) for i in range(30)]
        for k in range(5):
            val[k]['question'] = f'Train question {k}?'
        val[10]['question'] = 'validation question 11'       # duplicate of val row 11
        sizes = dict(dev_train=10, dev_test=5, train=15)
        with tempfile.TemporaryDirectory() as d:
            files = {'train0': self.write(d, 't0.parquet', train), 'validation': self.write(d, 'v.parquet', val),
                     'train1': (Path(d) / 'absent.parquet', 0, '')}
            items, rep = U.build_items(d, Path(d) / 'items', files=files, sizes=sizes)
            items2, _ = U.build_items(d, Path(d) / 'items2', files=files, sizes=sizes)
            self.assertEqual(items, items2)
            self.assertEqual(json.loads((Path(d) / 'items' / 'items.json').read_text()), items)
            usable = [r for r in train if U.usable(r)]
            self.assertEqual(rep['train0_usable_rows'], 38)
            perm = np.random.default_rng(2026102110).permutation(len(usable))
            want, seen = [], set()
            for i in perm:
                nq = U.normalize_answer(usable[i]['question'])
                if nq not in seen:
                    seen.add(nq)
                    want.append(usable[i]['question_id'])
            got = [x['qid'] for k in ('dev_train', 'dev_test', 'train') for x in items[k]]
            self.assertEqual(got, want[:30])
            self.assertEqual([len(items[k]) for k in ('dev_train', 'dev_test', 'train')], [10, 5, 15])
            used = {U.normalize_answer(x['question']) for k in ('dev_train', 'dev_test', 'train') for x in items[k]}
            test_q = [U.normalize_answer(x['question']) for x in items['test_pool']]
            self.assertFalse(used & set(test_q))
            self.assertEqual(len(test_q), len(set(test_q)))
            self.assertEqual(rep['test_removed_overlap'], sum(U.normalize_answer(r['question']) in used for r in val))
            self.assertEqual(rep['test_removed_duplicates'], 1)
            self.assertEqual(len(items['test_pool']), 30 - rep['test_removed_overlap'] - 1)
            self.assertTrue(all(x['window_match'] for k in ('dev_train', 'train') for x in items[k]))
            self.assertFalse(any(x['window_match'] for x in items['test_pool']))
            self.assertEqual(rep['alias_normalization_agreement'], 1.0)
            with self.assertRaises(SystemExit):        # too few rows and no second shard
                U.build_items(d, Path(d) / 'items3', files=files, sizes=dict(dev_train=30, dev_test=5, train=5))
            bad = dict(files, validation=(files['validation'][0], files['validation'][1] + 1, files['validation'][2]))
            with self.assertRaises(SystemExit):        # size check
                U.build_items(d, Path(d) / 'items4', files=bad, sizes=sizes)

    def test_fetch_guard_and_table(self):
        self.assertEqual(sum(fetch_data.FILES[k][1] for k in fetch_data.ITEMS['rcwiki']), 474_815_053)
        self.assertEqual(fetch_data.REVISION, '0f7faf33a3908546c6fd5b73a660e0f8ff173c2f')
        with self.assertRaises(SystemExit):
            fetch_data.main(['--researcher-said-yes', 'later', '--items', 'rcwiki'])
        with self.assertRaises(SystemExit):
            fetch_data.main(['--researcher-said-yes', 'yes', '--items', 'all'])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'f.bin'
            p.write_bytes(b'abc')
            sha = fetch_data.sha256_file(p)
            self.assertEqual(fetch_data.verify(p, 3, sha), sha)
            for size, h in ((4, sha), (3, '0' * 64)):
                with self.assertRaises(SystemExit):
                    fetch_data.verify(p, size, h)


# ---------------------------------------------------------------- cross-fit, probes and standardization

class TestFits(unittest.TestCase):
    def test_logistic_recovers_weights(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(2000, 3))
        y = (rng.random(2000) < 1 / (1 + np.exp(-(X @ [2.0, -1.0, 0.0] + 0.5)))).astype(float)
        w, b = U.fit_logistic(X, y, 1e-5)
        np.testing.assert_allclose(w.numpy(), [2.0, -1.0, 0.0], atol=0.25)
        self.assertAlmostEqual(float(b), 0.5, delta=0.2)

    def test_crossfit_folds_over_items(self):
        rng = np.random.default_rng(1)
        n_items = 60
        X = np.repeat(np.eye(n_items), 3, axis=0)              # three identical rows (candidates) for each item
        y = np.repeat(rng.integers(0, 2, n_items), 3).astype(float)
        g = np.repeat(np.arange(n_items), 3)
        oof_g, _, info_g = U.cv_logistic(X, y, X[:3], 0, groups=g)
        oof_r, _, info_r = U.cv_logistic(X, y, X[:3], 0, groups=None)
        self.assertGreater(info_g['cv_nll'], 0.6)              # no leak when the folds are over items
        self.assertLess(info_r['cv_nll'], info_g['cv_nll'] - 0.2)   # rows of one item in train and test leak
        fold = U.item_folds(n_items, 0)
        self.assertEqual(sorted(np.bincount(fold)), [12] * 5)
        np.testing.assert_array_equal(fold, U.item_folds(n_items, 0))

    def test_crossfit_probe_shapes_and_mlp(self):
        rng = np.random.default_rng(2)
        n, d = 40, 6
        m_fit = np.arange(3)[None] < rng.integers(1, 4, n)[:, None]
        m_eval = np.arange(3)[None] < rng.integers(1, 4, 10)[:, None]
        S_fit, S_eval = rng.normal(size=(n, 3, d)), rng.normal(size=(10, 3, d))
        y = np.where(m_fit, (S_fit[..., 0] + rng.normal(size=(n, 3)) > 0).astype(int), -1)
        for fam in ('logistic', 'mlp'):
            P_fit, P_eval, info = U.crossfit_probe(fam, S_fit, m_fit, y, S_eval, m_eval, 3, 3)
            self.assertEqual(info['family'], fam)
            self.assertTrue(np.isnan(P_fit[~m_fit]).all() and np.isnan(P_eval[~m_eval]).all())
            self.assertTrue(((P_fit[m_fit] >= 0) & (P_fit[m_fit] <= 1)).all())
            self.assertTrue(((P_eval[m_eval] >= 0) & (P_eval[m_eval] <= 1)).all())
        z_fit, z_eval = U.probe_z(P_fit, m_fit, P_eval, m_eval)
        self.assertAlmostEqual(float(z_fit[m_fit].mean()), 0.0, places=5)
        self.assertAlmostEqual(float(z_fit[m_fit].std()), 1.0, places=4)
        self.assertTrue((z_fit[~m_fit] == 0).all() and (z_eval[~m_eval] == 0).all())
        X_fit, X_eval = U.slot_standardize(S_fit.astype(np.float16), m_fit, S_eval.astype(np.float16), m_eval)
        np.testing.assert_allclose(X_fit[m_fit].mean(0), 0, atol=1e-5)
        self.assertTrue((X_fit[~m_fit] == 0).all() and (X_eval[~m_eval] == 0).all())

    def test_slot_permutation(self):
        mask = np.array([[1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 0]], bool)
        src = U.slot_permutation(mask, [5, 1])
        np.testing.assert_array_equal(src, U.slot_permutation(mask, [5, 1]))
        for j in range(3):
            have = np.flatnonzero(mask[:, j])
            self.assertEqual(sorted(src[have, j]), list(have))
            self.assertTrue((src[~mask[:, j], j] == -1).all())
        X = np.arange(15, dtype=np.float32).reshape(5, 3, 1) + 1
        Xp = U.apply_permutation(X * mask[..., None], src)
        for i in range(5):
            for j in range(3):
                self.assertEqual(Xp[i, j, 0], X[src[i, j], j, 0] if mask[i, j] else 0)


# ---------------------------------------------------------------- metrics, bootstrap, c_round and the reading rule

class TestMetrics(unittest.TestCase):
    def test_auroc_matches_pairs(self):
        rng = np.random.default_rng(0)
        s, y = np.round(rng.random(300), 1), rng.integers(0, 2, 300)
        pos, neg = s[y == 1], s[y == 0]
        brute = (pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()
        self.assertAlmostEqual(U.auroc(s, y), brute, places=12)
        self.assertEqual(U.auroc([1, 2, 3, 4], [0, 0, 1, 1]), 1.0)
        self.assertTrue(math.isnan(U.auroc([1, 2], [1, 1])))

    def test_nll_ece_brier(self):
        self.assertAlmostEqual(float(U.item_nll([0.0], [1])[0]), -math.log(1e-6), places=6)
        self.assertAlmostEqual(U.ece([0.5] * 4, [0, 1, 0, 1]), 0.0)
        self.assertAlmostEqual(U.ece([0.9] * 4, [1, 1, 1, 1]), 0.1)
        m = U.metrics([0.8, 0.2, 0.6, 0.4], [1, 0, 0, 1])
        self.assertAlmostEqual(m['brier'], (0.04 + 0.04 + 0.36 + 0.36) / 4)
        self.assertEqual(m['acc'], 0.5)

    def test_bootstrap(self):
        ci = U.bootstrap_ci({'c': np.full(50, 0.3), 'z': np.zeros(50)}, n_boot=1000)
        self.assertAlmostEqual(ci['c'][0], 0.3)
        self.assertAlmostEqual(ci['c'][1], 0.3)
        self.assertEqual(ci['z'], (0.0, 0.0))
        d = np.random.default_rng(3).normal(0.1, 1, 200)
        a = U.bootstrap_ci({'d': d}, n_boot=2000, chunk=500)['d']
        b = U.bootstrap_ci({'d': d}, n_boot=2000, chunk=2000)['d']
        self.assertEqual(a, b)
        idx = np.random.default_rng(U.BOOT_SEED).integers(0, 200, size=(2000, 200))
        np.testing.assert_allclose(a, np.percentile(d[idx].mean(1), [2.5, 97.5]))

    def test_c_round(self):
        rng = np.random.default_rng(4)
        n = 1200
        P = rng.uniform(0.41, 0.59, (n, 3))
        m = np.arange(3)[None] < rng.integers(1, 4, n)[:, None]
        P[~m] = np.nan
        y = (P[:, 0] > 0.5).astype(int)
        h = n // 2
        c3, info3 = U.c_round_cost(P[:h], m[:h], y[:h], P[h:], m[h:], y[h:], 'F3', 0)
        c2, info2 = U.c_round_cost(P[:h], m[:h], y[:h], P[h:], m[h:], y[h:], 'F2', 0)
        self.assertGreater(c3, 0.3)        # F3 puts every value in "uncertain": the text loses the signal
        self.assertGreaterEqual(c2, 0.0)
        self.assertLess(c2, c3)


ARMS = U.S_ARMS + ['T3', 'T4', 'L', 'L-shuf', 'L-testshuf', 'L-probe', 'Source', 'probe-direct']
BASE = dict(T0=0.30, T0c=0.28, T1=0.29, T2=0.25, T5=0.24, T6=0.26, T3=0.20, T4=0.10, L=0.10, **{'L-probe': 0.12},
            **{'L-shuf': 0.40, 'L-testshuf': 0.80, 'Source': 0.15, 'probe-direct': 0.05})


def nl_set(n=800, seed=0, **change):
    """Synthetic per-item NLL (3 seeds x n items) for every arm; an offset can be a list of three seed offsets."""
    off = dict(BASE, **{k.replace('_', '-'): v for k, v in change.items()})
    rng = np.random.default_rng(seed)
    base = rng.gamma(2.0, 0.25, n)
    return {a: base[None] + np.asarray(off[a], float).reshape(-1, 1) + rng.normal(0, 0.05, (3, n)) for a in ARMS}


class TestReadingRule(unittest.TestCase):
    def rule(self, c_round=0.0, v3=True, v4=True, **change):
        return U.reading_rule(nl_set(**change), c_round, v3, v4, n_boot=2000)

    def test_cases(self):
        self.assertEqual(self.rule()['case'], '2b')
        self.assertEqual(self.rule()['s_star'], 'T5')
        self.assertEqual(self.rule(T4=0.05)['case'], '2a')
        self.assertEqual(self.rule(L=0.03)['case'], '1')
        self.assertEqual(self.rule(L=0.03, c_round=0.5)['case'], '1m')
        self.assertEqual(self.rule(L=0.03, L_probe=0.03)['case'], '1m')
        self.assertEqual(self.rule(L=0.30)['case'], '3')
        self.assertEqual(self.rule(L_shuf=0.10)['case'], '0')
        # Amendment A1 (2026-10-21): Source is descriptive. A bad Source arm no longer makes the case invalid.
        self.assertEqual(self.rule(Source=0.35)['case'], '2b')
        self.assertFalse(self.rule(Source=0.35)['V2_descriptive'])
        self.assertNotIn('V2', self.rule()['validity'])
        self.assertEqual(self.rule(v3=False)['case'], '0')
        self.assertEqual(self.rule(v4=False)['case'], '0')

    def test_beats_conditions(self):
        r = self.rule(L=[0.0, 0.0, 0.11])          # mean and CI pass, but seed 3 has the wrong sign
        c = r['L_beats_T4']
        self.assertGreaterEqual(c['mean_b_minus_a'], 0.02)
        self.assertGreater(c['ci95'][0], 0)
        self.assertLess(min(c['per_seed']), 0)
        self.assertFalse(c['beats'])
        c = self.rule(L=0.085)['L_beats_T4']       # CI above 0, mean below the 0.02 margin
        self.assertGreater(c['ci95'][0], 0)
        self.assertLess(c['mean_b_minus_a'], 0.02)
        self.assertFalse(c['beats'])
        r = self.rule(L=0.03)
        self.assertTrue(r['L_beats_T4']['beats'])
        self.assertEqual(r['T4_beats_L']['mean_b_minus_a'], -r['L_beats_T4']['mean_b_minus_a'])
        self.assertAlmostEqual(r['route_loss']['T4']['mean'], 0.05, delta=0.01)
        self.assertEqual(len(r['secondary']), 18)


# ---------------------------------------------------------------- grid-end rule and weight-hash check

class TestGridAndHashes(unittest.TestCase):
    def fake(self, target, edge_epoch=False):
        calls = []

        def fit(spec):
            calls.append(spec)
            v = (math.log10(spec['lr']) - math.log10(target)) ** 2 + (0 if spec.get('rank') is None else 0.01)
            be = spec['max_epochs'] if edge_epoch else 3
            return dict(name=U.spec_name(spec), spec=spec, best_val_nll=v, best_epoch=be)
        return fit, calls

    def test_lr_rule(self):
        init = [dict(kind='L', rank=None, lr=lr) for lr in U.FULL_LRS] + \
               [dict(kind='L', rank=r, lr=lr) for r in U.RANKS for lr in U.LOW_LRS]
        fit, calls = self.fake(1e-7)
        best, res, log_ = U.grid_search(fit, init, 15, 25)
        self.assertEqual([x['added'] for x in log_], [3e-6, 1e-6])
        self.assertEqual(best['spec']['lr'], 1e-6)
        self.assertEqual(len(calls), 11)
        fit, calls = self.fake(3e-5)
        best, res, log_ = U.grid_search(fit, init, 15, 25)
        self.assertEqual((log_, best['spec']['lr'], len(calls)), ([], 3e-5, 9))
        fit, calls = self.fake(2e-2)
        best, res, log_ = U.grid_search(fit, init, 15, 25)
        self.assertEqual(log_[0]['added'], 1e-2)
        self.assertEqual(log_[0]['group'], ['L', 16])

    def test_epoch_rule(self):
        fit, calls = self.fake(3e-5, edge_epoch=True)
        best, res, log_ = U.grid_search(fit, [dict(kind='L-probe', lr=lr) for lr in U.LPROBE_LRS[:1]] +
                                        [dict(kind='L-probe', lr=lr) for lr in U.LPROBE_LRS[1:]], 15, 25)
        self.assertEqual(log_[-1]['rule'], 'epoch limit raised')
        self.assertEqual(best['spec']['max_epochs'], 25)
        self.assertEqual(sum(1 for r in res.values() if r['spec']['lr'] == best['spec']['lr']), 1)
        self.assertEqual(U.ladder_step(1e-5, -1), 3e-6)
        self.assertEqual(U.ladder_step(3e-3, 1), 1e-2)

    def test_hashes_ok(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'hashes.jsonl'
            p.write_text(json.dumps(dict(files={'a': '1', 'b': '2'})) + '\n')
            self.assertFalse(U.hashes_ok(d)[0])
            p.write_text(p.read_text() + json.dumps(dict(files={'a': '1', 'b': '2'})) + '\n')
            self.assertTrue(U.hashes_ok(d)[0])
            p.write_text(p.read_text() + json.dumps(dict(files={'a': '1', 'b': '3'})) + '\n')
            ok, info = U.hashes_ok(d)
            self.assertFalse(ok)
            self.assertEqual(info['changed'], ['b'])


# ---------------------------------------------------------------- all smoke stages on the CPU with tiny random models

class FakeChatTok:
    """A reversible character tokenizer with the methods that the sender and receiver stages use."""
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


class TestPipelineTiny(unittest.TestCase):
    """Every smoke stage after 'items' (sender to analysis) on synthetic items, with tiny random models on the CPU."""

    def test_smoke_stages(self):
        import argparse
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.manual_seed(1)
        sender = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=3, num_attention_heads=2,
            num_key_value_heads=1, head_dim=8, max_position_embeddings=512, tie_word_embeddings=True, eos_token_id=1,
            pad_token_id=0, bos_token_id=2)).eval()
        receiver = tiny_llama(2)
        saved = (U.load_sender, U.load_receiver, dict(U.SENDERS['main']), U.cv_logistic, U.MLP_WDS)
        cv = U.cv_logistic
        U.cv_logistic = lambda *x, **k: cv(*x, **dict(k, lams=[1e-2, 1.0]))   # fewer fits: this test checks the glue
        U.MLP_WDS = [1e-3]
        U.load_sender = lambda key, device: (FakeChatTok(), sender)
        U.load_receiver = lambda device: (FakeChatTok(), receiver, [5, 6])
        U.SENDERS['main'].update(blocks=(1, 3))
        rng = np.random.default_rng(0)
        mk = lambda i, s: dict(split=s, qid=f'{s}{i}', question=f'what is {i}?', aliases=['zzz'], value='zzz',
                               window='ev ' * 5, window_match=True)
        try:
            with tempfile.TemporaryDirectory() as d:
                d = Path(d)
                items = dict(dev_train=[mk(i, 'dev_train') for i in range(48)], dev_test=[mk(i, 'dev_test') for i in range(16)],
                             train=[], test_pool=[mk(i, 'test') for i in range(5)])
                U.write_json(d / 'items' / 'items.json', items)
                a = argparse.Namespace(mode='smoke', out=str(d / 'smoke'), items=str(d / 'items'), data=str(d / 'data'),
                                       sender='main', dev_dir=str(d / 'dev'), split=None, seeds=None, sender_bs=16,
                                       sample_items=4, recv_bs=64, stages=None)
                ctx = U.Ctx(a)
                ctx.device = CPU
                ctx.out.mkdir(parents=True)
                U.stage_sender(ctx)
                for name in ('dev_train', 'dev_test'):
                    # The tiny random sender gives empty or random answers. For mixed labels, the test gives the
                    # candidates fixed strings (same count) and makes the first candidate right for even items.
                    sj = U.read_json(ctx.out / f'sender_{name}.json')
                    for i, (x, r) in enumerate(zip(items[name], sj)):
                        r['cands'] = [f'ans{i % 3}'] + [f'alt{j}' for j in range(1, len(r['cands']))]
                        r['a1'] = r['cands'][0]
                        x['aliases'] = [f'ans{i % 3}'] if i % 2 == 0 else ['zzz']
                    U.write_json(ctx.out / f'sender_{name}.json', sj)
                U.write_json(d / 'items' / 'items.json', items)
                ctx._items = None
                for stage in ('gates', 'probes', 'text', 'latent', 'select', 'ctext', 'seeds', 'analysis'):
                    U.STAGES[stage](ctx)
                res = U.read_json(ctx.out / 'results.json')
                g = U.read_json(ctx.out / 'gates.json')
                self.assertEqual(g['G-D1']['value'], 0.5)
                self.assertTrue(math.isfinite(g['G-D2']['value']))
                self.assertTrue(res['status'].startswith('SMOKE'))
                self.assertIn(res['rule']['case'], ['0', '1', '1m', '2a', '2b', '3'])
                self.assertFalse(res['rule']['validity']['V3'])     # no weight-hash records in this test
                self.assertEqual(set(res['arms']), set(U.SEED_FREE_ARMS + U.SEEDED_TEXT_ARMS + U.LATENT_ARMS +
                                                       ['probe-direct', 'sender-direct', 'base-rate']))
                tp = np.load(ctx.out / 'test_probs.npz')
                self.assertEqual(tp['p_L_seed20261101'].shape, (16,))
                sel = U.read_json(ctx.out / 'selection.json')
                self.assertEqual(set(sel['formats']), set(U.FORMATS))
                chk = U.read_json(ctx.out / 'smoke_sender_checks.json')
                self.assertEqual(chk['greedy_batched_equals_single'], 1.0)
                self.assertLess(chk['logp_max_abs_diff'], 1e-3)
                sz, sj = U.load_sender_out(ctx.out, 'dev_train')
                self.assertEqual(sz['s1'].shape, (48, 3, 16))
                self.assertTrue(all(len(r['samples']) == 10 and r['cands'][0] == r['a1'] for r in sj))
                self.assertTrue(np.isnan(sz['logp'][~U.cand_mask(sz)]).all())
                self.assertTrue(np.isfinite(sz['logp'][U.cand_mask(sz)]).all())
                for f in FROZEN_FAKE:
                    (d / f).write_text(f)
                U.write_json(d / 'dev' / 'selection.json', sel)
                saved_here = U.HERE
                U.HERE = d
                try:
                    c = U.Ctx(argparse.Namespace(**dict(vars(a), mode='confirm', out=str(d / 'confirm'))))
                    with self.assertRaises(SystemExit):
                        U.check_freeze(c)                   # no FREEZE.json
                    U.write_json(d / 'FREEZE.json', dict(sha256=U.freeze_hashes(c)))
                    self.assertIsNotNone(U.check_freeze(c))
                    (d / 'uconfirm.py').write_text('changed')
                    with self.assertRaises(SystemExit):
                        U.check_freeze(c)
                    # confirm mode on fresh items: the freeze check in every stage, and the blind test labels
                    items['train'] = [dict(mk(i, 'train'), aliases=[f'ans{i % 3}'] if i % 2 == 0 else ['zzz'])
                                      for i in range(48)]
                    items['test_pool'] = [dict(mk(i, 'test'), aliases=[f'ans{i % 3}'] if i % 2 == 0 else ['zzz'])
                                          for i in range(20)]
                    U.write_json(d / 'items' / 'items.json', items)
                    U.write_json(d / 'dev' / 'selection.json', dict(sel, n_test=16, status='DEVELOPMENT (test)'))
                    c = U.Ctx(argparse.Namespace(**dict(vars(a), mode='confirm', out=str(d / 'confirm'))))
                    c.device = CPU
                    c.out.mkdir(parents=True)
                    with self.assertRaises(SystemExit):
                        U.stage_sender(c)                   # FREEZE.json does not match the files now
                    U.write_json(d / 'FREEZE.json', dict(sha256=U.freeze_hashes(c)))
                    U.stage_sender(c)
                    for name in ('train', 'test'):
                        sj = U.read_json(c.out / f'sender_{name}.json')
                        self.assertEqual(len(sj), 48 if name == 'train' else 16)
                        for i, r in enumerate(sj):
                            r['cands'] = [f'ans{i % 3}'] + [f'alt{j}' for j in range(1, len(r['cands']))]
                            r['a1'] = r['cands'][0]
                        U.write_json(c.out / f'sender_{name}.json', sj)
                    for stage in ('ctext', 'seeds', 'analysis'):
                        U.STAGES[stage](c)
                    self.assertFalse(U._GUARD['test_labels'])
                    res = U.read_json(c.out / 'results.json')
                    self.assertTrue(res['status'].startswith('CONFIRMATORY'))
                    self.assertEqual(res['sizes'], dict(train=48, test=16))
                    self.assertEqual(res['base_rate_train'], 0.5)
                finally:
                    U.HERE = saved_here
        finally:
            U.load_sender, U.load_receiver = saved[0], saved[1]
            U.SENDERS['main'] = saved[2]
            U.cv_logistic, U.MLP_WDS = saved[3], saved[4]

    def test_main_refuses_wrong_stages(self):
        with self.assertRaises(SystemExit):
            U.main(['dev', '--stages', 'freeze'])
        with self.assertRaises(SystemExit):
            U.main(['confirm', '--sender', 'fallback'])


FROZEN_FAKE = list(U.FROZEN_FILES)


if __name__ == '__main__':
    unittest.main()
