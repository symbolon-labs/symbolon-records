#!/usr/bin/env python
"""EXPLORATORY follow-up of the uncertainty pilot: does "T4 beats L" stay when the latent map does not
overfit? See PLAN.md and IMPLEMENTATION_NOTES.md. This file starts as a copy of
../latent_uncertainty_pilot_20261020/pilot.py; the pilot folder does not change.

Stages (each stage keeps its output in --out, and a second run skips a stage that has its output):
  data     synthetic witness cards with the exact Bayes posterior (new seeds)
  sender   Qwen3-1.7B answer, color logits, verbal digit, states after blocks 21 and 28
  probes   block-21 probes for T3 (answer correct) and T4 (true color), cross-fitted on train
  text     SmolLM2-360M final state and zero-shot color logits for T0-T6 and Card
  latent   L (2048 -> 960) and L-lowrank (2048 -> r -> 960), stacked grid on inner validation,
           final fits, shuffle controls
  analysis readouts, metrics, paired bootstrap, reading rule -> results.json
"""
import argparse
import itertools
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

HUB = Path.home() / '.cache/huggingface/hub'
SENDER = HUB / 'models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
RECV = HUB / 'models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/a10cc1512eabd3dde888204e902eca88bddb4951'
DEV = torch.device('mps')
COLORS = ['red', 'blue', 'green', 'yellow']
Q, N_REP = 0.65, 5
BLOCKS = (21, 28)  # 1-indexed; the state after block k is the output of layers[k - 1]
ID_CHARS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
SEEDS = dict(followup=(2026102101, 2026102102), smoke=(2026102191, 2026102192))
SEED = 20261021
SAMPLE_SEED, K_SAMPLES = 2026102103, 8
SENDER_Q = 'Which color is the box most likely? Answer with one word: red, blue, green, or yellow.'
VERBAL_Q = ('How sure are you that your answer is right? Reply with one digit from 0 (not sure at all) '
            'to 9 (completely sure).')
R_PREFIX, R_QUESTION = 'Message:', '\nWhat color is the box? Answer:'
LAMS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
FULL_LRS = [1e-5, 3e-5, 1e-4]  # PLAN.md addition 23:15 (after the smoke timing, before any follow-up data)
LOW_LRS = [3e-4, 1e-3, 3e-3]
RANKS = [16, 64]
FAMILY_LRS = {'L': FULL_LRS, 'L-lowrank': LOW_LRS}
N_FOLDS, N_BOOT, MARGIN, N_BINS = 5, 10_000, 0.02, 15
TEXT_ARMS = ['T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'Card']
LATENT_ARMS = ['L', 'L-lowrank', 'L-shuf', 'L-lowrank-shuf', 'L-testshuf', 'L-lowrank-testshuf']


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# ---------------------------------------------------------------- data

def bayes(reports):
    lp = np.array([sum(math.log(Q) if r == c else math.log((1 - Q) / 3) for r in reports) for c in range(4)])
    p = np.exp(lp - lp.max())
    return p / p.sum()


def make_items(n, seed, taken=()):
    rng = np.random.default_rng(seed)
    items, seen = [], set(taken)
    while len(items) < n:
        bid = ''.join(rng.choice(ID_CHARS, 3))
        if bid in seen:
            continue
        seen.add(bid)
        c = int(rng.integers(4))
        reps = [c if rng.random() < Q else int(rng.choice([k for k in range(4) if k != c])) for _ in range(N_REP)]
        items.append(dict(id=bid, color=c, reports=reps, bayes=bayes(reps).tolist()))
    return items


def bayes9_table():
    """Exact P(c | d) for the rounded Bayes vector d, over all 4^N_REP report sequences (pilot PLAN.md addition)."""
    table = {}
    for r in itertools.product(range(4), repeat=N_REP):
        joint = np.array([np.prod([Q if x == c else (1 - Q) / 3 for x in r]) for c in range(4)]) / 4
        d = tuple(rnd9(p) for p in joint / joint.sum())
        table[d] = table.get(d, 0) + joint
    return {d: v / v.sum() for d, v in table.items()}


def card(x):
    head = f"Box {x['id']}. Five witnesses saw this box. Each witness is right 65% of the time."
    return '\n'.join([head] + [f'Witness {i + 1} says {COLORS[r]}.' for i, r in enumerate(x['reports'])])


def rnd9(p):
    return int(math.floor(9 * p + 0.5))


def msg_answer(a):
    return f'The box is {COLORS[a]}.'


def msg_digit(a, d):
    return f'{msg_answer(a)} Confidence: {d} out of 9.'


def msg_dist(a, p):
    return msg_answer(a) + ' Confidence: ' + ', '.join(f'{c} {rnd9(pc)} out of 9' for c, pc in zip(COLORS, p)) + '.'


def msg_samples(a, smp):
    return msg_answer(a) + ' Eight sampled answers: ' + ', '.join(COLORS[s] for s in smp) + '.'


# ---------------------------------------------------------------- sender

def answer_prompt(tok, x):
    turn = [dict(role='user', content=f'{card(x)}\n\n{SENDER_Q}')]
    return tok.apply_chat_template(turn, tokenize=False, add_generation_prompt=True, enable_thinking=False) + 'The box is'


def verbal_prompt(tok, x, a):
    turns = [dict(role='user', content=f'{card(x)}\n\n{SENDER_Q}'), dict(role='assistant', content=msg_answer(a)),
             dict(role='user', content=VERBAL_Q)]
    return tok.apply_chat_template(turns, tokenize=False, add_generation_prompt=True, enable_thinking=False)


@torch.no_grad()
def sender_stage(splits, out, bs):
    tok = AutoTokenizer.from_pretrained(SENDER, local_files_only=True)
    tok.padding_side = 'right'
    m = AutoModelForCausalLM.from_pretrained(SENDER, local_files_only=True, dtype=torch.bfloat16).to(DEV).eval()
    color_ids = [tok.encode(' ' + c, add_special_tokens=False)[0] for c in COLORS]
    digit_ids = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(10)]
    caught = {}
    for k in BLOCKS:
        m.model.layers[k - 1].register_forward_hook(
            lambda mod, inp, o, k=k: caught.__setitem__(k, o[0] if isinstance(o, tuple) else o))

    def run(prompts, cand, states):
        lg, st = [], {k: [] for k in BLOCKS}
        for i in range(0, len(prompts), bs):
            enc = tok(prompts[i:i + bs], return_tensors='pt', padding=True, add_special_tokens=False).to(DEV)
            last = enc.attention_mask.sum(1) - 1
            ar = torch.arange(len(last), device=DEV)
            h = m.model(**enc, use_cache=False).last_hidden_state[ar, last]
            lg.append(m.lm_head(h)[:, cand].float().cpu())
            if states:
                for k in BLOCKS:
                    st[k].append(caught[k][ar, last].float().cpu())
        return torch.cat(lg).numpy(), {k: torch.cat(v).numpy() for k, v in st.items() if v}

    for split, items in splits.items():
        t0 = time.time()
        color_logits, st = run([answer_prompt(tok, x) for x in items], color_ids, True)
        ans = color_logits.argmax(1)
        digit_logits, _ = run([verbal_prompt(tok, x, int(a)) for x, a in zip(items, ans)], digit_ids, False)
        np.savez(out / f'sender_{split}.npz', color_logits=color_logits, digit_logits=digit_logits,
                 s21=st[21], s28=st[28])
        log(f'sender {split}: n={len(items)} acc={np.mean(ans == [x["color"] for x in items]):.3f} '
            f'{time.time() - t0:.0f}s')
    del m
    torch.mps.empty_cache()


# ---------------------------------------------------------------- logistic fits (CPU, float64)

def standardize(Xtr, *Xs):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-6] = 1.0
    return [(X - mu) / sd for X in (Xtr,) + Xs]


def fit_lr(X, y, K, lam):
    X, y = torch.as_tensor(X, dtype=torch.float64), torch.as_tensor(y, dtype=torch.long)
    W = torch.zeros(X.shape[1], K, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(K, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=1000, history_size=20, tolerance_grad=1e-8,
                            tolerance_change=1e-12, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(X @ W + b, y) + lam * W.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


def predict(Wb, X):
    W, b = Wb
    return torch.softmax(torch.as_tensor(X, dtype=torch.float64) @ W + b, 1).numpy()


def cv_logreg(Xtr, ytr, Xte, K, seed=SEED):
    """L2 logistic, weight by 5-fold CV NLL on train. Returns out-of-fold train probs, test probs, info."""
    Xtr, Xte = standardize(np.asarray(Xtr, np.float64), np.asarray(Xte, np.float64))
    folds = np.random.default_rng(seed).permutation(len(ytr)) % N_FOLDS
    best, cv = None, {}
    for lam in LAMS:
        oof = np.zeros((len(ytr), K))
        for f in range(N_FOLDS):
            tr, va = folds != f, folds == f
            oof[va] = predict(fit_lr(Xtr[tr], ytr[tr], K, lam), Xtr[va])
        nll = float(-np.log(np.clip(oof[np.arange(len(ytr)), ytr], 1e-12, None)).mean())
        cv[str(lam)] = nll
        if best is None or nll < best[0]:
            best = (nll, lam, oof)
    nll, lam, oof = best
    pte = predict(fit_lr(Xtr, ytr, K, lam), Xte)
    return oof, pte, dict(lam=lam, cv_nll=nll, cv_grid=cv)


# ---------------------------------------------------------------- probes and messages

def probes_stage(items, out):
    S = {s: np.load(out / f'sender_{s}.npz') for s in ('train', 'test')}
    y = {s: np.array([x['color'] for x in items[s]]) for s in items}
    ans = {s: S[s]['color_logits'].argmax(1) for s in S}
    corr = {s: (ans[s] == y[s]).astype(int) for s in S}
    p3_tr, p3_te, info3 = cv_logreg(S['train']['s21'], corr['train'], S['test']['s21'], 2)
    p4_tr, p4_te, info4 = cv_logreg(S['train']['s21'], y['train'], S['test']['s21'], 4)
    p4b_tr, p4b_te, info4b = cv_logreg(S['train']['s28'], y['train'], S['test']['s28'], 4)
    np.savez(out / 'probes.npz', p3_train=p3_tr[:, 1], p3_test=p3_te[:, 1], p4_train=p4_tr, p4_test=p4_te,
             p4b28_train=p4b_tr, p4b28_test=p4b_te)
    (out / 'probes.json').write_text(json.dumps(dict(T3=info3, T4=info4, T4_block28=info4b), indent=1))
    log(f'probes: T3 lam={info3["lam"]} cv_nll={info3["cv_nll"]:.4f}; T4 lam={info4["lam"]} '
        f'cv_nll={info4["cv_nll"]:.4f}; T4@28 cv_nll={info4b["cv_nll"]:.4f}')


def sender_softmax(color_logits):
    return torch.softmax(torch.as_tensor(color_logits, dtype=torch.float64), 1).numpy()


def sample_answers(p, split):
    """T6: K_SAMPLES draws from the sender's four-color softmax (temperature 1), from cached logits only."""
    rng = np.random.default_rng([SAMPLE_SEED, ('train', 'test').index(split)])
    return np.stack([rng.choice(4, size=K_SAMPLES, p=pi / pi.sum()) for pi in p])


def messages(items, out):
    P = np.load(out / 'probes.npz')
    msgs, digits, samples = {}, {}, {}
    for s in ('train', 'test'):
        S = np.load(out / f'sender_{s}.npz')
        ans = S['color_logits'].argmax(1)
        p_soft = sender_softmax(S['color_logits'])
        p_tok = p_soft[np.arange(len(ans)), ans]
        d1 = S['digit_logits'].argmax(1)
        d2 = np.array([rnd9(p) for p in p_tok])
        d3 = np.array([rnd9(p) for p in P[f'p3_{s}']])
        d5 = np.array([[rnd9(pc) for pc in p] for p in p_soft])
        samples[s] = sample_answers(p_soft, s)
        msgs[s] = dict(T0=[msg_answer(a) for a in ans],
                       T1=[msg_digit(a, d) for a, d in zip(ans, d1)],
                       T2=[msg_digit(a, d) for a, d in zip(ans, d2)],
                       T3=[msg_digit(a, d) for a, d in zip(ans, d3)],
                       T4=[msg_dist(a, p) for a, p in zip(ans, P[f'p4_{s}'])],
                       T5=[msg_dist(a, p) for a, p in zip(ans, p_soft)],
                       T6=[msg_samples(a, smp) for a, smp in zip(ans, samples[s])],
                       Card=[card(x) for x in items[s]])
        digits[s] = dict(T1=d1.tolist(), T2=d2.tolist(), T3=d3.tolist(), T5=d5.tolist())
    np.savez(out / 't6_samples.npz', train=samples['train'], test=samples['test'])
    return msgs, digits, samples


# ---------------------------------------------------------------- receiver

def load_receiver():
    tok = AutoTokenizer.from_pretrained(RECV, local_files_only=True)
    tok.padding_side = 'right'
    m = AutoModelForCausalLM.from_pretrained(RECV, local_files_only=True, dtype=torch.float32).to(DEV).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    color_ids = [tok.encode(' ' + c, add_special_tokens=False)[0] for c in COLORS]
    return tok, m, color_ids


@torch.no_grad()
def text_stage(msgs, out, bs=64):
    tok, m, color_ids = load_receiver()
    for arm in TEXT_ARMS:
        for s in ('train', 'test'):
            texts = [f'{R_PREFIX} {t}{R_QUESTION}' for t in msgs[s][arm]]
            H, Z = [], []
            for i in range(0, len(texts), bs):
                enc = tok(texts[i:i + bs], return_tensors='pt', padding=True, add_special_tokens=False).to(DEV)
                last = enc.attention_mask.sum(1) - 1
                h = m.model(**enc, use_cache=False).last_hidden_state[torch.arange(len(last), device=DEV), last]
                H.append(h.cpu())
                Z.append(m.lm_head(h)[:, color_ids].cpu())
            np.savez(out / f'recv_{arm}_{s}.npz', h=torch.cat(H).numpy(), z=torch.cat(Z).numpy())
        log(f'text arm {arm} done')
    # identity check: the embedding route must give the same final state as the token route
    enc = tok([f'{R_PREFIX} {msgs["test"]["T0"][0]}{R_QUESTION}'], return_tensors='pt', add_special_tokens=False).to(DEV)
    h_ids = m.model(**enc, use_cache=False).last_hidden_state[0, -1]
    h_emb = m.model(inputs_embeds=m.model.embed_tokens(enc.input_ids), use_cache=False).last_hidden_state[0, -1]
    return float((h_ids - h_emb).abs().max())


class Channel(torch.nn.Module):
    """rank=None: the pilot map 2048 -> 960. rank=r: 2048 -> r (no bias) -> 960 (with bias)."""

    def __init__(self, s_dim, emb, pre, post, rank=None):
        super().__init__()
        e_dim = emb.shape[1]
        if rank is None:
            self.map = torch.nn.Linear(s_dim, e_dim)
            torch.nn.init.normal_(self.map.weight, std=float(emb.std()) / math.sqrt(s_dim))
            bias = self.map.bias
        else:
            down, up = torch.nn.Linear(s_dim, rank, bias=False), torch.nn.Linear(rank, e_dim)
            torch.nn.init.normal_(down.weight, std=1 / math.sqrt(s_dim))
            torch.nn.init.normal_(up.weight, std=float(emb.std()) / math.sqrt(rank))
            self.map = torch.nn.Sequential(down, up)
            bias = up.bias
        with torch.no_grad():
            bias.copy_(emb.mean(0))
        self.readout = torch.nn.Linear(e_dim, 4)
        self.register_buffer('pre', pre)
        self.register_buffer('post', post)

    def inputs(self, s):
        B = len(s)
        return torch.cat([self.pre.expand(B, -1, -1), self.map(s)[:, None], self.post.expand(B, -1, -1)], 1)


def chunks(seq, n):
    seq = list(seq)
    return [seq[i:i + n] for i in range(0, len(seq), n)]


def latent_stage(items, out, args, bs=32):
    tok, R, color_ids = load_receiver()
    emb = R.model.embed_tokens.weight.detach()
    pre = R.model.embed_tokens(tok(R_PREFIX, return_tensors='pt', add_special_tokens=False).input_ids.to(DEV))
    post = R.model.embed_tokens(tok(R_QUESTION, return_tensors='pt', add_special_tokens=False).input_ids.to(DEV))
    S = {s: np.load(out / f'sender_{s}.npz')['s21'] for s in ('train', 'test')}
    Str, Ste = standardize(S['train'].astype(np.float64), S['test'].astype(np.float64))
    ytr = np.array([x['color'] for x in items['train']])
    yte = np.array([x['color'] for x in items['test']])
    T = lambda a: torch.as_tensor(a, dtype=torch.float32, device=DEV)
    s_dim = Str.shape[1]

    def make(spec):
        torch.manual_seed(SEED)
        ch = Channel(s_dim, emb, pre, post, spec['rank']).to(DEV)
        return ch, torch.optim.AdamW(ch.parameters(), lr=spec['lr'], weight_decay=0.01)

    def receiver_h(x):
        return R.model(inputs_embeds=x, use_cache=False).last_hidden_state[:, -1]

    @torch.no_grad()
    def evaluate(chs, Xs, ibs=64):
        """Stacked evaluation: channel k reads Xs[k]. Returns readout probs and zero-shot logits per channel."""
        for ch in chs:
            ch.eval()
        P, Z = [[] for _ in chs], [[] for _ in chs]
        for i in range(0, len(Xs[0]), ibs):
            for chunk in chunks(range(len(chs)), args.stack_max):
                xs = [chs[k].inputs(Xs[k][i:i + ibs]) for k in chunk]
                b = len(xs[0])
                h = receiver_h(torch.cat(xs))
                for c, k in enumerate(chunk):
                    hk = h[c * b:(c + 1) * b]
                    P[k].append(torch.softmax(chs[k].readout(hk), 1).cpu())
                    Z[k].append(R.lm_head(hk)[:, color_ids].cpu())
        return [torch.cat(p).double().numpy() for p in P], [torch.cat(z).numpy() for z in Z]

    def nll(P, y):
        return float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, None)).mean())

    def train(specs, Xs, y, epochs, Xva=None, yva=None, label=''):
        """Train the configurations of specs together. Channel k reads Xs[k] and trains for at most epochs[k]
        epochs; with validation data, it stops after `patience` epochs with no new best validation NLL.
        All configurations get the same batch order (one generator, seeded as in the pilot)."""
        made = [make(sp) for sp in specs]
        chs, opts = [m[0] for m in made], [m[1] for m in made]
        yt = torch.as_tensor(y, device=DEV)
        g = torch.Generator().manual_seed(SEED)
        hist, best = [[] for _ in specs], [(math.inf, 0) for _ in specs]
        active, t0 = list(range(len(specs))), time.time()
        for ep in range(1, max(epochs) + 1):
            active = [k for k in active if ep <= epochs[k]]
            if not active:
                break
            perm = torch.randperm(len(y), generator=g)
            tot = [0.0] * len(specs)
            for ch in chs:
                ch.train()
            for i in range(0, len(y), bs):
                j = perm[i:i + bs].to(DEV)
                for chunk in chunks(active, args.stack_max):
                    h = receiver_h(torch.cat([chs[k].inputs(Xs[k][j]) for k in chunk]))
                    losses = [F.cross_entropy(chs[k].readout(h[c * len(j):(c + 1) * len(j)]), yt[j])
                              for c, k in enumerate(chunk)]
                    for k in chunk:
                        opts[k].zero_grad()
                    torch.stack(losses).sum().backward()
                    for c, k in enumerate(chunk):
                        torch.nn.utils.clip_grad_norm_(chs[k].parameters(), 1.0)
                        opts[k].step()
                        tot[k] += losses[c].item() * len(j)
            Pva = evaluate([chs[k] for k in active], [Xva] * len(active))[0] if Xva is not None else None
            for c, k in enumerate(active):
                row = dict(epoch=ep, train_loss=tot[k] / len(y))
                if Xva is not None:
                    row['val_nll'] = nll(Pva[c], yva)
                    if row['val_nll'] < best[k][0]:
                        best[k] = (row['val_nll'], ep)
                hist[k].append(row)
            n_act = len(active)
            if Xva is not None:
                active = [k for k in active if ep - best[k][1] < args.patience]
            log(f'latent {label} epoch {ep}: {n_act} configs trained, {len(active)} continue ({time.time() - t0:.0f}s)')
        return chs, hist, best

    t0 = time.time()
    if args.bench:
        bench(R, make, receiver_h, T(Str), torch.as_tensor(ytr, device=DEV), bs, args.stack_max)
    inner = np.random.default_rng(SEED).permutation(len(ytr))
    n_va = max(1, len(ytr) // 5)
    va, tr = inner[:n_va], inner[n_va:]
    specs = ([dict(name=f'L_lr{lr:g}', family='L', rank=None, lr=lr) for lr in FULL_LRS] +
             [dict(name=f'L-lowrank_r{r}_lr{lr:g}', family='L-lowrank', rank=r, lr=lr) for r in RANKS for lr in LOW_LRS])
    Xin, Xva = T(Str[tr]), T(Str[va])
    chs, hist, best = train(specs, [Xin] * len(specs), ytr[tr], [args.max_epochs] * len(specs), Xva, ytr[va], 'grid')
    del chs
    grid = {sp['name']: dict(sp, best_val_nll=b[0], best_epoch=b[1], history=h) for sp, h, b in zip(specs, hist, best)}
    for name, v in grid.items():
        log(f'grid {name}: best val NLL {v["best_val_nll"]:.4f} at epoch {v["best_epoch"]} of {len(v["history"])}')
    sel = {fam: min((v for v in grid.values() if v['family'] == fam), key=lambda v: v['best_val_nll'])
           for fam in ('L', 'L-lowrank')}
    selected = min(sel, key=lambda fam: sel[fam]['best_val_nll'])
    (out / 'latent_grid.json').write_text(json.dumps(dict(selected=selected, grid=grid), indent=1))
    log(f'selected: L={sel["L"]["name"]} ({sel["L"]["best_val_nll"]:.4f}), L-lowrank={sel["L-lowrank"]["name"]} '
        f'({sel["L-lowrank"]["best_val_nll"]:.4f}); L* = {selected} ({time.time() - t0:.0f}s)')

    stack_check = None
    if args.check_stack:
        stack_check = {}
        for name in (specs[0]['name'], specs[len(FULL_LRS)]['name']):
            k = [sp['name'] for sp in specs].index(name)
            _, h1, _ = train([specs[k]], [Xin], ytr[tr], [args.max_epochs], Xva, ytr[va], f'alone {name}')
            a, b = [r['val_nll'] for r in grid[name]['history']], [r['val_nll'] for r in h1[0]]
            n = min(len(a), len(b))
            stack_check[name] = dict(epochs_stack=len(a), epochs_alone=len(b),
                                     max_abs_diff_val_nll=float(np.max(np.abs(np.array(a[:n]) - np.array(b[:n])))))
            log(f'stack check {name}: {stack_check[name]}')

    perm_tr = np.random.default_rng(SEED + 1).permutation(len(ytr))
    perm_te = np.random.default_rng(SEED + 2).permutation(len(yte))
    fspecs = [sel['L'], sel['L-lowrank'], sel['L'], sel['L-lowrank']]
    Xfull, Xshuf = T(Str), T(Str[perm_tr])
    chs, hist_f, _ = train(fspecs, [Xfull, Xfull, Xshuf, Xshuf], ytr, [sp['best_epoch'] for sp in fspecs], label='final')
    Xte, Xte_p = T(Ste), T(Ste[perm_te])
    P, Z = evaluate([chs[0], chs[1], chs[0], chs[1], chs[2], chs[3]], [Xte, Xte, Xte_p, Xte_p, Xte_p, Xte_p])
    order = ['L', 'L-lowrank', 'L-testshuf', 'L-lowrank-testshuf', 'L-shuf', 'L-lowrank-shuf']
    log('latent test NLL: ' + ', '.join(f'{a}={nll(p, yte):.4f}' for a, p in zip(order, P)) + f' ({time.time() - t0:.0f}s)')
    key = lambda a: a.replace('-', '_')
    np.savez(out / 'latent.npz', **{f'P_{key(a)}': p for a, p in zip(order, P)}, **{f'Z_{key(a)}': z for a, z in zip(order, Z)})
    nparams = {fam: sum(p.numel() for p in ch.map.parameters()) for fam, ch in zip(('L', 'L-lowrank'), chs[:2])}
    (out / 'latent.json').write_text(json.dumps(dict(
        selected=selected, selection={fam: {k: v for k, v in s.items() if k != 'history'} for fam, s in sel.items()},
        map_params=nparams, readout_params=960 * 4 + 4, n_inner_train=len(tr), n_inner_val=len(va),
        max_epochs=args.max_epochs, patience=args.patience, family_lrs=FAMILY_LRS, ranks=RANKS, grid=grid,
        final_history=dict(zip(['L', 'L-lowrank', 'L-shuf', 'L-lowrank-shuf'], hist_f)), stack_check=stack_check,
        seconds=time.time() - t0), indent=1))


def bench(R, make, receiver_h, X, y, bs, stack_max):
    """Timing only (smoke test): seconds per stacked train step for K configurations."""
    specs = ([dict(rank=None, lr=1e-4)] * 5 + [dict(rank=16, lr=1e-3)] * 4 + [dict(rank=64, lr=1e-3)] * 4)
    for K in (1, 4, 8, 13):
        made = [make(sp) for sp in specs[:K]]
        times = []
        for step in range(7):
            j = torch.randint(0, len(y), (bs,), device=DEV)
            torch.mps.synchronize()
            t = time.time()
            for chunk in chunks(range(K), stack_max):
                h = receiver_h(torch.cat([made[k][0].inputs(X[j]) for k in chunk]))
                loss = sum(F.cross_entropy(made[k][0].readout(h[c * bs:(c + 1) * bs]), y[j]) for c, k in enumerate(chunk))
                for k in chunk:
                    made[k][1].zero_grad()
                loss.backward()
                for k in chunk:
                    made[k][1].step()
            torch.mps.synchronize()
            times.append(time.time() - t)
        log(f'bench K={K}: {np.mean(times[2:]):.3f} s/step (stack_max {stack_max}); '
            f'MPS driver memory {torch.mps.driver_allocated_memory() / 2**30:.2f} GiB')
        del made
        torch.mps.empty_cache()


# ---------------------------------------------------------------- analysis

def metrics(P, y):
    P = np.clip(np.asarray(P, np.float64), 1e-12, None)
    P = P / P.sum(1, keepdims=True)
    n = len(y)
    conf, pred = P.max(1), P.argmax(1)
    hit = (pred == y).astype(float)
    b = np.minimum((conf * N_BINS).astype(int), N_BINS - 1)
    ece = sum(abs(hit[b == k].mean() - conf[b == k].mean()) * (b == k).sum() / n for k in range(N_BINS) if (b == k).any())
    return dict(nll=float(-np.log(P[np.arange(n), y]).mean()), brier=float(((P - np.eye(4)[y]) ** 2).sum(1).mean()),
                acc=float(hit.mean()), ece=float(ece))


def item_nll(P, y):
    P = np.clip(np.asarray(P, np.float64), 1e-12, None)
    P = P / P.sum(1, keepdims=True)
    return -np.log(P[np.arange(len(y)), y])


def auc(score, label):
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float(((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()))


def analysis_stage(items, out, digits, samples, msgs, id_check, args, times):
    ytr = np.array([x['color'] for x in items['train']])
    yte = np.array([x['color'] for x in items['test']])
    probs, zshot, readout_info = {}, {}, {}
    for arm in TEXT_ARMS:
        A = {s: np.load(out / f'recv_{arm}_{s}.npz') for s in ('train', 'test')}
        _, probs[arm], readout_info[arm] = cv_logreg(A['train']['h'], ytr, A['test']['h'], 4)
        zshot[arm] = torch.softmax(torch.as_tensor(A['test']['z'], dtype=torch.float64), 1).numpy()
        log(f'readout {arm}: lam={readout_info[arm]["lam"]} cv_nll={readout_info[arm]["cv_nll"]:.4f}')
    Lz = np.load(out / 'latent.npz')
    lat = json.loads((out / 'latent.json').read_text())
    for arm in LATENT_ARMS:
        probs[arm] = Lz[f'P_{arm.replace("-", "_")}']
        zshot[arm] = torch.softmax(torch.as_tensor(Lz[f'Z_{arm.replace("-", "_")}'], dtype=torch.float64), 1).numpy()
    Ste = np.load(out / 'sender_test.npz')
    Pr = np.load(out / 'probes.npz')
    d9 = np.array([[rnd9(p) for p in x['bayes']] for x in items['test']], np.float64)
    b9 = bayes9_table()
    refs = dict(Bayes=np.array([x['bayes'] for x in items['test']]),
                **{'Bayes-9': d9 / d9.sum(1, keepdims=True), 'Bayes-9-cal': np.array([b9[tuple(int(v) for v in d)] for d in d9])},
                Prior=np.full((len(yte), 4), 0.25),
                sender_softmax_direct=sender_softmax(Ste['color_logits']),
                probe4_block21_direct=Pr['p4_test'], probe4_block28_direct=Pr['p4b28_test'])
    np.savez(out / 'test_probs.npz', y=yte, **{f'readout_{k}': v for k, v in probs.items()},
             **{f'zeroshot_{k}': v for k, v in zshot.items()}, **{f'ref_{k}': v for k, v in refs.items()})

    bayes_nll = metrics(refs['Bayes'], yte)['nll']
    arms = {}
    for arm in probs:
        r, z = metrics(probs[arm], yte), metrics(zshot[arm], yte)
        r['gap_to_bayes_nll'] = r['nll'] - bayes_nll
        z['gap_to_bayes_nll'] = z['nll'] - bayes_nll
        arms[arm] = dict(readout=r, zero_shot=z)
    references = {k: dict(metrics(v, yte), gap_to_bayes_nll=metrics(v, yte)['nll'] - bayes_nll) for k, v in refs.items()}

    nl = {arm: item_nll(probs[arm], yte) for arm in probs}
    boot_idx = np.random.default_rng(SEED).integers(0, len(yte), size=(N_BOOT, len(yte)))

    def diff(a, b):
        d = nl[a] - nl[b]
        bs = d[boot_idx].mean(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return dict(mean=float(d.mean()), ci95=[float(lo), float(hi)])

    differences = {}
    for lat_arm in ('L', 'L-lowrank'):
        for a in ['T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'Card', f'{lat_arm}-shuf', f'{lat_arm}-testshuf']:
            differences[f'{a}_minus_{lat_arm}'] = diff(a, lat_arm)
    for a, b in [('L', 'L-lowrank'), ('T0', 'T2'), ('T2', 'T4'), ('T3', 'T4'), ('T0', 'Card'), ('T5', 'T4'),
                 ('T6', 'T5'), ('T2', 'T5'), ('T6', 'T4'), ('T6', 'T2')]:
        differences[f'{a}_minus_{b}'] = diff(a, b)

    def beats(a, b):
        """a beats b: mean of NLL(b) - NLL(a) >= MARGIN and the lower CI bound > 0."""
        d = differences.get(f'{b}_minus_{a}') or diff(b, a)
        return bool(d['mean'] >= MARGIN and d['ci95'][0] > 0)

    Ls = lat['selected']
    s_sel = lat['selection'][Ls]
    best_text = min(['T0', 'T1', 'T2'], key=lambda a: arms[a]['readout']['nll'])
    shuf_ok = arms[f'{Ls}-shuf']['readout']['nll'] >= math.log(4) - MARGIN
    card_ok = arms['Card']['readout']['nll'] <= arms['T0']['readout']['nll']
    valid = bool(shuf_ok and card_ok)
    t4_beats, l_beats = beats('T4', Ls), beats(Ls, 'T4')
    case = 0 if not valid else 1 if t4_beats else 3 if l_beats else 2
    grid_lrs = FAMILY_LRS[Ls]
    lr_edge = s_sel['lr'] in (min(grid_lrs), max(grid_lrs))
    ep_edge = s_sel['best_epoch'] >= lat['max_epochs']
    b_text, b3, b4 = beats(Ls, best_text), beats(Ls, 'T3'), beats(Ls, 'T4')
    pilot_case = 1 if (b_text and b3 and b4) else 2 if b_text else 3
    secondary = {f'{a} beats {b}': beats(a, b) for a, b in [(Ls, 'T5'), (Ls, 'T6'), ('T4', 'T5'), ('T5', 'T6'), ('T5', 'T2')]}
    secondary.update({f'{b} beats {a}': beats(b, a) for a, b in [(Ls, 'T5'), (Ls, 'T6'), ('T4', 'T5'), ('T5', 'T6'), ('T5', 'T2')]})
    rule = dict(selected_latent=Ls, selected_config=s_sel['name'], case=case,
                case_name={0: 'invalid', 1: 'the T4 win stays', 2: 'no winner by the rule', 3: 'the T4 win reverses'}[case],
                T4_beats_Lstar=t4_beats, Lstar_beats_T4=l_beats, T4_minus_Lstar=diff('T4', Ls),
                fit_limit_not_removed=bool(lr_edge or ep_edge), selected_lr_at_grid_end=bool(lr_edge),
                selected_epoch_at_max=bool(ep_edge), sanity_shuf_arm=f'{Ls}-shuf', sanity_shuf_ok=bool(shuf_ok),
                sanity_Card_ok=bool(card_ok), valid=valid, pilot_rule=dict(
                    best_of_T0_T2=best_text, Lstar_beats_best_text=b_text, Lstar_beats_T3=b3, Lstar_beats_T4=b4,
                    case=pilot_case),
                secondary=secondary, bayes9_clause_applies=bool(case == 3),
                margin_nats=MARGIN, ln4=math.log(4))

    ans = {s: np.load(out / f'sender_{s}.npz')['color_logits'].argmax(1) for s in ('train', 'test')}
    corr_te = (ans['test'] == yte).astype(int)
    hist = lambda v: {str(k): int(c) for k, c in zip(*np.unique(v, return_counts=True))}
    smp_te = samples['test']
    n_ans = (smp_te == ans['test'][:, None]).sum(1)
    d5 = np.array(digits['test']['T5'])
    fit_diag = {}
    for fam in ('L', 'L-lowrank'):
        s = lat['selection'][fam]
        row = lat['grid'][s['name']]['history'][s['best_epoch'] - 1]
        fit_diag[fam] = dict(config=s['name'], lr=s['lr'], rank=s['rank'], best_epoch=s['best_epoch'],
                             best_val_nll=s['best_val_nll'], train_loss_at_best_epoch=row['train_loss'],
                             val_minus_train_at_best_epoch=row['val_nll'] - row['train_loss'],
                             lr_at_grid_end=s['lr'] in (min(FAMILY_LRS[fam]), max(FAMILY_LRS[fam])),
                             map_params=lat['map_params'][fam])
    fit_diag['pilot_L_reference'] = dict(lr=3e-5, best_epoch=5, best_val_nll=0.5180, train_loss_at_best_epoch=0.230,
                                         val_minus_train_at_best_epoch=0.288)
    diagnostics = dict(
        sender_acc=dict(train=float((ans['train'] == ytr).mean()), test=float(corr_te.mean())),
        bayes_argmax_acc_test=float((refs['Bayes'].argmax(1) == yte).mean()),
        bayes9_cost_nll=references['Bayes-9-cal']['nll'] - bayes_nll,
        bayes9_true_color_zero_fraction_test=float((d9[np.arange(len(yte)), yte] == 0).mean()),
        sender_agrees_with_bayes_argmax_test=float((ans['test'] == refs['Bayes'].argmax(1)).mean()),
        digit_histograms_test={k: hist(v) for k, v in digits['test'].items() if k != 'T5'},
        digit_auc_for_sender_correct_test={k: auc(np.array(v, float), corr_te) for k, v in digits['test'].items() if k != 'T5'},
        t5_answer_digit_equals_t2_digit_fraction_test=float((d5[np.arange(len(yte)), ans['test']] == np.array(digits['test']['T2'])).mean()),
        t5_digit_sum_histogram_test=hist(d5.sum(1)),
        t6_answer_count_histogram_test=hist(n_ans),
        t6_all_samples_same_fraction_test=float((smp_te == smp_te[:, :1]).all(1).mean()),
        t6_answer_count_auc_for_sender_correct_test=auc(n_ans.astype(float), corr_te),
        probe3_auc_test=auc(Pr['p3_test'], corr_te),
        readout_info=readout_info, probe_info=json.loads((out / 'probes.json').read_text()),
        latent_info={k: v for k, v in lat.items() if k not in ('grid', 'final_history')},
        latent_grid={k: dict(best_val_nll=v['best_val_nll'], best_epoch=v['best_epoch'], epochs_run=len(v['history']))
                     for k, v in lat['grid'].items()},
        latent_fit=fit_diag,
        embedding_route_identity_max_abs_diff=id_check,
        example_messages_test={arm: msgs['test'][arm][:3] for arm in TEXT_ARMS},
        example_cards_test=[dict(card=card(x), color=COLORS[x['color']], bayes=x['bayes']) for x in items['test'][:3]])

    records = [dict(arm=arm, view=view, metric=k, value=v) for arm in arms for view in ('readout', 'zero_shot')
               for k, v in arms[arm][view].items()]
    records += [dict(arm=k, view='reference', metric=m, value=v) for k, r in references.items() for m, v in r.items()]
    records += [dict(arm=k.replace('_minus_', ' minus '), view='readout', metric='nll_difference_mean', value=v['mean'])
                for k, v in differences.items()]
    res = dict(status='EXPLORATORY', plan='PLAN.md', notes='IMPLEMENTATION_NOTES.md', sizes=dict(train=len(ytr), test=len(yte)),
               seeds=dict(data=SEEDS[args.seeds], other=SEED, t6_samples=SAMPLE_SEED), args=vars(args), stage_seconds=times,
               env=dict(torch=torch.__version__, transformers=transformers.__version__, python=platform.python_version(),
                        sender=str(SENDER), receiver=str(RECV)),
               arms=arms, references=references, differences=differences, reading_rule=rule,
               diagnostics=diagnostics, records=records)
    (out / 'results.json').write_text(json.dumps(res, indent=1))
    return res


def table(res):
    lines = ['| arm | NLL | gap to Bayes | Brier | acc | ECE | zero-shot NLL | zero-shot acc |', '|---|---|---|---|---|---|---|---|']
    for arm, v in res['arms'].items():
        r, z = v['readout'], v['zero_shot']
        lines.append(f"| {arm} | {r['nll']:.4f} | {r['gap_to_bayes_nll']:.4f} | {r['brier']:.4f} | {r['acc']:.3f} | "
                     f"{r['ece']:.4f} | {z['nll']:.4f} | {z['acc']:.3f} |")
    for k, r in res['references'].items():
        lines.append(f"| ref: {k} | {r['nll']:.4f} | {r['gap_to_bayes_nll']:.4f} | {r['brier']:.4f} | {r['acc']:.3f} | {r['ece']:.4f} | | |")
    lines += ['', '| difference (NLL) | mean | 95% CI |', '|---|---|---|']
    for k, d in res['differences'].items():
        lines.append(f"| {k.replace('_minus_', ' minus ')} | {d['mean']:+.4f} | [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] |")
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_train', type=int, default=2000)
    ap.add_argument('--n_test', type=int, default=1000)
    ap.add_argument('--seeds', choices=list(SEEDS), default='followup')
    ap.add_argument('--max_epochs', type=int, default=20)
    ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--sender_bs', type=int, default=16)
    ap.add_argument('--stack_max', type=int, default=13, help='most configurations in one stacked receiver pass')
    ap.add_argument('--check_stack', action='store_true', help='smoke test: train two configurations alone again')
    ap.add_argument('--bench', action='store_true', help='smoke test: time stacked train steps')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    times = {}

    t = time.time()
    if (out / 'items.json').exists():
        items = json.loads((out / 'items.json').read_text())
    else:
        tr_seed, te_seed = SEEDS[args.seeds]
        train = make_items(args.n_train, tr_seed)
        test = make_items(args.n_test, te_seed, taken={x['id'] for x in train})
        items = dict(train=train, test=test)
        (out / 'items.json').write_text(json.dumps(items))
    assert not {x['id'] for x in items['train']} & {x['id'] for x in items['test']}
    times['data'] = time.time() - t
    log(f"data: train={len(items['train'])} test={len(items['test'])}")

    t = time.time()
    if not (out / 'sender_test.npz').exists():
        sender_stage(items, out, args.sender_bs)
    times['sender'] = time.time() - t

    t = time.time()
    if not (out / 'probes.npz').exists():
        probes_stage(items, out)
    msgs, digits, samples = messages(items, out)
    (out / 'messages_test.json').write_text(json.dumps(msgs['test']))
    times['probes'] = time.time() - t

    t = time.time()
    id_check = text_stage(msgs, out) if not (out / 'recv_Card_test.npz').exists() else None
    if id_check is not None:
        (out / 'id_check.json').write_text(json.dumps(dict(max_abs_diff=id_check)))
    id_check = json.loads((out / 'id_check.json').read_text())['max_abs_diff']
    times['text'] = time.time() - t
    log(f'embedding route identity max abs diff: {id_check:.2e}')

    t = time.time()
    if not (out / 'latent.npz').exists():
        latent_stage(items, out, args)
    times['latent'] = time.time() - t

    t = time.time()
    res = analysis_stage(items, out, digits, samples, msgs, id_check, args, times)
    times['analysis'] = time.time() - t
    (out / 'table.md').write_text(table(res) + '\n')
    print(table(res))
    print(json.dumps(res['reading_rule'], indent=1))
    print(json.dumps({k: v for k, v in res['diagnostics'].items() if k in (
        'sender_acc', 'bayes_argmax_acc_test', 'sender_agrees_with_bayes_argmax_test', 'digit_histograms_test',
        'digit_auc_for_sender_correct_test', 'probe3_auc_test', 'latent_grid', 'latent_fit',
        't6_all_samples_same_fraction_test', 't6_answer_count_auc_for_sender_correct_test')}, indent=1))
    log('done')


if __name__ == '__main__':
    main()
