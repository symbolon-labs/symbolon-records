#!/usr/bin/env python
"""EXPLORATORY pilot: does a latent channel carry uncertainty that text loses? See PLAN.md and
IMPLEMENTATION_NOTES.md.

Stages (each stage keeps its output in --out, and a second run skips a stage that has its output):
  data     synthetic witness cards with the exact Bayes posterior
  sender   Qwen3-1.7B answer, color logits, verbal digit, states after blocks 21 and 28
  probes   block-21 probes for T3 (answer correct) and T4 (true color), cross-fitted on train
  text     SmolLM2-360M final state and zero-shot color logits for T0-T4 and Card
  latent   linear map from the block-21 state into one soft receiver input (L, L-shuf)
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
SEEDS = dict(pilot=(2026102001, 2026102002), smoke=(2026102091, 2026102092))
SEED = 20261020
SENDER_Q = 'Which color is the box most likely? Answer with one word: red, blue, green, or yellow.'
VERBAL_Q = ('How sure are you that your answer is right? Reply with one digit from 0 (not sure at all) '
            'to 9 (completely sure).')
R_PREFIX, R_QUESTION = 'Message:', '\nWhat color is the box? Answer:'
LAMS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
LATENT_LRS = [3e-5, 1e-4, 3e-4, 1e-3]
N_FOLDS, N_BOOT, MARGIN, N_BINS = 5, 10_000, 0.02, 15
TEXT_ARMS = ['T0', 'T1', 'T2', 'T3', 'T4', 'Card']


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
    """Exact P(c | d) for the rounded Bayes vector d, over all 4^N_REP report sequences (PLAN.md addition)."""
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


def messages(items, out):
    P = np.load(out / 'probes.npz')
    msgs, digits = {}, {}
    for s in ('train', 'test'):
        S = np.load(out / f'sender_{s}.npz')
        ans = S['color_logits'].argmax(1)
        p_tok = torch.softmax(torch.as_tensor(S['color_logits']), 1).numpy()[np.arange(len(ans)), ans]
        d1 = S['digit_logits'].argmax(1)
        d2 = np.array([rnd9(p) for p in p_tok])
        d3 = np.array([rnd9(p) for p in P[f'p3_{s}']])
        msgs[s] = dict(T0=[msg_answer(a) for a in ans],
                       T1=[msg_digit(a, d) for a, d in zip(ans, d1)],
                       T2=[msg_digit(a, d) for a, d in zip(ans, d2)],
                       T3=[msg_digit(a, d) for a, d in zip(ans, d3)],
                       T4=[msg_dist(a, p) for a, p in zip(ans, P[f'p4_{s}'])],
                       Card=[card(x) for x in items[s]])
        digits[s] = dict(T1=d1.tolist(), T2=d2.tolist(), T3=d3.tolist())
    return msgs, digits


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
    def __init__(self, s_dim, emb, pre, post):
        super().__init__()
        self.map = torch.nn.Linear(s_dim, emb.shape[1])
        torch.nn.init.normal_(self.map.weight, std=float(emb.std()) / math.sqrt(s_dim))
        with torch.no_grad():
            self.map.bias.copy_(emb.mean(0))
        self.readout = torch.nn.Linear(emb.shape[1], 4)
        self.register_buffer('pre', pre)
        self.register_buffer('post', post)

    def forward(self, R, s):
        B = len(s)
        x = torch.cat([self.pre.expand(B, -1, -1), self.map(s)[:, None], self.post.expand(B, -1, -1)], 1)
        h = R.model(inputs_embeds=x, use_cache=False).last_hidden_state[:, -1]
        return self.readout(h), h


def latent_stage(items, out, max_epochs, patience, bs=32):
    tok, R, color_ids = load_receiver()
    emb = R.model.embed_tokens.weight.detach()
    pre = R.model.embed_tokens(tok(R_PREFIX, return_tensors='pt', add_special_tokens=False).input_ids.to(DEV))
    post = R.model.embed_tokens(tok(R_QUESTION, return_tensors='pt', add_special_tokens=False).input_ids.to(DEV))
    S = {s: np.load(out / f'sender_{s}.npz')['s21'] for s in ('train', 'test')}
    Str, Ste = standardize(S['train'].astype(np.float64), S['test'].astype(np.float64))
    ytr = np.array([x['color'] for x in items['train']])
    yte = np.array([x['color'] for x in items['test']])
    T = lambda a: torch.as_tensor(a, dtype=torch.float32, device=DEV)

    @torch.no_grad()
    def evaluate(ch, X):
        ch.eval()
        P, Z = [], []
        for i in range(0, len(X), 128):
            lg, h = ch(R, T(X[i:i + 128]))
            P.append(torch.softmax(lg, 1).cpu())
            Z.append(R.lm_head(h)[:, color_ids].cpu())
        return torch.cat(P).double().numpy(), torch.cat(Z).numpy()

    def nll(P, y):
        return float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, None)).mean())

    def train(X, y, lr, epochs, Xva=None, yva=None):
        torch.manual_seed(SEED)
        ch = Channel(X.shape[1], emb, pre, post).to(DEV)
        opt = torch.optim.AdamW(ch.parameters(), lr=lr, weight_decay=0.01)
        g = torch.Generator().manual_seed(SEED)
        Xt, yt = T(X), torch.as_tensor(y, device=DEV)
        hist, best = [], (math.inf, 0)
        for ep in range(1, epochs + 1):
            ch.train()
            perm = torch.randperm(len(y), generator=g)
            tot = 0.0
            for i in range(0, len(y), bs):
                j = perm[i:i + bs].to(DEV)
                loss = F.cross_entropy(ch(R, Xt[j])[0], yt[j])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ch.parameters(), 1.0)
                opt.step()
                tot += loss.item() * len(j)
            row = dict(epoch=ep, train_loss=tot / len(y))
            if Xva is not None:
                row['val_nll'] = nll(evaluate(ch, Xva)[0], yva)
                if row['val_nll'] < best[0]:
                    best = (row['val_nll'], ep)
            hist.append(row)
            if Xva is not None and ep - best[1] >= patience:
                break
        return ch, hist, best

    t0 = time.time()
    inner = np.random.default_rng(SEED).permutation(len(ytr))
    n_va = max(1, len(ytr) // 5)
    va, tr = inner[:n_va], inner[n_va:]
    grid = {}
    for lr in LATENT_LRS:
        _, hist, best = train(Str[tr], ytr[tr], lr, max_epochs, Str[va], ytr[va])
        grid[str(lr)] = dict(best_val_nll=best[0], best_epoch=best[1], history=hist)
        log(f'latent lr={lr}: best val NLL {best[0]:.4f} at epoch {best[1]} ({time.time() - t0:.0f}s)')
    lr_best = min(LATENT_LRS, key=lambda lr: grid[str(lr)]['best_val_nll'])
    ep_best = grid[str(lr_best)]['best_epoch']

    ch, hist_L, _ = train(Str, ytr, lr_best, ep_best)
    P_L, Z_L = evaluate(ch, Ste)
    perm_te = np.random.default_rng(SEED + 2).permutation(len(yte))
    P_Lts, Z_Lts = evaluate(ch, Ste[perm_te])
    perm_tr = np.random.default_rng(SEED + 1).permutation(len(ytr))
    ch_s, hist_S, _ = train(Str[perm_tr], ytr, lr_best, ep_best)
    P_S, Z_S = evaluate(ch_s, Ste[perm_te])
    log(f'latent: lr={lr_best} epochs={ep_best} test NLL L={nll(P_L, yte):.4f} L-shuf={nll(P_S, yte):.4f} '
        f'L-testshuf={nll(P_Lts, yte):.4f} ({time.time() - t0:.0f}s)')
    np.savez(out / 'latent.npz', P_L=P_L, Z_L=Z_L, P_Lshuf=P_S, Z_Lshuf=Z_S, P_Ltestshuf=P_Lts, Z_Ltestshuf=Z_Lts)
    (out / 'latent.json').write_text(json.dumps(dict(lr=lr_best, epochs=ep_best, n_inner_train=len(tr),
                                                     n_inner_val=len(va), grid=grid, final_history=hist_L,
                                                     shuf_history=hist_S, seconds=time.time() - t0), indent=1))


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


def analysis_stage(items, out, digits, msgs, id_check, args, times):
    ytr = np.array([x['color'] for x in items['train']])
    yte = np.array([x['color'] for x in items['test']])
    probs, zshot, readout_info = {}, {}, {}
    for arm in TEXT_ARMS:
        A = {s: np.load(out / f'recv_{arm}_{s}.npz') for s in ('train', 'test')}
        _, probs[arm], readout_info[arm] = cv_logreg(A['train']['h'], ytr, A['test']['h'], 4)
        zshot[arm] = torch.softmax(torch.as_tensor(A['test']['z'], dtype=torch.float64), 1).numpy()
    Lz = np.load(out / 'latent.npz')
    for arm, key in (('L', 'L'), ('L-shuf', 'Lshuf'), ('L-testshuf', 'Ltestshuf')):
        probs[arm] = Lz[f'P_{key}']
        zshot[arm] = torch.softmax(torch.as_tensor(Lz[f'Z_{key}'], dtype=torch.float64), 1).numpy()
    Ste = np.load(out / 'sender_test.npz')
    Pr = np.load(out / 'probes.npz')
    d9 = np.array([[rnd9(p) for p in x['bayes']] for x in items['test']], np.float64)
    b9 = bayes9_table()
    refs = dict(Bayes=np.array([x['bayes'] for x in items['test']]),
                **{'Bayes-9': d9 / d9.sum(1, keepdims=True), 'Bayes-9-cal': np.array([b9[tuple(int(v) for v in d)] for d in d9])},
                Prior=np.full((len(yte), 4), 0.25),
                sender_softmax_direct=torch.softmax(torch.as_tensor(Ste['color_logits'], dtype=torch.float64), 1).numpy(),
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

    best_text = min(['T0', 'T1', 'T2'], key=lambda a: arms[a]['readout']['nll'])
    differences = {f'{a}_minus_L': diff(a, 'L') for a in ['T0', 'T1', 'T2', 'T3', 'T4', 'Card', 'L-shuf', 'L-testshuf']}
    differences['T0_minus_T2'] = diff('T0', 'T2')
    differences['T2_minus_T4'] = diff('T2', 'T4')
    differences['T3_minus_T4'] = diff('T3', 'T4')
    differences['T0_minus_Card'] = diff('T0', 'Card')

    def beats(x):
        d = differences[f'{x}_minus_L']
        return bool(d['mean'] >= MARGIN and d['ci95'][0] > 0)

    b_text, b3, b4 = beats(best_text), beats('T3'), beats('T4')
    case = 1 if (b_text and b3 and b4) else 2 if b_text else 3
    lshuf_ok = arms['L-shuf']['readout']['nll'] >= math.log(4) - MARGIN
    card_ok = arms['Card']['readout']['nll'] <= arms['T0']['readout']['nll']
    rule = dict(best_of_T0_T2=best_text, L_beats_best_text=b_text, L_beats_T3=b3, L_beats_T4=b4, case=case,
                sanity_Lshuf_ok=bool(lshuf_ok), sanity_Card_ok=bool(card_ok), valid=bool(lshuf_ok and card_ok),
                margin_nats=MARGIN, ln4=math.log(4))

    ans = {s: np.load(out / f'sender_{s}.npz')['color_logits'].argmax(1) for s in ('train', 'test')}
    corr_te = (ans['test'] == yte).astype(int)
    hist = lambda v: {str(k): int(c) for k, c in zip(*np.unique(v, return_counts=True))}
    diagnostics = dict(
        sender_acc=dict(train=float((ans['train'] == ytr).mean()), test=float(corr_te.mean())),
        bayes_argmax_acc_test=float((refs['Bayes'].argmax(1) == yte).mean()),
        bayes9_cost_nll=references['Bayes-9-cal']['nll'] - bayes_nll,
        bayes9_true_color_zero_fraction_test=float((d9[np.arange(len(yte)), yte] == 0).mean()),
        sender_agrees_with_bayes_argmax_test=float((ans['test'] == refs['Bayes'].argmax(1)).mean()),
        digit_histograms_test={k: hist(v) for k, v in digits['test'].items()},
        digit_auc_for_sender_correct_test={k: auc(np.array(v, float), corr_te) for k, v in digits['test'].items()},
        probe3_auc_test=auc(Pr['p3_test'], corr_te),
        readout_info=readout_info, probe_info=json.loads((out / 'probes.json').read_text()),
        latent_info={k: v for k, v in json.loads((out / 'latent.json').read_text()).items() if k not in ('grid', 'final_history', 'shuf_history')},
        latent_grid={k: dict(best_val_nll=v['best_val_nll'], best_epoch=v['best_epoch'])
                     for k, v in json.loads((out / 'latent.json').read_text())['grid'].items()},
        embedding_route_identity_max_abs_diff=id_check,
        example_messages_test={arm: msgs['test'][arm][:3] for arm in TEXT_ARMS},
        example_cards_test=[dict(card=card(x), color=COLORS[x['color']], bayes=x['bayes']) for x in items['test'][:3]])

    records = [dict(arm=arm, view=view, metric=k, value=v) for arm in arms for view in ('readout', 'zero_shot')
               for k, v in arms[arm][view].items()]
    records += [dict(arm=k, view='reference', metric=m, value=v) for k, r in references.items() for m, v in r.items()]
    records += [dict(arm=k.replace('_minus_', ' minus '), view='readout', metric='nll_difference_mean', value=v['mean'])
                for k, v in differences.items()]
    res = dict(status='EXPLORATORY', plan='PLAN.md', notes='IMPLEMENTATION_NOTES.md', sizes=dict(train=len(ytr), test=len(yte)),
               seeds=dict(data=SEEDS[args.seeds], other=SEED), args=vars(args), stage_seconds=times,
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
    ap.add_argument('--seeds', choices=list(SEEDS), default='pilot')
    ap.add_argument('--max_epochs', type=int, default=40)
    ap.add_argument('--patience', type=int, default=6)
    ap.add_argument('--sender_bs', type=int, default=16)
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
    msgs, digits = messages(items, out)
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
        latent_stage(items, out, args.max_epochs, args.patience)
    times['latent'] = time.time() - t

    t = time.time()
    res = analysis_stage(items, out, digits, msgs, id_check, args, times)
    times['analysis'] = time.time() - t
    (out / 'table.md').write_text(table(res) + '\n')
    print(table(res))
    print(json.dumps(res['reading_rule'], indent=1))
    print(json.dumps({k: v for k, v in res['diagnostics'].items() if k in (
        'sender_acc', 'bayes_argmax_acc_test', 'sender_agrees_with_bayes_argmax_test', 'digit_histograms_test',
        'digit_auc_for_sender_correct_test', 'probe3_auc_test', 'latent_info', 'latent_grid')}, indent=1))
    log('done')


if __name__ == '__main__':
    main()
