"""E1/E2 scratch sweep runner (v2, after the 2026-10-01 harness review).

Usage:  python -B sweep.py <queue-name> [--csv results.csv]
One process loads the receiver once and runs a queue of configs. Finished runs
(runs/<name>.json present) are skipped, so a queue is resumable. Every evaluation appends a
row to results.csv; the run JSON holds every arm on every development split.

Development splits (E0_data/make_splits.py v2):
  dev_known          fresh tables over the 56 trained IDs
  dev_oob_pure       tables made only of 8 out-of-block unseen IDs  -> selection target (mirrors FINAL)
  dev_inblock_pure   tables made only of the 8 burned in-block IDs
  dev_oob_mixed      one out-of-block unseen ID + 3 trained companions per table
  dev_inblock_mixed  one burned ID + 3 trained companions per table
Mixed-split novel rows can be answered by eliminating the trained companions, so they are
reported but never used for selection or for the H1/H2 thresholds.

Selection: best dev_oob_pure joint, then dev_inblock_pure joint, then dev_known joint, then the
earlier step. FINAL splits are never loaded.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE / 'runs'
CSV = HERE / 'results.csv'
LRS = [3e-4, 1e-3, 3e-3, 1e-2]
SHAPES = [(8, 64), (8, 128), (16, 64)]
DEV = ['dev_known', 'dev_oob_pure', 'dev_inblock_pure', 'dev_oob_mixed', 'dev_inblock_mixed']
LAB_SHA = lab.sha(lab.PROG / 'lab.py')
SWEEP_SHA = lab.sha(Path(__file__).resolve())


def cfg(experiment, variant, seed, lr, k=8, w=64, steps=2000, batch=32, eval_every=100, symbol_lr_mult=1.0,
        temp_init=None, learn_temp=False, gate_init=None):
    name = f'{experiment}_{variant}_k{k}w{w}_lr{lr:g}_s{seed}'
    if temp_init is not None:
        name += f"_t{temp_init:g}{'L' if learn_temp else 'F'}"
    if gate_init is not None:
        name += f'_g{gate_init:g}'
    if symbol_lr_mult != 1.0:
        name += f'_slr{symbol_lr_mult:g}'
    if steps != 2000:
        name += f'_st{steps}'
    if eval_every != 100:
        name += f'_ev{eval_every}'
    return dict(name=name, experiment=experiment, variant=variant, seed=seed, lr=lr, k=k, w=w,
                steps=steps, batch=batch, eval_every=eval_every, symbol_lr_mult=symbol_lr_mult,
                temp_init=temp_init, learn_temp=learn_temp, gate_init=gate_init)


QUEUES = {
    'smoke': [cfg('SMOKE', 'symbols', 1, 1e-3, steps=40, eval_every=20)],
    'phaseA': [cfg('E1', 'continuous', 1, lr) for lr in LRS] + [cfg('E2', 'symbols', 1, lr) for lr in LRS],
    # Phase B screen: does a sharper/learnable attention temperature let the bridge fit binding at all?
    'temp': [cfg('E1T', 'continuous', 1, lr, steps=1000, temp_init=t, learn_temp=L)
             for (t, L) in [(2.0, True), (10.0, True), (10.0, False)] for lr in [1e-3, 3e-3]],
    # Receiver-side ceiling: gold symbolic memory (color+letter+digit per line), no sender.
    'oracle': [cfg('R0', 'oracle', 1, lr, k=4) for lr in [1e-3, 3e-3]]
              + [cfg('R0', 'oracle', 1, lr, k=4, gate_init=1.0) for lr in [1e-3, 3e-3]],
    # Sender states with the oracle's per-line structure (no learned-query resampler).
    'lineslots': [cfg('R1', 'lineslots', 1, lr, k=4) for lr in [1e-3, 3e-3]],
    # Gate-init 1.0 across sender paths (the oracle showed the 0.01 gate is the main brake).
    'gate1': [cfg('R1', 'lineslots', 1, 3e-3, k=4, gate_init=1.0),
              cfg('E1', 'continuous', 1, 3e-3, gate_init=1.0),
              cfg('E2', 'symbols', 1, 3e-3, gate_init=1.0),
              cfg('R1', 'lineslots', 1, 1e-3, k=4, gate_init=1.0),
              cfg('E1', 'continuous', 1, 1e-3, gate_init=1.0)],
    # Structured read, gate 1.0: factorized ID maps, and sender-side symbols on line slots (E2-line).
    'lines2': [cfg('R2', 'lineslots_fact', 1, 3e-3, k=4, gate_init=1.0),
               cfg('E2L', 'lineslots_sym', 1, 3e-3, k=4, gate_init=1.0),
               cfg('E2L', 'lineslots_fact_sym', 1, 3e-3, k=4, gate_init=1.0),
               cfg('E2L', 'lineslots_sym', 2, 3e-3, k=4, gate_init=1.0)],
    # Force identity through the symbols: no continuous identity term, or 50% identity dropout.
    'lines3': [cfg('E2L', 'lineslots_symid', 1, 3e-3, k=4, gate_init=1.0),
               cfg('E2L', 'lineslots_sym_drop', 1, 3e-3, k=4, gate_init=1.0),
               cfg('E2L', 'lineslots_symid', 2, 3e-3, k=4, gate_init=1.0),
               cfg('E2L', 'lineslots_sym_drop', 2, 3e-3, k=4, gate_init=1.0)],
    # Sender-derived factored memory (heads on frozen sender states), pure latent and with aux heads.
    'factored': [cfg('E3F', 'factored', 1, 3e-3, k=4, gate_init=1.0),
                 cfg('E3F', 'factored_aux', 1, 3e-3, k=4, gate_init=1.0),
                 # seed-2 pure-latent replicate dropped: seed 1 never left step 0 (heads got no gradient)
                 cfg('E3F', 'factored_aux', 2, 3e-3, k=4, gate_init=1.0)],
    # Factored-aux grid: more seeds at 3e-3, five seeds at 1e-3 (select on DEV only).
    'factored_grid': [cfg('E3F', 'factored_aux', s, 3e-3, k=4, gate_init=1.0) for s in [3, 4, 5]]
                     + [cfg('E3F', 'factored_aux', s, 1e-3, k=4, gate_init=1.0) for s in [1, 2, 3, 4, 5]],
    # Frozen closed-form ridge heads (hard one-hot), train only embeddings/projector/reader.
    'factored_ridge': [cfg('E3R', 'factored_ridge', s, 3e-3, k=4, gate_init=1.0) for s in [1, 2, 3]],
    # Receiver-composition variance: oracle memory replicates at gate 1.0, lr 3e-3.
    'oracle_seeds': [cfg('R0', 'oracle', s, 3e-3, k=4, gate_init=1.0) for s in [2, 3, 4, 5]],
    # E3 secondary arm: best free-form continuous channel (per-line sender slots), retrained with checkpoints.
    'e3_freeform': [cfg('E3C', 'lineslots', s, 1e-3, k=4, gate_init=1.0) for s in [1, 2, 3]],
    # E5: free-form continuous slots under factoring pressure (aux probes on the slot), and ' =' input.
    'e5': [cfg('E5', 'lineslots_aux1', 1, 1e-3, k=4, gate_init=1.0),
           cfg('E5', 'lineslots_aux10', 1, 1e-3, k=4, gate_init=1.0),
           cfg('E5', 'lineslots_eq', 1, 1e-3, k=4, gate_init=1.0),
           cfg('E5', 'lineslots_eq_aux1', 1, 1e-3, k=4, gate_init=1.0),
           cfg('E5', 'lineslots_aux1', 2, 1e-3, k=4, gate_init=1.0),
           cfg('E5', 'lineslots_aux10', 2, 1e-3, k=4, gate_init=1.0)],
    'e1_full': [cfg('E1', 'continuous', s, lr, k, w) for (k, w) in SHAPES for lr in LRS for s in [1, 2, 3, 4, 5]],
}


class Context:
    def __init__(self):
        self.dev = lab.device()
        t = time.monotonic()
        self.model, self.tok, self.rhash = lab.load('receiver', self.dev)
        self.bank = lab.AnswerBank(self.tok)
        self.bank_tokens_dev = self.bank.tokens.to(self.dev)
        self.splits = {name: lab.load_split(name) for name in ['train'] + DEV}
        for s in self.splits.values():
            s['ids'] = self.bank.prompts([r['query'] for r in s['data']['rows']])
            s['H'] = s['hidden'].to(self.dev)
            s['L'], s['D'] = s['letter'].to(self.dev), s['digit'].to(self.dev)
            s['C'] = s['color'].to(self.dev)
        tr = self.splits['train']
        assert not tr['novel'].any()
        self.train_ids = tr['ids'].to(self.dev)
        self.train_labels = tr['labels'].to(self.dev)
        self.train_groups = tr['groups'].to(self.dev)
        self.subset = dict(tr, ids=tr['ids'][:224], labels=tr['labels'][:224], groups=tr['groups'][:224])
        self.receiver = None
        self.receiver_only = {}
        lab.event('context_ready', s=round(time.monotonic() - t, 1), device=str(self.dev), receiver_hash=self.rhash)

    def attach(self, bridge):
        if self.receiver is None:
            self.receiver = lab.Receiver(self.model, bridge)
        self.receiver.bridge = bridge

    def baseline(self, name):
        if name not in self.receiver_only:
            s = self.splits[name]
            self.receiver_only[name] = lab.score_cached(self.receiver, s['ids'], self.bank, None, dev=self.dev)
        return self.receiver_only[name]


def encode_split(bridge, split):
    with torch.no_grad():
        return torch.cat([bridge.encode(h, l, d, color=c) for h, l, d, c in
                          zip(split['H'].split(64), split['L'].split(64), split['D'].split(64), split['C'].split(64))]).cpu()


def correct_eval(ctx, bridge, split):
    z = encode_split(bridge, split)
    res = lab.score_cached(ctx.receiver, split['ids'], ctx.bank, z[split['groups']], dev=ctx.dev)
    return res, lab.counts(res, split['labels'], ctx.bank.tokens)


def rate(pred, target, mask=None):
    if mask is None:
        return (pred == target).float().mean().item()
    return (pred[mask] == target[mask]).float().mean().item() if mask.any() else None


def attention_entropy(ctx, bridge):
    """Mean resampler entropy over 19 source tokens (max 2.944) and reader entropy over k slots
    at the last query position (max ln k), on the first 64 dev_known rows."""
    s = ctx.splits['dev_known']
    ids, g = s['ids'][:64].to(ctx.dev), s['groups'][:64].to(ctx.dev)
    bridge.resampler.track = bridge.reader.track = True
    try:
        with torch.no_grad():
            z = bridge.encode(s['H'][g], s['L'][g], s['D'][g], color=s['C'][g])
            ctx.receiver.forward(ids, z, 1)
        return bridge.resampler.last_entropy, bridge.reader.last_entropy
    finally:
        bridge.resampler.track = bridge.reader.track = False


def fit_ridge_heads(bridge, tr, lam=10.0):
    """Closed-form ridge heads on TRAIN lines only (labels are the sender's own source text);
    heads are then frozen and used as hard one-hot symbols."""
    f = bridge.factored
    h = torch.nn.functional.layer_norm(tr['hidden'], (tr['hidden'].shape[-1],)).double()
    eq = torch.cat([h[:, a + 2] for a, _ in lab.ID_POSITIONS])
    col = torch.cat([h[:, a + 3] for a, _ in lab.ID_POSITIONS])
    L = torch.cat([tr['letter'][:, k] for k in range(4)])
    D = torch.cat([tr['digit'][:, k] for k in range(4)])
    C = torch.cat([tr['color'][:, k] for k in range(4)])
    for head, X, y, n in [(f.letter_head, eq, L, 16), (f.digit_head, eq, D, 8), (f.color_head, col, C, 4)]:
        Xa = torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype)], 1)
        Y = torch.nn.functional.one_hot(y, n).double()
        reg = lam * torch.eye(Xa.shape[1], dtype=X.dtype); reg[-1, -1] = 0
        W = torch.linalg.solve(Xa.T @ Xa + reg, Xa.T @ Y)
        with torch.no_grad():
            head.weight.copy_(W[:-1].T.float()); head.bias.copy_(W[-1].float())
        head.requires_grad_(False)
    f.hard = True


def head_accuracy(bridge, split):
    """Argmax accuracy of the factored heads per source line (letter, digit, color)."""
    with torch.no_grad():
        bridge.factored(split['H'])
        ll, dl, cl = bridge.factored.last_logits
    return dict(letter=(ll.argmax(-1) == split['L']).float().mean().item(),
                digit=(dl.argmax(-1) == split['D']).float().mean().item(),
                color=(cl.argmax(-1) == split['C']).float().mean().item())


def curve_point(ctx, bridge):
    out = {}
    if bridge.lineslots is not None and bridge.lineslots.probes is not None:
        for name, key in [('dev_known', 'known'), ('dev_oob_pure', 'oob')]:
            sp = ctx.splits[name]
            for comp, v in zip(['letter', 'digit', 'color'], bridge.lineslots.probe_accuracy(sp['H'], sp['L'], sp['D'], sp['C'])):
                out[f'probe_{key}_{comp}'] = v
    if bridge.factored is not None:
        for name, key in [('dev_known', 'known'), ('dev_oob_pure', 'oob')]:
            h = head_accuracy(bridge, ctx.splits[name])
            for comp, v in h.items():
                out[f'head_{key}_{comp}'] = v
    out['resampler_entropy'], out['reader_entropy'] = attention_entropy(ctx, bridge)
    out['temperature'] = float(torch.as_tensor(bridge.reader.scale()).detach())
    _, c = correct_eval(ctx, bridge, ctx.subset)
    out['train_subset_joint'] = c['joint'] / c['rows']
    for name, key in [('dev_known', 'dev_known'), ('dev_oob_pure', 'oob_pure'), ('dev_inblock_pure', 'inblock_pure')]:
        _, c = correct_eval(ctx, bridge, ctx.splits[name])
        out[f'{key}_joint'], out[f'{key}_seq'] = c['joint'] / c['rows'], c['sequential'] / c['rows']
    for name, key in [('dev_oob_mixed', 'oob_mixed'), ('dev_inblock_mixed', 'inblock_mixed')]:
        s = ctx.splits[name]
        res, c = correct_eval(ctx, bridge, s)
        out[f'{key}_novel_joint'] = rate(res['prediction'], s['labels'], s['novel'])
        out[f'{key}_all_joint'] = c['joint'] / c['rows']
    return out


def in_table(pred, s, mask=None):
    recs = s['data']['records']
    hits = torch.tensor([lab.IDS[int(p)] in recs[int(g)]['bindings'].values() for p, g in zip(pred, s['groups'])])
    return hits.float().mean().item() if mask is None else hits[mask].float().mean().item()


def collision_mask(s):
    """Rows whose answer shares its letter or digit with another line of the same table."""
    recs = s['data']['records']
    out = []
    for r in s['data']['rows']:
        ids = list(recs[r['group']]['bindings'].values())
        a = r['answer']
        others = [i for i in ids if i != a]
        out.append(any(i[0] == a[0] or i[1] == a[1] for i in others))
    return torch.tensor(out)


def full_arms(ctx, bridge, name):
    s = ctx.splits[name]
    arms = ['correct', 'shuffled', 'wrong_binding', 'zero', 'random_71001', 'random_71002']
    if bridge.symbols is not None or (bridge.lineslots is not None and bridge.lineslots.sym_letter is not None):
        arms += ['symbols_only', 'continuous_only', 'hybrid']
    values, meta = lab.memories(bridge, s, ctx.dev, arms)
    donor_index, positional = meta['donor_index'], meta['positional']
    labels, wrong = s['labels'], s['wrong_labels']
    pure = name in ('dev_known', 'dev_oob_pure', 'dev_inblock_pure')
    # rows whose target is an unseen ID; for dev_known and pure splits that is all rows
    nv = torch.ones(len(labels), dtype=torch.bool) if pure else s['novel']
    unseen_ids = torch.tensor([lab.IDS[i] in set(s['data']['unseen']) for i in range(128)]) if s['data']['unseen'] \
        else torch.ones(128, dtype=torch.bool)
    donor_lab = labels[donor_index]
    same = donor_lab == labels                      # donor table happens to give the same answer
    coll = collision_mask(s)
    distinct = (labels != donor_lab) & (labels != positional) & (donor_lab != positional)
    result = dict(rows=len(labels), scored_rows=int(nv.sum()), pure=pure, same_answer_donor_rows=int((same & nv).sum()),
                  collision_rows=int((coll & nv).sum()), hybrid_distinct_rows=int((distinct & nv).sum()))
    for arm in arms + ['receiver_only']:
        res = ctx.baseline(name) if arm == 'receiver_only' else \
            lab.score_cached(ctx.receiver, s['ids'], ctx.bank, values[arm], dev=ctx.dev)
        c = lab.counts(res, labels, ctx.bank.tokens)
        pred, seq = res['prediction'], c['sequential_prediction']
        r = dict(joint=rate(pred, labels, nv), seq=rate(seq, labels, nv),
                 joint_all_rows=c['joint'] / c['rows'], in_table=in_table(pred, s, nv),
                 mean_id_nll=-res['lp'][torch.arange(len(labels)), labels][nv].mean().item())
        if arm == 'correct':
            r['joint_collision'] = rate(pred, labels, nv & coll)
            r['joint_no_collision'] = rate(pred, labels, nv & ~coll)
        if arm == 'shuffled':
            # donor's target must be unseen too, and must differ from the row's own answer
            dmask = (nv if pure else nv[donor_index]) & ~same
            r['donor_following'] = rate(pred, donor_lab, dmask)
            r['joint_excluding_same_answer'] = rate(pred, labels, nv & ~same)
        if arm == 'wrong_binding':
            wmask = nv if pure else s['novel_wrong']
            r['changed_following'] = rate(pred, wrong, wmask)
        if arm == 'hybrid':
            dmask = nv if pure else nv & unseen_ids[donor_lab]
            pmask = nv if pure else nv & unseen_ids[positional]
            r['follows_own'] = rate(pred, labels, nv)
            r['follows_donor'] = rate(pred, donor_lab, dmask)
            r['follows_positional'] = rate(pred, positional, pmask)
            for key, tgt in [('own', labels), ('donor', donor_lab), ('positional', positional)]:
                r[f'follows_{key}_distinct_rows'] = rate(pred, tgt, nv & distinct)
        result[arm] = r
    return result


def run(ctx, c, csv_path=CSV):
    out_path = RUNS / f"{c['name']}.json"
    if out_path.exists():
        lab.event('skip', run=c['name'])
        return json.loads(out_path.read_text())
    t0 = time.monotonic()
    lab.seed_all(c['seed'])
    bridge = lab.Bridge(c['k'], c['w'], symbols=c['variant'] == 'symbols',
                        temp_init=c.get('temp_init'), learn_temp=c.get('learn_temp', False),
                        oracle=c['variant'] == 'oracle', gate_init=c.get('gate_init'),
                        lineslots=c['variant'].startswith('lineslots'),
                        factorized='fact' in c['variant'], line_symbols='sym' in c['variant'],
                        line_use_id='symid' not in c['variant'],
                        id_dropout=0.5 if 'drop' in c['variant'] else 0.0,
                        factored=c['variant'].startswith('factored'),
                        eq_input='_eq' in c['variant'], aux_probes='_aux' in c['variant'] and c['variant'].startswith('lineslots')).to(ctx.dev)
    ctx.attach(bridge)
    if c['variant'] == 'factored_ridge':
        fit_ridge_heads(bridge, ctx.splits['train'])      # CPU float64 solve; weights copied to the device heads
    params = sum(p.numel() for p in bridge.parameters())
    groups = [dict(params=[p for n, p in bridge.named_parameters() if not n.startswith('symbols.') and p.requires_grad],
                   lr=c['lr'])]
    if bridge.symbols is not None:
        groups.append(dict(params=list(bridge.symbols.parameters()), lr=c['lr'] * c['symbol_lr_mult']))
    opt = torch.optim.AdamW(groups, lr=c['lr'], weight_decay=0)
    gen = torch.Generator().manual_seed(c['seed'] + 1)
    n = len(ctx.train_ids)
    order, cursor = torch.randperm(n, generator=gen), 0
    tr = ctx.splits['train']
    base = dict(experiment=c['experiment'], variant=c['variant'], run=c['name'], seed=c['seed'], lr=c['lr'],
                slots=c['k'], slot_dim=c['w'], params=params, device=str(ctx.dev), lab_sha256=LAB_SHA)
    curve, states, losses = [], {}, []

    def symbol_norm():
        return bridge.symbols.table.norm(dim=-1).mean().item() if bridge.symbols is not None else ''

    def record(step):
        bridge.eval()
        point = curve_point(ctx, bridge)
        recent = losses[-c['eval_every']:]
        point.update(step=step, train_ce=sum(recent) / len(recent) if recent else None,
                     gate=bridge.reader.gate.item(), symbol_norm=symbol_norm(), wall_s=round(time.monotonic() - t0, 1))
        curve.append(point)
        states[step] = {k: v.detach().cpu().clone() for k, v in bridge.state_dict().items()}
        lab.append_csv(csv_path, dict(base, kind='curve', **point))
        lab.event('curve', run=c['name'], **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in point.items()})
        bridge.train()

    record(0)
    stopped = None
    for step in range(1, c['steps'] + 1):
        if cursor + c['batch'] > n:
            order, cursor = torch.randperm(n, generator=gen), 0
        ix = order[cursor:cursor + c['batch']].to(ctx.dev)
        cursor += c['batch']
        opt.zero_grad(set_to_none=True)
        g = ctx.train_groups[ix]
        z = bridge.encode(tr['H'][g], tr['L'][g], tr['D'][g], color=tr['C'][g])
        loss = lab.train_logits(ctx.receiver, ctx.bank_tokens_dev, ctx.train_ids[ix], ctx.train_labels[ix], z)
        total = loss
        if c['variant'].endswith('_aux') and bridge.factored is not None:
            total = loss + bridge.factored.aux_loss(tr['L'][g], tr['D'][g], tr['C'][g])
        if bridge.lineslots is not None and bridge.lineslots.probes is not None:
            weight = float(c['variant'].split('_aux')[1])
            total = total + weight * bridge.lineslots.probe_loss(tr['L'][g], tr['D'][g], tr['C'][g])
        total.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if not torch.isfinite(torch.tensor(losses[-1])):
            stopped = f'nonfinite loss at step {step}'
            lab.event('nonfinite', run=c['name'], step=step)
            break
        if step % c['eval_every'] == 0:
            record(step)
    assert all(p.grad is None for p in ctx.model.parameters())

    best = max(curve, key=lambda p: (p['oob_pure_joint'], p['inblock_pure_joint'], p['dev_known_joint'], -p['step']))
    final = curve[-1]
    summary = dict(config=c, params=params, lab_sha256=LAB_SHA, sweep_sha256=SWEEP_SHA, curve=curve, best_step=best['step'],
                   final_step=final['step'], stopped=stopped)
    for tag, point in [('best', best), ('final', final)]:
        bridge.load_state_dict(states[point['step']])
        bridge.eval()
        arms = {name: full_arms(ctx, bridge, name) for name in DEV}
        summary[tag] = dict(step=point['step'], curve=point, arms=arms)
        if bridge.factored is not None:
            summary[tag]['heads'] = {name: head_accuracy(bridge, ctx.splits[name]) for name in ['train'] + DEV}
        RUNS.mkdir(exist_ok=True)
        torch.save(states[point['step']], RUNS / f"{c['name']}_{tag}.pt")
        op, dk = arms['dev_oob_pure'], arms['dev_known']
        row = dict(base, kind=tag, step=point['step'], train_ce=point.get('train_ce'),
                   train_subset_joint=point['train_subset_joint'],
                   dev_known_joint=dk['correct']['joint'], dev_known_seq=dk['correct']['seq'],
                   oob_pure_joint=op['correct']['joint'], oob_pure_seq=op['correct']['seq'],
                   inblock_pure_joint=arms['dev_inblock_pure']['correct']['joint'],
                   inblock_pure_seq=arms['dev_inblock_pure']['correct']['seq'],
                   oob_mixed_novel_joint=arms['dev_oob_mixed']['correct']['joint'],
                   oob_mixed_all_joint=arms['dev_oob_mixed']['correct']['joint_all_rows'],
                   inblock_mixed_novel_joint=arms['dev_inblock_mixed']['correct']['joint'],
                   oob_pure_donor_following=op['shuffled']['donor_following'],
                   oob_pure_changed_following=op['wrong_binding']['changed_following'],
                   oob_pure_in_table=op['correct']['in_table'],
                   oob_pure_shuffled=op['shuffled']['joint'],
                   oob_pure_random_71001=op['random_71001']['joint'], oob_pure_random_71002=op['random_71002']['joint'],
                   oob_pure_zero=op['zero']['joint'], oob_pure_receiver_only=op['receiver_only']['joint'],
                   oob_pure_symbols_only=op.get('symbols_only', {}).get('joint', ''),
                   oob_pure_continuous_only=op.get('continuous_only', {}).get('joint', ''),
                   oob_pure_hybrid_own=op.get('hybrid', {}).get('follows_own', ''),
                   oob_pure_hybrid_donor=op.get('hybrid', {}).get('follows_donor', ''),
                   oob_pure_hybrid_positional=op.get('hybrid', {}).get('follows_positional', ''),
                   dev_known_donor_following=dk['shuffled']['donor_following'],
                   dev_known_changed_following=dk['wrong_binding']['changed_following'],
                   gate=point['gate'], symbol_norm=point['symbol_norm'], wall_s=round(time.monotonic() - t0, 1),
                   notes=stopped or '')
        lab.append_csv(csv_path, row)
        lab.event(tag, run=c['name'], step=point['step'], dev_known=round(row['dev_known_joint'], 3),
                  oob_pure=round(row['oob_pure_joint'], 3), inblock_pure=round(row['inblock_pure_joint'], 3),
                  oob_donor=round(row['oob_pure_donor_following'], 3), oob_changed=round(row['oob_pure_changed_following'], 3),
                  hybrid=[row['oob_pure_hybrid_own'], row['oob_pure_hybrid_donor'], row['oob_pure_hybrid_positional']])
    summary['wall_s'] = round(time.monotonic() - t0, 1)
    RUNS.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=1) + '\n')
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('queue')
    ap.add_argument('--csv', default=str(CSV))
    args = ap.parse_args()
    queue = QUEUES[args.queue]
    ctx = Context()
    for i, c in enumerate(queue):
        lab.event('start', run=c['name'], index=i, of=len(queue))
        run(ctx, c, Path(args.csv))
    assert lab.parameter_hash(ctx.model.cpu()) == lab.RECEIVER_HASH
    lab.event('queue_done', queue=args.queue)


if __name__ == '__main__':
    main()
