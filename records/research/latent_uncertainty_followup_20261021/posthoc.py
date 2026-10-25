#!/usr/bin/env python
"""POST-HOC secondary views (exploratory, the same views as the pilot's posthoc.py, written before the
follow-up data existed). They do not enter the reading rule. Input: run/results.json and run/test_probs.npz.
Output: RESULTS.json (results.json plus the key "post_hoc_exploratory")."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEED, N_BOOT = 20261021, 10_000


def main():
    res = json.loads((HERE / 'run/results.json').read_text())
    z = np.load(HERE / 'run/test_probs.npz')
    y = z['y']
    bayes = z['ref_Bayes']
    rows = {k[len('readout_'):]: z[k] for k in z.files if k.startswith('readout_')}
    rows.update({f'ref: {k[len("ref_"):]}': z[k] for k in z.files if k.startswith('ref_') and k != 'ref_Bayes-9'})
    norm = lambda P: np.clip(P, 1e-12, None) / np.clip(P, 1e-12, None).sum(1, keepdims=True)
    nll = {k: -np.log(norm(P)[np.arange(len(y)), y]) for k, P in rows.items()}
    kl = {k: (bayes * (np.log(np.clip(bayes, 1e-300, None)) - np.log(norm(P)))).sum(1) for k, P in rows.items()}
    idx = np.random.default_rng(SEED).integers(0, len(y), size=(N_BOOT, len(y)))

    def ci(d):
        lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
        return dict(mean=float(d.mean()), ci95=[float(lo), float(hi)])

    Ls = res['reading_rule']['selected_latent']
    probe = 'ref: probe4_block21_direct'
    texts = ['T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'Card']
    out = dict(
        note='POST HOC, exploratory. The same views as the pilot post hoc block. Not part of the reading rule.',
        selected_latent=Ls,
        kl_bayes_to_arm={k: float(v.mean()) for k, v in kl.items()},
        kl_differences={f'{a} minus {Ls}': ci(kl[a] - kl[Ls]) for a in texts + [x for x in ('L', 'L-lowrank') if x != Ls]},
        route_loss_vs_probe4_block21_nll={a: ci(nll[a] - nll[probe]) for a in ['L', 'L-lowrank', 'T4']},
        route_loss_vs_probe4_block21_kl={a: ci(kl[a] - kl[probe]) for a in ['L', 'L-lowrank', 'T4']},
        top_prob_corr_with_bayes={k: float(np.corrcoef(norm(P).max(1), bayes.max(1))[0, 1]) for k, P in rows.items()
                                  if k not in ('ref: Prior',)},
    )
    items = json.loads((HERE / 'run/items.json').read_text())
    bnll = lambda xs: float(np.mean([-np.log(x['bayes'][x['color']]) for x in xs]))
    inner = np.random.default_rng(SEED).permutation(len(items['train']))
    out['split_difficulty_bayes_nll'] = dict(
        note='Bayes NLL of the items of each split. The inner-validation part selects the latent configuration.',
        inner_val=bnll([items['train'][i] for i in inner[:len(inner) // 5]]),
        inner_train=bnll([items['train'][i] for i in inner[len(inner) // 5:]]),
        train=bnll(items['train']), test=bnll(items['test']))
    res['post_hoc_exploratory'] = out
    for rec in res['records']:
        rec['label'] = 'EXPLORATORY'
    (HERE / 'RESULTS.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
