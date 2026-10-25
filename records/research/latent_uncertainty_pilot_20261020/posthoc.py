#!/usr/bin/env python
"""POST-HOC secondary views (exploratory, written after the pilot data existed). They do not enter the
reading rule. Input: run/results.json and run/test_probs.npz. Output: RESULTS.json (results.json plus the
key "post_hoc_exploratory")."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEED, N_BOOT = 20261020, 10_000


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

    probe = 'ref: probe4_block21_direct'
    out = dict(
        note='POST HOC, exploratory. Written after the pilot data existed. Not part of the reading rule.',
        kl_bayes_to_arm={k: float(v.mean()) for k, v in kl.items()},
        kl_differences={f'{a} minus L': ci(kl[a] - kl['L']) for a in ['T0', 'T1', 'T2', 'T3', 'T4', 'Card']},
        route_loss_vs_probe4_block21_nll={a: ci(nll[a] - nll[probe]) for a in ['L', 'T4']},
        route_loss_vs_probe4_block21_kl={a: ci(kl[a] - kl[probe]) for a in ['L', 'T4']},
        top_prob_corr_with_bayes={k: float(np.corrcoef(norm(P).max(1), bayes.max(1))[0, 1]) for k, P in rows.items()
                                  if k not in ('ref: Prior',)},
    )
    res['post_hoc_exploratory'] = out
    (HERE / 'RESULTS.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
