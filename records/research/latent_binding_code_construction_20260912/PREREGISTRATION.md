# Stage 0 preregistration: reconstruct the composition diagnostic

Written BEFORE any receiver scoring. Read-only over all sealed trees; no fit, no
new candidate, no new data, no gates changed. One receiver-only process, 3600 s cap.

## What is being reconstructed

The claim (LATENT_BINDING_NEXT_SESSION_20260911.md; convergence INTERPRETATION):
"composed letter/digit codes reproduce the correct first token for A0 (A4),
I2 (I3) and C4 (C4) where the learned code does not."

No script/array of that in-session diagnostic was sealed, so the claim is
reconstructed mechanically from sealed artifacts. The learned baseline it
contrasts is sealed: novel first-token predictions A0->D1/D4, I2->G2/G3,
C4->D4 (822db91d PER_ID.json).

## Fixed recipes (nothing else is tried; no outcome-dependent extension)

Notation: m = selected-record memory (32x960, novel original condition);
`labels(name)` = 24-dim letter/digit one-hot; Y = 56x24 training label matrix;
`features` = alignment_source run/features.pt projected (56x960);
A = lstsq(Y, features @ B) additive component-code solve.

1. **f61-sham**: v = m W61^T (unchanged candidate; must replay sealed scores).
2. **f61-R3 exact composition**: q = labels(id) P; v = vexport + (q - m B61) H61^T.
   (The sealed oracle's `matched` arm; its novel readout was never opened.)
3. **f61-R1 fitted additive**: q = labels(id) A61; same formula.
4. **f61-R2 probe composition**: q = labels(probe_decode(id)) P; same formula.
5. **822d-sham**: unchanged 822db91d (must replay sealed scores).
6. **822d-R1 fitted additive**: A822 = lstsq(Y, features @ B822); q = labels(id) A822.
7. **822d-R2 probe composition**: q = labels(probe_decode(id)) A822.

Probe decodes (sealed RESULT.json): A0->H3, A2->M3, C4->C4, I2->I2, K0->K0,
K3->K3, M0->M0, M3->M3.

## Mandatory checks before any arm outcome is read

- Receiver parameter hash == sealed 6a3e1782...; bank tokens == sealed
  candidate_tokens; novel routing/joins per sealed `case`.
- Zero-correction payloads must reproduce the sealed prefill capture outputs
  EXACTLY (f61 and 822d), and the sham scores must equal the sealed
  trained_novel/correct.pt token_lp/lp/prediction EXACTLY.
- direct-score parity <= 1e-4 with same argmax.
- After scoring: no hooks, no gradients, receiver hash unchanged.

## Readout

Per arm and per held-out ID (4 colors each): predicted whole ID, first-token
correctness (predicted vs gold first candidate token), first-token-argmax
correctness, and joint count. Verdict: does any non-oracle recipe reproduce
A0 letter A / I2 letter I / C4 exact, and does the sham stay at the sealed
wrong first tokens? No repair, no candidate, no gates; diagnostic only.

## Environment note

The sealed oracle ran with tokenizers==0.22.2; the project venv had been bumped
to 0.23.1 on 2026-09-12 (collateral of the ember binding install) which broke
`import transformers`. Restored tokenizers==0.22.2 (uv, cached wheel) before this
stage; torch/transformers/numpy/safetensors unchanged; platform matches sealed.
