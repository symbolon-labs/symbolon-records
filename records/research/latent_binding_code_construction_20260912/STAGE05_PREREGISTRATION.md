# Stage 0.5 preregistration: LOO qualification of the fitted additive composition

Written BEFORE any receiver scoring. Training-ID-only data: the 56 ancestral
positive IDs. No novel-bank label enters the construction, the arms or the
decision. No candidate fit, no optimizer. One receiver-only process, 3600 s cap.

## Question

Stage 0 (STAGE0_RESULT.md) showed the fitted additive composition in
822db91d's own code space fixes first tokens for two novel IDs (I2 -> I3
exactly; C4 letter) but not A0. Is that construction a sound code in this
space, or is the novel-bank effect coincidence?

## Construction (fixed, nothing else is tried)

Training rows: the sealed 224 rows (56 IDs x 4 color queries) with the
candidate's training memories `features[i]` and sealed query ids / labels.

For each training ID i: fit A^{-i} = lstsq(Y_others, features_others B822) on
the other 55 IDs; compose q_i = labels(id_i) A^{-i}; payload for each of the
ID's four rows: v = m W^T + (q_i - m B) H^T with m = features[i], W = 822db91d
value matrix, B/H from its sealed coordinates.

Arms:
- `sham`: v = m W^T (candidate unchanged).
- `loo`: q_i as above.
- `null`: q_i = labels(pi(i)) A^{-i}, pi = next ID in sorted order (shifted
  label), scored against the true targets.

Scoring: `b.score` over the 224 rows, full 128 candidates, second run must
replay exactly. Mechanics: receiver hash unchanged before/after, no hooks, no
gradients.

## Decision rule (frozen)

Let letters = first-token-correct rows, joint = whole-ID-correct rows, on all
224 training rows.

- PASS iff letters(loo) >= 0.90 x letters(sham) AND joint(loo) > 3 x joint(null)
  AND letters(null) <= 0.10 x letters(sham).
- PASS => the fitted additive composition is a sound code construction in this
  space; the novel I2/C4 letter fixes are consistent with the mechanism; a
  revised Stage 1 (LOO-selected construction, bounded novel evaluation) may be
  designed and separately preregistered.
- FAIL => the composition direction is closed; the A0/I2/C4 sentence is not a
  foundation for further work.

Diagnostic only. No repair, no candidate, no optimizer, no change to any
existing gate or threshold, no novel outcome read in this stage. Sealed trees
read-only.
