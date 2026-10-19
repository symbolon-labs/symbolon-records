# Interpretation: what the convergence null does and does not show

## It shows

1. **The sealed 4/4/4 was not a truncated optimum.** The trajectory's held-out
   whole-ID margins peak near the sealed step-200 checkpoint: from d0 they rise
   +10 to +19 nats, then fall back for six of eight IDs by step 600 while the
   training set becomes exactly fit (224/224 letters and digits, loss 0.5465 to
   0.0954). The one success (M3) was marginal (+0.25) and crossed below zero.
   So "train the same objective longer" is empirically the wrong direction for
   improving held-out whole-ID transfer.

2. **There is a real trade-off between fitting the 56 training IDs and placing
   the 72 unseen combinations.** The map is linear and the unseen features lie
   inside the training feature span, so their codes are determined by the
   training codes' positions. Driving the training codes to exact
   classification moves the held-out codes out of their (thin) correct regions.

3. **The failures are not numerical.** Everything that could be checked
   mechanically is exact: bitwise start reproduction, exact environment replay of
   both historical baselines and the sealed candidate, exact parity, routing and
   reader contracts, unchanged receiver hash, and full restoration.

## It does not show

1. It is not an impossibility proof for composition, for low-rank codes, for
   other loss functions, or for the receiver. It bounds one parameterization, one
   objective and one supervision set along one trajectory.

2. It does not identify a causal stage. The per-ID margins are a finite-bank
   descriptive readout of a frozen nonlinear receiver, not a claim about which
   circuit inside it decides the answer.

3. It does not license tuning the stopping point on this bank. Because the peak
   lies before convergence, "stop earlier" is an obvious hypothesis — but
   selecting a checkpoint by held-out novel accuracy would be test-set
   selection. Any future step-count choice must be validated on training IDs
   only (for example leave-one-ID-out folds), never on these eight.

4. It does not mean the eight IDs are unreachable. The earlier supervised
   oracle work showed correct codes exist near these codes, and the component
   diagnostic in this session showed that composed letter/digit codes reproduce
   the correct first token for some IDs (A0->A4, I2->I3, C4->C4) where the
   learned code does not. Both point at code construction, not at capacity.

## Practical consequences

- Keep 822db91d (step 200, novel 4/4/4) as the standing best; this run does not
  replace it and must not be averaged or blended with it post hoc.
- Two directions remain evidence-backed and were not tested here: (a) build the
  code from the component structure that the training-only linear probe decodes
  with 6/8 held-out accuracy, instead of from a generic linear map; (b) choose
  any stopping or configuration point using training-ID-only held-out folds.
- No repair, adapter, threshold change, new data, FINAL/Arabic or generation work
  is licensed by this result.
