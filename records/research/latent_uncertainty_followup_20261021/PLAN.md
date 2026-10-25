# Follow-up: does "T4 beats L" stay when the latent map does not overfit?

Status: EXPLORATORY follow-up. One new pair of data seeds, one synthetic task, development scale.
Not confirmatory. Every value of this follow-up is exploratory.

Written 2026-10-20, 23:05, before any data of this follow-up exists. The folder name uses the date
20261021, as the parent session asked. The pilot folder `../latent_uncertainty_pilot_20261020/` does
not change.

Go-ahead: [researcher] started this session with a click on the card of the parent session. That click is
[researcher]'s go-ahead for this one GPU run, of about 1.5 h at most. No other GPU job starts.

## 1. Question

The pilot (2026-10-20) gave case 2, "better estimator, not a latent carrier". The test NLL values were:
L 0.522, T4 0.483, T2 0.800, Bayes 0.335. T4 beat L by 0.039 nats (95% CI 0.005 to 0.075).

The pilot had one known limit. The L map (2048 x 960, about 2 million parameters) overfit the 1,600
inner-train items, and the selected learning rate (3e-5) was the lowest value of the grid. Thus the
T4 win can be a limit of the map fit, and not a limit of the latent route.

This follow-up asks: does T4 still beat the latent message when the latent map does not overfit?
It also adds two text arms that the pilot did not have (T5 and T6).

## 2. What stays the same as the pilot

- Task: a box has a hidden color from {red, blue, green, yellow} (uniform). Five witness reports are
  each right with probability q = 0.65, else they name one of the three other colors (uniform).
  The exact Bayes posterior is known for each item. The card text is the same as in the pilot.
- Sizes: 2,000 train items and 1,000 test items. Train and test share no box ID.
- Sender: Qwen3-1.7B, snapshot 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e, bf16 on MPS, frozen.
  The prompts are the same as in the pilot.
- Receiver: SmolLM2-360M-Instruct, snapshot a10cc1512eabd3dde888204e902eca88bddb4951, fp32 on MPS,
  frozen. The input format is the same as in the pilot.
- Sender state: the residual stream after block 21, at the last token of the answer prompt.
  The state after block 28 is a secondary view only.
- Readout: one linear layer (960 -> 4, with bias) on the final hidden state of the receiver, for every
  arm. For the text arms and Card, it is an L2 logistic fit. Its weight comes from the grid
  {1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10} by 5-fold cross-validation NLL on the train split.
- Pilot arms, again on the new items: T0, T1, T2, T3, T4, L, L-shuf, Card. The secondary view
  L-testshuf is also again. The T3 and T4 probes and their cross-fit are the same as in the pilot.
- Reference rows, again: Bayes, Bayes-9, Bayes-9-cal, Prior, the sender softmax, and the 4-way probes
  on blocks 21 and 28.
- Metrics: test NLL (nats) is primary. Brier, accuracy, ECE (15 bins) and the gap to the Bayes NLL are
  secondary. A paired bootstrap (10,000 draws over test items) gives each 95% CI.
- The secondary zero-shot view of the receiver is again in the results.

## 3. What changes

### 3.1 Seeds (all new)

- Data: train seed 2026102101, test seed 2026102102. The pilot used 2026102001 and 2026102002.
- Smoke test: train seed 2026102191, test seed 2026102192. The pilot smoke test used 2026102091 and
  2026102092.
- Other seed: 20261021 (the pilot used 20261020). It sets the cross-validation folds, the inner split,
  the map initialization, the batch order, the shuffles and the bootstrap.
- T6 sample seed: 2026102103.

### 3.2 New text arm T5: the sender's own four-color softmax as text

The message has the T4 format. The digits come from the sender's softmax over its four color logits,
and not from a probe:

    The box is red. Confidence: red 7 out of 9, blue 1 out of 9, green 1 out of 9, yellow 0 out of 9.

The color after "The box is" is the sender answer (the argmax), as in all arms. Each digit is
round(9 x p_c), with round(x) = floor(x + 0.5). T5 needs no fit, so it needs no cross-fit.

### 3.3 New text arm T6: 8 sampled sender answers as text

For each item, 8 answers are sampled from the sender's four-color softmax (temperature 1). The samples
come from the cached color logits of the sender stage. There is no new sender pass. The message is:

    The box is red. Eight sampled answers: red, red, blue, red, red, green, red, red.

The samples stay in the order of the draw. A NumPy generator with seed 2026102103 draws them. The train
split uses the stream [2026102103, 0], and the test split uses the stream [2026102103, 1].

### 3.4 L: the pilot map with a wider learning-rate grid

L is the pilot arm: a linear map (2048 -> 960, with bias) from the standardized block-21 state into one
soft input vector of the receiver. The map and the readout train together with AdamW (weight decay 0.01,
batch 32, gradient norm clip 1.0, at most 40 epochs, patience 6), as in the pilot.

Change: the learning-rate grid goes down to 1e-5. The grid is {1e-5, 3e-5, 1e-4, 3e-4, 1e-3}.

### 3.5 New arm L-lowrank: a map with less capacity

The map is 2048 -> r -> 960. It is a product of two linear layers with no nonlinearity between them:
first 2048 -> r (no bias), then r -> 960 (with bias). The rank r comes from {16, 64}. Thus the map has
49,088 parameters (r = 16) or 193,472 parameters (r = 64), against 1,967,040 for L. The readout, the
optimizer, the batch, the clip, the epoch limit and the patience are the same as for L.

Initialization: the first layer has weight std 1/sqrt(2048). The second layer has weight std
(embedding std)/sqrt(r), and its bias is the mean receiver token embedding. Thus at the start, the output
of the map has the same scale as the output of the L map.

The learning-rate grid of L-lowrank is {1e-4, 3e-4, 1e-3, 3e-3}. Reason: AdamW moves each parameter by
about the learning rate in each step. The receiver embedding std is 0.122. Thus the L map weights start
with std 0.0027, and the low-rank factors start with std 0.015 to 0.030. These are 6 to 11 times
larger. Thus a low-rank map needs a learning rate that is about 10 times larger for the same relative
step. Values below 1e-4 move the
low-rank factors very slowly, and they would use most of the time budget. This is a deviation from
the parent task, which asked for a grid down to 1e-5. The L grid goes down to 1e-5, as asked.

### 3.6 Selection of the latent arm (inner validation only)

- The inner split is 20% of the train split (400 items) for validation, and 80% (1,600 items) for
  training, as in the pilot.
- For each family (L, and L-lowrank), the configuration with the lowest inner-validation NLL gives the
  learning rate, the rank (L-lowrank only) and the epoch count of that family.
- L* (the selected latent arm) is the family with the lower best inner-validation NLL.
- Then each family trains again on the full train split, with its selected values. Both families give a
  test row. The primary comparison uses L* only.
- The test split selects nothing.

### 3.7 Shuffle controls

- L-shuf and L-lowrank-shuf: the same procedure and the same selected values as L and L-lowrank. The
  train states and the test states are each permuted across items.
- L-testshuf and L-lowrank-testshuf (secondary): the trained arm, with only the test states permuted.

### 3.8 Implementation of the latent grid (speed only)

All latent configurations train at the same time. In each step, each configuration gets the same
batch of items (the pilot also used the same batch order for each learning rate). Their soft inputs go
through the frozen receiver in one stacked forward pass. Each configuration has its own map, readout,
optimizer, gradient clip and early stop. The math is the same as for separate runs. The smoke test
checks this: one configuration that trains alone must give the same validation curve as in the stack
(maximum absolute difference below 1e-3 nats).

## 4. Reading rule (fixed now)

Definitions:

- "A beats B": the mean over test items of NLL(B) - NLL(A) is at least 0.02 nats, AND the lower bound
  of its 95% CI (paired bootstrap, 10,000 draws) is above 0. This is the pilot rule.
- L*: the selected latent arm (section 3.6).
- L*-shuf: L-shuf if L* is L, or L-lowrank-shuf if L* is L-lowrank.

Step 1, sanity. L*-shuf must have NLL of at least ln 4 - 0.02 = 1.366. Card must have NLL not more than
T0 (point estimates). If one check fails, the follow-up is case 0 (invalid).

Step 2, the primary comparison. This step gives the case of the follow-up:

- Case 1, "the T4 win stays": T4 beats L*.
- Case 2, "no winner by the rule": T4 does not beat L*, and L* does not beat T4.
- Case 3, "the T4 win reverses": L* beats T4.

Step 3, fit qualifier. If the learning rate of L* is at an end of its grid, or its epoch count is 40,
then the report must say that the fit limit is not removed. This applies to all cases. The rank values
{16, 64} are both ends by design, so the rank is not part of this qualifier.

Step 4, the pilot rule again, with L* in place of L. "Best of T0 to T2" is the arm with the lowest test
NLL point estimate.

- Pilot case 1: L* beats the best of T0 to T2, L* beats T3, and L* beats T4.
- Pilot case 2: L* beats the best of T0 to T2, but L* does not beat T3 or T4.
- Pilot case 3: L* does not beat the best of T0 to T2.

Step 5, secondary rows (no case). The report gives the "beats" result, the mean and the CI for:
L* against T5, L* against T6, T4 against T5, T5 against T6, and T5 against T2.

Step 6, the Bayes-9 clause of the pilot. If case 3 holds by a margin that is not clearly larger than
the Bayes-9-cal cost, then a follow-up with finer text numbers is necessary before any claim.

The meaning of each case, fixed now:

- Case 1: A latent map that inner validation selects from two map sizes still carries less of the
  uncertainty to this receiver than probe numbers in text. The pilot result replicates on fresh items.
  The claim "uncertainty needs a latent carrier" gets no support from this task.
- Case 2: The follow-up does not replicate the T4 win by the rule. It also shows no latent advantage.
  Text with probe numbers carries about as much uncertainty as the latent message.
- Case 3: The T4 win of the pilot was a limit of the map fit. With the selected map, the latent route
  carries more uncertainty than probe text. A confirmatory run is necessary before any claim.
- Secondary: if L* beats T5, then the sender's own four-color distribution, as text, loses uncertainty
  that the latent route keeps. If T4 beats T5, then the probe reads more from the state than the sender's
  own softmax shows. T5 against T6 shows the cost of 8 samples against the full distribution.

## 5. Time and stop rule

- Expected time: sender about 16 min, probes about 4 min, text arms about 6 min, latent grid and final
  fits about 35 to 45 min, analysis about 4 min. The total is about 65 to 75 min.
- If the full run is still active after 100 min, stop it. Then report the run as incomplete, and do not
  start it again without the word of [researcher].
- Before each GPU job (smoke test and full run): no other job of the type pilot.py, followup.py,
  harness.py, smcheck.py, latent_capacity or latent_r06 is active; the swap use is known; the charger
  gives at least 60 W. Each job runs under `caffeinate -i -w <pid>`, and its log uses `tee -a`.

Nothing in this file changes after the data exists. Any later change is a new, dated section.

## Addition 2026-10-20, 23:15 (after the smoke test, before any follow-up data)

The text above does not change. This section changes the latent grid, the patience and the epoch
limit. The reason is time only. No follow-up data exists at this time. The smoke test used other seeds
and 64 items, and its result has no effect on this change.

Fact from the smoke test (`--bench`): on this laptop (M1 Pro, 32 GB), one train step of one latent
configuration takes 0.47 s. A stack of 13 configurations takes 5.18 s (0.40 s for each). Thus the
stack gives almost no speed gain, because the GPU is already at full use with one configuration. One
configuration needs about 22 s for each epoch of the inner grid. The grid of section 3 (13
configurations, patience 6, at most 40 epochs) needs about 70 min or more. With the sender stage
(about 16 min), the run does not fit in 1.5 h.

Changes:

1. Patience: 3 (was 6). Epoch limit: 20 (was 40). For all latent configurations.
   Check on the pilot curves (`../latent_uncertainty_pilot_20261020/run/latent.json`): for all four
   pilot learning rates, patience 3 with limit 20 selects the same epoch (5, 5, 2, 2) as patience 6
   with limit 40. The pilot's best epochs were 5 or less.
2. L grid: {1e-5, 3e-5, 1e-4} (was {1e-5, 3e-5, 1e-4, 3e-4, 1e-3}). The grid still goes down to 1e-5.
   In the pilot, 3e-4 and 1e-3 had a worse inner-validation NLL than 3e-5, and they overfit after
   epoch 2.
3. L-lowrank grid: {3e-4, 1e-3, 3e-3} for each rank in {16, 64} (was {1e-4, 3e-4, 1e-3, 3e-3}).
   With the factor of about 10 in section 3.5, this grid is equal to {3e-5, 1e-4, 3e-4} for the L map.
   It contains the equivalent of the best pilot value (3e-5). The value 1e-4 would need the most epochs.
4. All other items stay: both ranks, the selection rule of section 3.6, the four final fits (L,
   L-lowrank, L-shuf, L-lowrank-shuf), the shuffle controls and the reading rule of section 4.

Effect on the reading rule: the grid ends in step 3 (fit qualifier) are now 1e-5 and 1e-4 for L, and
3e-4 and 3e-3 for L-lowrank. The epoch limit in step 3 is now 20.

New time estimate: sender, probes and text arms about 24 min; the latent grid about 25 to 35 min
(9 configurations); the final fits about 15 min; the analysis about 4 min. The total is about 70 min.
The stop rule of section 5 (100 min) stays.
