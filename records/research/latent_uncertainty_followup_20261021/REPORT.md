# Uncertainty follow-up: report (EXPLORATORY)

Status: EXPLORATORY. One new pair of data seeds, one synthetic task, one sender and one receiver,
development scale. This is not a confirmatory result. Every number in this report is EXPLORATORY.

Date: 2026-10-21 (the run went from 2026-10-20, 23:19 to 2026-10-21, 00:35). Plan: `PLAN.md` (the
original text did not change; it has one dated addition of 23:15, written before any follow-up data).
Choices that the plan does not fix: `IMPLEMENTATION_NOTES.md`. All values: `RESULTS.json`.
Pilot: `../latent_uncertainty_pilot_20261020/` (no file in it changed).

## 1. Result (EXPLORATORY)

Case 1 of the reading rule applies: "the T4 win stays". The follow-up is valid. The fit qualifier
applies: the fit limit is not removed.

- Inner validation selected L* = L (the full map 2048 -> 960) at learning rate 1e-5, 17 epochs.
- T4 (probe distribution in text) is better than L* by 0.050 nats (95% CI 0.011 to 0.092).
  In the pilot, this difference was 0.039 nats (95% CI 0.005 to 0.075).
- The selected learning rate (1e-5) is the lower end of the L grid. Thus, by step 3 of the rule, the
  report must say that the fit limit is not removed. The L map at 1e-5 still overfits: at its best
  epoch, the train loss was 0.051 and the inner-validation NLL was 0.423.
- Secondary row (not part of the rule): the low-rank map (r = 16, learning rate 1e-3, an inside value
  of its grid) overfits much less (train loss 0.327, inner-validation NLL 0.439). T4 is also better than
  this map, by 0.080 nats (95% CI 0.036 to 0.123).
- New text arms: the sender's own four-color softmax as text (T5, NLL 0.708) and 8 sampled sender
  answers as text (T6, NLL 0.761) are much worse than L* (by 0.175 and 0.229 nats). T4 is better than
  T5 by 0.226 nats.

Thus, on this task, a text message with the numbers of a probe on the sender state carries the
uncertainty better than one latent vector. The sender's own text (its answer, its softmax, or its
samples) does not carry it.

## 2. Question

The pilot gave case 2, "better estimator, not a latent carrier". T4 beat L by 0.039 nats. But the pilot
L map (2 million parameters) overfit 1,600 inner-train items, and its selected learning rate (3e-5) was
the lowest value of its grid. This follow-up asks: does "T4 beats L" stay when the latent map does not
overfit? It adds a low-rank map (L-lowrank), a learning rate of 1e-5 for L, and two text arms (T5, T6).

## 3. What ran

- Task, sizes, models, snapshots, prompts, block-21 state, receiver format and readout: the same as in
  the pilot. Train 2,000 items, test 1,000 items, from new seeds (2026102101, 2026102102). Other seed
  20261021. T6 sample seed 2026102103.
- Arms: T0 to T6, Card, L, L-lowrank, and the shuffle controls L-shuf, L-lowrank-shuf, L-testshuf and
  L-lowrank-testshuf. References: Bayes, Bayes-9, Bayes-9-cal, Prior, the sender softmax and the
  4-way probes on blocks 21 and 28.
- Latent grid (after the PLAN.md addition of 23:15): L at {1e-5, 3e-5, 1e-4}; L-lowrank at
  r in {16, 64} and {3e-4, 1e-3, 3e-3}; patience 3, at most 20 epochs. The addition reduced the grid
  for time only, before any follow-up data. The original grid needed more than 1.5 h on this laptop.
- Preconditions before the run: no other job (no pilot.py, followup.py, harness.py, smcheck.py,
  latent_capacity or latent_r06 process), swap 0 MB, charger 140 W. The run went under
  `caffeinate -i -w 14413`, with `tee -a` logs and a time limit of 100 min.
- Smoke test 1: 23:09:29 to 23:14:56 (64 train and 32 test items, smoke seeds). Smoke test 2 (latent and
  analysis stages only, after the grid change): 23:16:38 to 23:19:10. Both ran without error. The stacked
  training agrees with separate training to 6.1e-6 nats or better.
- Full run: 23:19:21 to 00:34:35 (75 min), rc 0. Stage times: sender 943 s, probes 222 s, text arms 225 s,
  latent 2,938 s (grid 1,712 s, final fits and test passes 1,225 s), analysis 184 s. Swap after: 0 MB.
- Overlap with other jobs: no other job ran during the full run. The R.06 job `t5.py smoke` (attempt 1,
  CPU) ran from 23:10:10 to 23:11:10, during smoke test 1, because its preflight did not yet check for
  `followup.py`. This can affect only the smoke timing numbers. The R.06 smoke attempts 2 and 3 started
  at 23:15:25 and 00:35:10, outside the run.

## 4. Main table (EXPLORATORY; test split, n = 1,000, readout view)

NLL in nats. "Gap" is NLL minus the Bayes NLL. ECE uses 15 bins.

| arm | message | NLL | gap | Brier | acc | ECE |
|---|---|---|---|---|---|---|
| T0 | answer only | 0.8265 | 0.4821 | 0.4110 | 0.755 | 0.0253 |
| T1 | answer + verbal digit | 0.7945 | 0.4501 | 0.3998 | 0.755 | 0.0205 |
| T2 | answer + token-probability digit | 0.7587 | 0.4143 | 0.3771 | 0.755 | 0.0284 |
| T3 | answer + probe scalar digit | 0.6138 | 0.2694 | 0.3099 | 0.766 | 0.0354 |
| T4 | answer + probe distribution (4 digits) | **0.4827** | 0.1383 | 0.2294 | 0.844 | 0.0146 |
| T5 | answer + sender softmax distribution (4 digits) | 0.7081 | 0.3637 | 0.3627 | 0.750 | 0.0433 |
| T6 | answer + 8 sampled sender answers | 0.7615 | 0.4171 | 0.3874 | 0.744 | 0.0242 |
| Card | the receiver reads the whole card | 0.5471 | 0.2027 | 0.2929 | 0.799 | 0.0350 |
| L (= L*) | one soft vector, map 2048 -> 960, lr 1e-5 | 0.5330 | 0.1886 | 0.2551 | 0.828 | 0.0777 |
| L-lowrank | one soft vector, map 2048 -> 16 -> 960, lr 1e-3 | 0.5625 | 0.2181 | 0.2738 | 0.811 | 0.0884 |
| L-shuf | L, states shuffled (train and test) | 1.6321 | 1.2877 | 0.8568 | 0.256 | 0.2269 |
| L-lowrank-shuf | L-lowrank, states shuffled (train and test) | 1.4508 | 1.1063 | 0.7814 | 0.258 | 0.1123 |
| L-testshuf | trained L, test states shuffled only | 3.8749 | 3.5305 | 1.3859 | 0.244 | 0.6617 |
| L-lowrank-testshuf | trained L-lowrank, test states shuffled only | 4.2924 | 3.9480 | 1.3818 | 0.234 | 0.6609 |

Reference rows (EXPLORATORY; no receiver):

| reference | NLL | gap | Brier | acc | ECE |
|---|---|---|---|---|---|
| Bayes (exact posterior) | 0.3444 | 0 | 0.1836 | 0.858 | 0.0149 |
| Bayes-9-cal (exact P(c given the 1/9-rounded posterior)) | 0.3513 | 0.0069 | 0.1838 | 0.858 | 0.0148 |
| Bayes-9 (literal rounded posterior, renormalized) | 0.8522 | 0.5078 | 0.1860 | 0.858 | 0.0321 |
| Prior (uniform) | 1.3863 | 1.0419 | 0.7500 | 0.247 | 0.0030 |
| Sender softmax over the four color tokens | 0.8281 | 0.4837 | 0.3961 | 0.755 | 0.1538 |
| Probe on the block-21 state (4-way) | 0.4395 | 0.0951 | 0.2223 | 0.842 | 0.0295 |
| Probe on the block-28 state (4-way) | 0.4430 | 0.0986 | 0.2281 | 0.841 | 0.0346 |

Pilot values for comparison (other items): T4 0.4832, L 0.5223, T2 0.7995, Card 0.5703, Bayes 0.3348.

## 5. Key differences (EXPLORATORY; paired bootstrap, 10,000 draws over test items)

A positive value means that the arm after "minus" has the lower NLL.

| difference in NLL | mean | 95% CI |
|---|---|---|
| T2 minus L (T2 is the best of T0, T1, T2) | +0.2257 | +0.1609 to +0.2895 |
| T3 minus L | +0.0808 | +0.0220 to +0.1385 |
| **T4 minus L (primary)** | **-0.0503** | **-0.0921 to -0.0107** |
| T5 minus L | +0.1751 | +0.1128 to +0.2369 |
| T6 minus L | +0.2285 | +0.1641 to +0.2900 |
| Card minus L | +0.0141 | -0.0488 to +0.0726 |
| L-shuf minus L | +1.0991 | +1.0053 to +1.1876 |
| T4 minus L-lowrank | -0.0798 | -0.1230 to -0.0355 |
| T3 minus L-lowrank | +0.0514 | -0.0082 to +0.1106 |
| T2 minus L-lowrank | +0.1963 | +0.1292 to +0.2645 |
| L minus L-lowrank | -0.0294 | -0.0786 to +0.0213 |
| T5 minus T4 | +0.2255 | +0.1764 to +0.2755 |
| T6 minus T5 | +0.0533 | +0.0241 to +0.0821 |
| T2 minus T5 | +0.0506 | +0.0233 to +0.0775 |
| T6 minus T2 | +0.0027 | -0.0196 to +0.0250 |
| T3 minus T4 | +0.1312 | +0.0835 to +0.1784 |
| T0 minus Card | +0.2794 | +0.2224 to +0.3362 |

`RESULTS.json` has all 30 differences.

## 6. Reading rule (EXPLORATORY values; rule from PLAN.md section 4 and the addition of 23:15)

"A beats B" means: NLL(B) minus NLL(A) is at least 0.02 nats, and the lower CI bound is above 0.

| step | value | result |
|---|---|---|
| L* (lower best inner-validation NLL) | L at 1e-5: 0.4228; L-lowrank (r 16, 1e-3): 0.4394 | L* = L |
| 1. Sanity: L-shuf NLL at least 1.366 | 1.632 | pass |
| 1. Sanity: Card NLL not more than T0 NLL | 0.547 against 0.827 | pass |
| 2. T4 beats L* | NLL(L*) - NLL(T4) = +0.050, CI +0.011 to +0.092 | yes |
| 2. L* beats T4 | -0.050 | no |
| 3. Fit qualifier: lr of L* at an end of its grid {1e-5, 3e-5, 1e-4} | 1e-5 | yes: fit limit not removed |
| 3. Fit qualifier: epoch count of L* is 20 | 17 | no |
| 4. Pilot rule: L* beats the best of T0 to T2 (T2) | +0.226, CI +0.161 to +0.290 | yes |
| 4. Pilot rule: L* beats T3 | +0.081, CI +0.022 to +0.139 | yes |
| 4. Pilot rule: L* beats T4 | -0.050 | no: pilot case 2 |
| 6. Bayes-9 clause | applies only to case 3 | not used |

Verdict: case 1, "the T4 win stays". The follow-up is valid. The fit limit is not removed, because the
selected learning rate is the lower end of the L grid. The pilot rule again gives pilot case 2.

Step 5, secondary rows:

| comparison | mean (NLL of the second arm minus NLL of the first arm) | 95% CI | result |
|---|---|---|---|
| L* beats T5 | +0.1751 | +0.1128 to +0.2369 | yes |
| L* beats T6 | +0.2285 | +0.1641 to +0.2900 | yes |
| T4 beats T5 | +0.2255 | +0.1764 to +0.2755 | yes |
| T5 beats T6 | +0.0533 | +0.0241 to +0.0821 | yes |
| T5 beats T2 | +0.0506 | +0.0233 to +0.0775 | yes |

The fixed meanings (PLAN.md section 4): the sender's own four-color distribution, as text, loses
uncertainty that the latent route keeps. The probe reads more from the state than the sender's own
softmax shows. Eight samples lose 0.053 nats against the full sender distribution.

## 7. Latent fit (EXPLORATORY)

Inner split: 1,600 train items and 400 validation items. "Run" is the number of epochs before the stop.

| configuration | map parameters | best inner-val NLL | best epoch | run |
|---|---|---|---|---|
| L, lr 1e-5 | 1,967,040 | **0.4228** | 17 | 20 |
| L, lr 3e-5 | 1,967,040 | 0.4426 | 5 | 8 |
| L, lr 1e-4 | 1,967,040 | 0.4437 | 4 | 7 |
| L-lowrank r 16, lr 3e-4 | 49,088 | 0.4772 | 5 | 8 |
| L-lowrank r 16, lr 1e-3 | 49,088 | **0.4394** | 7 | 10 |
| L-lowrank r 16, lr 3e-3 | 49,088 | 0.5085 | 5 | 8 |
| L-lowrank r 64, lr 3e-4 | 193,472 | 0.4783 | 4 | 7 |
| L-lowrank r 64, lr 1e-3 | 193,472 | 0.4760 | 3 | 6 |
| L-lowrank r 64, lr 3e-3 | 193,472 | 0.5235 | 4 | 7 |

Overfit at the best epoch (inner-val NLL minus the mean train loss of that epoch):

- L, lr 1e-5: 0.423 - 0.051 = 0.372. The pilot L had 0.518 - 0.230 = 0.288.
- L-lowrank r 16, lr 1e-3: 0.439 - 0.327 = 0.113.

Thus the low-rank map overfits much less, but it is not better on inner validation or on test
(L minus L-lowrank: -0.029, CI -0.079 to +0.021). With a lower learning rate, the full map trains longer
and gets a lower inner-validation NLL, but it still overfits.

Test NLL against inner-validation NLL: L 0.533 against 0.423; L-lowrank 0.562 against 0.439. Post hoc:
the inner-validation items are easier than the test items (Bayes NLL 0.304 against 0.344). This explains
about 0.04 of the 0.11 to 0.12 nats. The final fit on 2,000 items also has 25% more steps for each epoch
than the inner fit on 1,600 items (the pilot used the same procedure).

## 8. Secondary views (EXPLORATORY; fixed in the plan)

- Zero-shot receiver (no readout): all zero-shot NLLs are above ln 4, except T6 (1.313). With T6, the
  zero-shot receiver gets accuracy 0.722, near the sender accuracy (0.755). With T0, T4, T5 and Card, it
  gets 0.26. Thus the main results come from the readout view, as in the pilot.
- Sender: answer accuracy 0.755. Bayes argmax accuracy 0.858. The sender agrees with the Bayes argmax
  on 79.4% of the test items. The 4-way probe on its block-21 state gets accuracy 0.842 and NLL 0.440.
- Sender confidence: the verbal digit (T1) is 7 for 887 items and 9 for 113 items (AUC 0.572 for "the
  answer is correct"). The token-probability digit (T2) has AUC 0.711. The probe scalar (T3) has AUC
  0.903 (0.900 after the rounding to a digit).
- T5: the digit of the answer color is equal to the T2 digit for all 1,000 test items, as expected.
  The four digits add up to 9 for 844 items, to 8 for 145 items and to 10 for 11 items.
- T6: for 654 of 1,000 test items, all 8 samples are the same color. The count of the answer color
  among the samples has AUC 0.650 for "the answer is correct" (T2: 0.711).
- Bayes-9: 22 of 1,000 test items (2.2%) get the digit 0 on the true color. The cost of the 1/9 rounding
  for a calibrated reader is 0.0069 nats on this test split.
- Readout L2 weights (cross-validation): T0 1, T1 1e-5, T2 1e-2, T3 1e-3, T4 1e-2, T5 1e-2, T6 1,
  Card 1e-2. No weight is at the end of the grid {1e-5, ..., 10} except T1 (1e-5, the lower end).
  Probes: L2 weight 0.1 for T3, T4 and the block-28 probe.

## 9. Post-hoc views (EXPLORATORY; the same views as the pilot; not part of the rule)

Source: `posthoc.py`, key `post_hoc_exploratory` in `RESULTS.json`.

- KL from the exact posterior to each arm (less noise than the NLL): T4 0.135, Card 0.200, L 0.206,
  L-lowrank 0.218, T3 0.286, T5 0.359, T6 0.421, T2 0.424, T0 0.495. T4 minus L: -0.071 (95% CI -0.094
  to -0.050). The order is the same as in the NLL view.
- Route loss against the shared block-21 probe (NLL 0.440): L adds 0.093 nats (95% CI 0.055 to 0.135),
  L-lowrank adds 0.123 (95% CI 0.080 to 0.166), and T4 adds 0.043 (95% CI 0.023 to 0.064). As in the
  pilot, the text route keeps more of the probe information than the trained latent routes.
- Correlation of the top probability of each arm with the Bayes top probability: probe 0.794, T4 0.736,
  L-lowrank 0.652, L 0.644, T3 0.598, Card 0.536, T2 0.332, T5 0.330, T6 0.245, T1 0.233, T0 0.005.
- Split difficulty (Bayes NLL): inner validation 0.304, inner train 0.340, train 0.333, test 0.344.

## 10. Exploratory reading

1. The pilot result replicates on fresh items. A text message with probe numbers (T4) beats the latent
   message selected by inner validation (0.050 nats, CI 0.011 to 0.092).
2. A map with less capacity does not change this. The low-rank map (r = 16) overfits much less, but T4
   beats it by 0.080 nats (CI 0.036 to 0.123). This row is secondary, because inner validation selected
   the full map.
3. By the rule, the fit limit is not removed: the selected learning rate (1e-5) is the lower end of its
   grid, and that map still overfits. Thus a full map with a still lower learning rate or more epochs
   is not excluded.
4. The sender's own text loses most of the uncertainty: its softmax (T5) and its samples (T6) lose to
   the latent message by 0.175 and 0.229 nats. This agrees with the paper's result for weak text
   carriers. But a probe can write the uncertainty as text, and that text is the best carrier here.
5. On this task, the result does not support the claim that uncertainty needs a latent carrier.

## 11. Limits

- One new seed pair, one synthetic task, one sender and receiver pair. This is not confirmatory.
- The fit limit is not removed (section 6, step 3). The L grid ends at 1e-5, and the selected epoch (17)
  is near the limit (20).
- The grid is smaller than the first plan (PLAN.md addition of 23:15, for time). Not tested: L at 3e-4
  and 1e-3 (worse in the pilot), L-lowrank at 1e-4, ranks other than 16 and 64, patience 6, 40 epochs.
- The rank r = 16 is the lower end of {16, 64}. A rank below 16 is not tested.
- The fit methods are different. The T4 probe and the text readouts use a convex L2 fit with
  cross-validation. The latent maps train through the receiver with AdamW and early stopping.
- The inner-validation items are easier than the test items (Bayes NLL 0.304 against 0.344). The
  selection uses only this part. The selection order of the two latent families can change on another
  split (their inner-validation difference is 0.017 nats).
- T6 lists the 8 samples in the order of the draw. A count format is not tested. The zero-shot receiver
  reads T6 better than the other messages, but the readout view of T6 is not better than T2.
- T3 to T5 carry numbers with 1/9 resolution. For a calibrated reader, this costs 0.0069 nats here.
- The receiver is small (360M). It does not use most messages zero-shot.

## 12. Next steps (not run; a new GPU run needs [researcher]'s word)

- If a claim is necessary, write a confirmatory plan with new seeds, more than one seed, and a fixed
  map (for example L at 1e-5 with up to 40 epochs, and L-lowrank r = 16 at 1e-3), with T4 as the
  primary comparison.
- To remove the fit limit, extend the L grid below 1e-5 (for example 3e-6) with an epoch limit of 40.

## 13. Files

- `PLAN.md`: the plan (original text unchanged) and the dated addition of 23:15 (grid, patience, epoch
  limit; after the smoke timing, before any follow-up data). `PLAN.original_20261020.md`: a copy of the
  plan before the addition.
- `IMPLEMENTATION_NOTES.md`: choices that the plan does not fix; section 5 has the smoke facts.
- `followup.py`: all stages (a copy of the pilot code with the new arms, seeds and the stacked latent
  trainer). `run_followup.sh`: precondition checks, `caffeinate`, `tee -a`, time limit. `posthoc.py`:
  section 9 and `RESULTS.json`.
- `RESULTS.json`: every value with its arm, view and metric (also as a flat `records` list, each with
  the label EXPLORATORY), the differences, the rule, the diagnostics and the post-hoc block.
- `run/`: items, sender outputs, probes, receiver states, `t6_samples.npz`, the latent grid
  (`latent_grid.json`) and curves (`latent.json`), per-item test probabilities of every arm and
  reference (`test_probs.npz`), `results.json`, `table.md`.
- `smoke/`: the smoke outputs (`smoke/v1_*` are the latent and analysis outputs of smoke test 1).
- Logs: `run.log`, `run_wrapper.log`, `smoke.log`, `smoke_wrapper.log`, `posthoc.log`.
