# Uncertainty pilot: report (EXPLORATORY)

Status: EXPLORATORY. One seed, one synthetic task, one sender and one receiver, development scale.
This is not a confirmatory result. Every number in this report is exploratory.

Date: 2026-10-20. Plan: `PLAN.md` (the original text did not change; it has two dated additions).
Choices that the plan does not fix: `IMPLEMENTATION_NOTES.md`. All values: `RESULTS.json`.

## 1. Result

Case 2 of the reading rule applies: "better estimator, not a latent carrier". The pilot is valid.

- The latent message (L) is better than the best sender-written text message (T2) by 0.277 nats
  (95% CI 0.223 to 0.331).
- L is better than the probe scalar in text (T3) by 0.124 nats (95% CI 0.076 to 0.171).
- But the probe distribution in text (T4) is better than L by 0.039 nats (95% CI 0.005 to 0.075).
- Thus the sender state holds uncertainty that the sender's own text loses. But a text message can
  carry this uncertainty when a probe reads it out of the state.

## 2. Motivation (corrected)

arXiv 2606.20662 ("Confidence Laundering in Agent Systems") argues that agent handoffs lose uncertainty,
and that a latent carrier keeps it. Its abstract reports no experiment, but its full text has one
(Sec. 5, Figure 6c, App. C.1, Table 2). There, a latent hidden-state carrier from Qwen-1.5B on HotpotQA
gets AUROC 0.874, against 0.785 for a textual uncertainty description, 0.667 for a scalar score and
0.618 for the answer only. The text carriers in that test are weak. The paper does not say who writes
the textual description. It has no text arm that carries the same information as the latent arm.

This pilot adds that arm. T3 and T4 write the numbers of a probe on the sender state into the text.
If such a text arm is as good as the latent arm, then the carrier does not need to be latent.

## 3. What ran

- Task: a box has a hidden color from {red, blue, green, yellow}. Five witness reports are each right
  with probability 0.65. The exact Bayes posterior is known for each item.
- Sizes: 2,000 train items and 1,000 test items, from separate seeds. Train and test share no box ID.
- Sender: Qwen3-1.7B (snapshot 70d244cc). State: the residual stream after block 21, at the last token
  of the answer prompt. Receiver: SmolLM2-360M-Instruct (snapshot a10cc151). Both models were frozen
  and came from local files only.
- Readout: one linear layer (960 -> 4) on the final hidden state of the receiver, for every arm.
- Preconditions before each GPU job: no other Python job, swap 0 MB, charger 140 W. Each job ran
  under `caffeinate -i -w <pid>`. Only one GPU job ran at a time.
- Smoke test: 22:00 to 22:02 (64 train and 32 test items, other seeds). It ran without error.
- Pilot: 22:05:00 to 22:50:20 (45 min). Stage times: sender 949 s, probes 232 s, text arms 165 s,
  latent 1,237 s.
- The analysis stage ran again at 22:50 (CPU only, on the cached outputs) to add the Bayes-9 rows of
  the first PLAN.md addition. Of 161 earlier values, 0 changed.

## 4. Main table (test split, n = 1,000, readout view)

NLL in nats. "Gap" is NLL minus the Bayes NLL. ECE uses 15 bins.

| arm | message | NLL | gap | Brier | acc | ECE |
|---|---|---|---|---|---|---|
| T0 | answer only | 0.8634 | 0.5286 | 0.4338 | 0.736 | 0.0142 |
| T1 | answer + verbal digit | 0.8335 | 0.4987 | 0.4240 | 0.736 | 0.0099 |
| T2 | answer + token-probability digit | 0.7995 | 0.4647 | 0.3982 | 0.736 | 0.0385 |
| T3 | answer + probe scalar digit | 0.6463 | 0.3116 | 0.3279 | 0.739 | 0.0449 |
| T4 | answer + probe distribution (4 digits) | **0.4832** | 0.1484 | 0.2406 | 0.837 | 0.0234 |
| L | one soft vector from the block-21 state | 0.5223 | 0.1876 | 0.2645 | 0.812 | 0.0549 |
| L-shuf | L, states shuffled (train and test) | 1.4701 | 1.1354 | 0.7885 | 0.236 | 0.1243 |
| L-testshuf | trained L, test states shuffled only | 2.7689 | 2.4341 | 1.2090 | 0.275 | 0.5567 |
| Card | the receiver reads the whole card | 0.5703 | 0.2356 | 0.3059 | 0.786 | 0.0375 |

Reference rows (no receiver):

| reference | NLL | gap | Brier | acc | ECE |
|---|---|---|---|---|---|
| Bayes (exact posterior) | 0.3348 | 0 | 0.1779 | 0.864 | 0.0075 |
| Bayes-9-cal (exact P(c given the 1/9-rounded posterior)) | 0.3362 | 0.0014 | 0.1780 | 0.864 | 0.0075 |
| Bayes-9 (literal rounded posterior, renormalized) | 0.7182 | 0.3834 | 0.1800 | 0.864 | 0.0237 |
| Prior (uniform) | 1.3863 | 1.0515 | 0.7500 | 0.217 | 0.0330 |
| Sender softmax over the four color tokens | 0.9117 | 0.5769 | 0.4350 | 0.736 | 0.1733 |
| Probe on the block-21 state (4-way) | 0.4359 | 0.1011 | 0.2224 | 0.851 | 0.0310 |
| Probe on the block-28 state (4-way) | 0.4266 | 0.0918 | 0.2184 | 0.853 | 0.0211 |

## 5. Key differences (paired bootstrap, 10,000 draws over test items)

A positive value means that L has the lower NLL.

| difference in NLL | mean | 95% CI |
|---|---|---|
| T2 minus L (T2 is the best of T0, T1, T2) | +0.2772 | +0.2229 to +0.3312 |
| T0 minus L | +0.3411 | +0.2846 to +0.3971 |
| T1 minus L | +0.3111 | +0.2544 to +0.3659 |
| T3 minus L | +0.1240 | +0.0758 to +0.1709 |
| T4 minus L | **-0.0391** | **-0.0747 to -0.0046** |
| Card minus L | +0.0480 | -0.0054 to +0.1012 |
| L-shuf minus L | +0.9478 | +0.8865 to +1.0077 |
| T0 minus Card | +0.2931 | +0.2355 to +0.3504 |
| T3 minus T4 | +0.1631 | +0.1197 to +0.2061 |

## 6. Reading rule

"L beats X" means: NLL(X) minus NLL(L) is at least 0.02 nats, and the lower CI bound is above 0.

| step | value | result |
|---|---|---|
| Best of T0, T1, T2 | T2 (0.7995) | |
| L beats T2 | +0.277, CI +0.223 to +0.331 | yes |
| L beats T3 | +0.124, CI +0.076 to +0.171 | yes |
| L beats T4 | -0.039, CI -0.075 to -0.005 | no (T4 is better, CI below 0) |
| Sanity: L-shuf NLL at least ln 4 - 0.02 = 1.366 | 1.470 | pass |
| Sanity: Card NLL not more than T0 NLL | 0.570 against 0.863 | pass |

Verdict: case 2, valid pilot. L beats T0 to T2, but L is not better than T4. Thus the uncertainty is in
the sender states, but text can carry it once a probe reads it out.

Bayes-9 clause (first PLAN.md addition): this clause applies only if case 1 holds. Case 1 does not hold.
For the record, the cost of the 1/9 rounding for a calibrated reader is 0.0014 nats on the test split
(0.0042 nats over the full generating model). The literal Bayes-9 row has NLL 0.718, because 17 of
1,000 test items (1.7%) get the digit 0 on the true color.

## 7. Secondary views (fixed in the plan)

- Zero-shot receiver (no readout). The receiver does not use the messages in this format. With the
  T0, T4 and Card inputs, it puts " blue" first for all 1,000 test items. All zero-shot NLLs are above
  ln 4 (from 1.773 for T3 to 2.746 for Card). Thus all results in this report come from the readout view.
- Sender. Its answer accuracy is 0.736. The Bayes argmax accuracy is 0.864. The sender agrees with the
  Bayes argmax on 79.5% of the test items. A linear probe on its block-21 state gets accuracy 0.851.
  Thus the state holds the vote counts better than the sender's own answer does.
- Sender confidence. The verbal digit (T1) is 7 for 899 items and 9 for 101 items. Its AUC for "the
  answer is correct" is 0.566. The token-probability digit (T2) has AUC 0.704. The probe scalar (T3)
  has AUC 0.910 (0.904 after the rounding to a digit).
- Block 28. The 4-way probe on block 28 (NLL 0.427) is close to the probe on block 21 (NLL 0.436).
- L fit. The inner validation selected learning rate 3e-5 (the lowest value of the grid) and 5 epochs.
  At epoch 5 the train loss was 0.23, and the inner validation NLL was 0.518. After epoch 5 the
  validation NLL increased. The map (2048 x 960, about 2 million parameters) overfits the 1,600
  inner-train items.

## 8. Post-hoc views (EXPLORATORY, written after the data existed, not part of the rule)

Source: `posthoc.py`, key `post_hoc_exploratory` in `RESULTS.json`.

- KL from the exact posterior to each arm. The labels come from the Bayes posterior, so the mean KL is
  equal to the expected NLL gap, with less noise. Mean KL: T4 0.143, L 0.178, Card 0.228, T3 0.302,
  T2 0.420, T1 0.467, T0 0.491. T4 minus L: -0.035 (95% CI -0.053 to -0.018). Card minus L: +0.050
  (95% CI +0.018 to +0.086). The order is the same as in the NLL view.
- Route loss against the shared probe. L and T4 both start from the block-21 state. The 4-way probe
  on that state has NLL 0.436. The latent route adds 0.086 nats (95% CI 0.056 to 0.119). The text route
  T4 adds 0.047 nats (95% CI 0.030 to 0.066). In this pilot, the text route keeps more of the probe
  information than the trained latent route.
- Correlation of the top probability of each arm with the Bayes top probability: probe 0.754, T4 0.660,
  L 0.645, T3 0.598, Card 0.569, T2 0.325, T1 0.249, T0 0.036.

## 9. Exploratory reading

1. The sender state holds much of the correct uncertainty. A linear probe gets NLL 0.436 (Bayes 0.335).
2. The sender's own text confidence loses most of it. T1 and T2 are 0.46 to 0.50 nats above Bayes.
3. A latent message keeps much of it (L 0.522). A text message that the same kind of probe writes
   keeps more (T4 0.483).
4. Thus on this task, the result does not support the claim that uncertainty needs a latent carrier.
   The result is: the uncertainty is in the state, and a probe can write it as text.
5. The weak text carriers (T0 to T2) also lose to L in this pilot, by 0.28 to 0.34 nats. This agrees
   with the paper's result, and it shows that a latent win against weak text carriers does not prove
   that the carrier must be latent.

## 10. Limits

- One seed, one synthetic task, one sender and receiver pair. This is not confirmatory.
- The L map overfits, and the selected learning rate is at the lower edge of the grid. A map with less
  capacity (for example a low-rank map, or a stronger L2 weight) can make L better. Thus "T4 is better
  than L" can be a limit of this map fit, and not a limit of the latent route.
- The fit methods are different. The T4 probe and the text readouts use a convex L2 logistic fit with
  cross-validation. The L map and readout train through the receiver with AdamW and early stopping.
  The readout capacity is the same (one linear layer 960 -> 4).
- T3 and T4 carry numbers with 1/9 resolution. For a calibrated reader, this rounding costs only
  0.0014 nats here.
- The receiver is small (360M). It does not use the messages zero-shot. All results depend on the
  trained readout.
- The verbal confidence (T1) is almost constant (7 or 9). Thus T1 is a weak arm.
- The card tells the reader that each witness is right 65% of the time.
- The pilot has no arm with the sender's own full four-color distribution as text. T2 carries only
  the probability of the answer. The pilot also has no arm with sampled sender answers as text.

## 11. Next steps (not run; a new GPU run needs [researcher]'s word)

- Add a text arm with the sender's own four-color softmax, in the T4 format.
- Add a text arm with K sampled sender answers.
- Give L a map with less capacity (for example rank 16 to 64, or a ridge fit), and a second seed.
  This tests whether "T4 is better than L" stays when the map does not overfit.
- If the result stays, write a confirmatory plan with new seeds and a fixed map capacity.

## 12. Files

- `PLAN.md`: the plan (original text unchanged) and two dated additions (22:13 Bayes-9 rows; 22:46
  motivation correction). `PLAN.original_20261020.md`: a copy of the plan before the additions.
- `IMPLEMENTATION_NOTES.md`: choices that the plan does not fix, written before the data; section 8
  (grid change after the smoke test) and section 9 (Bayes-9 rows).
- `pilot.py`: all stages. `run_pilot.sh`: precondition checks and `caffeinate`. `posthoc.py`: section 8.
- `RESULTS.json`: every value with its arm, view and metric (also as a flat `records` list), the
  differences, the rule, the diagnostics and the post-hoc block.
- `run/`: items, sender outputs, probes, receiver states, L training curves (`latent.json`), per-item
  test probabilities of every arm and reference (`test_probs.npz`), `results.json`, `table.md`.
  `run/results_first_analysis.json` is the analysis before the Bayes-9 rows.
- `smoke/`: the smoke test outputs. Logs: `run.log`, `run_first.log`, `run_analysis2.log`,
  `run_wrapper.log`, `smoke.log`, `smoke_wrapper.log`.
