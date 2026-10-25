# R.08 Uncertainty carrier: records

These are the records behind the entry [R.08 Uncertainty carrier](https://symbolonlabs.com/research/uncertainty-carrier/) on the Symbolon site. All paths below are relative to the root of this repository.

## What the record shows

The question: can a latent message from a frozen sender carry its uncertainty to a frozen receiver better than text from the same sender? The sender is Qwen3-1.7B, and the receiver is SmolLM2-360M-Instruct. `LICENSES.md` gives the pinned revisions. A linear readout on the final state of the receiver gives the probability of the label. Each arm has a readout of the same size.

The record has four studies. Two are confirmatory, and they lead the record. Two exploratory runs came before them.

**Confirmatory run (closed-book TriviaQA; preregistered, audited and sealed).** The label y is 1 if the answer of the sender is correct. The run has 4,000 train items, 6,000 test items and 3 seeds. The primary metric is the test NLL of y, in nats.

- **Case 3, "no state advantage over the sender's own text".** The latent message L does not beat S\*, the best arm that the sender writes from its own outputs. S\* is T5, the probability of the sender for each of its candidate answers, as words. L against T5 is +0.0009 nats (95% CI -0.0076 to +0.0093), and one of the 3 seeds is negative.
- **Probe text beats the text of the sender.** T4 sends the values of a probe on the same sender state, as words. T4 beats T5 by +0.0213 nats (95% CI +0.0163 to +0.0262), with a positive value in all 3 seeds. Thus a probe reads more from the sender state than the outputs of the sender show.
- **Probe text beats L in this run.** T4 beats L by +0.0204 nats (95% CI +0.0134 to +0.0274). The rule stops at case 3, so this row does not set the case.
- **Soft vectors for the probe values.** L-probe sends the same probe values as trained soft vectors. L-probe beats T4 by +0.0200 nats (95% CI +0.0128 to +0.0273). The mean is only 0.000013 above the 0.02 margin, and seed 20261102 gives only +0.0042.
- **Audit.** A separate agent session recomputed the labels, every arm, every rule row and the case from the saved per-item arrays. All means agree to 1e-6, and the case is the same.

**Format-gap test (closed-book TriviaQA; preregistered and audited, no seal).** This test asks if a finer text format of the probe values closes the gap to the soft vectors.

- **Case C, "the gap is not in the number format".** A dev gate selected the format WD2: the F3 word and the value to two decimals. On 7,829 fresh items, WD2 keeps a gap of +0.0387 nats (95% CI +0.0322 to +0.0456) to L-probe-R, the soft vectors with the readout rule of the text arms. The gain of WD2 over F3 is +0.0028 nats (95% CI -0.0002 to +0.0059), so the gain is not resolved.
- **Probe text against L on fresh items.** F3 (the T4 format) is better than L in each seed, by +0.0107 nats (95% CI +0.0046 to +0.0169). This is below the 0.02 margin, so F3 does not beat L on the fresh items. Thus the statement "T4 beats L" holds for the main run only.
- **Audit.** A separate agent, with its own code, recomputed every rule row from the saved per-item probabilities. All point values agree to 1.1e-16, and the case is the same. The audit took the checks V1 to V3 from the run and did not recompute them (`<G>/audit/AUDIT.md:95`).

**Exploratory runs (synthetic task).** A box has a hidden color, and five witness reports describe it. The label is the true color.

- **Pilot: case 2, "better estimator, not a latent carrier".** Probe text T4 beats L: test NLL 0.4832 against 0.5223. L beats the text that the sender writes from its own outputs.
- **Follow-up: case 1, "the T4 win stays".** With new seeds, T4 has 0.4827 and L has 0.5330. T4 minus L is -0.0503 nats (95% CI -0.0921 to -0.0107). A low-rank map (0.5625) also loses to T4.
- **Fit limit.** In both runs, the selected learning rate of L is the lowest value of its grid. Thus the fit limit of L is not removed.

Limits that the records state:

- The confirmatory run has one task, one sender, one receiver, one latent map family (selected on development data) and three seeds (`<C>/REPORT.md:93`).
- The receiver does not use the evidence passage (gate G-D3), so the run has no upper bound from the evidence (`<C>/REPORT.md:94`). Amendment A1 made G-D3 and the check V2 descriptive. A1 came after the development data of job R2a and before the freeze (`<C>/PREREGISTRATION.md:501-518`).
- L trained for its selected 2 epochs on 4,000 items. Another map, more data or more epochs can give another L (`<C>/REPORT.md:95`).
- The label is the correctness of the answer of the sender. Other uncertainty targets can give other results (`<C>/REPORT.md:96`).
- The format-gap test uses the train items of the main run. Only its test items are new. Its comparator L-probe-R and its six soft seeds came from a post hoc view of the sealed test arrays. The preregistration fixed them before any fresh item went through a model (`<G>/REPORT.md:154-157`).
- A result about how a receiver reads text applies to this receiver only (`<G>/REPORT.md:154`).
- Each exploratory run has one pair of data seeds, one synthetic task and one model pair. The exploratory runs are not confirmatory (`<P>/REPORT.md:160`, `<F>/REPORT.md:241`).

## Terms

`<R>` is `records/research`. The four study folders are:

- `<P>` is `<R>/latent_uncertainty_pilot_20261020` (exploratory pilot, 2026-10-20).
- `<F>` is `<R>/latent_uncertainty_followup_20261021` (exploratory follow-up, 2026-10-20 to 2026-10-21).
- `<C>` is `<R>/latent_uncertainty_confirm_20261021` (confirmatory run, 2026-10-21 to 2026-10-22, sealed).
- `<G>` is `<R>/format_gap_20261023` (confirmatory format-gap test, 2026-10-23 to 2026-10-24).

NLL values and NLL differences are rounded to four decimals. The cited key holds the full value.

### Confirmatory run

- **Items.** TriviaQA, subset `rc.wikipedia`, revision `0f7faf33a3908546c6fd5b73a660e0f8ff173c2f`. The 4,000 train items come from train shard 0. The 6,000 test items are the first 6,000 items of the validation pool. No test question is equal to a dev or train question (`<C>/PREREGISTRATION.md:117-128`, `:133`).
- **Candidates.** c1 is the greedy answer a1 of the sender. c2 and c3 are the two most frequent other answers in 10 samples (`<C>/PREREGISTRATION.md:150-151`).
- **Sender state.** The residual stream after block 21, at the last token of each candidate (`:159`).
- **Label.** y = 1 if c1 matches a gold alias (`:160`). The train base rate is 0.2648 (`<C>/confirm/results.json`, `base_rate_train`). The test mean of y is 0.219 (`<C>/AUDIT.md:128`).
- **Receiver input.** The text "Message: ", then the body of the arm, then "\nIs the proposed answer correct? Answer:" (`:178-180`).
- **Readout.** One linear layer (960 -> 1, with bias) on the final state of the receiver (`:242`). The text arms use an L2 logistic fit with 5-fold cross-validation. The latent arms train the map and the readout together through the frozen receiver (`:243-244`).
- **Metric.** The test NLL of y in nats, with probabilities clipped to [1e-6, 1 - 1e-6]. The NLL of an item is the mean over the 3 seeds (`:320-321`).
- **"A beats B".** All three conditions must be true (`:335-338`):
  1. The mean of NLL(B) - NLL(A) is at least 0.02 nats.
  2. The lower bound of its 95% CI is above 0.
  3. In each seed, the mean of NLL(B) - NLL(A) is above 0.

  The CI comes from a paired bootstrap of 10,000 draws over test items, with seed 20261021 (`:324`).
- **Order of the rule** (`:355-361`). If L does not beat S\*, the case is 3. Else, if L beats T4, the case is 1 or 1m. Else, the case is 2 (2a: T4 beats L; 2b: no winner). The records call this rule the "reading rule".

Arms. The test NLL is in `<C>/confirm/results.json`, key `rule.means`. The messages are in `<C>/PREREGISTRATION.md:188-203`.

| Arm | Message to the receiver | Test NLL |
|---|---|---|
| T0 | The question and the answer only | 0.4323 |
| T0c | T0 and the other candidate strings, with no numbers | 0.4132 |
| T1 | T0 and the verbal confidence digit of the sender | 0.4360 |
| T2 | T0 and the probability of the sender for its answer | 0.4085 |
| T5 | T0c and the probability of the sender for each candidate (S\*) | 0.4030 |
| T6 | T0 and 10 sampled answers, as counts | 0.4058 |
| Source | T0 and 120 words of evidence (descriptive under A1; not a ceiling) | 0.4459 |
| Q-only | The question only | 0.4571 |
| T3 | T0 and the probe value of the answer | 0.3840 |
| **T4** | T0c and the probe value of each candidate (**primary text arm**) | **0.3817** |
| **L** | T0c with one trained soft vector after each candidate, from a rank-64 map of the block-21 state (**primary latent arm**) | **0.4021** |
| L-shuf | L, with the states of each slot permuted across items, in train and in test | 0.4521 |
| L-testshuf | The trained L, with permuted test states | 0.5322 |
| L-probe | The L layout, with soft vectors made from the probe values (medium control) | 0.3617 |
| probe-direct | The probe value of c1, with no receiver (reference) | 0.3407 |
| sender-direct | The calibrated probability of the sender, with no receiver (reference) | 0.4189 |
| base-rate | The train base rate (reference) | 0.5312 |

T2, T3, T4 and T5 write each value as one of five words: "very unlikely" (below 0.2), "unlikely", "uncertain", "likely" and "very likely" (0.8 or more). This is format F3. Phase D selected the formats and the map of L on development items only (`<C>/PREREGISTRATION.md:61-73`, `:217`).

### Format-gap test

- **Question.** In the main run, L-probe beat T4 by 0.020 nats. Does a finer text format of the same probe values close this gap (`<G>/PREREGISTRATION.md:13`)?
- **Formats.** Six candidates against F3: D2 and D3 (the value to two or three decimals), LOs (the logit), ICD7 (seven words), WD2 (the F3 word and the D2 value) and ICD7D2 (the ICD7 word and the D2 value) (`<G>/PREREGISTRATION.md:63`).
- **Dev gate (Phase D').** On the 3,000 development items, F3 must give its Phase D value again (0.40904 ± 0.001). The selected format F\* must gain at least 0.005 nats over F3 (`:77-80`).
- **Fresh items.** 7,829 items from three pools, with no question that the main run used: V 1,991, S0 1,838 and S1 4,000 (`:95`).
- **C\* = L-probe-R.** The L-probe soft vectors, read with the readout rule of the text arms (`:114`, `:128`).
- **Seeds.** 3 for the text arms and L, and 6 for L-probe and L-probe-R (`:125`).
- **CI.** A two-level paired bootstrap over items and seeds, 10,000 draws, seed 20261023 (`:159`).
- **Labels of the rule.** "X closes against C": gap(X, C) is 0.005 or less, and the upper CI bound is below 0.010. "A gap remains": X does not close, and the lower CI bound is above 0. "F\* gains over F3": the lower CI bound of gain(F\*) is above 0 (`:179-181`).
- **Order of the rule** (`:186-195`). Validity first. Then case R (F3 closes), case A (F\* closes), case B (a gap remains, and F\* gains), case C (a gap remains, and F\* does not gain) and case D.

### Exploratory runs

- **Task.** Each item is a box with a hidden color from {red, blue, green, yellow}, with a uniform prior. Five witness reports describe it. Each report is right with probability 0.65. The exact Bayes posterior of each item is known (`<P>/PLAN.md:12-24`).
- **Sizes.** Each run has 2,000 train items and 1,000 test items, from separate seeds (`RESULTS.json`, `sizes` and `seeds.data`, in both folders).
- **Arms.** T0 to T6 are text messages. L is one trained soft vector. Card gives the full card to the receiver. T4 lists each color with round(9 x p), from a 4-way probe on the block-21 state (`<P>/PLAN.md:33-49`; `<F>/PLAN.md:59-105`).
- **Readout and metric.** One linear layer (960 -> 4) on the final state of the receiver. The metric is the test NLL of the true color (`<P>/PLAN.md:51-63`).
- **Rule.** "A beats B" needs a mean of at least 0.02 nats and a lower CI bound above 0 (`<P>/PLAN.md:65-76`; `<F>/PLAN.md:132-160`).
- **Plans, not preregistrations.** Each plan fixed its rule before its data. The two folders have no freeze file and no seal.

| Arm | Pilot NLL | Follow-up NLL |
|---|---|---|
| T2 (the best of T0 to T2) | 0.7995 | 0.7587 |
| T3 | 0.6463 | 0.6138 |
| **T4** | **0.4832** | **0.4827** |
| T5 (the softmax of the sender, as text) | not run | 0.7081 |
| **L** | **0.5223** | **0.5330** |
| L-lowrank (rank 16) | not run | 0.5625 |
| L-shuf | 1.4701 | 1.6321 |
| Card | 0.5703 | 0.5471 |
| Bayes (exact posterior, no receiver) | 0.3348 | 0.3444 |

Keys: `arms.<arm>.readout.nll` and `references.Bayes.nll` in `RESULTS.json` of each folder. The same values are in `run/table.md`.

### Relation to the motivation that the records name

The plans name arXiv 2606.20662 as the motivation. That paper argues that agent handoffs lose uncertainty, and that a latent carrier keeps it (`<P>/PLAN.md:8-10`, `<P>/REPORT.md:22-27`). The confirmatory run has a descriptive view on the label of that paper, the semantic entropy of 10 samples (`<C>/PREREGISTRATION.md:162-166`). On this label, the samples of the sender as text (T6) have AUROC 0.938, and L has 0.861 (`<C>/confirm/results.json`, `paper_view`). The tasks, models and readouts are different from the paper, so this is a comparison of patterns, not of values (`<C>/REPORT.md:80`). This record does not repeat the experiment of that paper.

## Site values and their sources

This table follows the evidence map of the site (`design/ENTRY_R08_EVIDENCE.md` in the site repository).

The site shows the first 12 hex digits of the SHA-256 of the **original** result file. For each file, `MANIFEST.json` gives `sha256_original` and `sha256_copy`. Most result files of this entry are scrubbed (a local path or the name of the researcher), so the copy hash differs from the site hash. `SCRUB_LOG.json` lists each change.

Only `<C>` has a seal. For `<P>`, `<F>` and `<G>`, `sha256_original` is the hash of the study file on 2026-10-25 (`MANIFEST.json`, `known_issues`).

A gate on the site is passed when the rule supports the claim in its name, and failed when the rule does not support it.

### Gates

| Gate on the site | Site hash | Result file (key) | Copy scrubbed | Rule source |
|---|---|---|---|---|
| Latent beats sender text (primary) (failed: case 3) | `77c6fdac3e38` | `<C>/confirm/results.json` (`rule.case`, `rule.s_star`, `rule.L_beats_S_star`) | yes (1 path) | `<C>/PREREGISTRATION.md:335-338`, `:355` |
| Probe text beats sender text (passed) | `77c6fdac3e38` | `<C>/confirm/results.json` (`rule.secondary`, row `a` = T4, `b` = T5) | yes (1 path) | `<C>/PREREGISTRATION.md:367`, `:397` |
| Validity V1, V3, V4 (passed) | `77c6fdac3e38` | `<C>/confirm/results.json` (`rule.validity`, `rule.means`, `validity_detail`) | yes (1 path) | `<C>/PREREGISTRATION.md:342-349`; A1 at `:385` |
| G-D3 Receiver reads the evidence (failed; descriptive under A1) | `0875ab9b2431` | `<C>/dev/gates.json` (`G-D3`) | no | `<C>/PREREGISTRATION.md:286` (design 9.2) and `:44-50`; A1 at `:501-518` |
| Format gap, dev gate (passed) | `ac50917964ef` | `<G>/dev/dev_selection.json` (`gates`, `F_star`, `decision`) | yes (1 name) | `<G>/PREREGISTRATION.md:77-80` |
| Finer format closes the gap (failed: case C) | `a597366bc484` | `<G>/confirm/results.json` (`rule.case`, `rule.primary`) | yes (1 name) | `<G>/PREREGISTRATION.md:176-195` |
| Probe text beats latent, fresh items (failed) | `a597366bc484` | `<G>/confirm/results.json` (`rule.secondary.beats`, row `a` = F3, `b` = L) | yes (1 name) | `<G>/PREREGISTRATION.md:201`, `:216` |
| Latent beats probe text, pilot (failed: case 2) | `d5b169c29aa3` | `<P>/RESULTS.json` (`reading_rule`) | yes (2 paths) | `<P>/PLAN.md:65-76` |
| Latent beats probe text, follow-up (failed: case 1) | `d6a92cf2813a` | `<F>/RESULTS.json` (`reading_rule`) | yes (2 paths) | `<F>/PLAN.md:132-152` |

In each scrubbed result file, the scrub changed one or two strings only: a local path (rules R5 and R12) or the name of the researcher (rule R14). No number changed.

### Controls

All values are test NLL in `<C>/confirm/results.json`, key `rule.means` (mean over 3 seeds). The methods are in `<C>/PREREGISTRATION.md:188-210`.

| Arm on the site | Site value | Key and method |
|---|---|---|
| Shuffled states | 0.4521 (L 0.4021) | `rule.means.L-shuf` = 0.452137; `rule.means.L` = 0.402062. One permutation for each slot, in train and in test (`:199`; P12 at `:469`) |
| Test-only shuffle | 0.5322 (L 0.4021) | `rule.means.L-testshuf` = 0.532180 (`:200`) |
| Strings only | 0.4132 (T4 0.3817) | `rule.means.T0c` = 0.413191; `rule.means.T4` = 0.381697 (`:191`) |
| Question only | 0.4571 | `rule.means.Q-only` = 0.457052 (`:203`) |
| Probe vectors | 0.3617 (L 0.4021, T4 0.3817) | `rule.means.L-probe` = 0.361684 (`:201`, `:254-258`) |
| Base rate | 0.5312 | `rule.means.base-rate` = 0.531237 (`:207`) |

### Other values

| Value on the site | Source |
|---|---|
| Pair Qwen3-1.7B to SmolLM2-360M-Instruct, both frozen | `<C>/PREREGISTRATION.md:105-106`; `<C>/REPORT.md:8` |
| Closed-book TriviaQA; label: the answer of the sender is correct | `<C>/REPORT.md:1`, `:9`; `<C>/PREREGISTRATION.md:160` |
| 4,000 train items, 6,000 test items, 3 seeds | `<C>/confirm/results.json`, `sizes` and `seeds`; `<C>/REPORT.md:8` |
| L: one soft vector after each candidate, from a rank-64 map of the block-21 state | `<C>/PREREGISTRATION.md:198`, `:70`; `<C>/confirm/results.json`, `selection.L` (`rank` 64) |
| The test of the claim of arXiv 2606.20662 | `<P>/PLAN.md:8-10`; `<C>/PREREGISTRATION.md:162` |
| T4 beats L by +0.0204 (95% CI +0.0134 to +0.0274) in the main run | `<C>/confirm/results.json`, `rule.T4_beats_L` |
| L-probe beats T4 by +0.0200, 0.000013 above the margin | `<C>/confirm/results.json`, `rule.secondary` (row `a` = L-probe, `b` = T4: `mean_b_minus_a` = 0.020013); `<C>/AUDIT.md:209` |
| An independent audit recomputed every value | `<C>/AUDIT.md:14-20`; `<G>/audit/AUDIT.md:3` |
| The main run is sealed | `<C>/SEAL.json` (`time` 2026-10-22 13:59:24, `files` 419, `root_sha256` `3b287fb41e94…`) |
| 7,829 fresh items | `<G>/confirm/results.json`, `sizes.fresh`; `<G>/REPORT.md:10` |
| WD2: a word and a value to two decimals | `<G>/PREREGISTRATION.md:63`; `<G>/dev/dev_selection.json`, `F_star` |
| Exploratory test NLL: T4 0.483 and L 0.522 (pilot); T4 0.483 and L 0.533 (follow-up) | `arms.T4.readout.nll` and `arms.L.readout.nll` in `<P>/RESULTS.json` and `<F>/RESULTS.json` |
| The selected learning rate of L is at the end of its grid | `<P>/RESULTS.json`, `diagnostics.latent_info.lr` = 3e-05, the lowest key of `diagnostics.latent_grid`; `<F>/RESULTS.json`, `reading_rule.selected_lr_at_grid_end` = true and `reading_rule.fit_limit_not_removed` = true |
| Opened 2026-10-20 | `<P>/smoke_wrapper.log:15` (start 2026-10-20 22:00:24); `<P>/run_wrapper.log:15` (start 2026-10-20 22:05:00) |
| Updated 2026-10-24 | `<G>/REPORT.md:3` (date 2026-10-24); `<G>/logs/A_wrapper.log:8` (end 2026-10-24 12:32:28) |

## Preregistration

Confirmatory run:

- `<C>/PREREGISTRATION.md` (Parts 1 to 9) and `<C>/FREEZE.json` (frozen at 2026-10-21 22:48:47, with 36 unit tests that pass). Part 4 copies the design text word for word. Part 6 gives amendment A1.
- `<C>/PREREG_DRAFT.md` (the draft, with A1 at its end) and `<C>/PHASE_D_STOP.md` (the stop at G-D3, A1 and the operations note).
- `<C>/dev/selection.json` and `<C>/dev/selection.md` (the Phase D selection, frozen).
- `<C>/PHASE_C_STOP_R5.md` (an unplanned stop of job R5 at its time limit, and the restart with the frozen code).

Format-gap test:

- `<G>/PREREGISTRATION.md`, `<G>/FREEZE_D.json` (2026-10-23 21:09:25, before the first model load) and `<G>/FREEZE.json` (2026-10-23 23:54:17, before the first fresh item went through a model).

Exploratory runs (plans, not preregistrations):

- `<P>/PLAN.md` with two dated additions, `<P>/PLAN.original_20261020.md` (the text before the additions) and `<P>/IMPLEMENTATION_NOTES.md`.
- `<F>/PLAN.md` with one dated addition, `<F>/PLAN.original_20261020.md` and `<F>/IMPLEMENTATION_NOTES.md`.

Notes:

- The two preregistrations copy the rule of a design file. The design files are not in this package. Each preregistration copies the design text that it uses word for word.
- `<C>/FREEZE.json` and `<C>/SEAL.json` give the hash of each original. The scrubbed and redacted copies do not agree with these hashes. In `MANIFEST.json`, `sha256_original` is the sealed hash.
- The frozen text of `<C>/PREREGISTRATION.md` starts with "PREREGISTRATION, not frozen". The session wrote this header before the freeze, and `<C>/FREEZE.json` froze the text with it. The frozen `<C>/uconfirm.py` also keeps its draft header (`<C>/AUDIT.md:173-175`). The analysis wrote the `status` of `<C>/confirm/results.json` before the audit, so it says that the audit is still to do (`<C>/AUDIT.md:174`). The same is true for `<G>/confirm/results.json`.

Ten files of this entry have a redaction: a direct quote of private chat (an approval) is replaced by `[redacted: private chat]`. In `<P>/PLAN.md` and `<P>/PLAN.original_20261020.md`, the quote fills two whole lines.

| File | Lines |
|---|---|
| `<P>/PLAN.md` | 5, 6 |
| `<P>/PLAN.original_20261020.md` | 5, 6 |
| `<C>/PREREGISTRATION.md` | 32, 443, 505 |
| `<C>/PREREG_DRAFT.md` | 414 |
| `<C>/PHASE_D_STOP.md` | 5, 29 |
| `<C>/PHASE_C_STOP_R5.md` | 26 |
| `<C>/AUDIT.md` | 6 |
| `<C>/data/FETCH_LOG.json` | 4 |
| `<G>/PREREGISTRATION.md` | 25 |
| `<G>/data/FETCH_LOG.json` | 4 |

`SCRUB_LOG.json` (`redactions`) gives each position. No number changed.

## Raw outputs

Confirmatory run:

- `<C>/confirm/results.json` and `<C>/confirm/table.md`: every arm, every rule row, the validity record and the paper view.
- `<C>/confirm/seed20261101/` to `seed20261103/`: the latent fit records (`latent*.json`) and the probe fits (`probes.json`) of each seed.
- `<C>/confirm/hashes.jsonl`: the weight hashes of each model job (check V3). `<C>/confirm/ctext.json` and `times.json`.
- `<C>/dev/`: the Phase D outputs (`gates.json`, `text.json`, `latent.json`, `latent_grid/`, `probes.json`, `selection.json`). `<C>/smoke/`: the smoke test outputs.
- The logs: `confirm.log`, `confirm_wrapper.log`, `dev.log`, `dev_wrapper.log`, `smoke.log` and `smoke_wrapper.log`.

Format-gap test:

- `<G>/confirm/results.json` and `<G>/confirm/table.md`: the rule rows, the rows for each pool, the "beats" rows and all arms.
- `<G>/confirm/seed20261101/` to `seed20261106/`: the fit records of each seed.
- `<G>/dev/dev_result.json` and `<G>/dev/dev_selection.json` (Phase D'). `<G>/fresh/fresh_manifest.json` (the pool counts).
- `<G>/logs/`: the job, wrapper, guard and chain logs.

Exploratory runs:

- `RESULTS.json` in `<P>` and `<F>`: every value, the differences, the rule, the diagnostics and the post hoc views.
- `run/` in each folder: `items.json` (synthetic items with the exact posterior), `messages_test.json` (every test message), `probes.json`, `latent.json`, `results.json` and `table.md`. `<F>/run/latent_grid.json` holds the follow-up grid.
- `smoke/` in each folder, and the logs `run.log`, `run_wrapper.log`, `smoke.log` and `smoke_wrapper.log`.

The per-item probabilities of each arm are binary files. See "What is missing and why".

## Audit report

- `<C>/AUDIT.md`: the independent audit of the confirmatory run (job AU, 2026-10-22). Its verdict is "audit passes". It recomputes the labels, every arm, every rule row, c_round and the case from `confirm/test_probs.npz` and the saved run files. It also fits the text readouts again from the saved receiver states (section 8). **The audit scripts are not in this package.** They were outside the study folder. `<C>/AUDIT.md:267-274` gives the SHA-256 of each script and output.
- `<G>/audit/AUDIT.md`, `<G>/audit/au_recompute.py` and `<G>/audit/au_recompute.json`: the independent recompute of the format-gap rule from `confirm/fresh_probs.npz`. Its verdict is "audit passes". The script does not import code from `ugap.py` or `uconfirm.py`.
- `<C>/test_uconfirm.py` (36 tests) and `<G>/test_ugap.py` (18 tests): the unit tests that each freeze ran.
- The exploratory runs have no audit file in this package. `posthoc.py` in each folder makes the post hoc views only.

The auditors are separate agent sessions of the same lab. No outside party ran them.

## Code

- `<C>/uconfirm.py` (all stages of the confirmatory run), `<C>/fetch_data.py` (the download and its checks) and `<C>/run_uconfirm.sh` (the machine checks and the time limit).
- `<G>/ugap.py` (all stages of the format-gap test). It imports `uconfirm.py` and `fetch_data.py` of `<C>` as libraries. `<G>/run_ugap.sh`, `<G>/guard.sh` (the power guard) and `<G>/chain.sh` (the job order).
- `<P>/pilot.py`, `<P>/run_pilot.sh` and `<P>/posthoc.py`.
- `<F>/followup.py` (a copy of `pilot.py` with the new arms, seeds and the latent grid), `<F>/run_followup.sh` and `<F>/posthoc.py`.

## What is missing and why

`MANIFEST.json` lists each file that is not included, with its SHA-256, its size and the reason. For this entry, 250 text files (3.6 MB) are included, and 544 files (3,723 MB) are not included:

| Reason | Files | Size | Status |
|---|---|---|---|
| Binary files (`*.npz`: sender and receiver states, probe and latent outputs, the per-item probabilities). 25 of them are larger than 20 MB. | 435 | 3,669 MB | Not included, available on request |
| Per-item files with TriviaQA questions, Wikipedia passages and model answers (for example `<C>/items/items.json`, `<C>/confirm/sender_*.json`, `<G>/fresh/fresh_items.json`, and five check files with example questions, such as `<C>/dev/parser_view.json` and `<C>/dev/messages_eval_examples.json`) | 103 | 54 MB | Not included, available on request. The frozen fetch and run code rebuilds them. |
| Approval records that quote private chat (`<C>/APPROVAL_REQUEST.md`, `<C>/PHASE_C_APPROVAL.md`, `<G>/APPROVALS.md`) | 3 | < 0.1 MB | Not included |
| Runtime markers of the jobs (lock files) | 3 | 0 MB | Not included |

Other gaps:

- **Per-item probabilities.** The audits start from `<C>/confirm/test_probs.npz` (SHA-256 `eeba84a3e152…`) and `<G>/confirm/fresh_probs.npz` (`9c4ca926725e…`). These are binary files. Without them, a reader cannot recompute the bootstrap rows from this package. The exploratory files `run/test_probs.npz` are binary too.
- **Frozen inputs.** `<C>/items/items.json` is one of the seven frozen files of `<C>/FREEZE.json`. It holds TriviaQA text, so it is not included. Its SHA-256 is in `<C>/FREEZE.json` and in `MANIFEST.json`.
- **Audit scripts of the main run.** See "Audit report".
- **Design files.** The design files that the two preregistrations copy are not in this package.
- **Seals.** Only `<C>` is sealed. `<P>`, `<F>` and `<G>` have no seal. For their files, `sha256_original` is the hash on 2026-10-25.
- **Model weights and the data set.** These are not included. `LICENSES.md` gives the pinned revisions of the models and of TriviaQA.

## Study copy

These folders are copies of four study folders of the Ember project (snapshot of 2026-10). They are not from the recovered snapshot of R.01 and R.06, so the recovery manifest has no lines for them. They are not a runnable environment.

- The scrub changed 47 files of this entry (286 replacements). Local paths became relative paths or start with `<local>` or `<tmp>` (rules R5, R9, R10 and R12). The name of the researcher became `[researcher]` in 100 places (rule R14). In the code identifiers `researcher_said_yes` and `--researcher-said-yes`, the word `researcher` replaces the name in 22 places in 8 files (rule R15).
- Ten files have a redaction (see "Preregistration").
- Some scripts and logs name the folders of other studies as start conditions of the job chain. Those studies are not part of this package.
- The scripts refer to the project layout, a virtual environment and model caches that are not here. Most scripts need the binary files or the per-item files that are not included.

Use the code to read the method. Do not expect it to run without changes.
