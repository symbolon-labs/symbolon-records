# CONFIRMATORY. Probe text against a latent message for uncertainty transfer (closed-book TriviaQA): report

Date: 2026-10-22. Written by the parent session after job R8 and the independent audit (AU).

- **Status:** CONFIRMATORY. The run followed `PREREGISTRATION.md`, frozen in `FREEZE.json` at 2026-10-21 22:48:47.
- **Audit:** `AUDIT.md`, an independent recompute in a separate session. **Verdict:** **audit passes.** **Case 3, "no state advantage over the sender's own text", S\* = T5.**
- **Seal:** `SEAL.json` holds the sha256 of every file of this folder at the time of the seal, this report included.
- **Pair:** sender Qwen3-1.7B, receiver SmolLM2-360M-Instruct, both frozen. Train 4,000 items, test 6,000 items, seeds 20261101, 20261102, 20261103.
- **Label:** y = 1 if the sender's answer is correct. Train base rate 0.265. Primary metric: test NLL (nats) of y, from a readout of the same capacity for each arm.

## 1. Result

**Case 3: "no state advantage over the sender's own text".** S* (the best arm that the sender writes from its own outputs) is T5.

The claim that this case allows (design section 13): "The latent message does not beat the sender's own outputs as text (S*)."

- L against S*: mean +0.0009 nats, 95% CI [-0.0076, +0.0093]. L does not beat S*.
- T4 against S* (the row that design section 13 adds for case 3): mean +0.0213, 95% CI [+0.0163, +0.0262], the same sign in all three seeds. T4 beats S*. Thus a probe on the sender state still reads more than the sender's outputs show.
- T4 against L: mean +0.0204, 95% CI [+0.0134, +0.0274], all seeds positive. T4 beats L by the rule of section 12.

What this report must not say (design section 13): that the sender state holds no uncertainty. The probe-direct row (0.3407) is the best row of the table, so the state holds the signal. Under amendment A1, no statement may call Source a ceiling.

Validity: V1, V3 and V4 pass ({"V1": true, "V3": true, "V4": true}). V2 is descriptive under A1 (`V2_descriptive` = False). c_round (the round-off cost of the T4 text format) = 0.0103.

## 2. Arms (test NLL, mean of 3 seeds)

| Arm | Message to the receiver | NLL | AUROC (y) | AUROC (y_SE) |
|---|---|---|---|---|
| T0 | answer only | 0.4323 | 0.791 | 0.822 |
| T0c | answer and the other candidate strings | 0.4132 | 0.803 | 0.862 |
| T1 | sender's verbal confidence digit | 0.4360 | 0.785 | 0.811 |
| T2 | sender's answer probability as text | 0.4085 | 0.820 | 0.907 |
| T5 | sender's candidate distribution as text (S*) | 0.4030 | 0.820 | 0.917 |
| T6 | 10 sampled answers as counts | 0.4058 | 0.817 | 0.938 |
| Source | evidence passage (descriptive under A1; not a ceiling) | 0.4459 | 0.775 | 0.788 |
| Q-only | question only | 0.4571 | 0.756 | 0.785 |
| T3 | probe scalar as text | 0.3840 | 0.844 | 0.834 |
| T4 | probe values for all candidates as text | 0.3817 | 0.841 | 0.875 |
| L | latent message (rank-64 map) | 0.4021 | 0.826 | 0.861 |
| L-shuf | latent, states shuffled in train and test | 0.4521 | 0.755 | 0.842 |
| L-testshuf | latent, test states shuffled | 0.5322 | 0.668 | 0.772 |
| L-probe | probe values as soft vectors (medium control) | 0.3617 | 0.871 | 0.875 |
| probe-direct | probe on the sender state, no receiver (reference) | 0.3407 | 0.885 | n/a |
| sender-direct | sender's own probability, no receiver (reference) | 0.4189 | 0.818 | n/a |
| base-rate | train base rate (reference) | 0.5312 | 0.500 | n/a |

## 3. Rows of the reading rule

"Mean" is NLL(B) - NLL(A); a positive mean favors A. "Beats" needs a mean of at least 0.02 nats, a CI lower bound above 0 and the same sign in each seed. Bootstrap: 10,000 paired draws, seed 20261021.

| A vs B | Mean | 95% CI | Per seed | A beats B |
|---|---|---|---|---|
| L vs T5 | +0.0009 | [-0.0076, +0.0093] | +0.009, +0.010, -0.017 | no |
| L vs T4 | -0.0204 | [-0.0274, -0.0134] | -0.012, -0.011, -0.038 | no |
| T4 vs L | +0.0204 | [+0.0134, +0.0274] | +0.012, +0.011, +0.038 | yes |
| L vs L-probe | -0.0404 | [-0.0473, -0.0335] | -0.040, -0.015, -0.066 | no |
| T4 vs T5 | +0.0213 | [+0.0163, +0.0262] | +0.022, +0.021, +0.021 | yes |
| T4 vs T3 | +0.0023 | [-0.0047, +0.0093] | +0.002, +0.003, +0.002 | no |
| T4 vs T0c | +0.0315 | [+0.0259, +0.0371] | +0.032, +0.031, +0.031 | yes |
| T5 vs T2 | +0.0055 | [-0.0011, +0.0121] | +0.005, +0.005, +0.005 | no |
| T6 vs T5 | -0.0028 | [-0.0079, +0.0025] | -0.003, -0.003, -0.003 | no |
| L vs T6 | +0.0037 | [-0.0045, +0.0120] | +0.012, +0.013, -0.014 | no |
| L vs T3 | -0.0180 | [-0.0263, -0.0098] | -0.010, -0.008, -0.036 | no |
| L vs L-testshuf | +0.1301 | [+0.1185, +0.1417] | +0.115, +0.157, +0.118 | yes |
| L-probe vs T4 | +0.0200 | [+0.0128, +0.0273] | +0.028, +0.004, +0.028 | yes |

Secondary row, no case: L-probe (the same probe values as T4, sent as soft vectors) beats T4. For this receiver, a soft vector carries the probe values better than the selected text format. Part of this gap is the round-off of the text numbers (c_round 0.0103).

## 4. Route loss (descriptive): NLL(arm) - NLL(probe direct)

| Arm | Mean | 95% CI |
|---|---|---|
| T3 | +0.0434 | [+0.0353, +0.0513] |
| T4 | +0.0410 | [+0.0321, +0.0497] |
| L | +0.0614 | [+0.0534, +0.0692] |
| L-probe | +0.0210 | [+0.0136, +0.0284] |

## 5. The paper view (descriptive; design 12.5)

arXiv 2606.20662, Table 2, uncertainty AUROC: latent 0.874, textual 0.785, scalar 0.667, answer only 0.618. Here, on the semantic-entropy label y_SE: T6 (the sender's own samples as text) 0.938, T2 0.907, L 0.861. T6 is better than L on y_SE, so the rule gives this sentence: "the paper's label is carried by its own samples as text". The tasks, models and readouts differ from the paper, so this is a comparison of patterns, not of values.

## 6. Deviations and events (all recorded in this folder)

1. **Amendment A1 (after the Phase D data of R2a, before the freeze).** G-D3 failed (Source 0.4632 against T0 0.4429 on dev-test). [researcher] made G-D3 and V2 descriptive. Records: `PREREG_DRAFT.md`, `PHASE_D_STOP.md`, `PREREGISTRATION.md` Part 6.
2. **Before the freeze:** one unit test was changed to the A1 rule, and the swap limit of `run_uconfirm.sh` changed from 2,048 MB to 3,584 MB (a machine check only). Record: `PHASE_D_STOP.md`.
3. **Chunk check (P33).** The parent session did the read-only check of the saved sender chunks with its own script, outside the frozen code, before R3b (14 of 14 chunks open) and R4b (13 of 13). R4c was not necessary.
4. **R5 reached its 120-min limit (an unplanned stop).** The seed-free text arms, the seed-1 probes and the T3 and T4 passes were complete; the first latent fit did not end. The R5 process held about 4 GB in swap. [researcher] approved a restart in a new process. The restart used the frozen code, skipped the saved parts and ended with rc 0 (the seed-1 latent fits took 55 min). Record: `PHASE_C_STOP_R5.md`.
5. **Power.** At 07:12 on 2026-10-22 the charger gave 8 W; the wrapper refused to start, and the restart began at 07:13 when the charger gave 140 W again. A power guard watched every Phase C job and made no pause.
6. R6 and R7 needed no restart. R8 is the only job that read the test labels.

## 7. Limits

- One task (closed-book TriviaQA, rc.wikipedia), one sender, one receiver, one latent map family (selected on development data), three seeds.
- The receiver does not use the evidence passage (G-D3), so the run has no evidence-reading upper bound.
- L trained for its selected 2 epochs on 4,000 items. A different map, more data or more epochs can give a different L. This result is about the tested configuration only.
- The label is the correctness of the sender's own answer; other uncertainty targets can behave differently.

## 8. Files

`PREREGISTRATION.md`, `FREEZE.json`, `PHASE_C_APPROVAL.md`, `PHASE_C_STOP_R5.md`, `PHASE_D_STOP.md`, `AUDIT.md`, `confirm/results.json`, `confirm/table.md`, `confirm/test_probs.npz`, the logs `confirm.log` and `confirm_wrapper.log`, and `SEAL.json`.
