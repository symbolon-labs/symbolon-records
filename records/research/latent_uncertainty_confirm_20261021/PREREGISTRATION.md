# PREREGISTRATION, not frozen: probe text against a latent message for uncertainty transfer (closed-book TriviaQA)

**Status: PREREGISTRATION, not frozen.** [researcher] has not approved this text yet. `FREEZE.json` does not exist. No Phase C job ran, and no Phase C item went through a model.

- Written: 2026-10-21, after Phase D (the last Phase D job ended at 19:48 local). Prepared for [researcher] (Symbolon Labs) by a Claude Code session. Style: ASD-STE100.
- Design: `reports/uncertainty_confirmatory_design_20261021.md` (below: "the design"). Draft: `PREREG_DRAFT.md`. Phase D stop record and amendment A1: `PHASE_D_STOP.md`. Phase D selection: `dev/selection.json` and `dev/selection.md`.
- Mode: Phase C is confirmatory mode (`ember/AGENT.md`, "Research modes"): preregistration, freeze, tests, independent audit, seal and report.
- When [researcher] approves this text, job F (Part 7) runs the unit tests and writes `FREEZE.json` with the sha256 of this file, the code, `items/items.json` and `dev/selection.json`. After F, any change to these files stops every Phase C stage.

---

# Part 1. Summary

- **Question** (design section 0): on closed-book TriviaQA, can a latent message from a frozen sender carry uncertainty that the best text from the same sender state loses?
- **Primary comparison:** T4 (the probe values of the candidates, as text) against L (one trained soft vector in each candidate slot). Both read the same block-21 states at the same slots. The primary metric is the test NLL of y (the sender's answer is correct).
- **Phase D selection** (development values; Part 3):
  - sender Qwen3-1.7B (the main sender; G-D1 passed);
  - probe family: L2 logistic regression;
  - formats: T2, T3, T4 and T5 in words (F3); T6 as counts;
  - L: low-rank map with r = 64, learning rate 3e-4, 2 epochs;
  - L-probe: learning rate 1e-3, 3 epochs;
  - n_test = 6,000.
- **Amendment A1** (Part 6): after the dev data of R2a, [researcher] made G-D3 and the validity check V2 descriptive. Source is a descriptive row, not a ceiling. No confirmatory item was touched.
- **Reading rule:** design section 12, word for word (Part 4), with V2 descriptive (A1).
- **Phase C:** 3 seeds (20261101, 20261102, 20261103), 4,000 train items, 6,000 test items. About 9 to 11.5 h of laptop time in 9 or 10 jobs (Part 7). R3 and R4 need more than 120 min. Each runs in two parts that resume from saved chunks.
- **[researcher]'s decisions:** Part 8. The exact approval sentence is at its end.

---

# Part 2. Record of Phase D (scratch mode; development values, not evidence)

[researcher]'s approval (copied from `data/FETCH_LOG.json`): [redacted: private chat]

| Job | Time (local, 2026-10-21) | Code (`uconfirm.py` sha256) | Result |
|---|---|---|---|
| D1 download | 13:43 | (`fetch_data.py`) | 2 files; bytes and sha256 equal to the Hub file list |
| R0 items and smoke | 13:44 to 13:52 | `690305212c48` | items built (58 s); parser: 0 of the 20 view items failed; batched decoding equals one-at-a-time decoding on MPS (8 items; P_seq and state differences 0.0) |
| R1 sender and gates | 13:53 to 15:29 | `690305212c48` | G-D1 pass, G-D4 pass, parser failure rate 2.83% |
| R2a probes and text | 15:29 to 15:51 | `690305212c48` | G-D2 pass; **G-D3 FAIL** |
| Amendment A1 | about 15:55 | code changed to `cba5e677c6e9` | Part 6 |
| R2b part 1 | 16:04 to 18:04 | `cba5e677c6e9` | stopped at its 120-min limit (planned two-part run); 7 of 13 latent configurations finished |
| R2b part 2 | 18:05 to 19:48 | `cba5e677c6e9` | rc 0; the other 6 configurations and the selection |

| Gate (design 9.2) | Rule | Value | Status |
|---|---|---|---|
| G-D1 | sender dev accuracy from 0.20 to 0.80 | 0.2347 | pass |
| G-D2 | probe direct on c1: dev-test AUROC of 0.65 or more | 0.889 (logistic) | pass |
| G-D3 | dev-test NLL(Source) lower than NLL(T0) | 0.4632 against 0.4429 | **FAIL, descriptive under A1** |
| G-D4 | alias rule and token-F1 rule agree on 90% of dev items or more | 0.9937 | pass |
| Parser (design 6.1) | failures on 5% of dev items or fewer | 2.83% | pass: no parser change |

- Weight files: in the 2 smoke records and the 7 dev records of `hashes.jsonl`, each of the 3 weight files has one sha256. The receiver parameter hash before and after the latent grid is the same (`fe5f2146...`).
- No confirmatory item was touched. Phase D used only dev-train (2,000 items) and dev-test (1,000 items). The folder `confirm/` does not exist. The item build (R0) read the validation file one time to make the test pool: it removed the questions that are equal to dev or train questions, and it made the evidence windows. It computed no sender output, no label of a sender answer and no metric for a train or test item.

---

# Part 3. Phase D selection (development values; selection only)

This part fills in the parts of design sections 4, 7.3, 8, 9.1 and 15.3 that Phase D selects. Source: `dev/selection.json` (sha256 `838799334c06...`), made by `uconfirm.py` `cba5e677c6e9...` (the current file).

| Design section | Selection | Basis (dev values) |
|---|---|---|
| 4 and G-D1 | Sender Qwen/Qwen3-1.7B (`70d244cc`); block 21 (primary); block 28 (stored only) | G-D1 pass. The fallback sender is not used. |
| 8.1 probe family | L2 logistic regression | Dev-test NLL of the probe direct on c1: logistic 0.3392, MLP 0.3535 (AUROC 0.889 and 0.880). In Phase C, each probe fit selects its L2 weight again by 5-fold CV (design 8.1). |
| 7.3 format of T2 | F3 (words) | Dev-test NLL: F1 0.4371, F2 0.4392, F3 0.4293 |
| 7.3 format of T3 | F3 | F1 0.4128, F2 0.4179, F3 0.3927 |
| 7.3 format of T4 | F3 | F1 0.4268, F2 0.4299, F3 0.4090 |
| 7.3 format of T5 | F3 | F1 0.4395, F2 0.4410, F3 0.4376 |
| 7.3 format of T6 | counts | list 0.4332, counts 0.4316 |
| 8.3 map of L | low-rank map, r = 64 (2048 -> 64 -> 960), learning rate 3e-4, 2 epochs | Lowest inner-validation NLL (0.4188) of the 10 L configurations (table below) |
| 8.4 L-probe | learning rate 1e-3, 3 epochs | Inner-validation NLL 0.3826 (1e-4: 0.4042 at the epoch limit; 1e-2: 0.4037) |
| 9.1 grid-end rule | One step added: r = 64 at 1e-4 (0.4224). The selection did not change. | 3e-4 was the lower end of {3e-4, 1e-3, 3e-3}. After the step, it is not an end. The epoch rule did not apply (best epoch 2 is less than 15). |
| 15.3 n rule | n_test = 6,000 | Dev-test SD of the per-item binary NLL(T4) - NLL(L): 0.328 (limit 0.60) |
| Choices C1 to C4 (Part 5) | the defaults | [researcher], 2026-10-21, about 13:42 |

L configurations (inner split: 1,600 train items, 400 validation items; patience 3; at most 15 epochs):

| Configuration | Best inner-validation NLL | Best epoch | Epochs run |
|---|---|---|---|
| full, 1e-5 | 0.4302 | 15 | 15 |
| full, 3e-5 | 0.4236 | 5 | 8 |
| full, 1e-4 | 0.4230 | 4 | 7 |
| r = 16, 3e-4 | 0.4506 | 2 | 5 |
| r = 16, 1e-3 | 0.4474 | 1 | 4 |
| r = 16, 3e-3 | 0.4482 | 3 | 6 |
| r = 64, 1e-4 (grid-end step) | 0.4224 | 6 | 9 |
| **r = 64, 3e-4 (selected)** | **0.4188** | 2 | 5 |
| r = 64, 1e-3 | 0.4466 | 2 | 5 |
| r = 64, 3e-3 | 0.4241 | 1 | 4 |

Known limit: in Phase C, L trains for 2 epochs on 4,000 items (250 steps). The inner fit had 2 epochs on 1,600 items (100 steps). Design section 10 fixes "its selected epoch count", so this does not change. The follow-up report (section 7) saw the same effect.

---

# Part 4. Fixed text of the design (word for word), with the Phase D values

The text between each start marker and end marker is a copy of the design. It does not change. After each block, a note gives the Phase D values for that block and the effect of amendment A1, if any.

*Design lines 127 to 133, word for word (start):*

## 4. Models

| Role | Model | Snapshot | Local cache | Use |
|---|---|---|---|---|
| Sender | Qwen/Qwen3-1.7B (Qwen family, 28 blocks, hidden 2048) | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | yes, 4.06 GB of weights | bf16 on MPS, frozen, chat template with thinking off |
| Receiver | HuggingFaceTB/SmolLM2-360M-Instruct (SmolLM family, Llama architecture, hidden 960) | `a10cc1512eabd3dde888204e902eca88bddb4951` | yes, 0.72 GB of weights | fp32 on MPS, frozen, no chat template |
| Fallback sender (only by the gate G-D1) | Qwen/Qwen3-4B-Instruct-2507 | `cdbee75f17c01a7cc42f958dc650907174af0554` | yes, 8.04 GB of weights | bf16 on MPS, frozen |

*Design lines 127 to 133 (end).*

> **Phase D value:** G-D1 passed with the main sender (dev accuracy 0.2347). Phase C uses Qwen/Qwen3-1.7B. The fallback sender is not used.

*Design lines 145 to 159, word for word (start):*

## 5. Items and splits

Data: Hub dataset `mandarjoshi/trivia_qa`, config `rc.wikipedia`, revision `0f7faf33a3908546c6fd5b73a660e0f8ff173c2f` (pin this revision).

| Split | Source | Size | Use |
|---|---|---|---|
| Dev-train | train shard `00000-of-00007` | 2,000 | Phase D fits |
| Dev-test | train shard `00000-of-00007` | 1,000 | Phase D selection and gates |
| Train | train shard `00000-of-00007` | 4,000 | Phase C fits (probes, readouts, maps) |
| Test | validation file | 6,000 (or all, see the rule in section 15) | Phase C, one time, after the freeze |

- Order of the items: one permutation of shard 0 with seed 2026102110 gives dev-train, dev-test and train, in that order. One permutation of the validation file with seed 2026102111 gives the test items.
- Shard 0 has about 8,000 rows (61,888 rows x its share of the bytes). If it has fewer than 7,000 usable rows, also use shard `00001-of-00007` (260.8 MB).
- Remove each test item whose normalized question is equal to the normalized question of a dev or train item.
- The train and validation files come from the same TriviaQA question pool. Thus no distribution shift is expected between them.

*Design lines 145 to 159 (end).*

> **Phase D values:** shard 0 has 8,842 usable rows; the item build skipped 1 duplicate question (choice C1). Shard 1 was not necessary (D2 was not downloaded). The validation file has 7,993 usable rows; 2 had a question equal to a dev or train question and were removed; 0 duplicates. The test pool has 7,991 items. The Phase C test items are the first 6,000 items of the pool (n rule, design 15.3 below). Share of items with an alias match in the evidence: 99.2% (train), 99.1% (test pool).

*Design lines 163 to 192, word for word (start):*

## 6. Sender stage

All sender outputs are computed one time and cached. The seeds of section 10 do not change them.

### 6.1 Prompts (fixed now)

- Answer prompt (Qwen3 chat template, thinking off). User turn: "Answer the question with a short answer only (a name, a number or a few words).\nQuestion: {question}". The assistant turn starts empty.
- Greedy answer: at most 16 new tokens. Stop at a newline or `<|im_end|>`. Remove one final period. This is the sender's answer a1.
- Verbal confidence (T1): the pilot prompt. The first user turn, the answer a1 as the assistant turn, and the new user turn "How sure are you that your answer is right? Reply with one digit from 0 (not sure at all) to 9 (completely sure)." The digit is the argmax over the ten digit tokens at the first assistant token.
- Phase 0 checks the answer parser on 20 dev items. If the parser fails on more than 5% of dev items, a dated change to the parser goes into the preregistration before the freeze. The prompt words do not change.

### 6.2 Samples and candidates

- K = 10 samples from the answer prompt, temperature 1.0, top-p 1.0, at most 16 new tokens, one sample stream with seed 2026102112. The same parser applies.
- Candidates: c1 = a1. Then c2 and c3 are the two most frequent normalized sample answers that are not equal to a1 (ties: first in draw order). If fewer than two such answers exist, the item has fewer candidates (1 to 3).
- Equal answers are equal normalized strings. This design does not put answers with the same meaning into one group (section 18).

### 6.3 Teacher-forced pass for each candidate

For each candidate cj, one forward pass over the answer prompt plus the text of cj plus `<|im_end|>`:

- P_seq(cj): the product of the token probabilities of cj and `<|im_end|>`. This is the sender's own probability of the full answer string.
- The state: the residual stream after block 21 (fixed now), at the last token of cj. The state after block 28 is a secondary view only.
- Label: y_j = 1 if cj matches a gold alias. The primary label of the item is y = y_1.

### 6.4 Secondary label (the 2606.20662 view)

- SE: the entropy (nats) of the normalized answers among the 10 samples.
- y_SE = 1 if SE is above the median SE of the train split.
- T6 sends the same 10 samples. Thus T6 carries all the inputs of y_SE by construction. This is the point of the view: it shows how a text route carries the paper's own label.

*Design lines 163 to 192 (end).*

> **Phase D values:** parser check (Phase 0): 0 of the 20 view items failed. Parser failure rate on the 3,000 dev items: 2.83%. This is not more than 5%, so the parser does not change. Smoke check on MPS: batched greedy decoding equals one-at-a-time decoding on 8 items; the P_seq and state differences are 0.0.

*Design lines 196 to 243, word for word (start):*

## 7. Arms

### 7.1 Receiver input (fixed now)

- Prefix: "Message: " (as in the pilot, so that no soft vector is at position 0).
- Body: see the table.
- Suffix: "\nIs the proposed answer correct? Answer:".
- Final state: the output of the final norm at the last token (":").
- Secondary zero-shot view: the softmax over the receiver logits of " Yes" and " No" at that position.

### 7.2 Arm table

{v} is a value in the selected format of that arm (section 7.3). [s] is one soft input vector.

| Arm | Body of the message | Source of the uncertainty part | Role |
|---|---|---|---|
| T0 | "Question: {q}\nProposed answer: {a1}." | none | answer only |
| T0c | T0 + " Other answers: {c2}; {c3}." | candidate strings, no numbers | baseline for all candidate-list arms |
| T1 | "... Proposed answer: {a1} ({digit} out of 9)." | the sender's verbal digit | the sender's own stated confidence |
| T2 | "... Proposed answer: {a1} ({v})." with v = P_seq(c1) | the sender's answer probability | the sender's own token confidence |
| T3 | "... Proposed answer: {a1} ({v})." with v = probe value of c1 | probe scalar | probe scalar as text |
| **T4** | "... Proposed answer: {a1} ({v1}). Other answers: {c2} ({v2}); {c3} ({v3})." with vj = probe value of cj | probe values for all candidates | **primary text arm** (the probe distribution over the candidates as text) |
| T5 | the T4 body with vj = P_seq(cj) | the sender's distribution over its candidates | the sender's token distribution as text |
| T6 | T0 + " Ten sampled answers: {s1}; ...; {s10}." or the count format (section 7.3) | 10 samples | K sampled answers as text |
| **L** | "... Proposed answer: {a1} [s1]. Other answers: {c2} [s2]; {c3} [s3]." with sj = map(block-21 state of cj) | the full sender states | **primary latent arm** |
| L-shuf | the L body, with the states of each slot permuted across items, in train and in test | none (content-free vectors) | shuffled-latent control |
| L-testshuf | the trained L, with only the test states permuted | none | secondary view |
| L-probe | the L body, with sj = b + w * z_j, z_j = the standardized probe logit of cj | the probe values, as vectors | medium control |
| Source | T0 + "\nEvidence: {window}" | the gold evidence | ceiling: the receiver reads the source itself |
| Q-only | "Question: {q}" | none, and no answer | reference: the difficulty of the question alone |

References with no receiver:

- the base rate (the train mean of y);
- the probe direct (the probe value of c1, read as a probability);
- the sender direct (P_seq(c1), with a logistic calibration fit on train).

Source window: 120 words of the first Wikipedia evidence page, centered on the first match of a normalized gold alias. If no page has a match, use the first 120 words of the first page. The report gives the share of items with a match.

### 7.3 Formats of the numeric text arms (selected in Phase D)

- F1: "7 out of 9" (d = floor(9p + 0.5)), as in the pilot.
- F2: "73%" (floor(100p + 0.5)).
- F3: words, five levels: "very unlikely" (p < 0.2), "unlikely" (0.2 to 0.4), "uncertain" (0.4 to 0.6), "likely" (0.6 to 0.8), "very likely" (p of 0.8 or more).

T2, T3, T4 and T5 each get their own format. T6 gets one format from two: the raw list in draw order, or the counts ("{n1} of 10 say {x1}; {n2} of 10 say {x2}; {n3} of 10 other"). T1 has no choice, because the sender writes the digit itself.

The SYNC result (2610.19696, Sec. 4.5, in the lab's literature review) shows that the text format can change the result by a large amount. Thus the format selection gives the text arms a fair chance.

*Design lines 196 to 243 (end).*

> **Phase D values:** selected formats: T2 F3, T3 F3, T4 F3, T5 F3, T6 counts (Part 3). **Amendment A1:** the Source arm stays in every run as a descriptive row. It is not a ceiling (Part 6).

*Design lines 247 to 276, word for word (start):*

## 8. Probes, readouts and maps

### 8.1 Probes (T3, T4, L-probe)

- Input: the standardized block-21 state of a candidate (train statistics). Target: y_j of that candidate. One probe for all candidate slots.
- Family (selected in Phase D), one of two:
  - an L2 logistic regression, with the L2 weight from {1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10} by 5-fold cross-validation NLL;
  - an MLP with one hidden layer of 256 units and ReLU, with weight decay from {1e-4, 1e-3, 1e-2} and early stopping on an inner fold.
- The MLP option prevents the result "L wins because its route is nonlinear and the probe is linear".
- Cross-fit: 5 folds over items (not over candidates). The train messages use the out-of-fold probe values. The test messages use the probe fit on the full train split.

### 8.2 Readout (the same capacity for each arm)

- One linear layer (960 -> 1, with bias) on the standardized final state of the receiver, with a logistic output.
- Text arms, Source and Q-only: an L2 logistic fit. The L2 weight comes from the grid above by 5-fold cross-validation NLL on the train split. Then the readout is fit again on the full train split.
- L, L-shuf and L-probe: the map and the readout train together (AdamW, weight decay 0.01, batch 32, gradient norm clip 1.0, binary cross-entropy). The readout has the same size. The fit method is different, because the map must train through the receiver. This is the same as in the pilot.

### 8.3 Latent map (L)

- One map for all slots. Families (selected in Phase D):
  - full: Linear(2048 -> 960, with bias); learning rate from {1e-5, 3e-5, 1e-4};
  - low-rank r = 16 and r = 64: Linear(2048 -> r, no bias), then Linear(r -> 960, with bias); learning rate from {3e-4, 1e-3, 3e-3}.
- Initialization as in the follow-up. The output bias is the mean receiver token embedding. At the start, the output scale is equal to the scale of the pilot map.
- L-shuf uses the configuration and the epoch count of L.

### 8.4 Medium control (L-probe)

- sj = b + w * z_j, with b and w in R^960. b starts at the mean token embedding. w starts with std equal to the embedding std.
- Learning rate from {1e-4, 1e-3, 1e-2} (Phase D).
- z_j comes from the same cross-fit probe values as T4. Thus T4 and L-probe carry the same numbers, in two media.

*Design lines 247 to 276 (end).*

> **Phase D values:** probe family: L2 logistic regression. L: low-rank map with r = 64 (2048 -> 64 -> 960), learning rate 3e-4, 2 epochs. L-probe: learning rate 1e-3, 3 epochs. The probes and the text readouts select their L2 weight again in each fit (5-fold CV).

*Design lines 280 to 309, word for word (start):*

## 9. Phase D: development (scratch mode)

Items: dev-train (2,000) and dev-test (1,000). The confirmatory train and test items are not touched.

### 9.1 What Phase D selects

1. The map family, the learning rate and the epoch count of L. Inner validation: 20% of dev-train. Patience 3, at most 15 epochs. The lowest inner-validation NLL wins.
2. The learning rate and the epoch count of L-probe, by the same method.
3. The probe family (section 8.1), by the dev-test NLL of the probe direct on c1.
4. The format of T2, T3, T4, T5 and T6, by the dev-test NLL of each arm.

Grid-end rule:

- If a selected learning rate is at an end of its grid, add one step (a factor of 3) beyond that end. Then select again. Do this two times at most.
- If the selected epoch count is 15, raise the limit to 25 one time.

### 9.2 Gates (Phase D stops if one fails)

- G-D1, base rate: the sender's dev accuracy (mean of y) is from 0.20 to 0.80. If it fails, use the fallback sender (section 4) and run Phase D again. If the fallback also fails, stop and report "this task does not suit this sender".
- G-D2, state signal: the probe direct on c1 has a dev-test AUROC of at least 0.65. If it fails, stop and report "the sender state carries little correctness signal on this task". There is then nothing to transfer.
- G-D3, receiver reads text: the Source arm has a lower dev-test NLL than T0.
- G-D4, label check: compare the alias rule with a second rule (token F1 of at least 0.8 with any alias). The two rules must agree on at least 90% of dev items. If they do not, examine 50 items where they disagree. Then correct the normalization before the freeze.

### 9.3 Outputs of Phase D

- The selected configuration, written into `PREREGISTRATION.md` of the Phase C folder, with the hash of the code.
- The dev-test SD of the per-item binary NLL(T4) - NLL(L), for the n rule of section 15.
- The measured times for each item and each train step, for the run plan of section 16.

Phase D results are development results. Do not report them as evidence for or against the claim.

*Design lines 280 to 309 (end).*

> **Phase D values:** G-D1 pass (0.2347); G-D2 pass (0.889); G-D3 **FAIL, descriptive under A1** (Source 0.4632 against T0 0.4429); G-D4 pass (0.9937). The selection is in Part 3; the SD for the n rule is 0.328. **Amendment A1:** G-D3 is descriptive. It is computed and reported, but it does not stop Phase D.

*Design lines 313 to 320, word for word (start):*

## 10. Phase C: confirmatory run

- Mode: confirmatory (`ember/AGENT.md`, "Research modes"): preregistration, freeze, tests, an independent audit, seal and report.
- The configuration is the Phase D selection. No grid runs in Phase C.
- Seeds: 20261101, 20261102 and 20261103. Each seed sets the cross-fit folds, the readout cross-validation folds, the map initialization, the batch order and the shuffle permutations. The sender outputs (section 6) are the same for all seeds.
- Each latent arm trains on the full train split (4,000 items) with its selected epoch count.
- Per seed: T3, T4, L, L-shuf, L-testshuf and L-probe are new. T0, T0c, T1, T2, T5, T6, Source and Q-only use the same receiver states in all seeds; only their readout folds change.
- The test split is read one time, after the freeze, by the analysis script.

*Design lines 313 to 320 (end).*

> **Phase C values:** train 4,000 items; test 6,000 items; seeds 20261101, 20261102 and 20261103. The freeze (job F) comes before C1 (Part 5, P18).

*Design lines 324 to 331, word for word (start):*

## 11. Metrics

- Primary: the test NLL (nats) of y, with probabilities clipped to [1e-6, 1 - 1e-6].
- For each arm and each item, NLL_i is the mean over the 3 seeds of the NLL of that item.
- Secondary: AUROC of y (the metric of 2606.20662), Brier score, ECE (15 equal bins) and accuracy at threshold 0.5.
- Secondary label: the same metrics for y_SE. The readouts are fit again on y_SE. The maps do not train again.
- Uncertainty: a paired bootstrap over test items, 10,000 draws, seed 20261021. The 95% CI is the 2.5 and 97.5 percentiles of the mean difference.
- The task has no exact posterior. Thus the KL view of the pilot is not available.

*Design lines 324 to 331 (end).*

*Design lines 335 to 385, word for word (start):*

## 12. Reading rule (fixed now)

### 12.1 Definitions

- "A beats B": all three conditions are true.
  1. The mean over test items of NLL_i(B) - NLL_i(A) is at least 0.02 nats.
  2. The lower bound of its 95% CI is above 0.
  3. In each of the 3 seeds, the mean of NLL(B) - NLL(A) for that seed alone is above 0.
- S*: the sender-output text arm with the lowest test NLL (point estimate) among T0, T0c, T1, T2, T5 and T6.
- c_round: the round-off cost of the T4 format. Fit two logistic regressions on the train split: one on the formatted T4 values of all slots (with slot-present flags), one on the unformatted values. c_round = the test NLL of the first minus the test NLL of the second (0 if negative).

### 12.2 Step 1: validity (case 0)

The run is invalid if one of these checks fails:

- V1: NLL(L-shuf) is at least NLL(T0c) - 0.02. (L-shuf sees the same text as T0c. Its vectors must add no item-specific gain.)
- V2: NLL(Source) is not more than NLL(T0) (point estimates).
- V3: the weight hashes of the sender and the receiver are the same before and after the run.
- V4: all 3 seeds of all latent arms finish with finite losses.

### 12.3 Step 2: the case

Apply the steps in this order.

1. If L does not beat S*: **Case 3, "no state advantage over the sender's own text"**.
2. Else, if L beats T4:
   - if L also beats L-probe, AND the lower CI bound of NLL(T4) - NLL(L) is above c_round: **Case 1, "latent carrier"**;
   - else: **Case 1m, "medium effect, not a latent carrier"**.
3. Else (L beats S*, but L does not beat T4): **Case 2, "probe text carries it"**.
   - 2a: T4 beats L.
   - 2b: neither beats the other.

### 12.4 Step 3: secondary rows (no case)

The report gives "beats", the mean and the CI for these pairs:

- T4 against S*, T3 and T0c;
- T5 against T2, and T6 against T5;
- L against T6, T3 and L-testshuf;
- L-probe against T4.

### 12.5 Step 4: the paper view (descriptive)

- AUROC of y and of y_SE for each arm, next to the 2606.20662 Table 2 values.
- If T6 is equal to or better than L on y_SE, the report says: "the paper's label is carried by its own samples as text".

### 12.6 Step 5: route loss (descriptive)

As in the pilot post-hoc view: NLL(arm) - NLL(probe direct) for T3, T4, L and L-probe, with CIs.

Nothing in this rule changes after the Phase C test data exists. Any later change is a new, dated section.

*Design lines 335 to 385 (end).*

> **Amendment A1:** in 12.2, V2 is descriptive. The run is invalid only if V1, V3 or V4 fails. `results.json` reports V2 as `V2_descriptive`. Sections 12.1 and 12.3 to 12.6 do not change.

*Design lines 389 to 400, word for word (start):*

## 13. The claim that each result allows

| Case | What the report may say | What the report must not say |
|---|---|---|
| 1, latent carrier | "On closed-book TriviaQA, with Qwen3-1.7B to SmolLM2-360M, a latent message carries uncertainty that the best probe-written text from the same states does not carry. The gain is not the number format (L-probe) and not round-off (c_round)." This supports the 2606.20662 claim for one pair and one task. | That uncertainty needs a latent carrier in general, or for other pairs and tasks. |
| 1m, medium effect | "A soft vector carries the probe values to this receiver better than the selected text format. The sender state adds nothing beyond the probe values." | That the state holds uncertainty that text cannot carry. |
| 2a, probe text wins | "The pilot result replicates on a real task: once a probe reads the sender state, text carries the uncertainty better than the latent message." | That no latent route can win (only this map family was tested). |
| 2b, no winner | "Text with probe values carries about as much uncertainty as the latent message." | That the latent message is worse. |
| 3, no state advantage | "The latent message does not beat the sender's own outputs as text (S*)." Add the T4 against S* row: if T4 beats S*, a probe still reads more from the state than the sender's outputs show. | That the state holds no uncertainty (the probe direct row shows that). |
| 0, invalid | Only the failed check and its value. | Any claim. |

In all cases, the report gives the T2 and T6 rows next to the 2606.20662 values. They show how much of the "textual uncertainty" gap in that paper comes from weak text arms.

*Design lines 389 to 400 (end).*

> **Amendment A1:** no statement of the report may call Source a ceiling.

*Design lines 457 to 461, word for word (start):*

### 15.3 The n rule (fixed now)

- n_test = 6,000.
- If the Phase D dev-test SD of the binary NLL(T4) - NLL(L) is above 0.60, use more test items. Then n_test is the number of validation items that pass the duplicate check (at most 7,993). At SD 0.60 and n = 7,993, the power at a true 0.02 is about 0.85.
- n_train = 4,000. This is twice the pilot train size, and it keeps the latent fits in the budget.

*Design lines 457 to 461 (end).*

> **Phase D value:** the dev-test SD of the binary NLL(T4) - NLL(L) is 0.328. This is not above 0.60. Thus n_test = 6,000.

*Design lines 522 to 533, word for word (start):*

## 17. Run procedure and stop rules

1. [researcher] approves the download (474.8 MB). Then Phase 0 builds the items on the CPU.
2. Before each GPU run, [researcher] gives the word for that run. Then check:
   - no other GPU or CPU job of the lab is active (for example `pilot.py`, `followup.py`, `harness.py`, `smcheck.py`, `latent_capacity`, `latent_r06`, `t5.py`);
   - the swap use (`sysctl vm.swapusage`);
   - the charger gives at least 60 W (`pmset -g ac`).
3. Run each job under `caffeinate -i -w <pid>`, with its log written by `tee -a`.
4. Stop each run after 120 min. Report it as incomplete. Do not start it again without [researcher]'s word.
5. After Phase D: write `PREREGISTRATION.md` with the selected configuration, the n_test value and this reading rule. Freeze it with a hash before C3.
6. If a Phase D gate fails, stop. A Phase D failure is not a result about the claim.
7. After Phase C: an independent audit recomputes the primary row and the case from the saved per-item arrays.

*Design lines 522 to 533 (end).*

> **Phase C values:** step 2: the wrapper also checks the swap (limit 3,584 MB, P32), the free disk and the data files. Step 4: R3 and R4 need more than 120 min; each runs in parts that resume (Part 7, P33); a planned part that stops at 120 min is not a failure. Step 5: the freeze (job F) comes before C1.

---

# Part 5. Implementation choices

The code comments cite the IDs C1 to C4 and P1 to P31. Parts 5.1 and 5.2 come from `PREREG_DRAFT.md`, Part B. Part 5.3 (P32 to P35) is new after Phase D.

## 5.1 Choices that [researcher] decided (2026-10-21, about 13:42: [redacted: private chat])

[researcher] kept the four defaults. `dev/selection.json` records them: `dedupe_questions` true, `t6_count_top` samples, `latent_readout_input` raw, `t5_raw` true.

| ID | Subject | Default (in the code now) | Alternative | Why it can matter | Status |
|---|---|---|---|---|---|
| C1 | Duplicate questions (`DEDUPE_QUESTIONS`) | Skip a row whose normalized question came before it in the permutation: in shard 0 and in the validation file. "Fewer than 7,000 usable rows" (design 5) then counts distinct questions. | Keep duplicates. Only the test-against-train rule of design 5 applies. | Without it, one question can be in dev and train, or two times in test. | decided: default |
| C2 | T6 count format (`T6_COUNT_TOP`) | x1 and x2 are the two most frequent non-empty sample answers (ties: first in draw order). The rest, with empty answers, is "other". A part with count 0 is not written. | x1 = a1 and x2 = c2, with their counts among the 10 samples. | Design 7.3 does not define x1 and x2. The format can change the T6 row, and T6 can be S*. | decided: default |
| C3 | Readout input of L, L-shuf, L-testshuf and L-probe (`LATENT_READOUT_INPUT`) | The raw final state, as in the pilot and the follow-up. | (h - mu) / sd, with fixed mu and sd from the T0c final states of the train split. | Design 8.2 says "on the standardized final state", and also says that the joint fit "is the same as in the pilot". The pilot used the raw state. The two sentences do not agree on this point. The choice changes the optimization of every latent arm. | decided: default |
| C4 | T5 values (`T5_RAW`) | v_j = P_seq(c_j), as the arm table of design 7.2 says. | v_j = P_seq(c_j) divided by the sum over the candidates of the item. | The role column says "the sender's distribution over its candidates". Raw P_seq values do not add up to 1. | decided: default |

## 5.2 Fixed here (unchanged from the draft)

| ID | Subject | Choice |
|---|---|---|
| P1 | Alias set and label (design 3.1, 6.3) | The gold set is the normalized forms of `normalized_aliases` and `aliases`, without empty strings. y_j = 1 if the normalized c_j is in the set. The normalization is the TriviaQA one: "_" to space, lower case, punctuation to space, no articles, single spaces. Thus "U.S.A." becomes "u s", and an answer "A" becomes empty and never matches. The items stage reports the agreement of this normalization with the `normalized_aliases` field of the data. |
| P2 | Answer parser (design 6.1) | Cut at the first newline, strip spaces, remove one final period, strip again. A parse "fails" if the answer is empty, or if no newline and no end token come in 16 new tokens. The flag "prefixed" (the answer starts with "answer") is reported only. An empty a1 stays in the data: its label is 0, and its state is the state at the last prompt token. |
| P3 | Candidates (design 6.2) | An empty normalized sample answer is never a candidate. The text of c2 and c3 is the first sample (in draw order) with that normalized answer. |
| P4 | SE and y_SE (design 6.4) | Empty sample answers are one class in SE. y_SE = 1 if SE is strictly above the median SE of the train split. |
| P5 | Usable rows (design 5) | A row is usable if it has a question, at least one non-empty normalized alias and at least one non-empty evidence page. The question text has single spaces. |
| P6 | Slot layout of L, L-shuf, L-testshuf and L-probe (design 7.2) | The text is the T0c text. One slot follows the tokens of each candidate directly; the text after the slot starts with "." or ";". No space token is next to a slot. Each text piece is tokenized alone, without special tokens. Right padding; the final state is at the last real token. |
| P7 | Source window (design 7.2) | The window comes from the first page (in list order) that contains a gold alias. The "first match" is the earliest position of any alias in that page. The center is the middle word of the match. Near a page end, the window moves inside the page. With no match in any page: the first 120 words of the first page. |
| P8 | T6 list (design 7.3) | An empty sample answer is written as "(no answer)". |
| P9 | Decoding and samples (design 6.1, 6.2) | A batched decode loop with a KV cache and left padding. Greedy: argmax. Sampling: the full softmax at temperature 1.0 and top-p 1.0 (no top-k), by the inverse CDF on the CPU in float64, with a numpy generator. The sampling defaults in the Qwen3 model files (temperature 0.6, top-k 20, top-p 0.95) are not used. The design says "one sample stream with seed 2026102112". Here each chunk of 250 items has its own stream, with seed [2026102112, split index, chunk index]. Reason: a stopped run continues with the same samples. Each sampling batch has 4 items x 10 samples. |
| P10 | MLP probe (design 8.1) | AdamW, learning rate 1e-3, batch 256, at most 200 epochs, patience 10 on an inner fold of 20% of the items, float32 on the CPU. The weight decay comes from {1e-4, 1e-3, 1e-2} by 5-fold CV NLL. The final fit on all train items has its own inner fold. |
| P11 | Probe and readout fits (design 8.1, 8.2) | As in the follow-up code: standardization with the train statistics (for the probe: over all present candidate slots); the L2 weight by 5-fold CV NLL; the out-of-fold values at the selected weight are the train values. The folds are over items, so all candidates of one item are in one fold. LBFGS in float64 on the CPU. |
| P12 | L-shuf and L-testshuf (design 7.2) | For each slot, one permutation among the items that have that slot. Seeds [s, 1] (train) and [s, 2] (test), with s the run seed. L-shuf and L-testshuf use the same test permutation. |
| P13 | Fallback sender (design 4) | Blocks 27 and 36 of 36: the same depth fractions as 21 and 28 of 28. |
| P14 | Precision and storage | Sender in bf16 on MPS. P_seq and the verbal digit come from a float32 copy of the output weights. Block states are stored in fp16 (no loss for bf16 values); the stage stops if a value is above 65,000. Receiver final states are stored in fp16. |
| P15 | c_round (design 12.1) | Features for each slot: the number that the text shows (F1: d/9; F2: percent/100) or a one-hot of the five words (F3), times the slot-present flag, plus the three flags. Probability scale, not logit. The fit is the readout fit (P11). Train values are out of fold. c_round is computed for each seed; the rule uses the mean over the 3 seeds. |
| P16 | Metrics and references (design 7.2, 11, 12.5) | ECE: 15 equal bins of p(y = 1). Accuracy: p of 0.5 or more. Each secondary metric is computed for each seed, then averaged. Zero-shot view: the softmax over the " Yes" and " No" logits, averaged over seeds. Sender direct: a logistic fit (P11) on logit P_seq(c1). Base rate: the train mean of y. The sentence of design 12.5 uses the y_SE AUROC point estimates. |
| P17 | Grid-end rule (design 9.1) | The added learning rate is the next value of the 1-3-10 ladder (for example 1e-5 to 3e-6, 3e-3 to 1e-2), in the family and rank of the selected configuration. The learning-rate rounds come first, then the epoch rule. The 25-epoch fit replaces the 15-epoch fit of that configuration. A tie goes to the first configuration in grid order. |
| P18 | Freeze time (design 10, 17) | The freeze comes before C1. The design allows any time before C3. |
| P19 | V3 (design 12.2) | The sha256 of each weight file of the sender and the receiver at the start and the end of each model job (`hashes.jsonl`), and the hash of the receiver parameters in memory before and after each latent fit. V3 passes if each file has one hash in all records (at least two records) and the memory hash never changes. |
| P20 | Phase D seed | 2026102113, for the folds, the inits, the batch order and the inner split. |
| P21 | SD for the n rule (design 9.3, 15.3) | The selected L is the best-epoch state of its inner fit (no new fit on all of dev-train). It is evaluated on dev-test against T4 with its selected format and probe family. |
| P22 | Order in Phase D (design 9.1) | The probe family is selected first. T3, T4 and L-probe in Phase D then use that family. |
| P23 | Blind test labels (design 10) | Only the analysis stage computes test labels. A code guard stops any other stage that tries. No stage before the analysis prints a test metric. |
| P24 | Gates (design 9.2) | G-D1 and G-D4 use all 3,000 dev items (dev-train and dev-test). The first 50 disagreements of G-D4 go to `gd4_disagreements.jsonl`. The parser view has the first 20 dev-train items. Gates stop dev mode only, not smoke mode. A failed G-D1 needs [researcher]'s yes for the fallback runs. |
| P25 | Phase C latent fits (design 10) | No validation: the state after the last selected epoch. The map init and the batch order come from the run seed. Each latent arm saves its own file, so that a stopped seed job continues at the next arm. |
| P26 | y_SE readouts of the latent arms (design 11) | An L2 logistic fit (P11) on the final receiver states of the trained latent arm. For L-testshuf, the train states are the L train states. |
| P27 | Batch sizes | Sender 16 (greedy, teacher-forced, verbal); sampling 4 items x 10 samples; receiver text 64, in length order; latent train 32 (design 8.2); latent evaluation 64. |
| P28 | Run limit (design 17) | The wrapper refuses a time limit above 120 min and stops the job at its limit. A stopped job is incomplete. It does not start again without [researcher]'s word. |
| P29 | Block 28 (design 6.3) | Stored for a secondary view. No rule row uses it. |
| P30 | Text readouts in Phase C | Fit in the analysis stage with the folds of each seed (design 10: "only their readout folds change"). |
| P31 | Item build | One job on the CPU before the smoke test. It checks the size and the sha256 of each data file. It writes `items/items.json` (the four splits and the source windows) and `items/manifest.json` (counts, removed rows, match share). |

## 5.3 Added after Phase D (fixed here)

| ID | Subject | Choice |
|---|---|---|
| P32 | Swap check of the wrapper | The limit is 3,584 MB (it was 2,048 MB) since 2026-10-21, 16:05 (`PHASE_D_STOP.md`, operations note). Reason: after R2a, the swap held about 2,573 MB of old pages with no job active, and `memory_pressure` showed 93% free memory. This is a machine check, not a rule of the design. |
| P33 | Jobs longer than 120 min (R3, R4) | Each runs in parts with the same command. Part a stops at its 120-min limit; this stop is planned and is not a failure. The next part continues from the saved chunks of 250 items. The chunk that the stop interrupts is computed again (about 8 to 9 min of lost work). Before each next part, a read-only check opens each saved chunk file. If a file does not open (the stop can come during a file write), it goes to `confirm/sender_chunks_bad/`, and the next part computes that chunk again. |
| P34 | Code of amendment A1 | `results.json` reports V2 as `V2_descriptive`. The validity record has V1, V3 and V4 only. The G-D3 record of the text stage has `"descriptive": true` for runs after A1 (`dev/gates.json` comes from R2a, before A1, and does not have this field). |
| P35 | Code versions in Phase D | R0, R1 and R2a ran with `uconfirm.py` `690305212c48`. R2b ran with `cba5e677c6e9`. The difference is the A1 code (P34). It does not change the sender, probe or text outputs of R1 and R2a. |

---

# Part 6. Amendment A1 (full text)

**When:** 2026-10-21, about 15:55 local. A1 came **after** the dev data of R2a (the G-D3 values), before R2b, and before any Phase C job.

**[researcher]'s words** (in the parent session): [redacted: private chat] (Note: "T5" and "C1" in this sentence are jobs of the R.06 line, not arms or choices of this test.)

**Reason:** on the dev-test items, NLL(Source) = 0.4632 and NLL(T0) = 0.4429 (`dev/text.json`). The receiver (SmolLM2-360M-Instruct) does not use the 120-word evidence window to check the answer. Source is also below T0 on AUROC (0.749 against 0.778). The text arms with numbers show that the receiver reads text: the best T2, T3 and T4 formats have dev-test NLL 0.4293, 0.3927 and 0.4090.

**Change:**
1. G-D3 (design 9.2) is descriptive. It is computed and reported, but it does not stop Phase D. Its Phase D status is **FAIL, descriptive under A1**.
2. The validity check V2 (design 12.2: NLL(Source) <= NLL(T0)) is descriptive in Phase C. It is reported as `V2_descriptive`. It does not enter case 0. The run is invalid only if V1, V3 or V4 fails.
3. The Source arm stays in every run as a descriptive row. Its role in the arm table (design 7.2, "ceiling: the receiver reads the source itself") and its kill-control row (design 14, "A receiver and readout that cannot use text") become descriptive.

**What A1 does not change:** the arms, the primary comparison, the margin, the seeds, the metrics, step 1 checks V1, V3 and V4, and steps 2 to 5 of the reading rule (design 12.3 to 12.6).

**Cost:** the claim loses the evidence-reading ceiling. No statement of the report may call Source a ceiling.

**No confirmatory item was touched:** Part 2 gives the facts. Phase D used only the 3,000 dev items. No train or test item of Phase C went through the sender, the receiver or a fit before A1 or after it.

**Code** (`uconfirm.py`, now `cba5e677c6e9...`): the G-D3 record of the text stage; `require_gates` of the stages latent and select (G-D3 is not required); the validity record of `reading_rule` (V2 moved to `V2_descriptive`). The R2a outputs did not change. **Test:** one assertion of `test_uconfirm.py` expected the old V2 rule. It now expects the amended rule (a bad Source arm gives case 2b, and `V2_descriptive` is false). The parent session ran the 36 tests after A1; all pass (`PHASE_D_STOP.md`). This session did not run them again; job F runs them before the freeze.

---

# Part 7. Phase C run plan

## 7.1 Basis of the times

- Sender: 1.90 s per item (dev-train) and 1.94 s per item (dev-test) in R1 (`dev/times.json`). The smoke projection used 2.04 s per item: C1 2.27 h, C2 3.40 h (`smoke/selection.json`).
- Receiver text passes: the measured time of each arm and format in R2a (`dev/times.json`). The 8 seed-free arms on 10,000 items: 1,727 s (29 min); Source is the slowest arm (0.060 to 0.086 s per pass).
- Latent fits: about 133 s for each inner epoch of 50 steps in R2b, thus about 2.55 s for each train step. In Phase C, one epoch on 4,000 items has 125 steps (about 5.3 min).
- One seed: L (2 epochs) + L-shuf (2 epochs) + L-probe (3 epochs) = 7 epochs, about 37 min; 36,000 latent receiver passes, about 9 min; T3 and T4 on 10,000 items each, about 5 min; probe cross-fit, 1 to 3 min. Total about 52 min.
- Analysis: about 3,000 logistic fits on 4,000 rows (14 arms, 3 seeds, 2 labels, 36 fits each). In R2a, a readout fit on 2,000 rows took about 0.25 s. This time for 4,000 rows is not measured.

## 7.2 Jobs

All GPU and CPU jobs go through `run_uconfirm.sh` (machine checks: no other heavy job, swap, charger of 60 W or more, disk, data; `caffeinate -i -w`; `tee -a`; a limit of 120 min or less). A job of the R.06 line (for example `t5.py`) blocks every job here.

| Job | Design | What | Command (in this folder) | Time (estimate) |
|---|---|---|---|---|
| **F** | 17, step 5 | Freeze: the 36 unit tests, then `FREEZE.json` | `../../.venv/bin/python -I -B uconfirm.py confirm --stages freeze` | under 2 min (CPU) |
| **R3a** | C1, part a | Sender on the 4,000 train items | `./run_uconfirm.sh confirm 120 --stages sender --split fit` | 120 min; stops at the limit after about 13 to 14 of 16 chunks |
| **R3b** | C1, part b | The rest of the chunks; merge | the same command | 0.3 to 0.5 h |
| **R4a** | C2, part a | Sender on the 6,000 test items | `./run_uconfirm.sh confirm 120 --stages sender --split eval` | 120 min; stops at the limit after about 13 to 14 of 24 chunks |
| **R4b** | C2, part b | The rest of the chunks; merge | the same command | 1.4 to 1.7 h |
| **R4c** | C2, part c | Only if R4b also stops at its limit (the sender is slower than 2.3 s per item) | the same command | under 0.5 h |
| **R5** | C3 and C4, seed 1 | The 8 seed-free text arms on 10,000 items; then seed 20261101: probes, T3, T4, L, L-shuf, L-testshuf, L-probe | `./run_uconfirm.sh confirm 120 --stages ctext,seeds --seeds 20261101` | 1.2 to 1.9 h (central 1.4 h) |
| **R6** | C4, seed 2 | Seed 20261102 | `./run_uconfirm.sh confirm 120 --stages seeds --seeds 20261102` | 0.8 to 1.2 h (central 0.9 h) |
| **R7** | C4, seed 3 | Seed 20261103 | `./run_uconfirm.sh confirm 120 --stages seeds --seeds 20261103` | 0.8 to 1.2 h (central 0.9 h) |
| **R8** | analysis | Readouts, metrics, bootstrap, reading rule; the only reader of the test labels | `./run_uconfirm.sh confirm 120 --stages analysis` | 0.3 to 0.7 h (CPU; not measured) |
| **AU** | 17, step 7 | Independent audit in another session: recompute the primary row and the case from `confirm/test_probs.npz`; no model | read-only script | under 0.5 h (CPU) |

Total: about 9 to 11.5 h of laptop time in 9 or 10 jobs, plus the audit. The design estimate was 5 to 12 h (section 16). The analysis job (R8) is new, and the measured latent step (2.55 s) is longer than the design basis (0.47 s with shorter messages).

## 7.3 Rules for the jobs

1. Order: F, R3a, R3b, R4a, R4b (R4c), R5, R6, R7, R8, AU. Each job starts only after the previous job ended with its expected result.
2. R3a and R4a stop at their limits by plan (P33). This is the only planned stop. Any other stop is a stop: an error, a gate or check that fails, or a limit reached in R3b, R4c, R5, R6, R7 or R8. Then the agent reports, and the job does not start again without [researcher]'s word (design 17, step 4).
3. Before R3b, R4b and R4c: the read-only check of the saved chunk files (P33).
4. If R4b stops at its limit, R4c continues (approved in advance, Part 8). If R4c also stops, the agent reports and waits.
5. If a seed job stops at its limit, the next start continues at the next latent arm (P25), but only with [researcher]'s word.
6. No stage before R8 reads a test label or prints a test metric (P23). R8 reads the test split one time.
7. After R8: the independent audit (AU). Then the seal and the report.

---

# Part 8. Decisions for [researcher] (Phase C)

Mark each item yes or no. If [researcher] wants a change to this text, the change comes before F. After F, the text does not change.

| Item | What | Answer |
|---|---|---|
| **PR** | Approve this preregistration as written: Parts 1 to 7, with the Phase D selection (Part 3) and amendment A1 (Part 6) | yes / no |
| **F** | The freeze (CPU, under 2 min) | yes / no |
| **R3** | R3a and R3b (sender on the train items; about 2.4 h in two parts) | yes / no |
| **R4** | R4a and R4b, and R4c only if R4b stops at its limit (sender on the test items; about 3.5 h in two or three parts) | yes / no |
| **R5** | Seed-free text arms and seed 20261101 (1.2 to 1.9 h) | yes / no |
| **R6** | Seed 20261102 (0.8 to 1.2 h) | yes / no |
| **R7** | Seed 20261103 (0.8 to 1.2 h) | yes / no |
| **R8** | The analysis (CPU, 0.3 to 0.7 h) | yes / no |
| **AU** | The independent audit in another session (CPU, read-only) | yes / no |

The sentence:

> I approve PREREGISTRATION.md as written. Yes to F, R3a, R3b, R4a, R4b, R4c (only if R4b stops at its limit), R5, R6, R7, R8 and AU.

---

# Part 9. Files and hashes now (before the freeze)

| File | sha256 on 2026-10-21 | Frozen by F |
|---|---|---|
| `uconfirm.py` | `cba5e677c6e98905632c2d5c5483b9155e84e13a3bb8282ddf11b5056d076d4c` | yes |
| `test_uconfirm.py` | `cd50a036019f336ddbf1a8d8fcc10f1ecd0567a86b980b06d33c47b6545da14f` | yes |
| `run_uconfirm.sh` | `fa76e46bd0098e150fb4cfdc23246fbdfbf55c5968f8dce25ff2297c840f9614` | yes |
| `fetch_data.py` | `fede2ed9d6f625ad5271c2f9d7a7d4ed5226740e08142a0692d7f59c8f961291` | yes |
| `dev/selection.json` | `838799334c06991f146e47dbf63fa089e8284743639c355aeaf6acb8a6f5def3` | yes |
| `items/items.json` | `e8dfd9c4e790fe3854b995d548faad41d2033a3684760be101195343191895f9` | yes |
| `PREREGISTRATION.md` | this file; F records its hash | yes |
| Data | validation `59dcbdde...`, train shard 0 `d4f952bc...` (`items/manifest.json`) | checked by the item build |

Other records: `PREREG_DRAFT.md` (the draft, with A1 at its end), `PHASE_D_STOP.md`, `APPROVAL_REQUEST.md` (Phase 0 and D approvals), `dev/` (Phase D outputs), `smoke/` (Phase 0 outputs), `dev.log`, `dev_wrapper.log`, `smoke.log`, `smoke_wrapper.log`.
