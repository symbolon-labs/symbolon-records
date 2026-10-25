# Preregistration DRAFT: probe text against a latent message for uncertainty transfer (closed-book TriviaQA)

**Status: DRAFT. Not frozen. Not approved by [researcher]. Nothing in this folder ran on a model or on real data.**

- Date: 2026-10-21. Prepared for [researcher] (Symbolon Labs) by a Claude Code session. Style: ASD-STE100.
- Design: `reports/uncertainty_confirmatory_design_20261021.md` (below: "the design").
- Part A copies the fixed text of the design word for word, with the line numbers of the design.
- Part B lists each implementation choice that the design leaves open. Each choice is "fixed here" or "open: [researcher] decides".
- Part C lists the facts that this session checked, and the points that are not checked.
- Code in this folder (EXPLORATORY code, DRAFT): `uconfirm.py` (the harness, modes test, smoke, dev and confirm), `run_uconfirm.sh` (the wrapper with the machine checks), `fetch_data.py` (the approval-gated download), `test_uconfirm.py` (36 unit tests). The code comments cite the IDs of Part B (P1 to P31, C1 to C4).
- Modes: Phase D is scratch mode. Phase C is confirmatory mode (`ember/AGENT.md`, "Research modes").

## How this draft becomes the preregistration

1. Phase D runs (after [researcher]'s yes for each run) and writes `dev/selection.json` and `dev/selection.md`.
2. The agent writes `PREREGISTRATION.md`: the text of this draft with Part B decided, plus `dev/selection.md` (the selected configuration, n_test, the code hash). [researcher] reads it and approves it.
3. `../../.venv/bin/python -I -B uconfirm.py confirm --stages freeze` runs the unit tests and writes `FREEZE.json` with the sha256 of `PREREGISTRATION.md`, the code, `items/items.json` and `dev/selection.json`.
4. Each confirm stage checks these hashes and stops on a difference. Phase C (each run with [researcher]'s yes) starts only after the freeze.

---

# Part A. Fixed by the design (copied word for word)

The text between the markers is a copy of the design. It does not change in this draft. Where Part B adds a detail, Part B names the design section.

*Design lines 127 to 133, word for word (start):*

## 4. Models

| Role | Model | Snapshot | Local cache | Use |
|---|---|---|---|---|
| Sender | Qwen/Qwen3-1.7B (Qwen family, 28 blocks, hidden 2048) | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | yes, 4.06 GB of weights | bf16 on MPS, frozen, chat template with thinking off |
| Receiver | HuggingFaceTB/SmolLM2-360M-Instruct (SmolLM family, Llama architecture, hidden 960) | `a10cc1512eabd3dde888204e902eca88bddb4951` | yes, 0.72 GB of weights | fp32 on MPS, frozen, no chat template |
| Fallback sender (only by the gate G-D1) | Qwen/Qwen3-4B-Instruct-2507 | `cdbee75f17c01a7cc42f958dc650907174af0554` | yes, 8.04 GB of weights | bf16 on MPS, frozen |

*Design lines 127 to 133 (end).*

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

*Design lines 313 to 320, word for word (start):*

## 10. Phase C: confirmatory run

- Mode: confirmatory (`ember/AGENT.md`, "Research modes"): preregistration, freeze, tests, an independent audit, seal and report.
- The configuration is the Phase D selection. No grid runs in Phase C.
- Seeds: 20261101, 20261102 and 20261103. Each seed sets the cross-fit folds, the readout cross-validation folds, the map initialization, the batch order and the shuffle permutations. The sender outputs (section 6) are the same for all seeds.
- Each latent arm trains on the full train split (4,000 items) with its selected epoch count.
- Per seed: T3, T4, L, L-shuf, L-testshuf and L-probe are new. T0, T0c, T1, T2, T5, T6, Source and Q-only use the same receiver states in all seeds; only their readout folds change.
- The test split is read one time, after the freeze, by the analysis script.

*Design lines 313 to 320 (end).*

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

*Design lines 457 to 461, word for word (start):*

### 15.3 The n rule (fixed now)

- n_test = 6,000.
- If the Phase D dev-test SD of the binary NLL(T4) - NLL(L) is above 0.60, use more test items. Then n_test is the number of validation items that pass the duplicate check (at most 7,993). At SD 0.60 and n = 7,993, the power at a true 0.02 is about 0.85.
- n_train = 4,000. This is twice the pilot train size, and it keeps the latent fits in the budget.

*Design lines 457 to 461 (end).*

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

---

# Part B. Implementation choices that the design leaves open

## B.1 Open choices ([researcher] decides)

Each open choice is one constant in `uconfirm.py`. The default is in the code now. If [researcher] chooses the alternative, the constant changes before the freeze, and the unit tests run again.

| ID | Subject | Default (in the code now) | Alternative | Why it can matter | Status |
|---|---|---|---|---|---|
| C1 | Duplicate questions (`DEDUPE_QUESTIONS`) | Skip a row whose normalized question came before it in the permutation: in shard 0 and in the validation file. "Fewer than 7,000 usable rows" (design 5) then counts distinct questions. | Keep duplicates. Only the test-against-train rule of design 5 applies. | Without it, one question can be in dev and train, or two times in test. | open: [researcher] decides |
| C2 | T6 count format (`T6_COUNT_TOP`) | x1 and x2 are the two most frequent non-empty sample answers (ties: first in draw order). The rest, with empty answers, is "other". A part with count 0 is not written. | x1 = a1 and x2 = c2, with their counts among the 10 samples. | Design 7.3 does not define x1 and x2. The format can change the T6 row, and T6 can be S*. | open: [researcher] decides |
| C3 | Readout input of L, L-shuf, L-testshuf and L-probe (`LATENT_READOUT_INPUT`) | The raw final state, as in the pilot and the follow-up. | (h - mu) / sd, with fixed mu and sd from the T0c final states of the train split. | Design 8.2 says "on the standardized final state", and also says that the joint fit "is the same as in the pilot". The pilot used the raw state. The two sentences do not agree on this point. The choice changes the optimization of every latent arm. | open: [researcher] decides |
| C4 | T5 values (`T5_RAW`) | v_j = P_seq(c_j), as the arm table of design 7.2 says. | v_j = P_seq(c_j) divided by the sum over the candidates of the item. | The role column says "the sender's distribution over its candidates". Raw P_seq values do not add up to 1. | open: [researcher] decides |

## B.2 Fixed here

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

---

# Part C. Checks of this session, and what is not checked

## C.1 Checked on 2026-10-21

- Hub API (metadata only, no download): at revision `0f7faf33a3908546c6fd5b73a660e0f8ff173c2f` (main), `rc.wikipedia/validation-00000-of-00001.parquet` has 234,778,394 bytes and `rc.wikipedia/train-00000-of-00007.parquet` has 240,036,659 bytes. The total is 474,815,053 bytes (474.8 MB). This agrees with the design. The LFS sha256 of each file is in `fetch_data.py`. The dataset is not gated. Its card tag is "license: unknown".
- Shard 0 has about 8,086 rows (61,888 rows x its share of the train bytes). The second shard (260,772,194 bytes) is thus probably not necessary.
- The local cache has the three models. SmolLM2-360M-Instruct `model.safetensors` resolves to 723,674,912 bytes (a sibling note says that the blob is a 79-byte stub; the file that the snapshot link points to is complete). Qwen3-4B-Instruct-2507 has 36 blocks and hidden size 2560. The SmolLM2 tokenizer has the single tokens " Yes" (9230) and " No" (2838).
- The 36 unit tests pass (`../../.venv/bin/python -I -B uconfirm.py test`, 37 s of CPU time). They include a run of every smoke stage and of the confirm stages (sender, ctext, seeds, analysis), with a fake tokenizer and tiny random Qwen3 and Llama models on the CPU. This checks the glue between the stages, the freeze check and the blind test labels. It does not check the real models.
- `./run_uconfirm.sh --check-only` reports the charger (8 W, below 60 W) and the missing data files, and starts no job.

## C.2 Not checked (the smoke run checks these)

- The real tokenizers and the Qwen3 chat template with thinking off.
- Batched decoding against one-at-a-time decoding on MPS in bf16 (the smoke run writes `smoke_sender_checks.json`).
- The time for each item, each receiver pass and each latent step. The smoke run writes them to `times.json` and a projection to `selection.json`.
- The analysis stage has about 3,000 small logistic fits (14 arms, 3 seeds, 2 labels, 36 fits each). The time estimate (20 to 40 min of CPU) is not measured. Design section 16 does not include this time.

## Amendment A1 (2026-10-21, about 15:55 local, after the Phase D data of R2a)

[researcher]'s words, in the parent session: [redacted: private chat]

- Reason: on the dev-test items, NLL(Source) = 0.4632 and NLL(T0) = 0.4429 (`dev/text.json`). The receiver (SmolLM2-360M-Instruct) does not use the 120-word evidence window to check the answer. The text arms with numbers show that the receiver reads text: the best T2, T3 and T4 formats have 0.4293, 0.3927 and 0.4090.
- Change: G-D3 is descriptive. It is computed and reported, but it does not stop Phase D. The validity check V2 (NLL(Source) <= NLL(T0)) is also descriptive in Phase C. It is reported as `V2_descriptive`, and it does not enter case 0. The Source arm stays in every run as a descriptive row.
- Cost: the claim loses the evidence-reading ceiling. No reading-rule statement may call Source a ceiling.
- This change came after the dev data of R2a. The preregistration must say so. No confirmatory item was touched.
- Code: `uconfirm.py` (the gate record of stage text, `require_gates` of stages latent and select, the validity dict). The R2a outputs do not change.
- Order: R2b runs after the R.06 T5 and C1 jobs. R2b probably needs more than 120 min; it runs in two parts with the resume of the harness.
