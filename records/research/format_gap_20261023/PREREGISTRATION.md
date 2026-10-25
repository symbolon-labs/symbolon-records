# PREREGISTRATION: a finer text format of the probe values against soft vectors (closed-book TriviaQA)

- **Status:** PREREGISTRATION. `FREEZE_D.json` freezes this file, the code, the tests and the scripts before the first model load of this test. After that freeze, this file does not change. Any later change is a new, dated file next to it.
- **Written:** 2026-10-23, by a Claude Code session for [researcher] (Symbolon Labs). Style: ASD-STE100.
- **Design:** `reports/format_gap_design_20261023.md` (below: "the design"). This file copies the rule of design section 10 word for word (Part 6) and fixes every choice that the design leaves open (IDs G1 to G34).
- **Parent run:** `ember/research/latent_uncertainty_confirm_20261021/` (below: "the sealed run"). CONFIRMATORY, audited, sealed. This test reads it and changes no file there. All code runs with `python -I -B`, so no `.pyc` file goes into it.
- **Modes:** Phase D' is scratch mode (development data; not evidence). Phase C' is confirmatory mode: preregistration, freeze, tests, a reading rule fixed before the data, and an independent audit.

---

# Part 1. Summary

- **Question** (design 0): in the sealed run, L-probe beat T4 by 0.020 nats. L-probe sends the probe values as trained soft vectors. T4 sends the same values as text in format F3. Does a finer text format of the same values close this gap?
- **Phase D'** (dev gate): one GPU job on the 3,000 dev items, receiver passes only. It selects one format F\* from six candidates and applies two gates. If a gate fails, the test stops, and no fresh item goes through a model.
- **Phase C'** (Plan M): only if both gates pass. 7,829 fresh items, text arms with 3 seeds, soft arms with 6 seeds. The primary row is gap(F\*, L-probe-R).
- **Rule** (design 10): "X closes against C" if the mean gap is 0.005 nats or less and the upper bound of its 95% CI is below 0.010. The CI is a two-level paired bootstrap over items and seeds.
- **Expected result** (design 0): the gate stops the test, or case B or C. Case A is improbable.

---

# Part 2. Approval and scope

[researcher]'s words (2026-10-23, at about 20:30 local, in the parent session), as the parent session gave them:

> [redacted: private chat]

What this approval covers (the parent session's statement of scope):

1. The dev gate D' (job Dp).
2. Only if the gate passes by the fixed rule of Part 4: the full Plan M, with its one download (TriviaQA `rc.wikipedia/train-00001-of-00007.parquet`, 260,772,194 bytes) and its GPU jobs (S1 to S4, Q1 to Q4) and the CPU jobs (I, F, A).

It covers nothing else. Plan S and Plan L are not approved. A stop rule or an error ends the work. Then the session reports and waits for [researcher].

---

# Part 3. Inputs and reuse

## 3.1 Read from the sealed run (no change; G24 checks the sha256 of each file against `SEAL.json`)

`uconfirm.py` and `fetch_data.py` (imported as libraries), `items/items.json`, `items/manifest.json`, `dev/selection.json`, `dev/probes.npz`, `dev/text.json`, `dev/text_probs_eval.npz`, `dev/recv/T4_F3_dev_train.npz`, `dev/recv/T4_F3_dev_test.npz`, `dev/sender_dev_train.*`, `dev/sender_dev_test.*`, `confirm/sender_train.*`, `confirm/hashes.jsonl`, `confirm/seed2026110{1,2,3}/probes.npz`, `.../latent.json`, `.../latent_L-probe.npz`, and the two parquet files of `data/trivia_qa/rc.wikipedia/`.

## 3.2 Reused without change (design 5)

- Sender Qwen3-1.7B (snapshot `70d244cc`, bf16 on MPS) and receiver SmolLM2-360M-Instruct (snapshot `a10cc151`, fp32 on MPS), both frozen, with the prompts, parser (P2), candidates (P3), block 21 and the label rule (P1) of the sealed run.
- Probe family: L2 logistic regression, L2 weight from {1e-5, ..., 10} by 5-fold CV over items, cross-fit (sealed P11).
- Readout rule of the text arms: L2 logistic on the receiver final state (960 -> 1), 5-fold CV (`uconfirm.cv_logistic`).
- L-probe: `ProbeMap`, learning rate 1e-3, 3 epochs. L: rank 64, learning rate 3e-4, 2 epochs. Joint fit as in the sealed run: AdamW (weight decay 0.01), batch 32, gradient clip 1.0, raw readout input (C3 default), no validation, the state after the last epoch (sealed P25).
- The train split: the 4,000 train items of the sealed run with their cached sender outputs (`confirm/sender_train.*`). No sender pass on train items.
- Probability clip 1e-6. Seeds 20261101, 20261102 and 20261103.

## 3.3 Pre-freeze check (CPU only, no model; done on 2026-10-23 before this file was frozen)

The probe cross-fit of seed 20261101 on the 4,000 train items repeated the sealed `confirm/seed20261101/probes.npz` (`fit`) with a maximum absolute difference of 0.0 (same NaN mask, L2 weight 0.1, 169 s). Thus V1 keeps its limit of 1e-6. The torch version is 2.13.0, the same as in the sealed run.

---

# Part 4. Formats and Phase D' (the gate rule)

## 4.1 Formats (design 3)

| ID | Choice |
|---|---|
| G1 | The text of v for the probe probability p (Python): **F3** = the sealed `fmt_value(p, 'F3')` ("very unlikely" p < 0.2, "unlikely" < 0.4, "uncertain" < 0.6, "likely" < 0.8, else "very likely"); **D2** = `f'{p:.2f}'`; **D3** = `f'{p:.3f}'`; **LOs** = `f'logit {x:.1f}'` with x = ln(q / (1 - q)) and q = p clipped to [1e-6, 1 - 1e-6]; **ICD7** = "almost no chance" (p < 0.05), "very unlikely" (< 0.20), "unlikely" (< 0.45), "roughly even chance" (< 0.55), "likely" (< 0.80), "very likely" (< 0.95), else "almost certain"; **WD2** = F3 word + ", " + D2; **ICD7D2** = ICD7 word + ", " + D2. The Python text stays as it is (for example "-0.0" can occur in LOs). |
| G2 | Message: the T4 layout of the sealed run, `join_tags(cand_segments(q, cands), [' (' + v + ')' for each present candidate])`. Only v changes. For F3, the message is equal to the sealed `text_messages('T4', 'F3', ...)` (unit test). |
| G3 | Probe values. Phase D': `dev/probes.npz` of the sealed run (`logistic_fit`: out of fold on dev-train; `logistic_eval`: dev-test), the values that selected F3 in Phase D. Phase C': for each seed, the probe cross-fit of that seed; train values out of fold; fresh-item values from the fit on all 4,000 train items. |
| G4 | Token counts: the SmolLM2-360M-Instruct tokenizer; each tag " (v)" alone, without special tokens; the mean over all present candidate slots. **Length rule:** a candidate is eligible for F\* only if no dev tag has more than 11 receiver tokens. |

## 4.2 Phase D' (design 7)

| ID | Choice |
|---|---|
| G5 | One GPU job (Dp, limit 45 min). For F3 and the six candidates (D2, D3, LOs, ICD7, WD2, ICD7D2), the receiver passes on dev-train (2,000) and dev-test (1,000): 21,000 passes, batch 64 in length order (`uconfirm.receiver_text`). Final states are stored in fp16, as in Phase D. Readout of each format: `cv_logistic(H_dev_train, y_dev_train, H_dev_test, 2026102113)` on the float64 values of the stored states. |
| G6 | **Pooled dev NLL** of a format: the mean over all 3,000 dev items of the item NLL. Dev-train items use the out-of-fold readout values. Dev-test items use the fit on all dev-train items. |

**Gate rule (fixed now; apply in this order):**

1. **G-F2 (reproduction):** the dev-test NLL of F3 is in [0.40804, 0.41004] (0.40904 ± 0.001, the Phase D value). If not: **stop**. The pipeline does not repeat.
2. **Selection:** the eligible candidates (G4). F\* is the eligible candidate with the lowest pooled dev NLL. Tie: all eligible candidates within 0.001 of the lowest value; of these, the format with the fewest mean tag tokens, then the lower pooled NLL, then the order D2, D3, LOs, ICD7, WD2, ICD7D2. F° is the best candidate of the other family by the same rule (digits: D2, D3, LOs; words: ICD7, WD2, ICD7D2).
3. **G-F1 (gain):** pooled dev NLL(F3) - pooled dev NLL(F\*) is 0.005 or more. If not: **stop**, outcome G. No fresh item is used.
4. **Decision:** "go" (Plan M runs) only if G-F2 and G-F1 pass. Else "stop".

Outputs: `dev/dev_result.json` (all values), `dev/dev_selection.json` (F\*, F°, the gates, the decision), `dev/dev_item_nll.npz`, `dev/recv/*.npz`, `dev/hashes.jsonl`. Descriptive only (no gate): item-bootstrap CIs of the pooled gain of each candidate (10,000 draws, seed 20261023); the difference of the F3 final states to the Phase D states; the receiver weight hash and parameter hash.

What outcome G means (design 12): "On the dev items, no tested format reads the probe values better than F3 by 0.005 nats or more. Development data, not evidence." The report must not say that text cannot close the gap. The open row stays as the sealed run left it.

---

# Part 5. Phase C' (Plan M; only after the decision "go")

## 5.1 Download and fresh items (design 4)

| ID | Choice |
|---|---|
| G31 | Job DL (stage `fetch`, network): `hf_hub_download` of `rc.wikipedia/train-00001-of-00007.parquet` from `mandarjoshi/trivia_qa` at revision `0f7faf33a3908546c6fd5b73a660e0f8ff173c2f` into `data/trivia_qa/` of **this** folder (a new folder). The stage checks the size (260,772,194 bytes) and the sha256 (`55cd82902467361c86f350f95c408e741bef13d23ad3ecbc07cb6eca2d9def59`, from the sealed `fetch_data.py`) and stops on a difference. `data/FETCH_LOG.json` records [researcher]'s words, the size and the sha256. The file is data only: `pyarrow` reads it in a `python -I` process. |
| G7 | Job I (stage `items`, CPU). Spent questions: the normalized questions of the dev-train, dev-test, train and confirmatory test items (the first 6,000 items of `test_pool`): 13,000 items. Pools in this order: **V** = `test_pool[6000:]` of the sealed `items.json`, in its order; **S0** = the usable shard-0 rows after the 7,000th pick of the seed-2026102110 permutation, in that permutation order; **S1** = the usable shard-1 rows in the order of a new permutation with seed 2026102320. A row is removed if its normalized question is spent or equal to an earlier fresh question. S1 stops after 4,000 kept rows. **Expected counts:** V 1,991, S0 1,838, S1 4,000 (total 7,829). If the counts differ, the stage writes `fresh/fresh_manifest_STOP.json` and stops. No fresh item then goes through a model. |
| G8 | A fresh record holds the pool, qid, question (single spaces), the alias set (P1) and the answer value. The build reads the question, the aliases and, for the usable rule (P5) only, the evidence pages. It writes no evidence window. |
| G9 | The S0 rows come from a copy of the frozen pick (`build_items` with C1). The stage checks that the copy picks the same 7,000 qids, in the same order, as the sealed dev and train items. If not, it stops. |

## 5.2 Freeze (job F)

| ID | Choice |
|---|---|
| G24 | `FREEZE_D.json` (mode `prefreeze`, before the first model load): the unit tests pass; the sha256 of `PREREGISTRATION.md`, `ugap.py`, `test_ugap.py`, `run_ugap.sh`, `guard.sh` and `chain.sh`; the seal check of the sealed inputs (3.1). Every stage of this test checks these hashes before it starts and stops on a difference. |
| G25 | `FREEZE.json` (job F, after job I): the unit tests and the seal check again; the hashes of G24, plus `fresh/fresh_items.json`, `fresh/fresh_manifest.json`, `dev/dev_selection.json`, `dev/dev_result.json` and the shard-1 file. Every Phase C' stage checks these hashes and stops on a difference. F\* is the value in the frozen `dev/dev_selection.json`. |

## 5.3 Arms and jobs (design 6, 8)

| Arm | Message | Readout | Seeds | Role |
|---|---|---|---|---|
| F3 | T4 layout, F3 values | L2 logistic, CV | 1 to 3 | reference text arm; replication row |
| F\* | T4 layout, the format of `dev_selection.json` | L2 logistic, CV | 1 to 3 | **primary text arm** |
| the other 5 candidates | T4 layout | L2 logistic, CV | 1 to 3 | descriptive |
| T0c | candidate strings, no values | L2 logistic, CV | passes one time; readout in seeds 1 to 3 | reference |
| L-probe-R | L-probe soft vectors | L2 logistic, CV, on the L-probe final states | 1 to 6 | **primary soft arm (C\*)** |
| L-probe | L-probe soft vectors | joint | 1 to 6 | secondary soft arm |
| L | rank-64 latent map of the block-21 states | joint | 1 to 3 | reference |
| probe-direct | none (probe value of c1) | none | 1 to 3 | reference |

Seeds 1 to 6 are 20261101 to 20261106. The seed of an arm sets the probe cross-fit folds, the readout folds, the map initialization and the batch order.

| ID | Choice |
|---|---|
| G10 | Sender (jobs S1 to S4): `uconfirm.sender_chunk` on the fresh items of one pool, chunks of 250 items. Sample stream seed [2026102112, k, c]: k = 4 (V), 5 (S0), 6 (S1); c = the chunk index inside its pool. Batch sizes: 16 (greedy, teacher-forced, verbal); 4 items x 10 samples. Block states in fp16. Each chunk file is written to a temporary name and then renamed, so a stop cannot leave a broken chunk file. After the last chunk, the stage merges the chunks in the order V, S0, S1 (`confirm/sender_fresh.*`). |
| G11 | Probes (each seed): `crossfit_probe('logistic', ...)` on the block-21 states of the 4,000 train items, with the train labels (P1) and the seed. |
| G12 | Text arms (F3, the six candidates, T0c): seeds 20261101 to 20261103. L-probe and L-probe-R: seeds 20261101 to 20261106. L: seeds 20261101 to 20261103. |
| G13 | Each text seed: receiver passes of F3 and the six candidates on the train items (4,000) and the fresh items (7,829), with the probe values of that seed. T0c: one set of passes (job Q1). The readouts are fit in job A, with the folds of each text seed. |
| G14 | L-probe (each of the 6 seeds) and L (seeds 1 to 3): `train_channel` on the 4,000 train items, then `predict_channel` on the fresh items and the train items. Saved: probabilities, final states (fp16) and the zero-shot logits. The receiver parameter hash before and after each fit is saved. |
| G15 | L-probe-R: `cv_logistic(h_L-probe_fit, y_train, h_L-probe_fresh, seed)` on the float64 values of the stored fp16 states. This is the readout rule of the text arms. |

Jobs (each through `run_ugap.sh`, limit 120 min of run time; `chain.sh M` runs S1 to A in this order):

| Job | Command (in this folder) | Estimate |
|---|---|---|
| DL | `../../.venv/bin/python -I -B ugap.py confirm --stages fetch --researcher-said-yes "<[researcher]'s words>"` | network |
| I | `../../.venv/bin/python -I -B ugap.py confirm --stages items` | under 5 min (CPU) |
| F | `../../.venv/bin/python -I -B ugap.py confirm --stages freeze` | under 5 min (CPU) |
| S1 | `./run_ugap.sh S1 120 confirm --stages sender --pool V` | 69 min |
| S2 | `./run_ugap.sh S2 120 confirm --stages sender --pool S0` | 64 min |
| S3 | `./run_ugap.sh S3 120 confirm --stages sender --pool S1 --chunks 0:8` | 70 min |
| S4 | `./run_ugap.sh S4 120 confirm --stages sender --pool S1 --chunks 8:16` | 70 min |
| Q1 | `./run_ugap.sh Q1 120 confirm --stages seeds --t0c --seeds 20261101` | about 68 min |
| Q2 | `./run_ugap.sh Q2 120 confirm --stages seeds --seeds 20261102` | about 65 min |
| Q3 | `./run_ugap.sh Q3 120 confirm --stages seeds --seeds 20261103` | about 65 min |
| Q4 | `./run_ugap.sh Q4 120 confirm --stages seeds --seeds 20261104,20261105,20261106` | about 66 min |
| A | `./run_ugap.sh A 120 confirm --stages analysis` | 0.3 to 0.7 h (CPU) |

---

# Part 6. Metrics and the reading rule

## 6.1 Metrics and CIs

*Design section 9, word for word (start):*

- Primary metric: the NLL (nats) of y (the sender's answer a1 is correct), with probabilities clipped to [1e-6, 1 - 1e-6].
- NLL_i(A): for item i, the mean over the seeds of arm A.
- gap(X, C) = the mean over fresh items of NLL_i(X) - NLL_i(C). A positive gap favors C.
- gain(F\*) = the mean over fresh items of NLL_i(F3) - NLL_i(F\*). A positive gain favors F\*.
- **CI: two-level paired bootstrap.** 10,000 draws, seed 20261023. Each draw resamples the fresh items with replacement (the same items for both arms). For each arm, it also resamples that arm's seeds with replacement, independently. The statistic is the mean over the drawn items of the mean over the drawn seeds. The 95% CI is the 2.5 and 97.5 percentiles.
- The report also gives the item-only CI (the method of the confirmatory run) and the per-seed values. The rule does not use them.
- Secondary metrics: AUROC, Brier score and ECE (15 bins), as in design 11. No y_SE label.

*Design section 9 (end).*

| ID | Choice |
|---|---|
| G16 | Two-level bootstrap code (`two_level_ci`): numpy `default_rng(20261023)`; chunks of 250 draws; in each chunk, first the item indices (250 x n), then the seed indices (250 x S) of each arm, with the arms in sorted order of their names. One call computes every row of the rule with the same draws. The point value of a row is the gap of the seed means. Percentiles: numpy default (linear). A unit test checks the code against a slow loop version. |
| G17 | Item-only CI (descriptive): `uconfirm.bootstrap_ci` on the seed-mean item gaps, 10,000 draws, seed 20261023. |
| G18 | Secondary metrics: computed for each seed, then averaged. ECE: 15 equal bins of p(y = 1). |
| G23 | Blind labels: only job A computes a fresh-item label (a code guard stops any other stage that tries). No stage before A prints a fresh-item metric. |

## 6.2 Reading rule

*Design section 10, word for word (start):*

### 10.1 Definitions

- C\* = L-probe-R (the primary soft arm).
- **"X closes against C":** gap(X, C) is 0.005 or less, AND the upper bound of its 95% CI is below 0.010.
- **"A gap remains between X and C":** X does not close against C, AND the lower bound of the 95% CI of gap(X, C) is above 0.
- **"F\* gains over F3":** the lower bound of the 95% CI of gain(F\*) is above 0.
- The numbers: 0.010 is c_round of F3 and half of the confirmatory gap. 0.005 is half of c_round and one quarter of the 0.02 margin of the confirmatory rule.

### 10.2 Steps (apply in this order)

0. **Validity (case 0, invalid)** if one check fails:
   - V1: for seeds 1 to 3, the train probe values are equal to `confirm/seed*/probes.npz` (`fit`) within 1e-6.
   - V2: the sha256 of each sender and receiver weight file is equal to the confirmatory value. The receiver parameter hash is `fe5f2146...` before and after each latent fit.
   - V3: all latent fits end with finite losses, and all their fresh-item probabilities are finite.
   - V4: no fresh question is equal to a spent question. The pool counts are equal to the frozen manifest.
1. **Replication.** If F3 closes against C\*: **case R, "the gap does not replicate on fresh items"**.
2. If F\* closes against C\*: **case A, "a finer text format closes the gap"**. A1 if F\* gains over F3. A2 if not.
3. Else, if a gap remains between F\* and C\*, and F\* gains over F3: **case B, "the format narrows the gap; a gap remains"**. The report gives the share of the F3 gap that F\* removes.
4. Else, if a gap remains between F\* and C\*: **case C, "the gap is not in the number format"**.
5. Else: **case D, "unresolved"**.

### 10.3 Secondary rows (no case)

- The same "closes" and "gap remains" labels for F\* and F3 against L-probe (joint readout). This is the open row as the confirmatory run defined it.
- gap and gain for F° and for each other candidate (descriptive; no claim from a candidate other than F\*).
- L against F3 and L against F\*, with the "beats" rule of the confirmatory run. That rule needs a mean of at least 0.02, a CI lower bound above 0 and the same sign in each seed. This row checks that text still beats the latent message on fresh items.
- L-probe against L-probe-R (the effect of the readout method).
- F3 and F\* against T0c (what the values add); probe-direct (route loss).
- c_round of each format on the fresh items (the recipe of 2.1).
- Each primary row for each pool (V, S0, S1), descriptive.
- Token counts of each format on the fresh items.

Nothing in this rule changes after the fresh-item data exists. Any later change is a new, dated section.

*Design section 10 (end).*

| ID | Choice |
|---|---|
| G19 | "Closes": mean <= 0.005 and CI upper bound < 0.010. "Remains": not "closes" and CI lower bound > 0. "Gains": CI lower bound of gap(F3, F\*) > 0. All CIs of the rule are two-level (G16). The full receiver parameter hash of V2 is `fe5f214635a19e4d8052924e712fb52e517279e8161d212e8d1e60d0ee9628f4`. V2 uses the weight-file records of every Phase C' model job (`confirm/hashes.jsonl`, at least two records, three files in each). |
| G20 | The share of the F3 gap that F\* removes = gain(F\*) / gap(F3, C\*), from the point values. |
| G21 | The "beats" rows (L against F3 and F\*, both directions) use the item-only CI (G17), as the confirmatory rule did. The per-seed values pair text seed k with L seed k (k = 1 to 3). |
| G22 | The rows for each pool use the two-level bootstrap on the items of that pool, seed 20261023. |
| G29 | c_round of each format (descriptive): the recipe of design 2.1 on the fresh items, for each text seed. The features of a format are the numbers that its text shows (D2, D3), the sigmoid of the shown logit (LOs), a one-hot of the word level (F3, ICD7), or both (WD2, ICD7D2). The report gives the raw value (it can be below 0) and the value clipped at 0. For F3, the clipped value is equal to the sealed `c_round_cost` (unit test). |
| G30 | Also descriptive: the L-probe train-side difference of seeds 1 to 3 against the sealed fits (design 15, risk 7). |

## 6.3 What each result means (design 12, word for word)

| Outcome | Condition | The report may say | The report must not say | Effect on the main claim |
|---|---|---|---|---|
| 0 | V1 to V4 fail | the failed check and its value | any claim | none |
| G (dev stop) | G-F1 fails; no fresh item used | "On the dev items, no tested format reads the probe values better than F3 by 0.005 nats or more. Development data, not evidence." | that text cannot close the gap | The open row stays as the confirmatory run left it. Use the words of B and C below as a limit of the claim. |
| R | F3 closes against L-probe-R | "On fresh items, F3 text is within 0.010 nats of the soft vectors (95% bound). The L-probe row of the confirmatory run does not replicate at its size." | that a finer format helped | The claim holds without a qualification, for this pair and task. |
| A1 | F\* closes; F\* gains over F3 | "A finer text format carries the probe values to this receiver as well as soft vectors do, within 0.010 nats (95% bound). The L-probe advantage of the confirmatory run was the text format." | that soft vectors never help; that this holds for other receivers, senders or tasks | The claim holds cleanly: neither the latent message (case 3) nor soft vectors for the same values carry more than well-built text. |
| A2 | F\* closes; the gain is not resolved | as A1, but "the gain over F3 is not resolved" | that the format change caused the closure | As A1 for the claim. The cause stays open. |
| B | a gap remains; F\* gains over F3 | "F\* removes a share s of the F3 gap. Soft vectors keep an advantage of g nats (CI) for the same values at this receiver." | that the sender state holds uncertainty that text cannot carry | One qualification: "Text carries the information. For a fixed set of values, this small receiver reads trained soft vectors better than any tested text format, by g nats." Case 3 does not change. |
| C | a gap remains; no gain | "No tested format reads the probe values better than F3. The soft-vector advantage is a medium effect at this receiver." | that the format cannot matter at other receivers | The same qualification as B, with "the gap is not in the number format". |
| D | none of the above | the values and CIs only | any closure or gap claim | No change. The open row stays open. Plan L is the next step if [researcher] wants it resolved. |

In every outcome, the comparison of L (the latent message) with text does not change.

---

# Part 7. Run procedure and stop rules

| ID | Choice |
|---|---|
| G26 | `run_ugap.sh` checks before each job: no other heavy job (no Python process whose command line or working folder is in the project tree `latent-communication-recovery`, and no process with a known job name); swap use known and at most 3,584 MB; AC power; charger 60 W or more; battery 55% or more; 10 GiB free disk; the venv Python. Then `caffeinate -i -w <pid>`, the job log by `tee -a` (`logs/<job>.log`), the models offline (`HF_HUB_OFFLINE=1`). |
| G27 | `guard.sh` (every 20 s): it pauses the job (SIGSTOP) if the Mac is on battery power, the battery is at 50% or less, or the charger gives less than 60 W. It resumes the job (SIGCONT) at AC power, 55% or more and 60 W or more. Only the time without a pause counts toward the limit; a step of more than 60 s (for example a system sleep) counts as 60 s. If the battery falls below 20%, it stops the job. If `pmset` gives no battery percentage, the guard keeps the last value. |
| G28 | Limits: Dp 45 min; each Phase C' job 120 min of run time. A job that reaches its limit, or that the battery rule stops, is incomplete. It does not start again without [researcher]'s word. |
| G32 | `chain.sh M` runs S1 to S4, Q1 to Q4 and A in this order. Before each job, it waits until the machine checks pass (a check every 60 s, at most 6 h). A job that does not end with exit code 0 ends the chain. Then the session reports and waits for [researcher]. |
| G33 | `chain.sh dev` waits for `ember/research/marl_replay_20261023/DONE`, then for the machine checks (this covers every Python process of the project, the MARL job included). Then it runs job Dp. |
| G34 | After the work (or a stop): `REPORT.md` with the result, each number tied to a file; then the empty file `DONE` (one line with the reason if the work stopped early). Another session waits for `DONE`. |

Order: prefreeze (G24), Dp, then only after the decision "go": DL, I, F, S1 to S4, Q1 to Q4, A. Then the audit (AU, Part 8).

---

# Part 8. Deviations from the design (known before the freeze)

1. **One preregistration for D' and C'.** The design (14.4, 16) sends a Phase C' preregistration to [researcher] after D', and asks [researcher]'s word for each Phase C' job. [researcher] gave the approval in advance, on the condition of the gate (Part 2). Thus this one file fixes both phases before D'. F\* comes from the fixed rule of Part 4 and enters the run through the frozen `dev/dev_selection.json` (G25). This file does not change after D'.
2. **Process lists of other wrappers.** The design (14.2) adds `ugap.py` to the process lists of other wrappers. This test does not change other folders. Instead, its own check looks at every Python process of the project tree, and the next test of the chain (split learning) waits for `DONE` of this folder.
3. **Download target.** The sealed `fetch_data.py` downloads into the sealed folder. Here the `fetch` stage downloads the same file (same repository, revision, path, size and sha256) into this folder (G31).
4. **Power guard.** [researcher]'s run rule: pause the job on battery power or at 50% battery or less (G27). The time limit counts run time only.
5. **Audit (AU).** The design gives the audit to an independent session. A session started by another session cannot start a new session. Thus the audit is a separate recompute from the saved per-item arrays (`confirm/fresh_probs.npz`) by an agent that does not use the statistics code of `ugap.py`. If that is not possible, the report marks the audit as open for [researcher].

---

# Part 9. Files

| File | What |
|---|---|
| `PREREGISTRATION.md` | this file (frozen by G24) |
| `ugap.py` | the harness (frozen) |
| `test_ugap.py` | the unit tests (frozen; synthetic data and tiny random models) |
| `run_ugap.sh`, `guard.sh`, `chain.sh` | the wrapper, the power guard and the job chain (frozen) |
| `FREEZE_D.json`, `FREEZE.json` | the freezes (G24, G25) |
| `dev/` | Phase D' outputs |
| `data/`, `fresh/` | the shard-1 file and the fresh items (Plan M only) |
| `confirm/` | Phase C' outputs: sender chunks, seeds, `results.json`, `table.md`, `fresh_probs.npz` |
| `logs/` | job, wrapper, guard and chain logs |
| `REPORT.md`, `DONE` | the report and the end marker (G34) |
