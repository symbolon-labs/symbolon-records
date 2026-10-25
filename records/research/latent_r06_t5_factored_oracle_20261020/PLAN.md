# EXPLORATORY. R.06 test T5 (factored oracle at the frozen interface) and control C1: plan

Written 2026-10-20, before any T5 or C1 data exists. Parent plan: `../../../reports/r06_localization_plan_20261020.md`, section 6.

**Authorization.** [researcher], in this session, 2026-10-20: [redacted: private chat] This plan covers T5 (with controls C2 to C4 and the descriptive novel fit, which are part of the T5 design) and C1. It does not cover T5b, T6 or T7.

**Status of all outputs:** EXPLORATORY. These are post hoc diagnostics. The fold IDs were in the training of the ancestral d0, and the eight novel IDs are a burned bank.

## 1. T5 as written in the parent plan (copied word for word)

### T5. Factored oracle at the frozen interface (H4 against H5; new model run, decisive)

- **Question.** Does a code built from one vector for each letter and one vector for each digit, fit through the frozen output path and the frozen receiver on training IDs only, bind held-out IDs?
- **Design.**
  - Code: v(L, D) = c0 + a_L + b_D in the 64-d selected-value space. This is 1,600 values (64 + 16 x 64 + 8 x 64). There is no ID-specific base, unlike f61.
  - Insertion: the same seam as the constrained oracle. The code replaces the selected value at both query sites. The output map, gate, norm scale, centers and receiver stay frozen.
  - Start: c0, a and b from an additive least-squares fit of the d0 codes of the fit IDs.
  - Objective and budget: the oracle objective (both-token, full-vocabulary cross-entropy), 200 Adam steps, the SVD-conditioned coordinates and learning rate of `<L>/latent_binding_constrained_code_extended_20260911/coordinates.pt`. Nothing is tuned.
  - Data: the 224 training rows (56 IDs x 4 colors) and the four fixed folds of the fold record (fit 41/42/42/43 IDs). Each ID is held out once. All letters and digits occur in each fit set.
  - Score: joint and constrained-sequential accuracy over 128 candidates on held-out rows. Also score the fit rows.
- **Controls.**
  - C1, shifted-label factored fit: the same fit with labels moved one place in sorted ID order, scored against the true targets.
  - C2, wrong parts: a held-out code built from the parts of another held-out ID. It must follow the inserted ID.
  - C3, random code with the norm of each fitted code.
  - C4, zero code and receiver only.
  - Descriptive only: a fit on all 56 training IDs, scored on the eight novel IDs.
- **Reading rule.** H is the held-out joint count (of 224). F is the fit-row joint count, summed over folds (of 672 fit rows; 90% is 605). N is the C1 held-out joint count.
  - If F >= 90% of fit rows, H >= 168 (75%) and N <= 22 (10%): a factored code works at this interface. **H5 is rejected.** The failure is upstream (H3 or H4). Go to T6.
  - If F >= 90% of fit rows and H <= 56 (25%): the interface accepts factored codes for the fit IDs, but they do not transfer. **H5 is supported.**
  - If F < 90% of fit rows: a factored code cannot even fit the training IDs here. **H5 is supported in its strong form.**
  - Otherwise: inconclusive. Report the counts. Do not move a threshold.
  - The 75% bar is the joint bar of E3.
- **Conditional follow-up (T5b).** If H5 is supported, repeat T5 with the output map also trainable. If T5b passes, the frozen d0 output path, not the receiver, is the barrier. This matches the E3 design, which trains its reader.
- **Cost.** One arm (four folds) is about 3.7 hours on the CPU. This value scales from the fold job (13,147.9 s for four folds of 200 steps with receiver backward). C1 is a second arm of the same size. A one-fold smoke test must measure the real time before the freeze.
- **Limit.** One seed and one learning rate, as in every R.06 test. The fold IDs were in the training of the ancestral d0. All held-out IDs are inside a block, so T5 tests in-block transfer only. For a cross-block ID, the training IDs do not fix the offset between the letter vector and the digit vector.

## 2. Fixed thresholds

- F >= 605 of 672 fit rows (90%).
- H >= 168 of 224 held-out rows (75%) for "H5 is rejected"; H <= 56 of 224 (25%) for "H5 is supported".
- N <= 22 of 224 (10%).
- 200 Adam steps. Coordinates and learning rate from `ember/research/latent_binding_constrained_code_extended_20260911/coordinates.pt` (`s`, `vh`, `lr` = 0.19364916731037085).
- Nothing is tuned. No rule, threshold or step count changes after the smoke test.

## 3. Fixed implementation details

These details fill in what section 1 leaves open. They are fixed here, before any data.

**Inputs** (paths relative to `ember/research/`):

| Input | File |
|---|---|
| 224 training rows, receiver prompts (224 x 5 tokens), labels | `latent_binding_training_fold_recovery_20260913/inputs.json` |
| Folds (fit rows 164/168/168/172, held-out rows 60/56/56/52) | `latent_binding_training_fold_recovery_20260913/folds.json` |
| Projected memory of each training ID (56 x 960) | `latent_binding_training_fold_recovery_20260913/features.pt` (`projected`) |
| d0 bridge (SHA-256 d0f16cb9705ba53daf6cc8f2bfad03a08efa96cbb4f77dd836c9533e948cba87) | `latent_binding_shared_id_query_mechanics_20260908/run/candidate.pt` |
| Coordinates and learning rate | `latent_binding_constrained_code_extended_20260911/coordinates.pt` |
| d0 parity reference (saved step-0 scores, 60 fold-0 held-out rows) | `latent_binding_training_fold_recovery_20260913/run/fold0/step0/scores.pt` |
| Novel bank prompts and targets (descriptive fit only) | `latent_binding_literal_id_swap_20260910/inputs.json` (`banks.novel`) |
| Scorer, answer bank, model loader | `latent_binding_v1_20260906/binding.py` and `latent_channel_v1_20260906/channel.py`, through the sealed import chain of `latent_binding_literal_id_swap_20260910/runtime.py` (imports with bytecode off) |
| Seam | `seam.py` in this folder, a byte copy of `latent_binding_constrained_code_extended_20260911/seam.py` |

**Receiver.** SmolLM2-360M-Instruct, revision a10cc1512eabd3dde888204e902eca88bddb4951, float32, eager attention, CPU only. The parameter hash before and after must be 6a3e1782ebad58d1e9323d5eeaff191d112848b5f0058394a998d1ac54c492b3. No parameter takes a gradient.

**d0 codes.** For training ID i: v_i = W_v (float64) x projected_i (float64), with W_v = `reader.value.weight` of d0. This is the native selected value.

**Render.** As in the oracle: state at sites 3 and 4 = query_center + norm_scale x (gate x float32(W_out (float64) x value (float64))).

**Code parameters.** theta_c (1 x 64), theta_a (16 x 64), theta_b (8 x 64), all zero at the start. For an ID with letter L and digit D: value(L, D) = v0(L, D) + ((theta_c + theta_a[L] + theta_b[D]) / s) x vh. Here v0(L, D) = c0 + a_L + b_D from the minimum-norm least-squares fit of the design [1 | letter one-hot | digit one-hot] (25 columns) to the d0 codes of the fit IDs. This is the "additive least-squares fit" of section 1.

**Loss and steps.** One Adam step for each full pass over the fit rows, in `fit_rows` order, in batches of 8 (the last batch can be smaller). For each batch, the input is the 5 prompt tokens plus the first answer token, and the logits of the last two positions are scored against both answer tokens. Batch loss = sum of token cross-entropy over the batch / (2 x number of fit rows). The gradients of all batches add before the step. Adam: lr from `coordinates.pt`, betas (0.9, 0.999), eps 1e-8, no weight decay, no clipping, no early stop, no checkpoint selection. 200 steps. Seed and threads: `seed_all(60113)` of the channel module (deterministic algorithms, 4 intra-op threads, 1 inter-op thread).

**Scoring.** `binding.score` (128 candidates, full vocabulary, both tokens, batch 8). Each scored arm runs twice and must replay exactly. Joint correct: the argmax over 128 of the summed log-probability equals the target. Constrained sequential: the oracle `counts` rule (best of 16 prefixes, then best digit in that prefix).

**Arms for each fold k.**

- Matched (T5): held-out rows with their factored code. Gives H_k (joint) and the sequential count.
- Fit rows (T5): fit rows with their factored code. Gives F_k.
- C2, wrong parts: for each held-out row, the code of the next held-out ID of fold k in sorted ID order (cyclic). Report intended and inserted joint counts.
- C3, random: Gaussian noise (torch Generator, seed 60113 + k), each row scaled to the norm of its fitted held-out code, rendered. Report the intended joint count.
- C4, zero code: value = 0, rendered. C4, receiver only: no payload. Report the intended joint counts.
- C1 (separate job): the same fit, but the target of each fit row is the next training ID in sorted order of the 56 IDs (cyclic). The code of each row is still built from the parts of its true ID. The start is the same as in T5. Score the held-out rows with their C1 codes against the true targets. This gives N_k.

H = sum of H_k, F = sum of F_k, N = sum of N_k, over the four folds.

**Descriptive novel fit (part of the T5 job, after the four folds).** The same fit on all 224 training rows (start from the additive fit of all 56 d0 codes). Score the 32 novel-bank rows with the code of the target ID of each row, for the original and for the changed targets. Report the counts only as description.

**Mechanical gates (before any fit; stop and report on the first failure).**

1. Receiver parameter hash as above; no hooks; no parameter requires a gradient.
2. Stock logits equal the seam output with no payload, and equal the seam output with the receiver's own sites-3/4 states inserted (exact).
3. d0 parity: the oracle seam with the d0 code of each row reproduces the saved step-0 scores of the 60 fold-0 held-out rows: maximum absolute difference of `lp` at most 1e-4, and the same predictions.
4. Direct-score parity: `score` against `direct_score` for one row, at most 1e-4, same argmax.
5. At the end: the receiver is restored (stock logits equal), and the parameter hash is unchanged.

## 4. Run protocol

1. Smoke test: fold 0 only, gates 1 to 4, then 3 Adam steps. It measures the time of one step and of one scored row. It scores no T5 or C1 code, so it gives no held-out or fit-row score. Record the time below as a dated addition.
2. Freeze: `FREEZE.json` with the SHA-256 of this file, the code and every input.
3. T5 job: folds 0 to 3, then the descriptive novel fit. Then the C1 job: folds 0 to 3. One job at a time.
4. Before each long run: no process `pilot.py`, `harness.py`, `smcheck.py` or `latent_capacity` is active; record `sysctl vm.swapusage` and `pmset -g ac`; run only on AC power at 60 W or more. Use `caffeinate -i -w <pid>` and `tee -a` for logs.
5. A guard checks the power every 60 s. If the machine is on battery, or the battery is at 50% or less, the guard sends SIGSTOP to the job and writes the time in the log. It sends SIGCONT only on AC power at 60 W or more with the battery above 50%, and writes the time.
6. If a job fails, report it as incomplete. Do not restart it or select from partial folds without a new written decision.

## 5. Dated additions

**2026-10-20, 20:10Z (UTC), before any data.** Smoke attempt 1 started at 20:10:10Z and was stopped by the operator at 20:10:44Z, during model load. Reason: a `followup.py` job of another session (pid 13500) was active, and the coordination rule of the parent session says not to start while one is active. The preflight did not yet check for `followup.py`. No gate, no fit step and no score ran; no `SMOKE.json` exists. Changes (coordination only; no rule, threshold or step count changed): the preflight also blocks on `followup.py`, and `launch.sh` starts `t5.py` by its full path, so that the text "latent_r06" is in the process command line.

**2026-10-20, 20:16Z (UTC). Smoke attempt 2: gate 3 FAILED; the run stopped before any fit step.** Started 20:15:25Z after the preflight passed (no blocking process; AC 140 W; swap 0 MB). Gates 1 and 2 passed (receiver hash; stock, disabled and self-substitution exact). Gate 3: on the 60 fold-0 held-out rows, the oracle seam with the d0 codes gives the same 60 predictions as the saved step-0 scores, but the maximum absolute `lp` difference is 6.418e-4 (limit 1e-4). Gate 5 at the end passed (receiver restored, hash unchanged). No fit step ran and no T5 code was scored. The output is kept as `SMOKE_attempt2_gate3_fail.json`. Timing from this attempt: setup 3.9 s; two scorings of 60 rows took 38.4 s (0.32 s per scored row).

Probable cause (checked in the saved files, without a model run): the reference scores come from a different platform. The fold record ran on Linux x86_64 (`latent_binding_training_fold_recovery_20260913/FREEZE.json`, `platform`), and this run uses macOS arm64, with identical package versions (torch 2.13.0, transformers 5.10.2, tokenizers 0.22.2, numpy 2.4.6, safetensors 0.8.0). The payload rule of this plan is the same as the record's `render` (`common.py`), and on its own platform the record's native-against-seam check gave an error of 0.0 (`run/fold0/step0/contract.pt`). Thus gate 3, as written, compares two CPU platforms as well as the seam. No rule is changed by this note. A change of gate 3 needs [researcher]'s decision.

**2026-10-20 (UTC 20:40, approx.). Amendment 1, approved by [researcher] in this session, before any T5 or C1 data.** [researcher]'s words: [redacted: private chat] New gate 3:

- Rows: the 60 fold-0 held-out rows (`validation_rows`; equal to the `indices` of the saved step-0 scores).
- Native path: the sealed d0 bridge (`bridge()` of `latent_binding_shared_id_native_corrected_20260908/core.py`) in the native receiver hook, with the saved memory templates (`latent_binding_training_fold_recovery_20260913/memories.pt`), on the 6-token input (prompt plus first answer token), last two positions.
- Oracle path: the seam of this folder with the rendered d0 codes, same rows and batches of 8.
- Pass: maximum absolute logit difference at most 1e-4 and the same argmax at both positions. The checks of the record's contract must also hold: hard routes and scores equal the template capture (`latent_binding_fresh_known_native_20260910/run/prefill_correct.pt`), routes equal the expected slot, the selected memory slot equals the projected feature, positions outside sites 3 and 4 are unchanged, and the rendered d0 payload equals the native sites-3/4 output exactly.
- The comparison with the saved Linux x86_64 step-0 scores is now descriptive only (smoke attempt 2: max 6.418e-4, 60/60 same predictions).
- The d0 codes are now computed as `F.linear(projected, W_v)` in float64, which is the record's `render` rule, so that the payload can equal the native output exactly. The math is unchanged.
- Inputs added: `memories.pt`, the familiar prefill capture and the `core.py` above.

**2026-10-20. Removal of the descriptive novel-bank fit (before any data).** The project file `ember/AGENTS.md` has this standing rule: "Do not launch any saved-array audit, adjoint trace, precision diagnostic, sender-tap sweep, multi-slot read or trial on the burned 8-ID bank." The descriptive fit of section 1 ("a fit on all 56 training IDs, scored on the eight novel IDs") is a trial on that bank. Thus the T5 job does not run it. It does not enter the reading rule. It runs only if [researcher] says so after this note. The four T5 folds and C1 use only the 56 training IDs.

**2026-10-20, 21:36Z (UTC). Smoke attempt 3: amended gate 3 FAILED on one sub-check; the run stopped before any fit step.** Started 21:35:10Z after the preflight passed (AC 140 W, swap 0 MB, no blocking process). Gate 3 values on the 60 fold-0 held-out rows: native path against oracle seam, max absolute logit difference 0.0 and the same argmax; routes exact; selected memory slot equal to the projected feature; positions outside sites 3 and 4 unchanged; rendered d0 payload equal to the native output. One sub-check failed: hard routes and scores equal the template capture (`template_hard_scores_exact` False). No fit step ran and no T5 code was scored. The output is kept as `SMOKE_attempt3_gate3_fail.json`.

Finding (saved files only, no model run): the template capture `latent_binding_fresh_known_native_20260910/run/prefill_correct.pt` also comes from the Linux x86_64 machine (the nested identity in `FREEZE.json` of that folder gives platform `Linux-7.2.3-arch1-2-x86_64-with-glibc2.44`). Thus this sub-check compares two CPU platforms, the same problem as the first gate 3. The code now also reports, as descriptive numbers, whether the hard routes equal the template and the maximum absolute score difference. The pass condition is unchanged.

**Proposed amendment 2 (put in force on 2026-10-21; see the addition below).** In gate 3, the template sub-check becomes descriptive (report "hard routes equal the template" and the maximum score difference as numbers). These stay pass conditions: routes equal the expected slot, selected memory slot equals the feature, positions outside sites 3 and 4 unchanged, payload exact, and native against seam at most 1e-4 with the same argmax.

**2026-10-21, 02:29 local. Power.** `pmset -g ac` gives "Wattage = 8W" on the 140 W adapter (battery 100%). The preflight needs 60 W or more, so no smoke, T5 or C1 run can start.

**2026-10-21, about 13:20 local. Coordination (no rule, threshold or step count changed).** The parent session relayed this order from [researcher]: [redacted: private chat] Thus the uncertainty confirmatory runs R0, R1, R2a and R2b (`latent_uncertainty_confirm_20261021/uconfirm.py`) go before T5. Changes: the preflight also blocks on `uconfirm.py` and `phase0.py`. T5 and C1 do not start while a `uconfirm.py` job runs, or in the gaps between its runs R0 to R2b. The preflight cannot see these gaps, so T5 starts only after the parent session or [researcher] says that the R0 to R2b sequence has ended. Amendment 2 is still not in force.

**2026-10-21. Amendment 2 in force, approved by [researcher] in this session, before any T5 or C1 data.** [researcher]'s words: [redacted: private chat] In the code, the template sub-check is `template_hard_scores_exact_descriptive`, with `template_hard_equal_descriptive` and `template_scores_max_abs_descriptive`. None of them enters the pass condition. The pass condition of gate 3 is: routes exact, memory slot equal to the feature, outside sites unchanged, payload exact, native against seam at most 1e-4 with the same argmax.

**2026-10-21, 12:57Z (UTC). Smoke attempt 4: gate 4 FAILED through a code error; the run stopped before any fit step.** Started 12:55:45Z after the preflight passed (AC 140 W; swap 2,757 MB used, left from the Phase D runs, with 93% of memory free; no blocking process). Gates 1, 2 and 3 (amendment 2) passed: native against seam max absolute logit 0.0 with the same argmax; routes, memory slot, outside sites and payload exact. Descriptive: template hard routes equal (True), template score difference 9.567e-6, cross-platform `lp` difference 6.418e-4 with 60/60 same predictions. Gate 4 compared the first row as scored in a batch of 8 with `direct_score` of that row in a batch of 1: 1.354e-4, same argmax. No fit step ran and no T5 code was scored. The output is kept as `SMOKE_attempt4_gate4_fail.json`. Timing: two scorings of 60 rows took 42.0 s (0.35 s per scored row).

Code fix (no rule change): section 3 says "`score` against `direct_score` for one row". The oracle runtime (`latent_binding_constrained_code_extended_20260911/runtime.py`) scores that one row alone for this check. The code now does the same: `score` of the single row (batch of 1) against `direct_score` of that row, limit 1e-4 and the same argmax, unchanged. The batch-8 against batch-1 difference is reported as `cross_batch_max_abs_descriptive`.

**2026-10-21, 13:00Z (UTC). Smoke attempt 5: gate 4 FAILED again; the run stopped before any fit step.** Gates 1 to 3 passed with the same values as attempt 4. Gate 4, now with `score` of the single row alone: 1.354e-4 against `direct_score` (limit 1e-4), same argmax. The batch-8 value is the same, 1.354e-4. Thus batch size is not the cause. The two paths of the scorer differ on this machine: `score` takes the first-token log-probability from a 5-token forward pass and the second from a 6-token pass; `direct_score` takes both from one 6-token pass. On the Linux machine the records give about 1e-5 for this check (for example `direct_error` 9.537e-6 in the fold-0 step-0 scores). No fit step ran and no T5 code was scored. The output is kept as `SMOKE_attempt5_gate4_fail.json`. Timing: two scorings of 60 rows took 39.7 s. A change of gate 4 needs [researcher]'s word.

**2026-10-21. Amendment 3 in force, approved by [researcher] in this session, before any T5 or C1 data.** [researcher]'s words: [redacted: private chat] In the code, the difference is `direct_max_abs_descriptive`, and the pass condition of gate 4 is the same argmax only. All other gates are unchanged.

**2026-10-21, 16:51Z (UTC). Smoke attempt 6 PASSED; this is the smoke time required by section 4.** Started after the preflight passed (AC 140 W; no blocking process; R2b of Phase D had ended at 19:48:30 local). All gates passed under amendments 1 to 3. Descriptive values: native against seam max absolute logit 0.0; template hard routes equal; template score difference 9.567e-6; cross-platform `lp` difference 6.418e-4 with 60/60 same predictions; direct-score difference 1.354e-4 with the same argmax. Fit: 3 Adam steps on the 164 fold-0 fit rows took 4.95, 5.00 and 4.95 s. Scoring: 0.308 s per row for one scoring (each arm scores twice). No T5 or C1 code was scored. Output: `SMOKE.json`.

Estimate from this time: one fold of the T5 job is about 200 x 5.0 s of fit plus about 450 rows x 2 x 0.31 s of scoring, about 21 min. Four folds take about 1.5 h. One fold of C1 is about 18 min, so C1 takes about 1.2 h. No rule, threshold or step count changes after this addition.

