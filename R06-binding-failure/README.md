# R.06 Binding failure mechanism: records

These are the records behind the entry [R.06 Binding failure mechanism](https://symbolonlabs.com/research/binding-failure/) on the Symbolon site. All paths below are relative to the root of this repository.

## What the record shows

A trained bridge of the R.01 design connects Qwen3-1.7B to SmolLM2-360M-Instruct (569,857 parameters). The bridge binds familiar IDs, but it fails on IDs that are absent from training. This series of sealed tests asks why.

- **Familiar IDs.** All six views passed.
- **Novel whole IDs.** Eight whole IDs that are absent from training scored 0/32. A text handoff scored 32/32 (original) and 29/32 (changed).
- **Oracle codes.** Supervised codes through the frozen output path of the bridge reached 26/32 after 40 steps and 32/32 after 200 steps. Thus the reader can bind when it gets a suitable code.
- **Identity.** No two of the eight IDs share a code at any stage of the bridge.
- **Learned codes.** The learned codes are specific to each training ID. An additive letter-plus-digit model of them has 30.5% error in sample and 51.3% error when one ID is left out. Codes built from parts score 32/224, against 15/224 for a shifted-label null.
- **More training.** More training fits all 224 training rows, but it lowers the novel IDs from 4/32 to 0/32.
- **Erasure.** The removal of at least 99.1% of the readable signal of one probe changed no outcome on 27 failed IDs. The internal path of the failure is still open.

Limits that the records state:

- Each design has one seed, one learning rate and one bridge size. The source of this summary is not public (see "Other values").
- The eight novel IDs had earlier evaluation history, so they are not a pristine test (`records/research/latent_binding_validation_ladder_20260910/REPORT.md:23`).
- These tests use three different banks of IDs: the eight novel whole IDs, the 56 training IDs held out in four folds, and the 224 training rows of the composition test. The site keeps them apart.

## Site values and their sources

This table follows the evidence map of the site (`design/ENTRY_R06_EVIDENCE.md` in the site repository). `<R>` is `records/research`.

The site shows the first 12 hex digits of the SHA-256 of the **original** result file. For each file, `MANIFEST.json` gives `sha256_original` and `sha256_copy`.

### Gates

| Gate on the site | Site hash | Result file | Copy scrubbed | Rule source |
|---|---|---|---|---|
| Familiar IDs (passed) | `d46612aa92eb` | `<R>/latent_binding_validation_ladder_20260910/SUMMARY.json`; table in `REPORT.md` | yes (128 paths) | Same folder, `REPORT.md:19` (minima 31/31/28; reversal 28/28/28) |
| Novel whole IDs (failed) | `f155c4075a79` | `<R>/latent_binding_novel_whole_native_20260910/analysis.json`; text arm `<R>/latent_binding_novel_whole_text_20260910/summary.json` (`b78cf0ce39b2`) and `REPORT.md` | no | `<R>/latent_binding_novel_whole_native_20260910/REPORT.md:5` and `PREREGISTRATION.md` |
| Oracle code, 40 steps (failed) | `44daaabc1e19` | `<R>/latent_binding_constrained_code_oracle_20260911/ANALYSIS.json`; table in `REPORT.md` | no | Same folder, `PLAN.md:27-30` |
| Oracle code, 200 steps (passed) | `c0282ae48521` | `<R>/latent_binding_constrained_code_extended_20260911/ANALYSIS.json`; table in `REPORT.md` | no | Same rule; the extension is registered in `INTERPRETATION.md` and `PLAN.md` of that folder |
| Code composition (failed) | `de4ea4594879` | `<R>/latent_binding_code_construction_20260912/run/RESULT05.json`; `STAGE05_RESULT.md:21-24` | no | Same folder, `STAGE05_PREREGISTRATION.md` |
| Longer training (failed) | `2549a4c8343f` | `<R>/latent_binding_convergence_fixed_20260911/ANALYSIS.json`; table in `REPORT.md` | no | Same folder, `PLAN.md:54-56` |
| Held-out training IDs (failed) | `91e8458e0d08` | `<R>/latent_binding_training_fold_recovery_20260913/ANALYSIS.json`; table in `REPORT.md` | no | Same folder, `REPORT.md` (selection rule) and `PLAN.md` |

The scrub of `SUMMARY.json` changed only paths: model cache paths and the paths of the source files that it hashes. `SCRUB_LOG.json` lists each change.

### Controls

| Arm on the site | Site value | Source |
|---|---|---|
| Wrong code | Follows the inserted ID 32/32; intended ID 0/32 | `<R>/latent_binding_constrained_code_extended_20260911/REPORT.md`, table, columns "Wrong inserted" and "Wrong intended" |
| Random code | Intended ID 0/32 | Same table, column "Random intended"; method `<R>/latent_binding_constrained_code_oracle_20260911/PLAN.md:30-31` |
| Shifted-label null | 15/224 joint (composed 32/224) | `<R>/latent_binding_code_construction_20260912/STAGE05_RESULT.md:21` |
| Text handoff | 32/32 and 29/32 (latent 0/32) | `<R>/latent_binding_novel_whole_text_20260910/REPORT.md`, table at line 11 |
| Orthogonal erasure | No correctness change on 27 IDs, same as treatment | `<R>/latent_binding_boundary_causal_native_20260914/REPORT.md:15-17` (baseline, treatment and control all 20/12/0); `RESULT.json` (`5897b59df2de`) |
| Shifted probe labels | 0/56 whole IDs | `<R>/latent_binding_matched_fold_probe_20260914/REPORT.md:42`. The report calls this a structured shifted-label control, not a chance baseline. |

### Other values

| Value on the site | Source |
|---|---|
| Pair Qwen3-1.7B to SmolLM2-360M-Instruct | `<R>/latent_binding_novel_whole_native_20260910/` (model IDs in `FREEZE.json`) |
| Bridge of 569,857 parameters | `<R>/latent_binding_novel_whole_native_20260910/REPORT.md:18` |
| "A bridge of the R.01 design" | The same pair and parameter count as R.01: `<R>/latent_channel_smollm_20260906/REPORT.md:59`. The weights are not the R.01 weights. |
| One seed, one learning rate and one bridge size for each design | Source not public. The summary comes from an outside review of the program (2026-09-30). The review is not included, because it quotes private chat. No included file states this summary. |
| Eight novel IDs with earlier evaluation history | `<R>/latent_binding_validation_ladder_20260910/REPORT.md:23` ("older evaluation/design-selection history ... Not pristine IDs") |
| No two of the eight IDs share a code at any stage | `<R>/latent_binding_identity_localization_20260910/REPORT.md:7-18` (decode 48/48 at each stage; no exact cross-ID collisions) |
| Additive model error 30.5% in sample, 51.3% leave-one-out | `<R>/latent_binding_code_construction_20260912/STAGE05_RESULT.md:27-37` |
| More training: 224/224 training rows; novel 4/32 to 0/32 | `<R>/latent_binding_convergence_fixed_20260911/REPORT.md` table |
| At least 99.1% of the readable signal of one probe removed | `<R>/latent_binding_diagnosis_20260914/REPORT.md:16` |
| The internal path of the failure is still open | `<R>/latent_binding_diagnosis_20260914/REPORT.md:24` |
| E3 uses a factored code and passes | `<R>/latent_anchor_E3_20261003/REPORT.md:3` (see R.01) |
| Opened 2026-09-09 | `<R>/latent_binding_novel_whole_native_20260910/seal.json` field `utc` = 2026-09-09T22:58:55Z |
| Updated 2026-09-14 | `<R>/latent_binding_diagnosis_20260914/REPORT.md` (ledger dated 2026-09-14) |

## Preregistration

The version 1.1 tests T5 and T5b have their own list in [Version 1.1](#version-11-tests-t5-and-t5b-2026-10-20-to-2026-10-22).

- `<R>/latent_binding_novel_whole_native_20260910/PREREGISTRATION.md` and `FREEZE.json`
- `<R>/latent_binding_novel_whole_text_20260910/PREREGISTRATION.md` and `FREEZE.json`
- `<R>/latent_binding_constrained_code_oracle_20260911/PLAN.md` and `FREEZE.json`
- `<R>/latent_binding_constrained_code_extended_20260911/PLAN.md`, `INTERPRETATION.md` and `FREEZE.json`
- `<R>/latent_binding_code_construction_20260912/PREREGISTRATION.md` and `STAGE05_PREREGISTRATION.md`
- `<R>/latent_binding_convergence_fixed_20260911/PLAN.md`, `FREEZE.json` and `CORRECTION.md`
- `<R>/latent_binding_training_fold_recovery_20260913/PLAN.md`, `DESIGN.md` and `FREEZE.json`
- `<R>/latent_binding_boundary_causal_native_20260914/PLAN.md` and `FREEZE.json`
- `<R>/latent_binding_matched_fold_probe_20260914/PLAN.md` and `FREEZE.json`
- `<R>/latent_binding_identity_localization_20260910/PREREGISTRATION.md` and `FREEZE.json`

Three files of this entry have one redaction each: a direct quote of private chat is replaced by `[redacted: private chat]`.

| File | Line |
|---|---|
| `<R>/latent_binding_training_fold_recovery_20260913/PLAN.md` | 15 |
| `<R>/latent_binding_matched_fold_probe_20260914/PLAN.md` | 3 |
| `<R>/latent_binding_diagnosis_20260914/REPORT.md` | 49 |

The SHA-256 of these copies does not agree with the recovery manifest, or with the seal of the folder where a seal exists. In `MANIFEST.json`, `sha256_original` is the sealed hash, and `SCRUB_LOG.json` (`redactions`) gives each position. No other text and no number changed.

## Raw outputs

The version 1.1 tests T5 and T5b have their own list in [Version 1.1](#version-11-tests-t5-and-t5b-2026-10-20-to-2026-10-22).

- The result files of each test: `analysis.json`, `ANALYSIS.json`, `RESULT.json`, `SUMMARY.json`, `summary.json`, `PER_ID.json` and `run/RESULT05.json`.
- The text files in each `run/` folder (for example `<R>/latent_binding_novel_whole_native_20260910/run/evaluation/per_example.json`).
- The text arm: `<R>/latent_binding_novel_whole_text_20260910/sender/` and `receiver/`.

## Audit report

The version 1.1 tests T5 and T5b have their own list in [Version 1.1](#version-11-tests-t5-and-t5b-2026-10-20-to-2026-10-22).

- `<R>/latent_binding_novel_whole_native_20260910/audit.py`, `audit.json` and `closing_checks.json`
- `<R>/latent_binding_novel_whole_text_20260910/audit.py`, `input_audit.json`, `sender_audit.json` and `receiver_audit.json`
- `<R>/latent_binding_matched_fold_probe_20260914/audit_saved.py` and `INDEPENDENT_AUDIT.json`
- `<R>/latent_binding_identity_localization_20260910/audit.py` and `AUDIT.json`
- `closing_checks.json` in each sealed folder
- The synthesis of the series: `<R>/latent_binding_diagnosis_20260914/REPORT.md`

These audits are second scripts in the same project. No outside party ran them.

## Code

The version 1.1 tests T5 and T5b have their own list in [Version 1.1](#version-11-tests-t5-and-t5b-2026-10-20-to-2026-10-22).

The code of each test is in its folder: `runtime.py`, `analysis.py` or `analyze.py`, `core.py`, `common.py`, `seam.py`, `stage0.py`, `stage05.py`, `diagnostic.py` and their tests (`test_*.py`). The bridge design is the R.01 design: `<R>/latent_channel_smollm_20260906/channel_cf.py`.

## What is missing and why

`MANIFEST.json` lists each file that is not included, with its SHA-256, its size and the reason. For this entry, 339 files (2,211 MB) are not included:

| Reason | Files | Size | Status |
|---|---|---|---|
| Binary files (checkpoints, codes, coordinates, saved arrays, logits) | 264 | 252 MB | Not included, available on request |
| Files larger than 20 MB (`prefill_*.pt`, `six_*.pt`, `contract.pt`, `logits.pt`) | 32 | 1,959 MB | Not included, available on request |
| Job folders (`job/`, `job_sender/`, `job_receiver/`: process state and logs) | 27 | 0.1 MB | Not included, available on request |
| Agent session notes (`HANDOFF.md`, `SESSION_HANDOFF.md`, `AUTHORIZATION.md`) | 7 | < 0.1 MB | Not included |
| Empty lock files (`job.lock`) | 8 | 0 MB | Not included |
| The outside review of the program (`RESEARCH_REVIEW_20260930.md`): it quotes private chat | 1 | < 0.1 MB | Not included |

Other gaps:

- **Folders without a seal.** `latent_binding_code_construction_20260912/` and `latent_binding_diagnosis_20260914/` have no seal file. Their hashes are only in the recovery manifest (`integrity/recovery-manifest-lines.jsonl`).
- **Model weights.** The models are not copied. `LICENSES.md` gives the pinned revisions.
- **Other binding folders.** The snapshot has 153 folders named `latent_binding_*`. This entry includes only the 12 that the site cites.

## Historical copy

This folder is a copy of records from a recovered historical snapshot of the old Ember project. It is not a runnable environment.

- The code refers to paths, virtual environments and model caches that do not exist here.
- The scrub changed local paths in 17 files of this entry. Code paths are now relative to `records/` or start with `<local>`.
- Three files of this entry have a redaction (see "Preregistration").
- Most scripts need the binary files that are not included.

Use the code to read the method. Do not expect it to run without changes.

## Version 1.1: tests T5 and T5b (2026-10-20 to 2026-10-22)

Version 1.1 adds three later R.06 folders. Two of them hold tests T5 and T5b, with their shifted-label controls C1 and C1b. The third folder holds an earlier CPU run that this entry does not use (see "Other folder of version 1.1").

Both tests are exploratory: each plan and each report calls them post hoc diagnostics. Each test has a plan with a frozen decision rule that was written before any data. Each test also has a freeze file with the SHA-256 of the plan, the code and every input. The three folders have no seal file.

In this section, `<T5>` is `<R>/latent_r06_t5_factored_oracle_20261020` and `<T5b>` is `<R>/latent_r06_t5b_wide_20261022`.

### Design of both tests

- **IDs.** Both tests use only the 56 training IDs and their 224 rows, in the four fixed folds of `<R>/latent_binding_training_fold_recovery_20260913/`. Each ID is held out once. This is the bank of the gate "Held-out training IDs", not the bank of the eight novel whole IDs.
- **Bridge and receiver.** The bridge is d0 (SHA-256 `d0f16cb9…`), the candidate of the gate "Novel whole IDs". The receiver SmolLM2-360M-Instruct is frozen, and no receiver parameter takes a gradient. Its parameter hash is the same before and after each job.
- **Factored code.** The code of an ID with letter L and digit D is v(L, D) = c0 + a_L + b_D. It has one shared vector, one vector for each letter and one vector for each digit. The code replaces the selected value of the bridge at both query sites, as in the oracle tests of version 1.0.
- **Fit.** For each fold, the fit has 200 Adam steps on the fit rows. It uses the coordinates and the learning rate of the 200-step oracle (`<R>/latent_binding_constrained_code_extended_20260911/coordinates.pt`). Nothing is tuned. There is one seed (60113). The jobs ran on the CPU only.
- **Counts.** F is the joint count on the fit rows, summed over the four folds (of 672). H is the joint count on the held-out rows (of 224). N is the joint count of the shifted-label control on the held-out rows, against the true targets (of 224).
- **Shifted-label control.** This is the same fit, but the target of each fit row is the next training ID in sorted order. The code of each row is still built from the parts of its true ID.

### T5: factored code at the frozen interface

**Question** (`<T5>/PLAN.md:13`): "Does a code built from one vector for each letter and one vector for each digit, fit through the frozen output path and the frozen receiver on training IDs only, bind held-out IDs?"

**Design.** The code has 1,600 values in the 64-d selected-value space of the bridge. The output map, the gate, the norm scale, the centers and the receiver stay frozen. The start is the additive least-squares fit of the d0 codes of the fit IDs. The other controls are wrong parts (C2: the code of another held-out ID), a random code (C3), a zero code and the receiver only (C4).

**Frozen rule** (`<T5>/PLAN.md:27-31` and section 2):

1. If F >= 605, H >= 168 and N <= 22: "a factored code works at this interface. **H5 is rejected.**"
2. If F >= 605 and H <= 56: "the interface accepts factored codes for the fit IDs, but they do not transfer. **H5 is supported.**"
3. If F < 605: "a factored code cannot even fit the training IDs here. **H5 is supported in its strong form.**"
4. Otherwise: "inconclusive. Report the counts. Do not move a threshold."

The bar of 168 (75%) is the joint bar of E3.

**Amendments before any data** (`<T5>/PLAN.md`, section 5). Three amendments changed the mechanical gates 3 and 4, and each amendment came before any T5 or C1 data. The reference scores of the first gate 3 come from another CPU platform (Linux x86_64; this run used macOS arm64). Amendment 1 compares the native path and the seam on the same machine. Amendment 2 makes the template sub-check of gate 3 descriptive. Amendment 3 makes gate 4 need the same argmax only, and the difference (1.354e-4) becomes a descriptive value.

No decision rule, threshold or step count changed. The smoke test needed six attempts, and section 5 records each attempt. Also before any data, the plan removed a descriptive fit on the eight novel IDs. A project rule does not permit trials on that burned bank.

**Freeze.** `<T5>/FREEZE.json`, SHA-256 `ce382b1f8abd…`. Both result files give this value in `freeze_sha256`.

**Result** (`<T5>/REPORT.md:9-13` and the table at line 27):

| Count | Value | Bar | Source |
|---|---:|---|---|
| F, fit rows | 672 / 672 | at least 605 | `RESULT_t5.json`, `totals.fit_joint` |
| H, held-out rows | 32 / 224 | at least 168 rejects H5; at most 56 supports H5 | `RESULT_t5.json`, `totals.heldout_joint` |
| N, control C1 | 15 / 224 | at most 22 | `RESULT_c1.json`, `totals.heldout_joint` |
| C2 wrong parts, intended ID | 8 / 224 | | `RESULT_t5.json`, `folds.<k>.wrong_intended.joint`, sum over the four folds |
| C2 wrong parts, inserted ID | 32 / 224 | | Same file, `wrong_inserted.joint` |
| C3 random code | 2 / 224 | | Same file, `random_intended.joint` |
| C4 zero code | 4 / 224 | | Same file, `zero_intended.joint` |
| C4 receiver only | 4 / 224 | | Same file, `receiver_only_intended.joint` |

**Reading** (`<T5>/REPORT.md:13`): F >= 605 and H <= 56. Thus "the interface accepts factored codes for the fit IDs, but they do not transfer. **H5 is supported.**" Test T6 did not run, because its rule needs T5 to reject H5 (`<T5>/REPORT.md:15`).

**Descriptive notes** (`<T5>/REPORT.md:31-34`; these are not readings):

- C1 also fits all of its fit rows (teacher-forced accuracy 100% in each fold). Thus, through this frozen path, a sum of a letter vector and a digit vector can carry an arbitrary answer for each fit ID.
- In each fold, the C2 count for the inserted ID equals the T5 held-out count. When a held-out code works, the parts of another ID move the answer to that other ID.
- H = 32 and N = 15 are equal to the counts of the composition test of version 1.0 (composed codes 32/224, null 15/224). H = 32 is also the step-0 held-out count of the fold record. The report says that these are different tests with different codes, and that no inference follows from the equal numbers.

### T5b: factored code at full boundary width

**Question** (`<T5b>/PLAN.md:53`). T5 fit a factored code in the 64-d value space. The frozen output map of d0 sends this space to a 64-d subspace of the 960-d boundary state at block 15. T5b asks if a factored code binds held-out IDs when it can also use the other 896 boundary directions.

**Design** (`<T5b>/PLAN.md:59-75`). The parent plan says "repeat T5 with the output map also trainable". The T5b plan uses an equivalent form. A factored code has 25 free vectors, and 25 <= 64, so a trainable output map can give any factored code in the 960-d boundary space. T5b keeps the T5 code and adds a factored code in the 896-d complement of the output map, for 24,000 trainable values. The script `complement.py` computed the basis of the complement once, before the freeze (`COMPLEMENT.json`).

All other parts are as in T5. The controls are C1b (shifted labels), C2b (wrong parts), C3b (random code) and C4b (zero code and receiver only).

**Frozen rule** (`<T5b>/PLAN.md:136-145`, section B, 4.5):

1. If Fb >= 605, Hb >= 168 and Nb <= 22: "T5b passes." "**The frozen d0 output path is the barrier (H5-path), not the receiver.**"
2. If Fb >= 605 and Hb <= 56: "factored codes fit the training IDs at full width, but they do not transfer. **The frozen receiver at this seam is the barrier (H5-receiver).**"
3. If Fb < 605: "**H5-receiver is supported in its strong form.**"
4. Otherwise: "inconclusive. Report the counts. Do not move a threshold."

The bars are the T5 bars.

**Freeze.** `<T5b>/FREEZE.json`, SHA-256 `50b7d07096c0…`. The smoke test passed at the first attempt (`<T5b>/PLAN.md`, section D). The plan has no amendment.

**Result** (`<T5b>/REPORT.md:9-11` and the table at line 32):

| Count | Value | Bar | Source |
|---|---:|---|---|
| Fb, fit rows | 672 / 672 | at least 605 | `RESULT_t5b.json`, `totals.fit_joint` |
| Hb, held-out rows | 57 / 224 | at least 168 passes; at most 56 supports H5-receiver | `RESULT_t5b.json`, `totals.heldout_joint` |
| Nb, control C1b | 19 / 224 | at most 22 | `RESULT_c1b.json`, `totals.heldout_joint` |
| C2b wrong parts, intended ID | 0 / 224 | | `RESULT_t5b.json`, `folds.<k>.wrong_intended.joint`, sum over the four folds |
| C2b wrong parts, inserted ID | 57 / 224 | | Same file, `wrong_inserted.joint` |
| C3b random code | 1 / 224 | | Same file, `random_intended.joint` |
| C4b zero code | 4 / 224 | | Same file, `zero_intended.joint` |
| C4b receiver only | 4 / 224 | | Same file, `receiver_only_intended.joint` |

**Reading** (`<T5b>/REPORT.md:13-20`): **inconclusive.**

1. A pass needs Hb >= 168. Hb is 57, so T5b does not pass.
2. Rule 2 needs Hb <= 56. Hb is 57, one row above the bar, so rule 2 does not apply.
3. Rule 3 needs Fb < 605. Fb is 672, so rule 3 does not apply.
4. Thus rule 4 gives "inconclusive". The report applies the bars as written.

T5b does not locate the barrier between the frozen d0 output path (H5-path) and the frozen receiver at this seam (H5-receiver). The T5 reading is unchanged (`<T5b>/REPORT.md:20`).

**Descriptive values** (`<T5b>/REPORT.md:36-45`; no rule):

- Hb - H = +25 (32 to 57). Fb - F = 0. Nb - N = +4.
- Held-out transfer rises from 32 to 57 of 224, but it stays far below the 75% bar of E3 (`<T5b>/REPORT.md:60`).
- Over the 56 held-out IDs, the median share of the code norm in the complement is 0.346 (range 0.293 to 0.419). Thus the fitted codes use the complement directions to a large degree.
- C1b also fits all of its fit rows. In each fold, the C2b count for the inserted ID equals the T5b held-out count.

### Limits that the records state

- Both tests are exploratory post hoc diagnostics. The 56 fold IDs were in the training of the ancestral d0 (`<T5>/PLAN.md:7`).
- All held-out IDs are inside a letter-digit block. Thus the tests check transfer inside a block only, and they do not test cross-block IDs (`<T5>/PLAN.md:35`; `<T5b>/REPORT.md:62`).
- Neither test used the eight novel IDs (`<T5>/REPORT.md:3`; `<T5b>/REPORT.md:3`, gate 7).
- Each test has one seed, one learning rate and one step budget. T5b also has one complement basis (`<T5>/REPORT.md:52`; `<T5b>/PLAN.md:165`).
- T5b keeps one shared vector at both sites, as the native path does. It does not test a code for each site, or a trainable gate, norm scale or centers (`<T5b>/PLAN.md:166-167`).
- The jobs saved the counts and the codes, but not the prediction of each row (`<T5>/REPORT.md:53`; `<T5b>/REPORT.md:63`).
- The two PLAN.md files cite parent plans in the `reports/` folder of the project. These parent plans are not in this package. Each PLAN.md copies the sections of its test word for word.

### Site values and their sources (version 1.1)

The site shows the first 12 hex digits of the SHA-256 of the original result file, as in version 1.0. For the files of these folders, `sha256_original` in `MANIFEST.json` is the hash of the original on 2026-10-25 (`known_issues`). The result files are not scrubbed, so each copy has the same hash as its original.

| Gate on the site | Site hash | Result file | Copy scrubbed | Rule source |
|---|---|---|---|---|
| Factored code, frozen output path (failed) | `5609ca1b9db5` | `<T5>/RESULT_t5.json` (`totals`); control `<T5>/RESULT_c1.json` (`2c4ce823bf4e`); table in `REPORT.md` | no | `<T5>/PLAN.md:27-31` and section 2 (`PLAN.md:37-43`) |
| Factored code, full boundary width (open) | `b928eb689816` | `<T5b>/RESULT_t5b.json` (`totals`); control `<T5b>/RESULT_c1b.json` (`fd4627b8ee26`); table in `REPORT.md` | no | `<T5b>/PLAN.md:136-145` (section B, 4.5) |

The site marks T5 "failed", because the branch "a factored code works at this interface" needs H >= 168, and H is 32. The site marks T5b "open", because its frozen rule gives "inconclusive". T5b does not pass, but the receiver branch does not apply either.

| Value on the site | Source |
|---|---|
| Updated 2026-10-22 | `<T5b>/run_c1b.log:109`: event `job_end` at 2026-10-22T17:38:08Z, the end of the last T5b job |

### Plans, raw outputs, audit and code (version 1.1)

- **Plans and freeze files.** `<T5>/PLAN.md` (with the dated additions of section 5), `FREEZE.json`, `SMOKE.json` and the four failed smoke outputs `SMOKE_attempt2_gate3_fail.json` to `SMOKE_attempt5_gate4_fail.json`. `<T5b>/PLAN.md`, `FREEZE.json`, `SMOKE.json` and `COMPLEMENT.json`.
- **Raw outputs.** `RESULT_t5.json`, `RESULT_c1.json`, `RESULT_t5b.json` and `RESULT_c1b.json`. The counts and the fit history of each fold are in `run/<arm>/fold0.json` to `fold3.json`. The job logs are the `run_*.log` files.
- **Audit.** No second audit script exists for T5 or T5b. The synthetic tests are `<T5>/test_t5.py` and `<T5b>/test_t5b.py`. Each scored arm runs twice and must replay exactly (`<T5>/PLAN.md:73`). The `gates` field of each result file gives the values of the mechanical gates.
- **Code.** `<T5>/t5.py` and `seam.py` (a byte copy of the oracle seam), `<T5b>/t5b.py` and `complement.py`, and the job scripts `preflight.sh`, `guard.sh` and `launch.sh`. The T5b folder holds byte copies of `t5.py` and `seam.py`. Gate 2 of T5b checks their SHA-256 against the T5 freeze.

### Changes in the copies (version 1.1)

The version 1.1 files are copies of the study folders in a later snapshot (2026-10), not of the recovered snapshot of version 1.0.

- **Name.** In five process files of the three folders, the name of the researcher became `[researcher]` (rule R14, 19 replacements).
- **Local paths.** In `launch.sh` of both tests and in `<T5>/run_smoke.log`, local paths became relative paths or start with `<local>` (rules R5 and R12).
- **Quotes of private chat.** Seven direct quotes of private chat became `[redacted: private chat]`:

| File | Lines |
|---|---|
| `<T5>/PLAN.md` | 5, 113, 133, 135, 143 |
| `<T5b>/PLAN.md` | 5 |
| `<R>/latent_r06_localization_cpu_20261020/REPORT.md` | 106 |

No number changed. `SCRUB_LOG.json` gives the position of each change.

### Files that are not included (version 1.1)

For T5 and T5b, 24 files (8.8 MB) are not included. `MANIFEST.json` lists each file with its SHA-256, its size and the reason.

| Reason | Files | Size | Status |
|---|---|---|---|
| Binary files: the fitted codes (`run/<arm>/fold<k>_code.pt`) and the complement basis (`<T5b>/complement.pt`) | 17 | 8.8 MB | Not included, available on request |
| Runtime markers of the jobs (`smoke_attempt*.pid`) | 7 | < 0.1 MB | Not included, available on request |

### Other folder of version 1.1

An earlier CPU run of T2 and T3 (`<R>/latent_r06_localization_cpu_20261020/`) broke a permanent project rule: no saved-array audit or trial on the eight burned IDs. Its `REPORT.md` has a rule note (lines 104 to 112), so this entry does not use its results as evidence.
