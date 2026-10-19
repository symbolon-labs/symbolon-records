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

- The result files of each test: `analysis.json`, `ANALYSIS.json`, `RESULT.json`, `SUMMARY.json`, `summary.json`, `PER_ID.json` and `run/RESULT05.json`.
- The text files in each `run/` folder (for example `<R>/latent_binding_novel_whole_native_20260910/run/evaluation/per_example.json`).
- The text arm: `<R>/latent_binding_novel_whole_text_20260910/sender/` and `receiver/`.

## Audit report

- `<R>/latent_binding_novel_whole_native_20260910/audit.py`, `audit.json` and `closing_checks.json`
- `<R>/latent_binding_novel_whole_text_20260910/audit.py`, `input_audit.json`, `sender_audit.json` and `receiver_audit.json`
- `<R>/latent_binding_matched_fold_probe_20260914/audit_saved.py` and `INDEPENDENT_AUDIT.json`
- `<R>/latent_binding_identity_localization_20260910/audit.py` and `AUDIT.json`
- `closing_checks.json` in each sealed folder
- The synthesis of the series: `<R>/latent_binding_diagnosis_20260914/REPORT.md`

These audits are second scripts in the same project. No outside party ran them.

## Code

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
