# R.01 Latent channel: records

These are the records behind the entry [R.01 Latent channel](https://symbolonlabs.com/research/latent-channel/) on the Symbolon site. All paths below are relative to the root of this repository.

## What the record shows

A small trained bridge (569,857 parameters) carries a two-bit symbol from a frozen Qwen3-1.7B sender into a frozen receiver.

- **Gates 0-2 (tiny set, cross family).** Mechanics, tiny accuracy and tiny controls passed.
- **Gate 3 (same family, Qwen3-0.6B receiver).** The held-out test passed.
- **Gate 4 (cross family, SmolLM2-360M-Instruct receiver, 1,024 new rows).** The held-out test passed with the Gate 3 rule, unchanged. Correct 100%, shuffled 26.37%, random 25.49% and 25.68%, zero 25.00%.
- **Text handoff.** A text handoff also scores 100%. Thus the channel shows no accuracy advantage over text.
- **Two earlier attempts failed.** Channel v1 failed Gate 1. A Llama-3.2-1B-Instruct receiver failed the tiny gate.
- **Binding gate V1 failed.** The best tiny result was 14/32 at step 240, and the result was 2/32 at step 300. The gate needs at least 95%.
- **Binding test E3 passed.** A factored letter, digit and color code scored 93.8% on 56 never-seen IDs. Its free-form arm failed (14.3%).

Limits that the records state:

- The task is a synthetic two-bit task, not natural-language transfer (`records/research/latent_channel_smollm_20260906/REPORT.md:10-13`).
- Each design has one training seed.
- "Determinism" means a second run with the same code, hardware and batches. It is not an independent replication (`REPORT.md:249-250` in the same folder).

## Site values and their sources

The site shows the first 12 hex digits of the SHA-256 of the file that holds each gate decision. This is the hash of the **original** file. For each file, `MANIFEST.json` gives `sha256_original` and `sha256_copy`. If `scrubbed` is false, the two hashes are equal.

`<C1>` is `records/research/latent_channel_v1_20260906`, `<V2>` is `records/research/latent_channel_v2_20260906`, `<CF>` is `records/research/latent_channel_smollm_20260906`, `<BV1>` is `records/research/latent_binding_v1_20260906` and `<E3>` is `records/research/latent_anchor_E3_20261003`.

### Gates

| Gate on the site | Site hash | File with the decision | Copy scrubbed | Rule |
|---|---|---|---|---|
| Channel v1, Gate 1 (failed) | `8d500bfebdf6` | `<C1>/run_tiny/decision.json` | no | `<C1>/PREREGISTRATION.md:96-117`; result `<C1>/REPORT.md:7` |
| Llama receiver, tiny gate (failed) | `a93cbe3a2f7e` | `records/research/latent_channel_crossfamily_20260906/seal.json` (field `verdict`) | no | `PREREGISTRATION.md` and `REPORT.md` in the same folder |
| Gate 0, Gate 1, Gate 2 (passed) | `8be13af87f99` | `<CF>/run_tiny/decision.json` (fields `gate0`, `gate1`, `gate2`) | no | `<C1>/PREREGISTRATION.md:96-117`; `<CF>/PREREGISTRATION.md` |
| Gate 3 (passed) | `b268b1a8472e` | `<V2>/run_final/analysis.json` (field `gate3`) | yes (2 paths) | `<V2>/PREREGISTRATION.md` |
| Gate 4 (passed) | `18e4a0f5df4b` | `<CF>/run_final/analysis.json` (field `gate4`) | yes (1 path) | `<CF>/PREREGISTRATION.md:75-81` |
| Binding gate V1 (failed) | `5a6f3408a767` | `<BV1>/final_checks.json` | no | `<BV1>/PREREGISTRATION.md`; result `<BV1>/REPORT.md:3` and `:170-177` |
| Binding E3 (passed) | `987dc2f8d726` | `<E3>/RESULT.json` | no | `<E3>/PREREGISTRATION.md`, `<E3>/FREEZE.json`; result `<E3>/REPORT.md:3-4` |

In the two scrubbed `analysis.json` files, the scrub changed only file paths in the code inventory. `SCRUB_LOG.json` lists each change.

### Controls (Gate 4 run)

All values are in `<CF>/run_final/analysis.json`, object `accuracy`.

| Arm on the site | Site value | Field | Method |
|---|---|---|---|
| Shuffled | 26.37% | `accuracy.shuffled.mean` = 0.263671875 | `<CF>/PREREGISTRATION.md:53-56` (label-blind derangement, seed 61001) |
| Random | 25.49% / 25.68% | `accuracy.random_71001.mean` = 0.2548828125; `accuracy.random_71002.mean` = 0.2568359375 | same lines (per-slot norm-matched) |
| Zero | 25.00% | `accuracy.zero.mean` = 0.25; `mechanical_checks.zero_exact` = true | same lines |
| Text-handoff | 100% (latent 100%) | `accuracy.text_sender.mean` = 1.0; `accuracy.correct.mean` = 1.0 | `<CF>/TEXT_BASELINE_PREREGISTRATION.md`; outputs in `<CF>/text_final/` |

### Other values

| Value on the site | Source |
|---|---|
| Bridge of 569,857 parameters | `<CF>/REPORT.md:59` |
| 1,024 new rows (Gate 4) | `<CF>/run_final/analysis.json` field `n_examples` = 1024; `<CF>/REPORT.md:161` |
| E3: 93.8% on 56 never-seen IDs; free-form arm 14.3% | `<E3>/REPORT.md:3-4` and the table at `:12-28`; `<E3>/RESULT.json` |
| The receiver reads the table as text at only 25-30% | `records/research/latent_anchor_program_20260930/RESULTS.md:324-339`, section "receiver competence check (before EXP 1)". The table at lines 330-334 gives 29.9% and 30.4% (dev_known), 25.0% and 25.0% (dev_oob_pure), 28.1% and 28.9% (dev_oob_mixed). Line 339 gives "~25-30%". |
| Opened 2026-09-06 | Channel v1 tree starts after the R.02 run closed and before the Llama seal (`utc` 2026-09-06T05:43:41Z in `records/research/latent_channel_crossfamily_20260906/seal.json`) |
| Updated 2026-10-03 | `<E3>/SEAL.json` field `utc` |

`RESULTS.md` is the results log of the anchor program in scratch mode. It has no seal and no preregistration. The script that made the 25-30% values (`E0_data/smol_text_competence.py`) is not in this package.

## Preregistration

- `<C1>/PREREGISTRATION.md` (Gates 0-3)
- `<V2>/PREREGISTRATION.md` and `<V2>/TEXT_BASELINE_PREREGISTRATION.md`
- `<CF>/PREREGISTRATION.md` and `<CF>/TEXT_BASELINE_PREREGISTRATION.md` (Gate 4)
- `<BV1>/PREREGISTRATION.md`
- `<E3>/PREREGISTRATION.md` and `<E3>/FREEZE.json`

The copy of `<E3>/PREREGISTRATION.md` has one redaction: a direct quote of private chat (line 4) is replaced by `[redacted: private chat]`. Thus the SHA-256 of the copy does not agree with `<E3>/SEAL.json` and `<E3>/FREEZE.json`. In `MANIFEST.json`, `sha256_original` is the sealed hash, and `SCRUB_LOG.json` (`redactions`) gives the position.

## Raw outputs

- `<CF>/run_tiny/` and `<CF>/run_final/` (Gates 0-2 and Gate 4)
- `<CF>/text_final/` (the text-handoff arm of Gate 4)
- `<V2>/run_final/` (Gate 3)
- `records/research/latent_communication_dossier_20260906/evidence/` (per example, per class, donor swaps, paired contrasts)
- `<BV1>/run_tiny/` (binding gate V1)
- `<E3>/RESULT.json` and `<E3>/outputs/` (E3; see "What is missing")
- The inputs of each run: `data/*/examples.json` and `data/*/manifest.json` in each folder.

## Audit report

- `<CF>/independent_core_audit.py`, `<CF>/independent_core_audit.json` and `<CF>/VERIFICATION.json`
- `<V2>/independent_core_audit.py` and `<V2>/independent_core_audit.json`
- `records/research/latent_communication_dossier_20260906/AUDIT_AND_LIMITATIONS.md`
- `<BV1>/audit.py` and `<BV1>/audit.json`
- `<E3>/audit.py` and `<E3>/AUDIT.json`

These audits are second scripts in the same project. No outside party ran them.

## Code

- `<C1>/channel.py`
- `<V2>/channel_v2.py` and `<V2>/final_v2.py`
- `<CF>/channel_cf.py` and `<CF>/final_cf.py`
- `<BV1>/binding.py` and `<BV1>/run.py`
- `<E3>/e3.py`, with the program files that it imports: `records/research/latent_anchor_program_20260930/lab.py`, `sweep.py`, `E0_data/make_splits.py`, `E0_data/capture.py` and `E0_data/text_arms.py`

**Caution:** `lab.py` and `sweep.py` in this package are **not** the versions that E3 ran. Their hashes do not agree with `<E3>/FREEZE.json`. Both files changed on 2026-10-03, after the E3 seal (16:37:01Z). The frozen versions are not in the snapshot. The other three program files agree with the freeze. `MANIFEST.json` shows this in the field `matches_freeze`.

## Failed attempts

These folders are included in full (text files only), because they hold the negative results:

- `records/research/latent_channel_v1_20260906/` (channel v1, failed Gate 1)
- `records/research/latent_channel_crossfamily_20260906/` (Llama-3.2-1B-Instruct receiver, failed the tiny gate)
- `records/research/receiver_qualification_20260906/` (the receiver qualification that selected SmolLM2-360M-Instruct)

## What is missing and why

`MANIFEST.json` lists each file that is not included, with its SHA-256, its size and the reason. For this entry, 279 files (1,210 MB) are not included:

| Reason | Files | Size | Status |
|---|---|---|---|
| Large binary captures (`data/*/sender.pt`) | 12 | 863 MB | Not included, available on request |
| Other binary files (checkpoints `*.pt`, saved output tensors) | 80 | 100 MB | Not included, available on request |
| Files larger than 20 MB (`<E3>/outputs/final_mixed.pt`) | 1 | 34 MB | Not included, available on request |
| Subfolders outside the file list of this entry: `run_full/`, `run_tiny/` (v2), `text_dev/`, `text_tiny/`, `dev_reproduction/`, `sender_reproduction/`, `confirmation_job/`, `development_job/`, `tiny_job/`, `initial_code/`, `<E3>/captures/` | 182 | 214 MB | Not included, available on request |
| Agent session notes (`SESSION_HANDOFF.md`) | 4 | < 0.1 MB | Not included |

Other gaps:

- **E3 raw outputs.** The per-row scores of E3 are binary files (`<E3>/outputs/*.pt`). `<E3>/RESULT.json` holds all the reported numbers. `<E3>/audit.py` recomputes them from the binary files.
- **Hashes for E3 and the anchor program.** These folders are newer than the recovery (2026-09-30), so the recovery manifest has no lines for them. `<E3>/SEAL.json` and `<E3>/FREEZE.json` are the integrity records.
- **Model weights.** The models are not copied. `LICENSES.md` gives the pinned revisions.
- **Binaries and pickles.** Saved PyTorch files can run code when you load them. This is one more reason that this package contains text only.

## Historical copy

This folder is a copy of records from a recovered historical snapshot of the old Ember project. It is not a runnable environment.

- The code refers to paths, virtual environments and model caches that do not exist here.
- The scrub changed local paths in 77 files of this entry. Code paths are now relative to `records/` or start with `<local>`.
- One file of this entry has a redaction (`<E3>/PREREGISTRATION.md`, see "Preregistration").
- Many scripts need the binary files that are not included.

Use the code to read the method. Do not expect it to run without changes.
