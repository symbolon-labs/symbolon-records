# R.09 Split learning at equal bits: records

These are the records behind the entry [R.09 Split learning at equal bits](https://symbolonlabs.com/research/split-equal-bits/) on the Symbolon site. All paths below are relative to the root of this repository.

## What the record shows

This entry is not about language models. It is a split-learning test on CIFAR-10 images. A client runs a frozen part of a small image network and sends one fixed-length message for each image to a server. At an equal cap of bits, the test compares a symbolic message with learned activation messages on two axes: test accuracy, and leakage. Leakage is the PSNR of the image that a trained decoder rebuilds from the message. A higher PSNR means that more of the input leaks.

- **Sealed study (3 seeds, cut C2, cap of 64 bits).** The symbolic message TOPK@64 (58 bits) is 13.67 points more accurate than the learned bottleneck BOT@64 (64 bits), with a 95% CI of 13.09 to 14.27. BOT@64 leaks 0.550 dB more, with a CI of 0.527 to 0.572. The preregistered rule gives case S, "symbolic dominates".
- **A thin leakage margin.** The margin is 0.5 dB. Seed 2 alone gave 0.489 dB. The CI came from a bootstrap over test images only, without the variance between seeds.
- **Replication (6 new seeds, Decoder-L, two-level CI).** The leakage gap is 0.609 dB, with a CI of 0.547 to 0.664 over seeds, then images. The reading is R1, "L+ replicates". One seed of six is below 0.5 dB (0.480). Decoder-S gives 0.615 dB. The accuracy gap is 13.85 points.
- **Other budgets (secondary rows).** TOPK is at least as accurate as the activation messages at each budget. At 1,024 bits, BOT has the accuracy of TOPK (148 bits), and it leaks 3.8 dB more.
- **Audits.** For each study, a second script recomputed the values from the saved files. Both audits passed.

Limits that the records state:

- The leakage values are lower bounds. They come from one decoder family that learns from 20,000 pairs, and stronger attackers exist (`records/research/split_equal_bits_20261023/REPORT.md:8` and `:144`).
- The bottleneck was weaker than the design expected: 13.7 points below TOPK, not about 3 points (`REPORT.md:143` in the same folder).
- The study has one data set, one model family, a frozen client and forward messages only (`REPORT.md:146`). The replication changed only the seeds, not the data split (`records/research/split_leak_replication_20261024/REPORT.md:109`).
- PSNR measures pixel similarity only (`records/research/split_equal_bits_20261023/REPORT.md:145`).
- A symbolic message is not private. TOPK@64 still leaks 0.36 dB above the floor of the true label (`REPORT.md:136` in the same folder).

## Site values and their sources

This table follows the evidence map of the site (`design/ENTRY_R09_EVIDENCE.md` in the site repository). `<R>` is `records/research`, `<SE>` is `<R>/split_equal_bits_20261023` (the sealed study) and `<SL>` is `<R>/split_leak_replication_20261024` (the replication).

The site shows the first 12 hex digits of the SHA-256 of the **original** result file. For each file, `MANIFEST.json` gives `sha256_original` and `sha256_copy`. The two result files of this entry are not scrubbed, so the two hashes are equal.

### Gates

| Gate on the site | Site hash | Result file | Copy scrubbed | Rule source |
|---|---|---|---|---|
| Validity checks, sealed study (passed) | `c4f91b4430ba` | `<SE>/results/results.json` (fields `checks`, `check_failures`); the records are in `<SE>/results/s<seed>/checks.jsonl` | no | `<SE>/PREREGISTRATION.md:256-270` (Part 7.3) |
| Accuracy A+ (passed) | `c4f91b4430ba` | `<SE>/results/results.json` (fields `primary.d_acc`, `d_acc_ci`, `d_acc_seeds`, `A`) | no | `<SE>/PREREGISTRATION.md:246-250` and `:274-278` |
| Leakage L+ (passed) | `c4f91b4430ba` | `<SE>/results/results.json` (fields `primary.d_leak`, `d_leak_ci`, `d_leak_seeds`, `L`) | no | `<SE>/PREREGISTRATION.md:280-285` |
| Decoder dependence (passed) | `c4f91b4430ba` | `<SE>/results/results.json` (fields `decoder_check`, `primary.case_final`) | no | `<SE>/PREREGISTRATION.md:287-302` (Parts 7.5 and 7.6) |
| Validity checks, replication (passed) | `5f66717e496a` | `<SL>/results/results.json` (fields `checks`, `failed_records_all`, `valid`) | no | `<SL>/PREREGISTRATION.md:75-84` and `:118` |
| Replication R1 (passed) | `5f66717e496a` | `<SL>/results/results.json` (fields `primary.d_leak`, `d_leak_ci`, `d_leak_seeds`, `reading_final`) | no | `<SL>/PREREGISTRATION.md:101-108` and `:116-134` |
| Replication, accuracy (passed; secondary) | `5f66717e496a` | `<SL>/results/results.json` (fields `accuracy.d_acc`, `d_acc_ci`, `outcome`) | no | `<SL>/PREREGISTRATION.md:136` |

The sealed report gives the same values in `<SE>/REPORT.md:16-23`. The replication report gives them in `<SL>/REPORT.md:16-25`.

### Controls

All values are means over the 3 sealed seeds on the 10,000 test images, unless a row says otherwise.

| Arm on the site | Site value | Source |
|---|---|---|
| Shuffled pairs (V2; seed 1, development data) | BOT 12.672 dB, TOPK 12.669 dB (limit 12.881 dB) | `<SE>/results/s20261111/checks.jsonl`, the two V2 records (`dev_psnr`; `F0` = 12.6806 on D_dev, so the limit F0 + 0.2 is 12.881); `<SE>/REPORT.md:54`. Rule: `<SE>/PREREGISTRATION.md:263` and `:269` |
| No message (F0) | 12.693 dB | `<SE>/results/results.json` field `floors.F0.psnr` = 12.6928. Method: `<SE>/PREREGISTRATION.md:212-218` |
| True label only (FY) | 13.065 dB (TOPK 13.427, BOT 13.976) | `floors.FY.psnr` = 13.0651; `descriptive.TOPK@64.psnr` = 13.4265; `descriptive.BOT@64.psnr` = 13.9762 |
| Label message (LAB) | 13.092 dB at accuracy 0.826 (TOPK 0.826) | `descriptive.LAB.psnr` = 13.0922; `descriptive.LAB.acc` = 0.8263; `descriptive.TOPK@64.acc` = 0.8263 |
| No bit cap (RAW-native) | 22.24 dB at accuracy 0.879 | `descriptive.RAW_native.psnr` = 22.2378; `acc` = 0.87937; `bits` = 32768. Arm: `<SE>/PREREGISTRATION.md:102` |

### Other values

| Value on the site | Source |
|---|---|
| CIFAR-10; 10,000 test images | `<SE>/PREREGISTRATION.md:43-48`; class counts in `<SE>/FREEZE.json` (`data.class_counts`) |
| Cut C2; the client is the frozen stages 1 and 2 of the network | `<SE>/PREREGISTRATION.md:15` and `:77` |
| TOPK@64: the first class and the gaps to the other nine class scores, 58 bits | `<SE>/PREREGISTRATION.md:161` and `:168` (format F-all(6)); `<SE>/selection.json` (`TOPK.64`); `<SE>/REPORT.md:14` |
| BOT@64: 16 channels, 2×2 grid, 1 bit, 64 bits | `<SE>/selection.json` (`BOT.64` = `BOT_c16_s2_q1`); `<SE>/REPORT.md:14` |
| Accuracy 0.826 against 0.690 | `<SE>/results/results.json` fields `descriptive.TOPK@64.acc` = 0.8263 and `descriptive.BOT@64.acc` = 0.6896 |
| Decoder-S and Decoder-L; one family that learns from 20,000 pairs | `<SE>/PREREGISTRATION.md:198-201` |
| Decoder-L seed 1: +0.504 dB; Decoder-S seed 1: +0.512 dB | `<SE>/results/results.json` fields `decoder_check.d_leak_L_seed1`, `d_leak_S_seed1` |
| Replication, Decoder-S: 0.615 dB | `<SL>/results/results.json` field `secondary.decoder_S_new_seeds.d_leak` = 0.6149; `<SL>/REPORT.md:36` |
| CI width 0.117 dB (two-level) against 0.045 dB (images only) | `<SL>/REPORT.md:42`; from `primary.d_leak_ci` and `secondary.sealed_rule_L.d_leak_ci` in `<SL>/results/results.json` |
| At 1,024 bits: same accuracy, BOT leaks 3.8 dB more | `<SE>/results/results.json` field `secondary.TOPK_vs_BOT@1024` (`d_acc` = 0.03, CI -0.40 to 0.46; `d_leak` = 3.805); `<SE>/REPORT.md:79` and `:85` |
| TOPK at least as accurate at each budget | `<SE>/REPORT.md:85`; the secondary rows at `:75-84` |
| Label alone at 64 bits: 0.33 dB less leakage, NLL worse by 0.26 nats | `<SE>/results/results.json` field `secondary.LAB_vs_TOPK@64` (`d_leak` = 0.334; `nll.d_nll` = 0.2554); `<SE>/REPORT.md:86` and `:134` |
| TOPK@64 leaks 0.36 dB above the floor of the true label | `<SE>/results/results.json` field `descriptive.TOPK@64.excess_over_FY` = 0.361; `<SE>/REPORT.md:136` |
| The bottleneck was weaker than the design expected | `<SE>/REPORT.md:143` and `:167` |
| Cut C1 and the end-to-end client did not run | `<SE>/PREREGISTRATION.md:19`; `<SE>/REPORT.md:158` |
| Opened 2026-10-23 | `<SE>/FREEZE.json` field `time` = 2026-10-23 21:26:40 (the first freeze) |
| Updated 2026-10-25 | `<SL>/REPORT.md:3` (date of the replication report) |

## Preregistration

- `<SE>/PREREGISTRATION.md` and `<SE>/FREEZE.json`. The freeze is from 2026-10-23 21:26:40, before job J0. The rule of the primary pair is Part 7.
- `<SE>/selection.json`: the selection of each configuration on development data, by accuracy and NLL only. Leakage had no part in the selection (`<SE>/PREREGISTRATION.md:236`).
- `<SL>/PREREGISTRATION.md` and `<SL>/FREEZE.json`. The freeze is from 2026-10-24 18:39:26, before the first job. The differences from the sealed study are in Part 3 (D1 to D8). The reading rule is Part 7.
- `<SL>/selection.json` is a byte copy of `<SE>/selection.json`. The replication selects nothing.

The sealed preregistration and the sealed report name a design document (`reports/split_equal_bits_design_20261023.md`). It is not in this package. `<SE>/PREREGISTRATION.md` copies its rule (Part 7) and lists each difference from it (Part 10).

Two files of this entry have redactions: a direct quote of private chat is replaced by `[redacted: private chat]`.

| File | Line |
|---|---|
| `<SE>/PREREGISTRATION.md` | 429 |
| `<SL>/PREREGISTRATION.md` | 214 and 215 |

In both files, and in other process files, the name of the researcher is also replaced by `[researcher]` (see "Source of this copy"). Thus the SHA-256 of these copies does not agree with `FREEZE.json` or `SEAL.json` of their folder. In `MANIFEST.json`, `sha256_original` is the frozen and sealed hash. `SCRUB_LOG.json` gives each position. No number changed.

## Raw outputs

- `<SE>/results/results.json` and `<SE>/results/table.md`: all test metrics, the rule and the case (job J6).
- `<SE>/results/acc_psnr.svg`: a plot of accuracy against PSNR for all arms and budgets.
- `<SE>/results/s20261111/`, `s20261112/` and `s20261113/`: for each seed, `N.json`, `arms/*.json` (development values, bits and temperature of each arm), `dec/*.json` (decoder histories and development PSNR), `checks.jsonl`, `times.jsonl`, `topk_formats.json`, `raw_rank.json` and `done/`.
- `<SE>/results/j0/`, `<SE>/j0_result.json`, `<SE>/times.json` and `<SE>/job_plan.json`: the scratch job J0 (unit tests, time projections).
- `<SL>/results/results.json` and `<SL>/results/table.md`: all values of the replication and its reading.
- `<SL>/results/s20261121/` to `s20261126/`: the same layout for each new seed.
- The logs of each job: `<SE>/logs/` and `<SL>/logs/`.

The per-image values are in binary files that are not included: the reconstructions (`dec/*.npz`), the messages, predictions and logits (`msgs/*.npz`). The result files hold all the numbers that the site gives. The two audits recompute these numbers from the binary files.

## Audit report

- `<SE>/AUDIT.md`, `<SE>/audit/stats_check.py` and `<SE>/audit/stats_check.json`. All 92 rows agree with `results/results.json`. The largest float difference is 3.6e-15. Verdict: PASS.
- `<SL>/AUDIT.md`, `<SL>/audit/recheck.py`, `<SL>/audit/recheck.json` and `<SL>/audit/table_rows.md`. All 343 compared values agree. The largest float difference is 2.0e-13. All 15 file hashes agree with `FREEZE.json`. Verdict: PASS.
- The seals: `<SE>/SEAL.json` (2026-10-24 16:36:00, 562 files) and `<SL>/SEAL.json` (2026-10-25 09:44:20, 308 files). Each seal lists the SHA-256 of each file of its folder, with the binary files that are not included here.

Each audit script uses numpy and the Python standard library only. An agent session wrote each script from the written rules and the file formats. It did not read or import the main code (`<SE>/AUDIT.md:7`; `<SL>/AUDIT.md:9-15`). These audits are second scripts in the same project. No outside party ran them.

## Code

- `<SE>/sebits.py`: data, models, arms, the bit code (pack and unpack), decoders, probes, analysis and jobs.
- `<SE>/test_sebits.py`: 44 unit tests on synthetic data. `<SE>/run_sebits.sh`: the job wrapper.
- `<SL>/sleak.py`: the driver of the replication. It imports `<SL>/sebits.py`, a byte copy of the sealed file, and changes only two module paths.
- `<SL>/test_sleak.py` (12 unit tests), `<SL>/test_run_sleak.zsh`, `<SL>/run_sleak.sh` and `<SL>/wait_and_run.sh`: tests, the job wrapper and the start script.

**Caution:** the copies of both `sebits.py` files, `run_sebits.sh` and `run_sleak.sh` each have one name replacement in a comment. Thus their SHA-256 does not agree with `FREEZE.json`. In `MANIFEST.json`, `sha256_original` agrees with the freeze (`db0f25b468ee` for `sebits.py`). The two copies of `sebits.py` have the same SHA-256 (`6ea826210a32`).

The pre-freeze review of the replication found a defect in the zsh wrapper: after a pause and a resume, `wait` loses the exit code of the job. The sealed `run_sebits.sh` has the same defect. No job paused in either run, so no result changed (`<SL>/REPORT.md:121-122`).

## What is not included and why

`MANIFEST.json` lists each file that is not included, with its SHA-256, its size and the reason. For this entry, 354 files (4,216 MB) are not included:

| Reason | Files | Size | Status |
|---|---|---|---|
| Binary files: decoder reconstructions of the test images (`dec/*.npz`) | 102 | 3,030 MB | Not included. Not available on request: the preregistration allows no rebuilt image beyond small figure panels |
| Binary files: messages, predictions and logits (`msgs/*.npz`) | 76 | 707 MB | Not included, available on request |
| Binary files: model weights (`N.pt`, `arms/*.pt`) | 171 | 125 MB | Not included, available on request |
| Binary files: the CIFAR-10 archive and the three npz files made from it (`<SE>/data/`) | 4 | 354 MB | Not included. Get CIFAR-10 from its source; `<SE>/FREEZE.json` gives the hashes |
| Agent session note: the approval record of the sealed study (`<SE>/APPROVALS.md`), which quotes private chat | 1 | < 0.1 MB | Not included |

Other gaps:

- **The data.** CIFAR-10 is a public data set. `<SE>/FREEZE.json` (field `data`) gives the archive address, its size, md5 and SHA-256, and the SHA-256 of the three npz files. The preregistration treats the images as research-only data and allows no rebuilt image beyond small figure panels (`<SE>/PREREGISTRATION.md:28`).
- **Not run.** Cut C1 (jobs J7, J8a and J8b) and the end-to-end variant (J9) are not in the frozen plan. They did not run.
- **Model weights.** The models are small image networks that the lab trained in these runs. Their weights are binary files (see the table).
- **Binaries and pickles.** Saved PyTorch files can run code when you load them. This is one more reason that this package contains text only.

## Source of this copy

These folders are copies of two study folders of the Ember project, from the 2026-10 snapshot. `sha256_original` is the hash on 2026-10-25. Both folders are sealed. The copy is not a runnable environment.

- The code refers to paths, virtual environments and data files that do not exist here.
- The scrub changed 15 files of this entry. In 11 files, the name of the researcher became `[researcher]` (rule R14). In 4 files, local paths became relative paths or `<local>` (rules R5 and R12). `SCRUB_LOG.json` lists each change.
- Two files of this entry have a redaction (see "Preregistration").
- Some scripts and logs name the folders of other studies as start conditions of the job chain. Those studies are not part of this entry.
- Most scripts need the binary files that are not included.

Use the code to read the method. Do not expect it to run without changes.
