# PREREGISTRATION: replication of the leakage result of the split-learning study (CIFAR-10)

**Status: PREREGISTRATION. `FREEZE.json` freezes this file before the first job.** After the freeze, a change to a frozen file (Part 11) stops every job, because each job checks the sha256 values.

- Written: 2026-10-24, by a Claude Code session for [researcher] (Symbolon Labs). Style: ASD-STE100.
- Sealed study (below: "the sealed study"): `../split_equal_bits_20261023/` (`PREREGISTRATION.md`, `REPORT.md`, `SEAL.json`). This replication changes no file in that folder.
- Mode: confirmatory (`ember/AGENT.md`, "Research modes"). The configuration comes from the D_dev selection of the sealed study. This run selects nothing.
- Purpose: defensive measurement. The test measures how much of a client's input a trained decoder can rebuild from each message. It uses a public data set and the lab's own small models. It does not attack a real system.

---

# Part 1. Summary

- **Question.** Does the leakage half of case S hold with new seeds, a larger decoder and a confidence interval that includes the variance between seeds?
- **Claim under test** (sealed `REPORT.md`, section 1). At an uplink cap of 64 bits, BOT@64 leaks +0.550 dB more than TOPK@64 (Decoder-S, 3 seeds; CI +0.527 to +0.572 from an image bootstrap only; seeds +0.512, +0.489, +0.647; margin 0.5 dB).
- **Primary pair.** TOPK@64 (F-all(6), 58 bits) against BOT@64 (BOT (16, 2, 1), 64 bits), at cut C2.
- **Primary decoder.** Decoder-L (the larger decoder of the sealed study).
- **Seeds.** Six new run seeds: 20261121 to 20261126.
- **Primary statistic.** The pooled Δleak over the six seeds and the 10,000 test images, with a two-level bootstrap CI (seeds, then images).
- **Primary rule.** Part 7: R1 "L+ replicates", R2 "leak advantage, below margin", R3 "no leak difference shown", R4 "reversed".
- **Accuracy.** A secondary row (Part 8).
- **Jobs.** Six seed jobs on MPS (about 28 to 40 min each; limit 100 min) and one analysis job on the CPU (limit 30 min). Total about 3.0 h of laptop time.

---

# Part 2. What stays the same as the sealed study

- **Code.** `sebits.py` in this folder is a byte copy of the sealed file (sha256 `db0f25b468ee55046a4ab94b3398c875b5157409f6c2577fef0496045de08696`). The sealed study froze it on 2026-10-23 at 21:26:40, after 44 unit tests passed. The driver `sleak.py` imports this copy. It calls these functions without a change: `unit_N`, `check_v6`, `unit_head`, `unit_topk`, `unit_bot`, `unit_msgs`, `unit_decoder`, `run_units`, `load_data`, `psnr_u8`, `compare`, `boot_means`, `ci95`. It changes two module paths only: `DATA_DIR` (the sealed data folder) and `RES` (`results/` of this folder).
- **Selection.** `selection.json` in this folder is a byte copy of the sealed file (sha256 `b5a24e75a717249de91e0b5f82d1ef9f453e76070d151db8c524b789cd0ac907`). It is the D_dev selection of the sealed J1 (seed 20261111). TOPK@64 is F-all(6) (g_fill multiplier 0.0; F-all has no fill). BOT@64 is BOT_c16_s2_q1. `unit_topk` also fits the TOPK formats at 16 and 256 bits; this run does not use them.
- **Data.** The npz files of the sealed study, in `../split_equal_bits_20261023/data/`, read only. Their sha256 values come from the sealed `FREEZE.json`. Split seed 20261110: the same D_task (35,000), D_dev (5,000), D_aux (10,000) and test (10,000) sets. No download.
- **Recipes.** N, the TOPK head, BOT and the decoders train as in sealed Parts 3 and 5. N: 30 epochs of SGD. Head and BOT: 15 epochs of AdamW. Decoders: 20 epochs of Adam on D_aux and its flips (20,000 pairs), with the epoch of the lowest D_dev MSE. The quantizer cells, the temperatures and the LAB confusion rows are fit again in each seed, as in sealed seeds 2 and 3.
- **Devices.** MPS for the seed jobs (`PYTORCH_ENABLE_MPS_FALLBACK=0`). CPU for the analysis. MPS results are not bit-exact between runs.

---

# Part 3. Differences from the sealed study

| # | Sealed study | This replication | Reason |
|---|---|---|---|
| D1 | The primary uses Decoder-S | The primary uses Decoder-L (twice the channels: 512, 256, 128, 64) | The task brief. A stronger decoder is the more careful attacker. In sealed seed 1, Decoder-L gave +0.504 dB and Decoder-S +0.512 dB. |
| D2 | Seeds 20261111 to 20261113 | Six new seeds, 20261121 to 20261126 | New seeds, and more of them |
| D3 | CI from a bootstrap over test images only | CI from a two-level bootstrap: seeds, then images (Part 6.3) | The sealed CI left out the variance between seeds |
| D4 | L+: Δleak ≥ 0.5 dB, CI low > 0, each seed > 0 | R1 to R4 (Part 7) | The task brief. The sealed rule stays as a secondary row (Part 8, row 3). |
| D5 | All arms at 4 budgets, reference rows, probes, SSIM | TOPK@64, BOT@64 and LAB only (LAB for V1). No RAW, TOPK-full, LAB-full, RAW-native or TOPK-float decoder; no probe; no SSIM. | Only the primary pair is under test |
| D6 | V1 with Decoder-S in each seed; V2 with Decoder-S in seed 1 | V1 with Decoder-S and with Decoder-L in each seed; V2 with Decoder-L in the first new seed | The primary decoder is Decoder-L. The Decoder-L V1 uses the closed form of `sebits.unit_decoder` (the same formula), in `sleak.v1_large`. |
| D7 | Bootstrap seed 20261024 | Two-level bootstrap seed 20261127. The sealed-rule rows keep 20261024 (inside `sebits.compare`). | A new procedure gets a new seed |
| D8 | Seeds 2 and 3 run `unit_prep`, `unit_full` and the RAW units | These units do not run | They do not touch the primary arms. Each unit seeds its own generators by its tag (sealed Part 9, item 31), so the kept units do not depend on them. |

---

# Part 4. Seeds

- **Run seeds:** 20261121, 20261122, 20261123, 20261124, 20261125, 20261126. They are fixed now.
- A search on 2026-10-24 of `ember/research/`, `reports/` and `research_notes/` found no file that contains one of these numbers or 20261127.
- Each run seed sets the initialization and the data order of N, of the head, of BOT and of each decoder, through `sub_seed(run_seed, tag)` (sealed Part 9, item 31).
- Split seed 20261110 stays (Part 2). Two-level bootstrap seed: 20261127.
- No seed is replaced. Part 9.3 gives the stop rules.

---

# Part 5. Units of each seed (`sleak.seed_units`)

1. `N`: `unit_N`, then V6 (the D_dev accuracy of N must be 0.80 or more; a failure is a stop rule).
2. `head`: `unit_head` (the TOPK head, and the TOPK-float temperature on D_dev).
3. `topk`: `unit_topk` with the sealed selection (TOPK formats and LAB).
4. `BOT_c16_s2_q1`: `unit_bot`.
5. `msgs`: `unit_msgs` for TOPK@64, BOT@64 and LAB, on D_aux, D_aux flipped, D_dev, D_dev flipped, test, test flipped and the membership set. V4 on each message.
6. Decoders, in this order: `L1_TOPK@64`, `L1_BOT@64`, `L1_LAB` (then V1 for Decoder-L), `S1_TOPK@64`, `S1_BOT@64`, `S1_LAB` (V1 for Decoder-S inside `unit_decoder`).
7. First new seed only (20261121): `L1_BOT@64_ctrl` and `L1_TOPK@64_ctrl` (V2 with Decoder-L: shuffled message-image pairs).

- V3 (frozen client) runs before and after each unit, as in the sealed study.
- Each unit writes a "done" marker. A rerun skips finished units.
- Projected times, from the sealed `times.jsonl` files: N 2.6 min, head 0.6 min, topk 0.1 min, BOT 1.1 min, messages 0.3 min, each Decoder-L 5.7 min, each Decoder-S 2.0 min. A seed job takes about 28 min; the first seed job about 40 min.

## 5.1 Validity checks

| Check | Condition | Where |
|---|---|---|
| V1 decoder | Decoder-S on LAB and Decoder-L on LAB each get at least PSNR(closed form) − 0.2 dB on D_dev | Each seed |
| V2 content-free | A Decoder-L trained on shuffled pairs gets at most PSNR(F0) + 0.2 dB on D_dev, for BOT@64 and TOPK@64 | First new seed |
| V3 frozen client | The sha256 of stages 1 and 2 does not change in any unit | Each seed |
| V4 bits | Every message has its declared length, and unpack(pack(m)) = m | Each seed |
| V5 finish | Every planned unit has its "done" marker and every planned decoder output exists; no loss is non-finite | Each seed |
| V6 base model | The D_dev accuracy of N is at least 0.80 | Each seed |

---

# Part 6. Measurement and statistics

## 6.1 Per image

- P(A, s, i): the PSNR of the uint8 test reconstruction of the rep-1 decoder of arm A in seed s, for test image i (`sebits.psnr_u8`; MSE over the 3×32×32 values in [0, 1]).
- c(A, s, i): 1 if the prediction of arm A is the test label, else 0. TOPK predicts the first class of its message. BOT predicts the first class of a stable descending sort of the server logits (sealed Part 9, item 18).

## 6.2 Differences

- D(s, i) = P(BOT@64, s, i) − P(TOPK@64, s, i), with Decoder-L. A positive value means that BOT leaks more.
- **Δleak** = the mean of D over the six seeds and the 10,000 test images. Δleak_s = the mean over the images of seed s.
- **Δacc** = 100 × the mean of [c(TOPK@64, s, i) − c(BOT@64, s, i)], in points.

## 6.3 Two-level bootstrap

- Generator: `numpy.random.default_rng(20261127)`. 10,000 draws in 40 chunks of 250 draws.
- In each chunk, first the seed indices `si = rng.integers(0, K, size=(250, K))`, then the image indices `ii = rng.integers(0, n, size=(250, n))`.
- The value of draw b is the mean over k = 1 to K of the mean over j = 1 to n of D(si[b, k], ii[b, j]). Thus one image draw serves all the drawn seeds of that draw.
- The 95% CI is the 2.5 and 97.5 percentiles of the 10,000 draw values (numpy linear method).
- K = 6 and n = 10,000 for the primary row. The same draws serve Δleak (Decoder-L), Δacc and Δleak (Decoder-S, new seeds). The pooled row with nine seeds (Part 8, row 4) uses the same procedure with K = 9.
- Reason for one image draw for all seeds: the test images are the same in each seed. A shared image draw keeps the image effect that all seeds have in common.

## 6.4 A check of the bootstrap with few seeds (descriptive)

- The standard deviation of the six Δleak_s values, and a Student t 95% interval over them (5 degrees of freedom, t = 2.5706). This interval is not part of the rule.

---

# Part 7. Reading rule (primary; fixed before any job)

**Validity first.** If V1, V2, V3, V4, V5 or V6 fails, the reading is **void**. The report then gives the numbers as descriptive values only. Exception: if V5 fails (a planned output is absent), the analysis computes no comparison, as in the sealed J6.

**Then the reading of Δleak (Decoder-L) and its two-level 95% CI:**

| Reading | Condition |
|---|---|
| **R1: L+ replicates** | Δleak ≥ 0.5 dB and CI lower bound > 0 |
| **R2: leak advantage, below margin** | CI lower bound > 0 and Δleak < 0.5 dB |
| **R3: no leak difference shown** | CI lower bound ≤ 0 and CI upper bound ≥ 0 |
| **R4: reversed (BOT leaks less)** | CI upper bound < 0 |

**Wording in the report.**

- R1: "The leakage half of case S replicates with six new seeds and Decoder-L."
- R2: "TOPK@64 leaks less than BOT@64, but the difference is below the 0.5 dB margin. The L+ label of the sealed study does not replicate."
- R3 and R4: "The leakage half of case S does not replicate."
- In each case, the report also says whether the CI lower bound is at or above 0.5 dB. This is descriptive only.

**Accuracy (secondary).** "A+ replicates" if the two-level CI lower bound of Δacc is above −1.0 point.

Nothing in this rule changes after a test metric exists. A later change is a new, dated section.

---

# Part 8. Secondary rows (labeled secondary or descriptive; no claim beyond the label)

1. Δacc with the two-level CI, and the A+ outcome.
2. Δleak with Decoder-S on the six new seeds, with the two-level CI and its R label.
3. The sealed rule on the new seeds (`sebits.compare`: image bootstrap with seed 20261024; L+ needs Δleak ≥ 0.5, CI low > 0 and each seed > 0), for Decoder-L and for Decoder-S.
4. Δleak with Decoder-S, pooled over the three sealed seeds and the six new seeds (K = 9), with the two-level CI. Descriptive. The analysis reads the sealed reconstructions `S1_TOPK@64.npz` and `S1_BOT@64.npz` and does not change them.
5. The decoder gap: Δleak_s(L) − Δleak_s(S) for each seed, and PSNR(L) − PSNR(S) for each arm.
6. Descriptive values: test accuracy and PSNR (Decoder-L and Decoder-S) of TOPK@64, BOT@64 and LAB in each seed; the D_dev accuracy of N; the best decoder epochs; the V1 and V2 values; the floors F0 and FY on test; the sealed seed-1 Δleak with Decoder-L (reference).

---

# Part 9. Jobs, stop rules and operations

## 9.1 Order

1. **Freeze.** `sleak.py freeze` checks the two byte copies and the three npz hashes. It runs the 12 unit tests of `test_sleak.py` (synthetic data, CPU, about 75 s). They include an end-to-end run of two seed jobs and the analysis with 1 epoch for each part, a test of the check collection, and `test_run_sleak.zsh` (16 cases of the wrapper and the waiter, with a shell command in place of a job and a file in place of `pmset`). Then it writes `FREEZE.json`.
2. **Wait.** `wait_and_run.sh` runs in the background. It waits until `../bridge_safety_deep_20261024/DONE` exists and no Python process of that folder runs (a process whose command names the folder, or whose current directory is in the folder). The bridge-safety preregistration writes `DONE` at the end, also if its work stops early. Then the waiter waits 120 s more. If it has not started after about 24 h (1,440 checks), it stops with return code 7. A lock prevents a second waiter.
3. **Jobs.** The six seed jobs in the order of Part 4, then the analysis. Each job runs through `run_sleak.sh`. The waiter stops at the first job with a return code other than 0.
4. **After the analysis.** An independent recomputation (`AUDIT.md`), then `REPORT.md`, then `SEAL.json`, then the empty file `DONE`.

## 9.2 Machine checks and power guard (`run_sleak.sh`)

- Before each job: no other heavy lab job runs (a Python process whose command names a lab script, a lab venv or a folder of `ember/research/`); the swap use is at most 3,584 MB; the Mac draws from AC power, the battery is at 55% or more, and `pmset -g ac` gives 60 W or more.
- If one of these checks fails, the wrapper does not fail. It waits and checks again every 60 s, for at most 720 min. Then it stops with return code 5. The charger often falls to 8 W when the Mac is idle; this rule lets the job wait for it.
- Hard checks: 10 GiB or more of free disk; the Python of the venv exists. A failure stops the wrapper at once.
- During the job: `caffeinate -i -w <pid>`; logs with `tee -a`. Every 20 s, a guard pauses the job (SIGSTOP) if the Mac is on battery power, the battery is at 50% or less, or the charger gives less than 60 W. It resumes the job (SIGCONT) when the Mac is on AC power, the battery is at 55% or more and the charger gives 60 W or more.
- While the job is paused, the guard writes a line every 10 min. After 120 min in one pause, the guard stops the job (SIGTERM, return code 125). Reason: a paused job makes the Mac idle, and the charger possibly does not come back to 60 W by itself.
- The time limit counts running time only: 100 min for a seed job, 30 min for the analysis. At the limit the guard stops the job (return code 124). If the battery falls below 20%, the guard stops the job (return code 126).
- Exit code: zsh loses the exit code of a job that was paused. Thus `sleak.py` writes its exit code into a file (`SLEAK_EXIT_FILE`), and the wrapper uses that file.
- A lock (`logs/.lock`) prevents a second wrapper. Another `sleak.py` job also counts as another heavy job.

## 9.3 Stop rules

- **V6.** If N fails V6 in a seed, the job stops. The session reports to [researcher] and waits. A replacement seed needs a dated amendment and [researcher]'s word.
- **Errors and checks.** If the code raises an error, or V3, V4 or a non-finite loss stops a job, the waiter stops. The session records the event, reports to [researcher] and waits. If a rerun of the same frozen code cannot repair the event, the reading is void.
- **Restarts.** After a stop at the time limit, the pause limit, a battery stop, or the wait limit of the wrapper or of the waiter, a restart of the same job with the same frozen code is allowed. The "done" markers make the job continue. `REPORT.md` records each restart.
- **Check records after a rerun.** For each seed, check and place ("where"), the analysis uses only the last record. A rerun of a unit thus replaces the records of the attempt before it. `results.json` also lists every failed record (`failed_records_all`), and `REPORT.md` reports each one.
- **Re-freeze.** A code change after the freeze needs a dated amendment in this file and [researcher]'s word. Then `sleak.py freeze <label>` keeps the old file as `FREEZE_<label>.json`.

## 9.4 Test-label boundary

- The seed jobs open the train file and the test images (`load_data("confirm")`). They compute no test metric.
- Only the analysis opens `cifar10_test_labels.npz`.

## 9.5 Limits of the work

- No download, no commit, no push, no public post. No other experiment.
- No file in `../split_equal_bits_20261023/` changes. The jobs read its npz files and the analysis reads its saved reconstructions.

---

# Part 10. Expectations before the run

- Sealed Decoder-S values: +0.512, +0.489 and +0.647 dB (mean 0.549, SD 0.085). Sealed seed-1 Decoder-L: +0.504 dB.
- We expect Δleak (Decoder-L) of about 0.45 to 0.60 dB, a seed SD of about 0.05 to 0.10 dB, and a two-level CI of about ±0.07 to ±0.10 dB.
- Thus a CI lower bound above 0 is very likely. The choice between R1 and R2 depends on the point estimate near the margin. Our rough prior: R1 about 70%, R2 about 28%, R3 or R4 about 2%.
- Accuracy: Δacc of about +13 points; A+ replicates almost certainly.

---

# Part 11. Files

- Frozen (sha256 in `FREEZE.json`): `PREREGISTRATION.md`, `sebits.py` (byte copy), `selection.json` (byte copy), `sleak.py`, `test_sleak.py`, `run_sleak.sh`, `wait_and_run.sh`, `test_run_sleak.zsh`. `FREEZE.json` also holds the data record of the sealed study and the Python, torch and numpy versions.
- Outputs: `results/s<seed>/` (`N.pt`, `N.json`, `arms/`, `msgs/`, `dec/`, `checks.jsonl`, `times.jsonl`, `done/`); `results/results.json`; `results/table.md`; `logs/`.
- After the analysis: `audit/`, `AUDIT.md`, `REPORT.md`, `SEAL.json`, `DONE`.

---

# Part 12. Approval record

[researcher]'s words, relayed by the parent session:

- 2026-10-23 at about 21:30: [redacted: private chat].
- 2026-10-24 at about 18:10: [redacted: private chat].

The parent session states that this approval covers the GPU jobs of this replication, one heavy job at a time, with the CIFAR-10 data that is already on disk. It covers no download, no other experiment, no commit and no public post.

---

# Part 13. Pre-freeze review

- A separate reviewer (a code-review subagent of this session) read this file, `sleak.py`, `test_sleak.py`, `run_sleak.sh`, `wait_and_run.sh` and the `sebits.py` functions that `sleak.py` calls. It ran the unit tests (10 of 10 passed), `zsh -n` on both scripts and `run_sleak.sh --check-only`. It found 1 blocker, 3 items to fix and 3 minor items. It confirmed the bootstrap, the rule, the signs, the unit list, the closed form of V1 and the test-label boundary.
- All 7 items were fixed before the freeze:
  1. Blocker: after a SIGSTOP and a SIGCONT, zsh `wait` gives 145, not the exit code of the job. A paused job that ends with 0 would stop the waiter. Fix: `sleak.py` writes its exit code into `SLEAK_EXIT_FILE`; the wrapper reads it (Part 9.2). The session reproduced the defect before the fix.
  2. The guard had no pause limit and wrote no line while paused. Fix: a line every 10 min and a stop after 120 min in one pause (Part 9.2).
  3. The waiter had no maximum wait. Fix: a stop after about 24 h (Part 9.1). `DONE` stays the start signal: the bridge-safety preregistration writes it also when its work stops early.
  4. The heavy-job check did not count another `sleak.py` job, and no lock prevented two wrappers. Fix: `sleak.py` counts as a heavy job; a lock in the wrapper and in the waiter.
  5. A failed check record stayed after a rerun that repaired the event. Fix: the last record for each seed, check and place counts; every failed record is reported (Part 9.3).
  6. No test reached a valid check set. Fix: a test of `collect_checks` with synthetic records (full set; absent V1, V2 and V6 records; an absent done marker; a replaced failed record).
  7. With V5 failed, the analysis computes no numbers. Fix: Part 7 states this exception.
- The new self-test `test_run_sleak.zsh` found one more defect before the freeze: the waiter compared the current directory of a process with a path that had a symbolic link (`/var` against `/private/var`). Fix: the waiter uses the real path of the watched folder.
- After the fixes: 12 unit tests pass on the CPU (16 of 16 shell cases).
