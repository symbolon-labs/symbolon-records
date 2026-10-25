# REPORT: replication of the leakage result of the split-learning study (CIFAR-10)

**CONFIRMATORY. Reading R1: "L+ replicates".** Date: 2026-10-25. Prepared for [researcher] (Symbolon Labs) by a Claude Code session. Style: ASD-STE100.

- Preregistration: `PREREGISTRATION.md`, frozen on 2026-10-24 at 18:39:26 (`FREEZE.json`), before the first job. Sealed study: `../split_equal_bits_20261023/` (no file there changed).
- All numbers below come from `results/results.json` unless a different file is named. `results/table.md` has the main rows.
- **Independent check: PASS** (`AUDIT.md`). A subagent wrote `audit/recheck.py` (standard library and numpy only) from the preregistration text and the saved files. It did not open `sleak.py` or `sebits.py`. It calculated every value of `results.json` again, with the two-level bootstrap as a plain loop over the draws. All 343 compared values agree (208 floats; the largest difference is 2.0e-13). All 23 lines of `results/table.md` agree. All 15 file hashes agree with `FREEZE.json`.
- Scope: the same data, split, selection, model family and frozen-client design as the sealed study. The new parts are six new training seeds, Decoder-L as the primary decoder, and a CI that includes the variance between seeds.

---

## 1. Result

At an uplink cap of 64 bits, a trained Decoder-L rebuilds more of the input from the learned bottleneck BOT@64 (64 bits) than from the symbolic message TOPK@64 (58 bits). The gap is above the 0.5 dB margin, also with the variance between seeds in the CI.

| Row (6 new seeds, 10,000 test images) | Value | 95% CI (two-level: seeds, then images) | Seed values (20261121 to 20261126) | Outcome |
|---|---|---|---|---|
| **PRIMARY** Δleak, Decoder-L (dB; + means BOT leaks more) | **+0.609** | **+0.547 to +0.664** | +0.655, +0.580, +0.480, +0.687, +0.647, +0.607 | **R1: L+ replicates** |
| Δacc (points; + favors TOPK; secondary) | +13.85 | +13.15 to +14.57 | +13.75, +13.60, +14.70, +14.29, +13.67, +13.11 | A+ replicates |

- **Reading R1** (Part 7): Δleak (+0.609 dB) is at least 0.5 dB, and the CI lower bound (+0.547 dB) is above 0. All validity checks pass (Section 3).
- Wording that Part 7 sets for R1: **the leakage half of case S replicates with six new seeds and Decoder-L.**
- Descriptive: the CI lower bound is also above the 0.5 dB margin. The Student t 95% interval over the six seed values (+0.532 to +0.687 dB) is also above the margin. The seed SD is 0.074 dB.
- One seed (20261123, +0.480 dB) is below the margin. The rule uses the pooled value, not each seed.
- The accuracy half also replicates: TOPK@64 is 13.85 points more accurate than BOT@64 (test accuracy 0.827 against 0.688).

---

## 2. Comparison with the sealed study

| Row | Seeds | Decoder | Δleak (dB) | 95% CI | CI type |
|---|---|---|---|---|---|
| Sealed primary | 3 sealed | S | +0.550 | +0.527 to +0.572 | images only |
| Sealed rule on the new seeds (secondary) | 6 new | L | +0.609 | +0.587 to +0.632 | images only; L+ |
| Sealed rule on the new seeds (secondary) | 6 new | S | +0.615 | +0.593 to +0.636 | images only; L+ |
| Decoder-S on the new seeds (secondary) | 6 new | S | +0.615 | +0.547 to +0.675 | two-level; R1 |
| **Primary** | 6 new | L | **+0.609** | **+0.547 to +0.664** | two-level; R1 |
| Pooled (descriptive) | 3 sealed + 6 new | S | +0.593 | +0.536 to +0.649 | two-level (K = 9); R1 |

- The new seeds give a slightly larger gap than the sealed seeds (Decoder-S: +0.615 against +0.550 dB). The difference (0.065 dB) is smaller than the SD between seeds (about 0.08 dB). The nine Decoder-S seed values are: +0.512, +0.489, +0.647 (sealed) and +0.650, +0.591, +0.473, +0.713, +0.657, +0.605 (new). Two of nine are below 0.5 dB.
- The larger gap comes mainly from BOT@64. Its mean Decoder-S test PSNR is 14.040 dB in the new seeds and 13.976 dB in the sealed seeds. TOPK@64 is 13.425 dB in the new seeds and 13.43 dB in the sealed seeds.
- **The seed variance is the main source of uncertainty.** On the same data, the two-level CI of the primary row is 0.117 dB wide. The CI from images only is 0.045 dB wide. The sealed CI thus showed about two fifths of the real width.
- **Decoder size does not change the result.** In each seed, Δleak with Decoder-L and with Decoder-S differ by at most 0.026 dB (L minus S: +0.004, −0.012, +0.007, −0.026, −0.010, +0.002 dB). Decoder-L gains at most 0.007 dB of PSNR over Decoder-S on any of the three arms. This agrees with the sealed seed-1 check (+0.504 against +0.512 dB). Thus, at 64 bits, the message limits what this decoder family can rebuild, not the decoder size.

---

## 3. Validity checks (`results.json` `checks`; source records `results/s<seed>/checks.jsonl`)

| Check | Result | Values |
|---|---|---|
| V1 decoder | Pass, 12 records | LAB on D_dev against the closed form, in each seed. Decoder-L: 13.066/13.065, 13.071/13.071, 13.069/13.068, 13.066/13.066, 13.057/13.057, 13.059/13.060 dB. Decoder-S: 13.058 to 13.071 dB. The largest shortfall is 0.0016 dB (limit 0.2 dB). |
| V2 content-free | Pass, 2 records | Decoder-L on shuffled pairs, seed 20261121: BOT@64 12.683 dB and TOPK@64 12.675 dB, against F0 + 0.2 = 12.881 dB on D_dev |
| V3 frozen client | Pass, 124 records | The client sha256 did not change in any unit |
| V4 bits | Pass, 18 records | TOPK@64 58 bits, BOT@64 64 bits, LAB 4 bits; unpack(pack(m)) = m |
| V5 finish | Pass, 6 seeds | Every planned unit has its done marker; all 38 planned decoder outputs exist |
| V6 base model | Pass, 6 seeds | N D_dev accuracy 0.8872, 0.8882, 0.8866, 0.8848, 0.8812, 0.8846 |

- No check record failed, also not in a replaced attempt (`failed_records_all` is empty).

---

## 4. What ran

| Job | Local time | Running time | rc | Log |
|---|---|---|---|---|
| Freeze (CPU) | 2026-10-24 18:38 to 18:39 | 72 s | 0 | 12 unit tests pass; `FREEZE.json` |
| Wait for the bridge-safety run | 2026-10-24 18:39 to 2026-10-25 06:32 | 11 h 53 min | | `logs/waiter.log` |
| Seed 20261121 (with the 2 V2 controls) | 06:34 to 07:13 | 39 min | 0 | `logs/seed_20261121.log` |
| Seed 20261122 | 07:13 to 07:41 | 27 min | 0 | `logs/seed_20261122.log` |
| Seed 20261123 | 07:41 to 08:09 | 27 min | 0 | `logs/seed_20261123.log` |
| Seed 20261124 | 08:09 to 08:37 | 27 min | 0 | `logs/seed_20261124.log` |
| Seed 20261125 | 08:37 to 09:05 | 27 min | 0 | `logs/seed_20261125.log` |
| Seed 20261126 | 09:05 to 09:33 | 27 min | 0 | `logs/seed_20261126.log` |
| Analysis (CPU) | 09:33 | 21 s | 0 | `logs/analysis.log` |

- The bridge-safety run wrote its `DONE` file at 06:32:02. The waiter saw it at 06:32:27, when no Python process of that folder ran. It started the first job at 06:34:27.
- Total laptop time of the seed jobs: about 2.9 h (projection: about 3.0 h). Unit times agree with the projection: N 2.65 min, each Decoder-L about 5.7 min, each Decoder-S about 2.1 min.
- Before each job, all machine checks passed at once: AC power, battery 100%, charger 140 W, swap at most 1.64 GB, no other heavy job. No wrapper waited. No job paused. No job reached a limit. No job restarted.
- Test-label boundary: from the "files opened" line of each log, the six seed jobs opened the train file and the test images only. Only the analysis opened the test labels.

---

## 5. Descriptive rows (means over the 6 new seeds; `descriptive`)

| Arm | Bits | Test acc | PSNR Decoder-L (dB) | PSNR Decoder-S (dB) | Decoder-L over FY (dB) |
|---|---|---|---|---|---|
| TOPK@64 (F-all(6)) | 58 | 0.8269 | 13.426 | 13.425 | +0.361 |
| BOT@64 (16 channels, 2×2, 1 bit) | 64 | 0.6884 | 14.036 | 14.040 | +0.971 |
| LAB | 4 | 0.8269 | 13.089 | 13.089 | +0.024 |

- Floors on test: F0 12.693 dB (no message), FY 13.065 dB (the true label).
- Best decoder epochs: TOPK@64 12 to 14, BOT@64 6 to 10, LAB 18 to 20 (of 20). The BOT decoders reach their best D_dev MSE early. This agrees with the sealed study.
- A Δleak of +0.609 dB means that the mean reconstruction MSE from BOT@64 is about 13% lower than from TOPK@64.

---

## 6. What this result says, and what it does not say

- **It says:** In this setup, the leakage advantage of TOPK@64 over BOT@64 is not an effect of the three sealed seeds or of the smaller decoder. With six new seeds, a decoder with twice the channels, and a CI that includes the seed variance, the gap is +0.61 dB and its CI lower bound is above the 0.5 dB margin. With the nine Decoder-S seeds together, the gap is +0.59 dB (CI +0.54 to +0.65).
- **It does not say:**
  1. That a symbolic message is private. TOPK@64 still gives 0.36 dB over the true-label floor FY.
  2. That the result holds for other splits, data sets, model families, bottleneck families, an end-to-end trained client or a malicious server. Only the training seeds changed.
  3. That stronger attackers would see the same gap. Both decoders are of one family and learn from 20,000 pairs.

---

## 7. Limits

1. **Seeds only.** The data split, the selection and the frozen recipes are those of the sealed study. This replication tests the variance from training seeds (N, head, BOT and decoders). It does not test the variance from the data split.
2. **A weak bottleneck.** BOT@64 is 13.9 points less accurate than TOPK@64. The comparison is a strong symbolic message against a weak activation message at 64 bits (sealed `REPORT.md`, limit 2).
3. **Small effect size in absolute terms.** The gap is 0.6 dB, and both arms are within 1 dB of the true-label floor FY.
4. **Few seeds for a bootstrap.** A percentile bootstrap over 6 seeds can give too small a seed variance. The Student t interval over the six seed values is wider (+0.532 to +0.687 dB against +0.547 to +0.664 dB). Its lower bound is also above 0.5 dB, so the reading and the margin statement do not change.
5. **Pixel metrics.** PSNR measures pixel similarity only.
6. **MPS.** MPS results are not bit-exact between runs.

---

## 8. Events and differences from the plan

- No deviation from `PREREGISTRATION.md`. No stop rule fired. No rerun occurred. No amendment.
- Before the freeze, a code-review subagent found 1 blocker and 6 smaller items. All were fixed before the freeze (`PREREGISTRATION.md`, Part 13). The blocker: in zsh, `wait` gives 145 for a job that was paused and then resumed, so a paused job would lose its exit code. The new self-test `test_run_sleak.zsh` found one more defect (a symbolic-link path in the waiter); it was fixed before the freeze. No pause occurred in the run, so the fix was not used.
- Note for the sealed study: its `run_sebits.sh` (zsh) has the same `wait` defect. No pause occurred in the sealed run (sealed `REPORT.md`, section 2), so its results are not affected.
- The waiter log has no line from 05:25 to 06:32 on 2026-10-25, but the waiter saw `DONE` 25 s after the bridge-safety session wrote it.
- Order of the last files: `PREREGISTRATION.md` Part 9.1, item 4, lists AUDIT, REPORT, SEAL, DONE. A first draft of this report was written before `AUDIT.md`. The final report came after `AUDIT.md`. No number in the draft changed after the audit. `SEAL.json` came after the final report.
- Notes of the audit (no effect on a reading): (1) the FY floor uses round half to even; round half up gives 1.2e-7 dB less (the sealed study had the same note). (2) Part 6.4 writes t = 2.5706; the code uses 2.570582; the descriptive interval moves by about 5e-7 dB. (3) In 10 of 12 LAB decoders, the best epoch is the last (20). This touches only V1 and the LAB rows; V1 passes with a shortfall of at most 0.0016 dB against a limit of 0.2 dB.

---

## 9. Files

- Rules and code: `PREREGISTRATION.md`, `FREEZE.json`, `sleak.py`, `sebits.py` (byte copy), `selection.json` (byte copy), `test_sleak.py`, `test_run_sleak.zsh`, `run_sleak.sh`, `wait_and_run.sh`.
- Per seed: `results/s<seed>/` (`N.pt`, `N.json`, `arms/`, `msgs/`, `dec/`, `checks.jsonl`, `times.jsonl`, `done/`).
- Analysis: `results/results.json`, `results/table.md`.
- Logs: `logs/waiter.log`, `logs/<job>.log`, `logs/<job>_wrapper.log`.
- Independent check: `AUDIT.md`, `audit/`. Seal: `SEAL.json`.
- Data: read from `../split_equal_bits_20261023/data/` (research-only; do not publish rebuilt images beyond small figure panels).
