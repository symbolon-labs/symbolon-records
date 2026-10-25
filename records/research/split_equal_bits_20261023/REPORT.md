# REPORT: split learning at equal bits, activations against a symbolic message (CIFAR-10)

**CONFIRMATORY. Primary case S (symbolic dominates), with a small leakage margin.** Date: 2026-10-24. Prepared for [researcher] (Symbolon Labs) by a Claude Code session. Style: ASD-STE100.

- Preregistration: `PREREGISTRATION.md` (frozen 2026-10-23 21:26:40, `FREEZE.json`). Design: `reports/split_equal_bits_design_20261023.md`.
- All numbers below come from `results/results.json` (written by J6) unless a different file is named. `results/table.md` has the same numbers as tables. The seal is `SEAL.json`.
- **Independent check: PASS** (`AUDIT.md`). A separate script (`audit/stats_check.py`; standard library and numpy only; it does not import or copy `sebits.py`) calculated the primary pair, the CIs, the case, the decoder check, the descriptive values of five arms, the bits of all 49 message files and the floors again from the saved files. All 92 rows agree with `results/results.json`. The largest float difference is 3.6e-15.
- Scope: one data set (CIFAR-10), one model family, a frozen client, test-time forward messages only, one decoder family (two sizes). The values are lower bounds on what a decoder can rebuild. The claim is about the order of the arms.

---

## 1. Result

At an uplink cap of 64 bits, at cut C2, the symbolic message TOPK (F-all(6), 58 bits) is more accurate than the learned bottleneck BOT (16 channels, 2×2 grid, 1 bit; 64 bits), and a trained decoder rebuilds less of the input from it.

| Primary pair: TOPK@64 against BOT@64 | Value | 95% CI | Seed 1 | Seed 2 | Seed 3 | Key in `results/results.json` |
|---|---|---|---|---|---|---|
| Δacc (points; + favors TOPK) | +13.67 | +13.09 to +14.27 | +14.42 | +13.36 | +13.23 | `primary.d_acc`, `d_acc_ci`, `d_acc_seeds` |
| Δleak (dB; + means BOT leaks more) | +0.550 | +0.527 to +0.572 | +0.512 | +0.489 | +0.647 | `primary.d_leak`, `d_leak_ci`, `d_leak_seeds` |

- Accuracy outcome **A+**: the CI lower bound (+13.09) is above −1.0.
- Leakage outcome **L+**: Δleak (+0.550) is at least 0.5 dB, the CI lower bound (+0.527) is above 0, and each seed is above 0.
- **Case S** (`primary.case_final`). All validity checks pass (Section 3). The case is not decoder-dependent (Section 4).

**Caution about the size of the leakage result.** The point estimate is only 0.05 dB above the 0.5 dB margin. The value of seed 2 (+0.489 dB) is below the margin; the rule asks only that each seed is above 0. The CI comes from a bootstrap over test images. It does not include the variance between seeds (design section 12, item 7). The seed values have a standard deviation of about 0.085 dB. Thus "L+" is a pass of the preregistered rule, but the leakage gap at 64 bits is small: about 12% in MSE terms (0.5 dB) at the margin. The accuracy gap is large and clear.

---

## 2. What ran

| Job | Local time, 2026-10-24 | Running time | rc | Content | Log |
|---|---|---|---|---|---|
| Download | 12:40 to 13:03 | about 23 min | 0 | `cifar-10-binary.tar.gz`, 170,052,171 bytes, md5 `c32a1d4a...` = the dataset page, sha256 `c4a38c50...` | `FREEZE.json` (`data`) |
| Data | 13:04 | below 1 min | 0 | npz files; 5,000 train and 1,000 test images for each class | `logs/data_wrapper.log` |
| J0 (scratch) | 13:04 to 13:11 | 6 min | 0 | 44 unit tests pass; N seed 20261100 D_dev 0.889 in 2.7 min; projected Decoder-S 2.6 min; no stop rule | `j0_result.json`, `times.json` |
| J1 | 13:12 to 13:41 | 29 min | 0 | Seed 20261111: N, TOPK head, formats, 24 candidates, selection, BOT-float | `logs/j1.log`, `selection.json` |
| J2 | 13:42 to 14:19 | 36 min | 0 | Seed 1 messages and 18 Decoder-S | `logs/j2.log` |
| J3 | 14:19 to 14:55 | 35 min | 0 | 5 Decoder-L and 4 extra Decoder-S | `logs/j3.log` |
| J4 | 14:55 to 15:38 | 42 min | 0 | Seed 20261112: N, arms, messages, 15 Decoder-S | `logs/j4.log` |
| J5 | 15:38 to 16:21 | 42 min | 0 | Seed 20261113: as J4 | `logs/j5.log` |
| J6 | 16:21 to 16:26 | 4 min | 0 | Test metrics, probes, bootstrap, rule | `logs/j6.log`, `results/` |

- Total laptop time of J0 to J6: about 3.2 h (the design estimate was 4.6 to 9.3 h). No job paused for power. No job reached its time limit. Swap stayed at about 1.34 GB.
- Each job ran through `run_sebits.sh` (machine checks, `caffeinate -i -w`, power guard). Each job checked the sha256 of the frozen files and of the data files.
- Test-label boundary: from the "files opened" line of each job log, J1 opened the train file only, J2 to J5 the train file and the test images, and J6 the test labels. Note: `logs/j0.log` also has "files opened" lines from 13:04 to 13:06 that name test labels. They come from the end-to-end unit test, which runs on tiny synthetic files in a temporary folder, not on the CIFAR-10 files.

---

## 3. Validity checks (`results/results.json` `checks`; source records `results/s<seed>/checks.jsonl`)

| Check | Result | Values |
|---|---|---|
| V1 decoder | Pass in 3 seeds | LAB Decoder-S on D_dev against the closed form: 13.055 / 13.056, 13.065 / 13.066, 13.066 / 13.068 dB (`results/s<seed>/dec/S1_LAB.json`) |
| V2 content-free | Pass, both controls | BOT@64 12.672 dB and TOPK@64 12.669 dB, against F0 + 0.2 = 12.881 dB on D_dev (`results/s20261111/dec/S1_*_ctrl.json`) |
| V3 frozen client | Pass, 226 records | The client sha256 did not change in any unit |
| V4 bits | Pass, 49 records | Every message has its declared length; unpack(pack(m)) = m |
| V5 finish | Pass | All 57 planned decoder outputs exist; no job stopped on a non-finite loss. (By design, V5 has no record in `checks.jsonl`: a non-finite loss stops the job, and J6 checks the decoder outputs. The 3 V5 records of `results.json` are these J6 checks.) |
| V6 base model | Pass in 3 seeds | N D_dev accuracy 0.8924, 0.8886, 0.8872 (`results/s<seed>/N.json`) |

---

## 4. Decoder dependence (Part 7.6; `decoder_check`, `decoder_seed_sd`)

- Seed-1 Δleak with Decoder-L: +0.504 dB. With Decoder-S: +0.512 dB. Same sign; difference 0.008 dB (limit 1 dB). Thus the case is **not** decoder-dependent.
- Decoder seed noise (seed 1, three Decoder-S seeds each): TOPK@64 13.387, 13.393, 13.386 dB (SD 0.004); BOT@64 13.900, 13.906, 13.902 dB (SD 0.003). The decoder seed is not a source of the seed spread of Δleak. The spread comes from the arms and N of each seed (BOT@64 test PSNR: 13.900, 13.902, 14.126 dB; `descriptive.BOT@64.psnr_seeds`).
- In seed 1, Decoder-L gains almost nothing over Decoder-S at 64 bits: TOPK@64 13.406 against 13.387 dB, BOT@64 13.910 against 13.900 dB, RAW@64 14.416 against 14.380 dB. It gains 0.46 dB on RAW-native (22.661 against 22.202 dB). Thus at the small budget, the limit is the message, not the decoder size, for this decoder family.

---

## 5. Secondary rows (labeled secondary; no case claim; `secondary`)

| Row | Δacc (points) | 95% CI | Δleak (dB) | 95% CI | Rule cell |
|---|---|---|---|---|---|
| TOPK against RAW at 64 | +19.04 | +18.39 to +19.69 | +0.996 | +0.967 to +1.025 | S |
| TOPK against BOT at 16 | +22.44 | +21.74 to +23.15 | +0.151 | +0.134 to +0.168 | E |
| TOPK against RAW at 16 | +37.51 | +36.76 to +38.27 | +0.333 | +0.311 to +0.354 | E |
| TOPK against BOT at 256 | +5.86 | +5.38 to +6.33 | +1.731 | +1.697 to +1.764 | S |
| TOPK against RAW at 256 | +7.10 | +6.57 to +7.63 | +2.368 | +2.333 to +2.403 | S |
| TOPK against BOT at 1,024 | +0.03 | −0.40 to +0.46 | +3.805 | +3.764 to +3.847 | S |
| TOPK against RAW at 1,024 | +1.03 | +0.57 to +1.50 | +4.360 | +4.318 to +4.402 | S |
| LAB-full against BOT at 64 | +18.93 | +18.28 to +19.59 | +0.899 | +0.871 to +0.927 | S |
| TOPK-full against BOT at 64 | +18.93 | +18.28 to +19.59 | +0.621 | +0.597 to +0.645 | S |
| SSIM version of the primary pair (margin 0.02) | +13.67 | +13.09 to +14.27 | +0.023 SSIM | +0.022 to +0.024 | S |

- **Frontier.** At every budget, TOPK is at least as accurate as the activation arms. At 1,024 bits, TOPK (148 bits) and BOT have the same accuracy (+0.03 points, CI −0.40 to +0.46), and BOT leaks 3.8 dB more. At 16 bits, TOPK is 22 points more accurate and the leakage difference is inside ±0.5 dB.
- **LAB against TOPK** (the leakage cost of the extra TOPK bits; `secondary.LAB_vs_TOPK@<B>`): TOPK leaks +0.13 dB (at 16 bits) to +0.33 and +0.34 dB (at 64 bits and more) more than LAB. All four rows are inside ±0.5 dB (cell E). TOPK beats LAB on NLL at every budget, by 0.161 to 0.257 nats (CI lower bounds 0.150 to 0.246; `secondary.LAB_vs_TOPK@<B>.nll`). The accuracy is the same by construction.
- **c_quant** (`descriptive.TOPK@<B>.c_quant`): 0.096 nats at 16 bits, 0.0016 nats at 64 bits, 0 at 148 bits. Thus the 64-bit cap does not limit the TOPK message (below 0.01 nats).

---

## 6. Arms (descriptive; means over 3 seeds; `descriptive`)

| Arm | Bits | Test acc | NLL | PSNR (dB) | SSIM | R²_pix | Excess over FY (dB) |
|---|---|---|---|---|---|---|---|
| RAW@16 / @64 / @256 / @1024 | 16 / 64 / 256 / 1,024 | 0.451 / 0.636 / 0.755 / 0.816 | 1.503 / 1.027 / 0.701 / 0.529 | 13.55 / 14.42 / 15.80 / 17.79 | 0.148 / 0.172 / 0.264 / 0.457 | 0.176 / 0.331 / 0.519 / 0.699 | 0.49 / 1.36 / 2.74 / 4.73 |
| BOT@16 / @64 / @256 / @1024 | 16 / 64 / 256 / 1,024 | 0.602 / 0.690 / 0.768 / 0.826 | 1.138 / 0.887 / 0.669 / 0.505 | 13.37 / 13.98 / 15.16 / 17.24 | 0.146 / 0.180 / 0.290 / 0.507 | 0.137 / 0.261 / 0.446 / 0.664 | 0.30 / 0.91 / 2.10 / 4.17 |
| TOPK@16 / @64 / @256 (= @1024) | 16 / 58 / 148 | 0.826 | 0.598 / 0.504 / 0.502 | 13.22 / 13.43 / 13.43 | 0.148 / 0.157 / 0.156 | 0.104 / 0.146 / 0.148 | 0.15 / 0.36 / 0.37 |
| LAB | 4 | 0.826 | 0.759 | 13.09 | 0.143 | 0.076 | 0.03 |
| TOPK-float | 320 | 0.826 | 0.502 | 13.42 | 0.157 | 0.144 | 0.36 |
| TOPK-full (client runs all of N) | 58 | 0.879 | 0.354 | 13.36 | 0.153 | 0.132 | 0.29 |
| LAB-full (client runs all of N) | 4 | 0.879 | 0.571 | 13.08 | 0.143 | 0.073 | 0.01 |
| RAW-native (standard split) | 32,768 | 0.879 | 0.350 | 22.24 | 0.764 | 0.892 | 9.17 |
| BOT-float (seed 1 only) | 2,048 | 0.781 | 0.630 | n/a | n/a | n/a | n/a |

- Floors on test (`floors`): F0 12.693 dB, FY 13.065 dB. The floor images are rounded to uint8 with round half to even (`numpy.round`). The FY image has 10 values that are exactly at a half. Round half up would give a FY PSNR 1.2e-7 dB lower (`AUDIT.md`). The effect is negligible.
- Selection (`selection.json`, D_dev only): every selected RAW and BOT configuration uses 1-bit values with the most channels or grid cells: RAW (16,1,1), (64,1,1), (64,2,1), (64,4,1); BOT (16,1,1), (16,2,1), (16,4,1), (16,8,1). TOPK: F-top(4,1) at 16 bits, F-all(6) at 64 bits, F-all(16) at 256 and 1,024 bits. TOPK-full: F-all(6).
- Client and server MACs for each image (`descriptive.<arm>.client_macs`, `server_macs`): RAW 24.48 M and 14.16 M; BOT 24.54 M and 14.16 to 14.22 M; TOPK and LAB 24.81 M and 0; TOPK-full and LAB-full 38.63 M and 0.
- The plot of accuracy against PSNR for all arms and budgets: `results/acc_psnr.svg`.

---

## 7. Attribute and membership rows at B* (descriptive; `attributes`)

| Arm | Flip probe acc (chance 0.5) | Brightness excess over the label-only probe | Membership AUROC |
|---|---|---|---|
| RAW@64 | 0.502 (0.496 to 0.508) | +0.232 (+0.222 to +0.243) | 0.501 (0.494 to 0.508) |
| BOT@64 | 0.500 (0.494 to 0.506) | +0.122 (+0.112 to +0.132) | 0.508 (0.501 to 0.514) |
| TOPK@64 | 0.497 (0.490 to 0.505) | +0.059 (+0.050 to +0.068) | 0.503 (0.496 to 0.511) |
| LAB | 0.506 (0.497 to 0.516) | +0.003 (−0.002 to +0.009) | 0.512 (0.503 to 0.520) |
| LAB-full | 0.502 (0.494 to 0.510) | +0.003 (−0.002 to +0.007) | 0.532 (0.522 to 0.542) |
| TOPK-full | 0.496 (0.489 to 0.503) | +0.045 (+0.036 to +0.054) | 0.529 (0.521 to 0.537) |

- The label-only brightness probe gets 0.427 accuracy (`attributes.label_only_bright_acc`).
- No 64-bit message carries flip information (all CIs contain 0.5).
- Brightness that the class does not explain: RAW@64 carries the most (+0.23), BOT@64 half of that, TOPK@64 a quarter, LAB none. This order agrees with the PSNR order of the activation arms against TOPK.
- Membership: all AUROC values are near 0.5. The two full-client rows are the highest (0.53). They read the logits of N, which trained for 30 epochs on D_task; the TOPK head trained for 15 epochs on frozen features. This run does not test why the full-client rows are higher.

---

## 8. What a defender can take from case S (design section 11)

- At an uplink cap of 64 bits, a client that can add a small head (0.33 M MACs, about 1.3% of the client compute) should send top-k logit gaps, not a learned bottleneck. The accuracy is 13.7 points higher, and this decoder family rebuilds slightly less of the input (0.55 dB lower PSNR).
- Over the budgets of this run, BOT reaches the accuracy of TOPK only at 1,024 bits, where it leaks 3.8 dB more. RAW at 1,024 bits is still 1.0 point lower and leaks 4.4 dB more. Extra activation bits that give no accuracy over TOPK give only leakage.
- If the server does not need probabilities, the label alone (4 bits) has the same accuracy and the lowest leakage (0.03 dB over FY). If the server needs calibrated probabilities, TOPK@64 costs about 0.33 dB of PSNR over LAB and gives 0.26 nats lower NLL.

**What this report does not say.** It does not say that a symbolic message is private. TOPK@64 still carries image brightness beyond its class (+0.06) and leaks 0.36 dB over the true-label floor FY. It does not say that the result holds for other data sets, other models, other decoders, an end-to-end trained client, or a malicious server.

---

## 9. Limits

1. **Small leakage margin at B*.** Section 1: the primary L+ passes by 0.05 dB, and one seed is below 0.5 dB. A replication with more seeds could move Δleak below the margin. The accuracy result does not depend on this.
2. **The bottleneck was weaker than expected.** The design expected BOT@64 within about 3 points of TOPK. It was 13.7 points lower. Thus the primary pair compares a strong symbolic message with a weak activation message at 64 bits. A different bottleneck family, or an end-to-end trained client (J9, not run), can be stronger.
3. **Leakage depends on the decoder.** The values are what one decoder family learns from 20,000 pairs. Stronger attackers exist (optimization with the client weights, image priors, more data). The Decoder-L check tests one direction only (more capacity) and agrees.
4. **Pixel metrics.** PSNR and SSIM measure pixel similarity. A blurred image can still show the class and the scene.
5. **One data set, one model family, a frozen client, forward messages only.** Design section 12 applies in full.
6. **Statistics.** The bootstrap is over test images only. The seed variance enters only through the same-sign condition with 3 seeds.
7. **Selection on utility only.** Leakage had no part in the selection.

---

## 10. Events and differences from the plan

- No deviation from `PREREGISTRATION.md` occurred during J0 to J6. No stop rule fired. No rerun occurred.
- Order of the last files: `PREREGISTRATION.md` Part 8.4, item 5 lists AUDIT, SEAL, REPORT, DONE. A first draft of this report was written before `AUDIT.md`. The final report came after `AUDIT.md`. `SEAL.json` came after the final report, so that the seal also covers this report. No number in the draft depended on the audit.
- Before the freeze (2026-10-23), a code review found no blocker and 8 minor items. All were fixed before the freeze (`PREREGISTRATION.md`, Part 13). Planned differences from the design are in `PREREGISTRATION.md`, Part 10 (D1 to D7).
- Approvals: `APPROVALS.md`. The relayed approval of 2026-10-23 at about 21:30 allowed restarts after a time limit or a machine-check refusal. No restart was necessary.
- Not run: cut C1 (J7, J8a, J8b) and the end-to-end variant (J9). They are not in the frozen plan. The design recommends a decision on them after the primary case.

---

## 11. Expectations before the run (design section 13) against the result

| Expectation | Result |
|---|---|
| TOPK at C2 about 0.75 to 0.82 accuracy | 0.826 |
| BOT@64 within about 3 points of TOPK | 13.7 points lower |
| F0 about 12 to 13 dB | 12.69 dB |
| L+ most likely; differences of 0.5 to 1.5 dB | L+ at +0.55 dB |
| Most likely case S or T | S |
| At 1,024 bits, BOT and RAW gain accuracy and leak several dB more | Yes: +3.8 and +4.4 dB more than TOPK |

---

## 12. Files

- Code and rules: `PREREGISTRATION.md`, `sebits.py`, `test_sebits.py`, `run_sebits.sh`, `FREEZE.json`, `APPROVALS.md`.
- J0: `times.json`, `job_plan.json`, `j0_result.json`, `results/j0/`.
- J1: `selection.json`; per seed `results/s<seed>/`: `N.pt`, `N.json`, `arms/` (models, specs, D_dev values), `msgs/` (packed messages, test and D_dev logits and predictions), `dec/` (test reconstructions as uint8, D_dev PSNR, histories), `checks.jsonl`, `times.jsonl`, `done/`.
- J6: `results/results.json`, `results/table.md`, `results/acc_psnr.svg`.
- Logs: `logs/<job>.log`, `logs/<job>_wrapper.log`.
- Independent check: `AUDIT.md`, `audit/stats_check.py`, `audit/stats_check.json`. Seal: `SEAL.json`.
- Data (research-only; do not publish rebuilt images beyond small figure panels): `data/`.
