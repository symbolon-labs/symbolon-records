# AUDIT: independent statistics check of the split equal-bits run

- What: This audit compares the values in `results/results.json` with values that it calculates again from the saved files. It applies Parts 2.2, 4.4, 5.3, 5.4, 7.1, 7.4, 7.5 and 7.6 of `PREREGISTRATION.md` (Part 8.4, item 5).
- When: 2026-10-24 16:35:26 (local time). The `results.json` file has the time 2026-10-24 16:25:21 and the sha256 `c4f91b4430ba93175e0e8898b0767b83e6db7c94e5a9c54a2807a85ad136ba95`.
- Who: A Claude Code subagent wrote and ran the check for [researcher].
- How: The script `audit/stats_check.py` reads the npz, json, jsonl and log files of the run folder. It uses Python 3.12.14 and numpy 2.4.6 on the CPU. It trains no model and downloads no file. It changes no existing file. The command is `../../.venv/bin/python -I -B audit/stats_check.py`, from the run folder.
- Independence: The script shares no code with `sebits.py`. It does not import `sebits.py`. The author of the script did not read `sebits.py`. The author wrote the script from the written rules and the file formats only.
- Tolerance: Integers, strings, case letters and true or false values must be equal. Means and CI bounds must agree to an absolute difference of 1e-09 or less.
- All values: `audit/stats_check.json`.

## Results

| Quantity | results.json | Independent value | Difference | Pass |
|---|---|---|---|---|
| sha256 cifar10_train.npz (FREEZE.json) | 3718a9a09265eb5e1b478fc87f49ff46f36397e04fd03f151173b44ee928f95c | 3718a9a09265eb5e1b478fc87f49ff46f36397e04fd03f151173b44ee928f95c | exact | yes |
| sha256 cifar10_test_images.npz (FREEZE.json) | 4b19da3a0393bc5bc233d475551b7b4cd6fc3bd725bb6c897b4468ddef00cf4e | 4b19da3a0393bc5bc233d475551b7b4cd6fc3bd725bb6c897b4468ddef00cf4e | exact | yes |
| sha256 cifar10_test_labels.npz (FREEZE.json) | fec3b140ee9cc86565710c177dc1ff8abc36d8f06e158568576a36648d6b63c1 | fec3b140ee9cc86565710c177dc1ff8abc36d8f06e158568576a36648d6b63c1 | exact | yes |
| SYM arm | TOPK@64 | TOPK@64 | exact | yes |
| ACT arm | BOT@64 | BOT@64 | exact | yes |
| d_acc (points) | 13.67 | 13.67 | 0 | yes |
| d_acc CI low | 13.09 | 13.09 | 0 | yes |
| d_acc CI high | 14.26666666666667 | 14.26666666666667 | 0 | yes |
| d_acc seed 20261111 | 14.42 | 14.42 | 0 | yes |
| d_acc seed 20261112 | 13.36 | 13.36 | 0 | yes |
| d_acc seed 20261113 | 13.23 | 13.23 | 0 | yes |
| d_leak (dB) | 0.5496044431405254 | 0.5496044431405254 | 0 | yes |
| d_leak CI low | 0.5268899954619487 | 0.5268899954619487 | 0 | yes |
| d_leak CI high | 0.572313210778331 | 0.572313210778331 | 0 | yes |
| d_leak seed 20261111 | 0.5124317022466987 | 0.5124317022466987 | 0 | yes |
| d_leak seed 20261112 | 0.48907973210776673 | 0.48907973210776673 | 0 | yes |
| d_leak seed 20261113 | 0.6473018950671108 | 0.6473018950671109 | 1.1e-16 | yes |
| accuracy outcome A | A+ | A+ | exact | yes |
| leakage outcome L | L+ | L+ | exact | yes |
| leakage conditions that hold (L_hits) | L+ | L+ | exact | yes |
| case | S | S | exact | yes |
| case_final | S | S | exact | yes |
| seed-1 d_leak, Decoder-L (L1_) | 0.5039099120393447 | 0.5039099120393447 | 0 | yes |
| seed-1 d_leak, Decoder-S (S1_) | 0.5124317022466987 | 0.5124317022466987 | 0 | yes |
| decoder_dependent | false | false | exact | yes |
| primary.decoder_dependent | false | false | exact | yes |
| PSNR Decoder-L seed 1 TOPK@64 | 13.405752370570763 | 13.405752370570763 | 0 | yes |
| PSNR Decoder-L seed 1 BOT@64 | 13.909662282610109 | 13.909662282610109 | 0 | yes |
| acc RAW@64 | 0.6359333333333334 | 0.6359333333333334 | 0 | yes |
| PSNR RAW@64 | 14.422334877591195 | 14.422334877591199 | 3.6e-15 | yes |
| acc RAW@64 seed 20261111 | 0.6361 | 0.6361 | 0 | yes |
| PSNR RAW@64 seed 20261111 | 14.38004617454138 | 14.38004617454138 | 0 | yes |
| acc RAW@64 seed 20261112 | 0.6366 | 0.6366 | 0 | yes |
| PSNR RAW@64 seed 20261112 | 14.425793449245306 | 14.425793449245306 | 0 | yes |
| acc RAW@64 seed 20261113 | 0.6351 | 0.6351 | 0 | yes |
| PSNR RAW@64 seed 20261113 | 14.46116500898691 | 14.46116500898691 | 0 | yes |
| acc BOT@64 | 0.6896 | 0.6896 | 0 | yes |
| PSNR BOT@64 | 13.976150582177118 | 13.976150582177118 | 0 | yes |
| acc BOT@64 seed 20261111 | 0.6849 | 0.6849 | 0 | yes |
| PSNR BOT@64 seed 20261111 | 13.899521147994744 | 13.899521147994744 | 0 | yes |
| acc BOT@64 seed 20261112 | 0.6916 | 0.6916 | 0 | yes |
| PSNR BOT@64 seed 20261112 | 13.902494419167104 | 13.902494419167104 | 0 | yes |
| acc BOT@64 seed 20261113 | 0.6923 | 0.6923 | 0 | yes |
| PSNR BOT@64 seed 20261113 | 14.126436179369508 | 14.126436179369508 | 0 | yes |
| acc TOPK@64 | 0.8263000000000001 | 0.8263000000000001 | 0 | yes |
| PSNR TOPK@64 | 13.426546139036592 | 13.426546139036594 | 1.8e-15 | yes |
| acc TOPK@64 seed 20261111 | 0.8291 | 0.8291 | 0 | yes |
| PSNR TOPK@64 seed 20261111 | 13.387089445748042 | 13.387089445748042 | 0 | yes |
| acc TOPK@64 seed 20261112 | 0.8252 | 0.8252 | 0 | yes |
| PSNR TOPK@64 seed 20261112 | 13.413414687059337 | 13.413414687059337 | 0 | yes |
| acc TOPK@64 seed 20261113 | 0.8246 | 0.8246 | 0 | yes |
| PSNR TOPK@64 seed 20261113 | 13.479134284302397 | 13.479134284302397 | 0 | yes |
| acc LAB | 0.8263000000000001 | 0.8263000000000001 | 0 | yes |
| PSNR LAB | 13.092185133295992 | 13.092185133295992 | 0 | yes |
| acc LAB seed 20261111 | 0.8291 | 0.8291 | 0 | yes |
| PSNR LAB seed 20261111 | 13.088535912720365 | 13.088535912720365 | 0 | yes |
| acc LAB seed 20261112 | 0.8252 | 0.8252 | 0 | yes |
| PSNR LAB seed 20261112 | 13.096666937444681 | 13.09666693744468 | 1.8e-15 | yes |
| acc LAB seed 20261113 | 0.8246 | 0.8246 | 0 | yes |
| PSNR LAB seed 20261113 | 13.091352549722929 | 13.091352549722929 | 0 | yes |
| acc RAW_native | 0.8793666666666667 | 0.8793666666666667 | 0 | yes |
| PSNR RAW_native | 22.237798854124964 | 22.237798854124964 | 0 | yes |
| acc RAW_native seed 20261111 | 0.8799 | 0.8799 | 0 | yes |
| PSNR RAW_native seed 20261111 | 22.20154937737925 | 22.20154937737925 | 0 | yes |
| acc RAW_native seed 20261112 | 0.8781 | 0.8781 | 0 | yes |
| PSNR RAW_native seed 20261112 | 22.13746659308628 | 22.13746659308628 | 0 | yes |
| acc RAW_native seed 20261113 | 0.8801 | 0.8801 | 0 | yes |
| PSNR RAW_native seed 20261113 | 22.37438059190936 | 22.37438059190936 | 0 | yes |
| nbits BOT@1024 (s1, s2, s3) | 1024 | 1024/1024/1024 (rule: 1024) | exact | yes |
| nbits BOT@16 (s1, s2, s3) | 16 | 16/16/16 (rule: 16) | exact | yes |
| nbits BOT@256 (s1, s2, s3) | 256 | 256/256/256 (rule: 256) | exact | yes |
| nbits BOT@64 (s1, s2, s3) | 64 | 64/64/64 (rule: 64) | exact | yes |
| nbits BOT_float (s1) | 2048 | 2048 (rule: 2048) | exact | yes |
| nbits LAB (s1, s2, s3) | 4 | 4/4/4 (rule: 4) | exact | yes |
| nbits LAB_full (s1, s2, s3) | 4 | 4/4/4 (rule: 4) | exact | yes |
| nbits RAW@1024 (s1, s2, s3) | 1024 | 1024/1024/1024 (rule: 1024) | exact | yes |
| nbits RAW@16 (s1, s2, s3) | 16 | 16/16/16 (rule: 16) | exact | yes |
| nbits RAW@256 (s1, s2, s3) | 256 | 256/256/256 (rule: 256) | exact | yes |
| nbits RAW@64 (s1, s2, s3) | 64 | 64/64/64 (rule: 64) | exact | yes |
| nbits RAW_native (s1, s2, s3) | 32768 | 32768/32768/32768 (rule: 32768) | exact | yes |
| nbits TOPK@16 (s1, s2, s3) | 16 | 16/16/16 (rule: 16) | exact | yes |
| nbits TOPK@256 (s1, s2, s3) | 148 | 148/148/148 (rule: 148) | exact | yes |
| nbits TOPK@64 (s1, s2, s3) | 58 | 58/58/58 (rule: 58) | exact | yes |
| nbits TOPK_float (s1, s2, s3) | 320 | 320/320/320 (rule: 320) | exact | yes |
| nbits TOPK_full (s1, s2, s3) | 58 | 58/58/58 (rule: 58) | exact | yes |
| pred_test equals message class or logit argmax (49 files) | n/a | 0 files with a mismatch | exact | yes |
| F0 PSNR (test) | 12.692791981008561 | 12.692791981008565 | 3.6e-15 | yes |
| FY PSNR (test) | 13.065110518763419 | 13.065110518763419 | 0 | yes |
| every checks.jsonl record has ok true (283 records) | all ok | all ok | exact | yes |
| planned decoder outputs exist (57 files) | n/a | 0 missing | exact | yes |
| no job except j6 opened the test labels (job lines) | n/a | pass | exact | yes |
| every job wrapper log ends with rc=0 (data, j0 to j6) | n/a | all rc=0 | exact | yes |

## Notes

1. `logs/j0.log` has a data line with "test labels" (line 568). This line is in the output of the unit tests, before the line "Ran ... tests" (line 577). The docstring of `test_sebits.py` says that the tests use synthetic data only and read no CIFAR-10 file. Its end-to-end test runs J0 to J6 in a temporary folder. The J0 job line after the unit tests (line 580) shows the train file only. Thus the audit counts the line as a unit-test line, not as a J0 job line.
2. The mean images have 0 values in F0 and 10 values in FY that are exactly at a half. The audit rounds a half to the even integer (numpy `round`). This gives the `results.json` values. If a half goes up, FY is 13.065110399168814 dB (a difference of 1.2e-07 dB). Part 5.4 does not give the direction for a half.
3. The three `checks.jsonl` files have 283 records (V1 3, V2 2, V3 226, V4 49, V6 3). `results.json` also counts 3 V5 records, which are not in `checks.jsonl`. The audit checked one part of V5 again: all 57 planned decoder outputs exist.
4. The L+ outcome holds with a small margin. d_leak is 0.5496 dB, which is 0.0496 dB above the 0.5 dB bar. The CI lower bound is 0.5269 dB. The seed values are 0.5124, 0.4891, 0.6473 dB. Part 7.4 requires only that each seed value is above 0. Decoder-L in seed 1 gives 0.5039 dB.
5. Extra checks: in all 49 message files, `pred_test` agrees with the first class in the bits (TOPK, TOPK_full, LAB, LAB_full) or with the argmax of `logits_test` (the other arms). The padding bits after `nbits` are 0. All 8 job wrapper logs end with rc=0.

## Verdict

**Verdict: PASS.** All 92 rows pass. The largest difference of a float value is 3.6e-15. The primary case is S (A+, L+). The case is not decoder-dependent.
