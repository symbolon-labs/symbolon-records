# AUDIT: independent recomputation of the split-leak replication

- Date: 2026-10-25.
- Auditor: a Claude Code subagent for [researcher] (Symbolon Labs). Style: ASD-STE100.
- Script: `audit/recheck.py`. Python 3.12.14, numpy 2.4.6 and the standard library only. CPU only. Run time: about 10 s.
- Outputs: `audit/recheck.json` (all recomputed values and all comparisons) and `audit/table_rows.md` (the table of Section 4).
- Command: `../../.venv/bin/python -I -B audit/recheck.py` (from this folder).

## 1. Independence

- The script uses the text of `PREREGISTRATION.md` (Parts 5 to 9) and of the sealed `PREREGISTRATION.md` (Parts 2.2, 5.3, 5.4, 7.1, 7.4 and 7.5).
- The script reads only data files: the npz, json and jsonl files of `results/`, the sealed data and the sealed reconstructions.
- The auditor did not open, import or copy `sleak.py`, `sebits.py`, `test_sleak.py` or a file of the sealed `audit/` folder. The script reads their bytes only to calculate the sha256.
- The script uses no torch and no scipy.
- The auditor changed no file outside `audit/` and `AUDIT.md`. The sealed folder did not change.

## 2. What I checked, and how

1. Frozen files: the script calculates the sha256 of the 8 frozen files and compares each value with `FREEZE.json`.
2. Data files: it calculates the sha256 of the 3 sealed npz files and compares them with `data.npz_sha256`.
3. Extra hashes: it compares the sealed `FREEZE.json` with `sealed_freeze_sha256`, and `sebits.py` and `selection.json` with the sealed byte copies.
4. PSNR: for each test image, PSNR = 10 log10(1 / MSE), with the MSE over the 3x32x32 values of uint8 / 255 in float64.
5. Accuracy: for each seed and arm, the mean of (`pred_test` = test label).
6. Primary: D(s, i), the pooled Δleak and Δleak for each seed, with Decoder-L.
7. Two-level bootstrap: Part 6.3 with a plain loop over the 10,000 draws and over the K drawn seeds of each draw.
8. Reading: R1 to R4 of Part 7 on Δleak and its CI.
9. Δacc: 100 × mean(c(TOPK@64) − c(BOT@64)) with the same draws, and the A+ outcome.
10. Decoder-S row: Δleak of Decoder-S on the new seeds with the same draws, and its R label.
11. Sealed rule: the image bootstrap of sealed Part 7.1 (seed 20261024) on the across-seed mean of each image, then sealed Parts 7.4 and 7.5.
12. Pooled row: the two-level bootstrap with K = 9 (3 sealed seeds, then 6 new seeds) and a new generator with seed 20261127.
13. Seed spread: the seed SD (ddof = 1) and the Student t 95% interval (t = 2.570582 for 5 df, t = 2.306004 for 8 df).
14. Sealed seed 1: the mean of PSNR(BOT@64) − PSNR(TOPK@64) of the sealed Decoder-L outputs of seed 20261111.
15. Floors: the split of sealed Part 2.2 (seed 20261110), then F0 and FY on test from D_aux and its horizontal flips, each floor image rounded to uint8.
16. V1 and V2 values: the V1 closed form on D_dev in each seed (from the LAB messages) and F0 on D_dev, against the check records.
17. Validity: the last record of `checks.jsonl` for each seed, check and place; V1 to V6 as the audit brief specifies; a count of all records with ok = false.
18. V5: each planned done marker, each decoder output and each message file exists; each loss in each decoder history is finite.
19. Bits: `nbits` of each message file and `bits` of each arm file.
20. Table: the script builds `results/table.md` again from the recomputed values and compares the text line by line.
21. Compare: the script compares each leaf of `results.json` with its recomputed value. Floats use an absolute tolerance of 1e-9. Labels, counts and readings must be equal.

## 3. Result summary

- `results.json` leaves compared: 343 (208 floats, 73 integers, 55 labels and booleans, 7 empty lists). Agree: 343. Disagree: 0.
- Not compared: `time` and `descriptive.minutes`. The analysis job wrote these values; they are not data.
- Largest float difference: 2.04e-13, at `accuracy.d_acc_ci[1]`. The largest difference of a Δleak value is 7.8e-16.
- `results/table.md`: all 23 lines (64 decimal numbers) are equal to the text that the script builds from the recomputed values.
- Hashes: 15 of 15 agree.
- Validity: V1, V2, V3, V4, V5 and V6 pass. There are 162 check records and 162 last records. No record has ok = false.
- Primary: Δleak (Decoder-L) = 0.609 dB, two-level CI 0.547 to 0.664 dB. Reading R1, "L+ replicates". The CI lower bound is at or above 0.5 dB.
- Δacc = 13.85 points, CI 13.15 to 14.57. Outcome: A+ replicates.

## 4. Comparison table (results.json against the recomputation)

Each list is one row. The "Largest difference" column gives the largest absolute difference in that row. The rows `descriptive.V1[]`, `descriptive.V2[]`, `descriptive.N_dev_acc[]` and `best_epoch` compare a copy of the source record (`checks.jsonl`, `N.json`, `dec/<tag>.json`). Section 5 gives the independent recomputation of the V1 closed form and of F0 on D_dev.

| Quantity | results.json value | Recomputed value | Largest difference | Agree |
|---|---|---|---|---|
| `seeds[]` (6 values) | 20261121, 20261122, 20261123, 20261124, 20261125, 20261126 | 20261121, 20261122, 20261123, 20261124, 20261125, 20261126 | 0 | yes |
| `floors.F0` | 12.692792 | 12.692792 | 8.88e-15 | yes |
| `floors.FY` | 13.065111 | 13.065111 | 0.00e+00 | yes |
| `checks.V1.ok` | true | true | exact | yes |
| `checks.V1.records` | 12 | 12 | 0 | yes |
| `checks.V2.ok` | true | true | exact | yes |
| `checks.V2.records` | 2 | 2 | 0 | yes |
| `checks.V3.ok` | true | true | exact | yes |
| `checks.V3.records` | 124 | 124 | 0 | yes |
| `checks.V4.ok` | true | true | exact | yes |
| `checks.V4.records` | 18 | 18 | 0 | yes |
| `checks.V5.ok` | true | true | exact | yes |
| `checks.V5.records` | 6 | 6 | 0 | yes |
| `checks.V6.ok` | true | true | exact | yes |
| `checks.V6.records` | 6 | 6 | 0 | yes |
| `check_failures.V1` | [] | [] | exact | yes |
| `check_failures.V2` | [] | [] | exact | yes |
| `check_failures.V3` | [] | [] | exact | yes |
| `check_failures.V4` | [] | [] | exact | yes |
| `check_failures.V5` | [] | [] | exact | yes |
| `check_failures.V6` | [] | [] | exact | yes |
| `failed_records_all` | [] | [] | exact | yes |
| `valid` | true | true | exact | yes |
| `primary.d_leak` | 0.609203 | 0.609203 | 0.00e+00 | yes |
| `primary.d_leak_ci[]` (2 values) | 0.546743, 0.664031 | 0.546743, 0.664031 | 7.77e-16 | yes |
| `primary.d_leak_seeds[]` (6 values) | 0.654725, 0.579531, 0.479826, 0.686805, 0.64746, 0.606869 | 0.654725, 0.579531, 0.479826, 0.686805, 0.64746, 0.606869 | 0.00e+00 | yes |
| `primary.seed_sd` | 0.073732 | 0.073732 | 0.00e+00 | yes |
| `primary.t_ci[]` (2 values) | 0.531826, 0.68658 | 0.531826, 0.68658 | 0.00e+00 | yes |
| `primary.reading` | "R1" | "R1" | exact | yes |
| `primary.ci_low_ge_margin` | true | true | exact | yes |
| `primary.reading_final` | "R1" | "R1" | exact | yes |
| `primary.reading_text` | "L+ replicates" | "L+ replicates" | exact | yes |
| `primary.sym` | "TOPK@64" | "TOPK@64" | exact | yes |
| `primary.act` | "BOT@64" | "BOT@64" | exact | yes |
| `primary.decoder` | "L" | "L" | exact | yes |
| `primary.bits.TOPK@64` | 58 | 58 | 0 | yes |
| `primary.bits.BOT@64` | 64 | 64 | 0 | yes |
| `accuracy.d_acc` | 13.853333 | 13.853333 | 0.00e+00 | yes |
| `accuracy.d_acc_ci[]` (2 values) | 13.146625, 14.568375 | 13.146625, 14.568375 | 2.04e-13 | yes |
| `accuracy.d_acc_seeds[]` (6 values) | 13.75, 13.6, 14.7, 14.29, 13.67, 13.11 | 13.75, 13.6, 14.7, 14.29, 13.67, 13.11 | 0.00e+00 | yes |
| `accuracy.outcome` | "A+ replicates" | "A+ replicates" | exact | yes |
| `secondary.decoder_S_new_seeds.d_leak` | 0.614891 | 0.614891 | 0.00e+00 | yes |
| `secondary.decoder_S_new_seeds.d_leak_ci[]` (2 values) | 0.546944, 0.674893 | 0.546944, 0.674893 | 3.33e-16 | yes |
| `secondary.decoder_S_new_seeds.d_leak_seeds[]` (6 values) | 0.650299, 0.5914, 0.472552, 0.712989, 0.657311, 0.604794 | 0.650299, 0.5914, 0.472552, 0.712989, 0.657311, 0.604794 | 0.00e+00 | yes |
| `secondary.decoder_S_new_seeds.seed_sd` | 0.081964 | 0.081964 | 0.00e+00 | yes |
| `secondary.decoder_S_new_seeds.t_ci[]` (2 values) | 0.528875, 0.700907 | 0.528875, 0.700907 | 0.00e+00 | yes |
| `secondary.decoder_S_new_seeds.reading` | "R1" | "R1" | exact | yes |
| `secondary.decoder_S_new_seeds.ci_low_ge_margin` | true | true | exact | yes |
| `secondary.sealed_rule_L.d_acc` | 13.853333 | 13.853333 | 0.00e+00 | yes |
| `secondary.sealed_rule_L.d_acc_ci[]` (2 values) | 13.355, 14.361667 | 13.355, 14.361667 | 0.00e+00 | yes |
| `secondary.sealed_rule_L.d_acc_seeds[]` (6 values) | 13.75, 13.6, 14.7, 14.29, 13.67, 13.11 | 13.75, 13.6, 14.7, 14.29, 13.67, 13.11 | 1.78e-15 | yes |
| `secondary.sealed_rule_L.d_leak` | 0.609203 | 0.609203 | 1.11e-16 | yes |
| `secondary.sealed_rule_L.d_leak_ci[]` (2 values) | 0.586545, 0.631816 | 0.586545, 0.631816 | 1.11e-16 | yes |
| `secondary.sealed_rule_L.d_leak_seeds[]` (6 values) | 0.654725, 0.579531, 0.479826, 0.686805, 0.64746, 0.606869 | 0.654725, 0.579531, 0.479826, 0.686805, 0.64746, 0.606869 | 0.00e+00 | yes |
| `secondary.sealed_rule_L.A` | "A+" | "A+" | exact | yes |
| `secondary.sealed_rule_L.L` | "L+" | "L+" | exact | yes |
| `secondary.sealed_rule_L.case` | "S" | "S" | exact | yes |
| `secondary.sealed_rule_L.L_hits[0]` | "L+" | "L+" | exact | yes |
| `secondary.sealed_rule_S.d_acc` | 13.853333 | 13.853333 | 0.00e+00 | yes |
| `secondary.sealed_rule_S.d_acc_ci[]` (2 values) | 13.355, 14.361667 | 13.355, 14.361667 | 0.00e+00 | yes |
| `secondary.sealed_rule_S.d_acc_seeds[]` (6 values) | 13.75, 13.6, 14.7, 14.29, 13.67, 13.11 | 13.75, 13.6, 14.7, 14.29, 13.67, 13.11 | 1.78e-15 | yes |
| `secondary.sealed_rule_S.d_leak` | 0.614891 | 0.614891 | 0.00e+00 | yes |
| `secondary.sealed_rule_S.d_leak_ci[]` (2 values) | 0.593056, 0.636193 | 0.593056, 0.636193 | 0.00e+00 | yes |
| `secondary.sealed_rule_S.d_leak_seeds[]` (6 values) | 0.650299, 0.5914, 0.472552, 0.712989, 0.657311, 0.604794 | 0.650299, 0.5914, 0.472552, 0.712989, 0.657311, 0.604794 | 0.00e+00 | yes |
| `secondary.sealed_rule_S.A` | "A+" | "A+" | exact | yes |
| `secondary.sealed_rule_S.L` | "L+" | "L+" | exact | yes |
| `secondary.sealed_rule_S.case` | "S" | "S" | exact | yes |
| `secondary.sealed_rule_S.L_hits[0]` | "L+" | "L+" | exact | yes |
| `secondary.pooled_S_sealed_and_new.d_leak` | 0.593129 | 0.593129 | 0.00e+00 | yes |
| `secondary.pooled_S_sealed_and_new.d_leak_ci[]` (2 values) | 0.535891, 0.649027 | 0.535891, 0.649027 | 2.22e-16 | yes |
| `secondary.pooled_S_sealed_and_new.d_leak_seeds[]` (9 values) | 0.512432, 0.48908, 0.647302, 0.650299, 0.5914, 0.472552, 0.712989, 0.657311, 0.604794 | 0.512432, 0.48908, 0.647302, 0.650299, 0.5914, 0.472552, 0.712989, 0.657311, 0.604794 | 0.00e+00 | yes |
| `secondary.pooled_S_sealed_and_new.seed_sd` | 0.084191 | 0.084191 | 0.00e+00 | yes |
| `secondary.pooled_S_sealed_and_new.t_ci[]` (2 values) | 0.528414, 0.657843 | 0.528414, 0.657843 | 0.00e+00 | yes |
| `secondary.pooled_S_sealed_and_new.reading` | "R1" | "R1" | exact | yes |
| `secondary.pooled_S_sealed_and_new.ci_low_ge_margin` | true | true | exact | yes |
| `secondary.pooled_S_sealed_and_new.seeds[]` (9 values) | 20261111, 20261112, 20261113, 20261121, 20261122, 20261123, 20261124, 20261125, 20261126 | 20261111, 20261112, 20261113, 20261121, 20261122, 20261123, 20261124, 20261125, 20261126 | 0 | yes |
| `secondary.sealed_seed1_L` | 0.50391 | 0.50391 | 0.00e+00 | yes |
| `secondary.decoder_gap_L_minus_S.d_leak_seeds[]` (6 values) | 0.004426, -0.011868, 0.007274, -0.026183, -0.009851, 0.002075 | 0.004426, -0.011868, 0.007274, -0.026183, -0.009851, 0.002075 | 0.00e+00 | yes |
| `secondary.decoder_gap_L_minus_S.psnr.TOPK@64[]` (6 values) | 0.002105, 0.000361105, -0.001115, 0.002747, 0.003677, 0.001832 | 0.002105, 0.000361105, -0.001115, 0.002747, 0.003677, 0.001832 | 0.00e+00 | yes |
| `secondary.decoder_gap_L_minus_S.psnr.BOT@64[]` (6 values) | 0.006531, -0.011507, 0.006159, -0.023437, -0.006174, 0.003906 | 0.006531, -0.011507, 0.006159, -0.023437, -0.006174, 0.003906 | 0.00e+00 | yes |
| `secondary.decoder_gap_L_minus_S.psnr.LAB[]` (6 values) | 0.001949, -0.000321433, 0.001429, -8.6763e-05, -0.001121, 0.000389606 | 0.001949, -0.000321433, 0.001429, -8.6763e-05, -0.001121, 0.000389606 | 0.00e+00 | yes |
| `descriptive.N_dev_acc[]` (6 values) | 0.8872, 0.8882, 0.8866, 0.8848, 0.8812, 0.8846 | 0.8872, 0.8882, 0.8866, 0.8848, 0.8812, 0.8846 | 0.00e+00 | yes |
| `descriptive.TOPK@64.acc_seeds[]` (6 values) | 0.8291, 0.8299, 0.8315, 0.8197, 0.8247, 0.8266 | 0.8291, 0.8299, 0.8315, 0.8197, 0.8247, 0.8266 | 0.00e+00 | yes |
| `descriptive.TOPK@64.acc` | 0.826917 | 0.826917 | 0.00e+00 | yes |
| `descriptive.TOPK@64.psnr_L_seeds[]` (6 values) | 13.395524, 13.406449, 13.434741, 13.431066, 13.440228, 13.45025 | 13.395524, 13.406449, 13.434741, 13.431066, 13.440228, 13.45025 | 0.00e+00 | yes |
| `descriptive.TOPK@64.psnr_L` | 13.426376 | 13.426376 | 0.00e+00 | yes |
| `descriptive.TOPK@64.psnr_S_seeds[]` (6 values) | 13.393419, 13.406088, 13.435856, 13.42832, 13.436551, 13.448418 | 13.393419, 13.406088, 13.435856, 13.42832, 13.436551, 13.448418 | 0.00e+00 | yes |
| `descriptive.TOPK@64.psnr_S` | 13.424775 | 13.424775 | 0.00e+00 | yes |
| `descriptive.TOPK@64.best_epoch_L[]` (6 values) | 12, 12, 12, 12, 12, 14 | 12, 12, 12, 12, 12, 14 | 0 | yes |
| `descriptive.TOPK@64.best_epoch_S[]` (6 values) | 12, 12, 12, 12, 14, 14 | 12, 12, 12, 12, 14, 14 | 0 | yes |
| `descriptive.BOT@64.acc_seeds[]` (6 values) | 0.6916, 0.6939, 0.6845, 0.6768, 0.688, 0.6955 | 0.6916, 0.6939, 0.6845, 0.6768, 0.688, 0.6955 | 0.00e+00 | yes |
| `descriptive.BOT@64.acc` | 0.688383 | 0.688383 | 0.00e+00 | yes |
| `descriptive.BOT@64.psnr_L_seeds[]` (6 values) | 14.050249, 13.985981, 13.914567, 14.117872, 14.087688, 14.057119 | 14.050249, 13.985981, 13.914567, 14.117872, 14.087688, 14.057119 | 0.00e+00 | yes |
| `descriptive.BOT@64.psnr_L` | 14.035579 | 14.035579 | 0.00e+00 | yes |
| `descriptive.BOT@64.psnr_S_seeds[]` (6 values) | 14.043718, 13.997488, 13.908408, 14.141308, 14.093862, 14.053212 | 14.043718, 13.997488, 13.908408, 14.141308, 14.093862, 14.053212 | 0.00e+00 | yes |
| `descriptive.BOT@64.psnr_S` | 14.039666 | 14.039666 | 0.00e+00 | yes |
| `descriptive.BOT@64.best_epoch_L[]` (6 values) | 8, 8, 10, 6, 6, 8 | 8, 8, 10, 6, 6, 8 | 0 | yes |
| `descriptive.BOT@64.best_epoch_S[]` (6 values) | 8, 8, 8, 8, 6, 6 | 8, 8, 8, 8, 6, 6 | 0 | yes |
| `descriptive.LAB.acc_seeds[]` (6 values) | 0.8291, 0.8299, 0.8315, 0.8197, 0.8247, 0.8266 | 0.8291, 0.8299, 0.8315, 0.8197, 0.8247, 0.8266 | 0.00e+00 | yes |
| `descriptive.LAB.acc` | 0.826917 | 0.826917 | 0.00e+00 | yes |
| `descriptive.LAB.psnr_L_seeds[]` (6 values) | 13.085167, 13.090841, 13.095181, 13.090378, 13.086424, 13.087128 | 13.085167, 13.090841, 13.095181, 13.090378, 13.086424, 13.087128 | 0.00e+00 | yes |
| `descriptive.LAB.psnr_L` | 13.089187 | 13.089187 | 0.00e+00 | yes |
| `descriptive.LAB.psnr_S_seeds[]` (6 values) | 13.083218, 13.091163, 13.093751, 13.090465, 13.087545, 13.086739 | 13.083218, 13.091163, 13.093751, 13.090465, 13.087545, 13.086739 | 0.00e+00 | yes |
| `descriptive.LAB.psnr_S` | 13.088813 | 13.088813 | 0.00e+00 | yes |
| `descriptive.LAB.best_epoch_L[]` (6 values) | 20, 20, 20, 20, 18, 20 | 20, 20, 20, 20, 18, 20 | 0 | yes |
| `descriptive.LAB.best_epoch_S[]` (6 values) | 20, 20, 20, 20, 18, 20 | 20, 20, 20, 20, 18, 20 | 0 | yes |
| `descriptive.V1[].seed` (12 values) | 20261121, 20261121, 20261122, 20261122, 20261123, 20261123, 20261124, 20261124, 20261125, 20261125, 20261126, 20261126 | 20261121, 20261121, 20261122, 20261122, 20261123, 20261123, 20261124, 20261124, 20261125, 20261125, 20261126, 20261126 | 0 | yes |
| `descriptive.V1[].where` (12 values) | "L1_LAB", "S1_LAB", "L1_LAB", "S1_LAB", "L1_LAB", "S1_LAB", "L1_LAB", "S1_LAB", "L1_LAB", "S1_LAB", "L1_LAB", "S1_LAB" | "L1_LAB", "S1_LAB", "L1_LAB", "S1_LAB", "L1_LAB", "S1_LAB", "L1_LAB", "S1_LAB", "L1_LAB", "S1_LAB", "L1_LAB", "S1_LAB" | exact | yes |
| `descriptive.V1[].dev_psnr` (12 values) | 13.065557, 13.063392, 13.070976, 13.071062, 13.068734, 13.067018, 13.065959, 13.065629, 13.056657, 13.058397, 13.058969, 13.05877 | 13.065557, 13.063392, 13.070976, 13.071062, 13.068734, 13.067018, 13.065959, 13.065629, 13.056657, 13.058397, 13.058969, 13.05877 | 0.00e+00 | yes |
| `descriptive.V1[].closed` (12 values) | 13.065025, 13.065025, 13.070536, 13.070536, 13.068061, 13.068061, 13.066002, 13.066002, 13.057471, 13.057471, 13.060261, 13.060261 | 13.065025, 13.065025, 13.070536, 13.070536, 13.068061, 13.068061, 13.066002, 13.066002, 13.057471, 13.057471, 13.060261, 13.060261 | 0.00e+00 | yes |
| `descriptive.V1[].ok` (12 values) | true, true, true, true, true, true, true, true, true, true, true, true | true, true, true, true, true, true, true, true, true, true, true, true | exact | yes |
| `descriptive.V2[].seed` (2 values) | 20261121, 20261121 | 20261121, 20261121 | 0 | yes |
| `descriptive.V2[].where` (2 values) | "L1_BOT@64_ctrl", "L1_TOPK@64_ctrl" | "L1_BOT@64_ctrl", "L1_TOPK@64_ctrl" | exact | yes |
| `descriptive.V2[].dev_psnr` (2 values) | 12.683111, 12.675274 | 12.683111, 12.675274 | 0.00e+00 | yes |
| `descriptive.V2[].F0` (2 values) | 12.680559, 12.680559 | 12.680559, 12.680559 | 0.00e+00 | yes |
| `descriptive.V2[].ok` (2 values) | true, true | true, true | exact | yes |

## 5. Checks outside results.json

| Check | Expected value | Recomputed or found value | Difference | Agree |
|---|---|---|---|---|
| sha256 of the 8 frozen files | `FREEZE.json` | equal (8 of 8) | none | yes |
| sha256 of the 3 sealed npz files | `data.npz_sha256` | equal (3 of 3) | none | yes |
| sha256 of the sealed `FREEZE.json` | `sealed_freeze_sha256` | equal | none | yes |
| npz record of `FREEZE.json` | the record of the sealed `FREEZE.json` | equal | none | yes |
| `sebits.py` and `selection.json` | byte copies of the sealed files | equal sha256 | none | yes |
| V1 closed form on D_dev, 6 seeds (recomputed from the LAB messages) | the `closed` values of the V1 records | equal | 0.0 | yes |
| F0 PSNR on D_dev (recomputed) | 12.68055937201839 | 12.680559372018397 | 7.1e-15 | yes |
| V1 ok, re-derived as dev_psnr ≥ closed − 0.2 | 12 records ok | 12 ok | none | yes |
| V2 ok, re-derived as dev_psnr ≤ F0 + 0.2 | 2 records ok | 2 ok | none | yes |
| V1 dev_psnr of `checks.jsonl` against `dec/<tag>.json` | equal | equal (12 of 12) | 0.0 | yes |
| `dev_psnr` of `dec/<tag>.json` against the mean of `psnr_dev` of the npz | equal | equal (38 of 38) | 0.0 | yes |
| V3: a start record and an end record for each unit after N | 124 | 124, all ok | none | yes |
| V4: `msgs TOPK@64`, `msgs BOT@64`, `msgs LAB` in each seed | 18 records, bits 58, 64, 4 | 18, all ok | none | yes |
| V5: done markers (13 in seed 20261121, 11 in each other seed) | 68 | 68 | none | yes |
| V5: decoder outputs (8 in seed 20261121, 6 in each other seed) | 38 | 38 | none | yes |
| V5: decoder losses | all finite | all finite | none | yes |
| V6: one record for each seed, dev_acc ≥ 0.80, equal to `N.json` | 6 | 6 | none | yes |
| Records with ok = false (all records, not only the last) | 0 | 0 | none | yes |
| `nbits` of the message files, each seed | TOPK@64 58, BOT@64 64, LAB 4 | 58, 64, 4 | none | yes |
| `bits` of the arm files, each seed | 58, 64, 4 | 58, 64, 4 | none | yes |
| BOT@64 `pred_test` = first maximum of `logits_test` | 6 seeds | 6 of 6 | none | yes |
| LAB `pred_test` = the LAB message (bits >> 4) and = TOPK@64 `pred_test` | 6 seeds | 6 of 6 | none | yes |
| Distinct reconstruction arrays (38 new, 6 sealed) | 44 | 44 | none | yes |
| Test labels for each class | 1,000 | 1,000 | none | yes |

Sensitivity check (not saved in `audit/`): if the image indices come before the seed indices in each chunk, the primary CI bounds move by 4.5e-4 and 9.1e-4. Thus the agreement at 1e-13 confirms the exact draw order of Part 6.3.

## 6. Disagreements

None. Each recomputed value agrees with `results.json` and with `results/table.md`.

## 7. Notes on the data (no effect on a reading)

1. FY tie rule. The FY value of `results.json` agrees with a round half to even (the numpy and torch default). A round half up gives FY = 13.0651104 dB (a difference of 1.2e-7). The sealed Part 5.4 gives no tie rule. F0 and the V1 closed form do not change.
2. Value of t. Part 6.4 of the preregistration gives t = 2.5706. `results.json` uses t = 2.570582 (the implied value of its intervals). With t = 2.5706, the bounds move by about 5e-7 dB. This interval is descriptive only.
3. Few seeds. With K = 6, the two-level percentile CI (0.547 to 0.664 dB) is narrower than the Student t interval (0.532 to 0.687 dB). A percentile bootstrap over 6 seeds can give too small a seed variance. Both lower bounds are above 0.5 dB, so R1 and the margin statement do not change.
4. One seed is below the margin. Seed 20261123 gives Δleak = 0.480 dB (Decoder-L) and 0.473 dB (Decoder-S). The rule uses the pooled value, so the reading does not change. In the pooled K = 9 row, 2 of 9 seeds are below 0.5 dB.
5. Decoder size. Decoder-L and Decoder-S give almost the same values. The largest gap is 0.026 dB in Δleak for a seed and 0.023 dB in the PSNR of an arm. Here the larger decoder does not rebuild more of the image.
6. LAB decoders. In 10 of 12 LAB decoders, the best epoch is the last epoch (20). These decoders possibly did not reach their lowest D_dev MSE. This touches only V1 (which passes by about 0.2 dB) and the LAB rows.
7. File order. `REPORT.md` has a time of 09:35, before this audit. Part 9.1, item 4, puts `AUDIT.md` before `REPORT.md`. The auditor did not read `REPORT.md`. Make sure that `REPORT.md` agrees with this audit before `SEAL.json`.

## 8. Verdict

Independent check: PASS
