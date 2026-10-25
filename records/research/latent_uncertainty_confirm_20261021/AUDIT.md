# AUDIT (job AU): independent audit of the confirmatory uncertainty run

- **Status:** independent audit of a CONFIRMATORY run. The run follows `PREREGISTRATION.md`, frozen by `FREEZE.json` at 2026-10-21 22:48:47.
- **Verdict:** **audit passes.** **Case 3, "no state advantage over the sender's own text", S\* = T5.**
- **Date:** 2026-10-22. **Auditor:** a separate Claude Code session (job AU). The parent session started it.
- **Approval:** [researcher] approved AU in `PHASE_C_APPROVAL.md`. [researcher] confirmed it in `PHASE_C_STOP_R5.md` ([redacted: private chat]).
- **Limits of the audit:** CPU only, numpy only. No model load, no GPU, no download, no commit. The audit read the run files and did not change them. It wrote one file in this folder: `AUDIT.md`. The audit scripts are outside the folder (section 11).
- **Style:** ASD-STE100.

---

## 1. Verdict

**Audit passes.**

- My independent values give **case 3, "no state advantage over the sender's own text"**, with **S\* = T5**. R8 reports the same case and the same S\*.
- All 17 arm means and all 13 row means agree with `confirm/results.json` to 1e-6. The largest difference is 1.1e-16.
- All 95% CI ends agree within 0.003. The largest difference is 0.00024. The cause is the different bootstrap seed.
- All per-seed values agree to 1e-6. All "beats" flags, V1, V3, V4 and `V2_descriptive` are the same.
- No difference is outside the tolerances of the task. Section 7 lists every difference.

---

## 2. What the audit checked

1. **Freeze.** I computed the sha256 of the 7 files in `FREEZE.json`. Each value is equal to the frozen value (section 11). The code hash in `results.json` (`cba5e677...`) is equal to the frozen hash of `uconfirm.py`.
2. **Layout.** I read `uconfirm.py` only for the file layout and the label definition. I read these parts:
   - the constants (seeds, clip, margin, grid, F3 levels);
   - `normalize_answer`, `alias_set`, `label_alias`, `parse_answer`, `select_candidates`, `sample_entropy`, `se_labels`, `fmt_value`, `fmt_feature`;
   - `load_sender_out`, `cand_mask`, `cand_labels` and the start of `stage_probes`;
   - `analysis_body`, to learn which file gives which key of `confirm/test_probs.npz`.

   I did not read the bodies of `reading_rule`, `bootstrap_ci`, `c_round_cost`, `cv_logistic` or `fit_logistic`. I did not import `uconfirm.py`, and I did not copy its code.
3. **Blind order.** My script `au_recompute.py` wrote all my values to `au_values.json` first. I opened `confirm/results.json` and `confirm/table.md` only after that.
4. **Labels.** I wrote my own TriviaQA normalization from the rule in P1. y = 1 if the normalized c1 is in the normalized alias set of the item. The inputs are `items/items.json` and `confirm/sender_*.json`.
5. **NLL.** For each arm and seed, I clipped p to [1e-6, 1 - 1e-6]. The item NLL is -(y log p + (1 - y) log(1 - p)). NLL_i of an arm is the mean over the 3 seeds (design 11).
6. **S\*.** S\* is the arm with the lowest mean NLL among T0, T0c, T1, T2, T5 and T6 (design 12.1).
7. **Rows.** For "A vs B", d_i = NLL_i(B) - NLL_i(A).
   - The mean is the mean of d_i over the 6,000 test items.
   - The 95% CI comes from a paired bootstrap over test items: 10,000 draws, my seed 77720261022 (the run used 20261021), the 2.5 and 97.5 percentiles.
   - The per-seed value is the mean of NLL(B) - NLL(A) for that seed alone.
   - "A beats B" if the mean is at least 0.02, the CI lower bound is above 0, and each per-seed value is above 0.
8. **c_round.** I made my own L2 logistic fit. It does Newton steps on the mean NLL plus lam/2 times the sum of the squared weights. The bias has no penalty.
   - Features (P15): for each slot, the one-hot of the F3 word of the probe value (formatted) or the probe value (unformatted), times the slot-present flag. Then the 3 slot-present flags.
   - The fit standardizes with train statistics. It selects the L2 weight from the grid of design 8.1 by 5-fold CV NLL over items, with my own folds. Then it fits one time on the full train split.
   - c_round for a seed is max(0, test NLL(formatted) - test NLL(unformatted)). The rule uses the mean over the 3 seeds.
9. **Validity.** V1 comes from my arm means. V2 is descriptive (amendment A1). V3 comes from `confirm/hashes.jsonl` and from the receiver parameter hashes in `seed*/latent.json` and `seed*/latent_*.json`. V4 comes from the `finite` flags, the train losses in `seed*/latent_*.json` and the test probabilities of the 4 latent arms.
10. **Case.** I applied design 12.2 and 12.3, with A1, in the given order.
11. **Origin of the probabilities.** In `test_probs.npz`, the probabilities of L, L-shuf, L-testshuf and L-probe are equal, bit for bit, to `p_<arm>_eval` in `seed*/latent.npz` (12 of 12). probe-direct is equal to column 0 of `eval` in `seed*/probes.npz` (3 of 3).
12. **Extra check (descriptive).** I fit the text readouts of T5, T6, T2, T0c and T4 again from the saved receiver final states, with my own fit (section 8).
13. **Logs.** `confirm_wrapper.log` has one analysis job: start 2026-10-22 10:08:44, end 10:37:28, rc 0. `confirm.log` has one "stage analysis" line. The files `test_probs.npz`, `results.json` and `table.md` have the time 10:37:28.

---

## 3. Primary rows: audit values next to the run values

"Mean" is the mean of NLL(B) - NLL(A) in nats. A positive mean is in favor of A. "Largest CI difference" is the larger of the two CI end differences.

| Row (A vs B) | Mean, audit | Mean, run | Mean difference | 95% CI, audit | 95% CI, run | Largest CI difference | Per-seed signs (audit / run) | A beats B (audit / run) |
|---|---|---|---|---|---|---|---|---|
| L vs S\* (T5) | +0.000896 | +0.000896 | 0.0 | [-0.0075, +0.0091] | [-0.0076, +0.0093] | 0.00017 | + + - / + + - | no / no |
| L vs T4 | -0.020365 | -0.020365 | 0.0 | [-0.0273, -0.0136] | [-0.0274, -0.0134] | 0.00022 | - - - / - - - | no / no |
| T4 vs L | +0.020365 | +0.020365 | 0.0 | [+0.0136, +0.0273] | [+0.0134, +0.0274] | 0.00022 | + + + / + + + | yes / yes |
| L vs L-probe | -0.040378 | -0.040378 | 0.0 | [-0.0474, -0.0333] | [-0.0473, -0.0335] | 0.00021 | - - - / - - - | no / no |
| L-probe vs T4 | +0.020013 | +0.020013 | 0.0 | [+0.0127, +0.0272] | [+0.0128, +0.0273] | 0.00011 | + + + / + + + | yes / yes |
| T4 vs T5 | +0.021261 | +0.021261 | 0.0 | [+0.0164, +0.0262] | [+0.0163, +0.0262] | 0.00012 | + + + / + + + | yes / yes |

Per-seed values (audit; the run values are the same to 1e-6), seeds 20261101, 20261102 and 20261103:

- L vs T5: +0.0093, +0.0103, -0.0169.
- L vs T4: -0.0123, -0.0109, -0.0379.
- L vs L-probe: -0.0399, -0.0151, -0.0661.
- L-probe vs T4: +0.0276, +0.0042, +0.0282.
- T4 vs T5: +0.0216, +0.0212, +0.0209.

---

## 4. S\*, c_round, validity and the case

| Item | Audit | Run | Same |
|---|---|---|---|
| S\* | T5 | T5 | yes |
| Mean NLL of the S\* candidates | T5 0.402958; T6 0.405786; T2 0.408453; T0c 0.413191; T0 0.432276; T1 0.435987 | the same (section 5) | yes |
| c_round (mean of 3 seeds) | 0.010549 | 0.010314 | difference 0.00024 (section 7) |
| c_round per seed | 0.011300, 0.010945, 0.009403 | 0.011034, 0.010706, 0.009203 | differences 0.00020 to 0.00027 |
| Lower CI bound of NLL(T4) - NLL(L) above c_round | no (-0.0273 is below 0.0105) | no | yes |
| V1: NLL(L-shuf) is at least NLL(T0c) - 0.02 | pass (0.452137 is at least 0.393191) | pass | yes |
| V2_descriptive: NLL(Source) is not more than NLL(T0) | false (0.445940 is more than 0.432276) | false | yes |
| V3 | pass: 13 records; 1 hash for each of the 3 weight files; 1 receiver parameter hash (`fe5f2146...`) before and after all 9 latent fits | pass | yes |
| V4 | pass: 9 of 9 latent fits have `finite` true; all train losses are finite; all test probabilities of the 4 latent arms are finite | pass | yes |
| **Case** | **3** | **3** | **yes** |

How the case follows (design 12.2 and 12.3, with A1):

1. Step 1 (validity): V1, V3 and V4 pass. Thus the run is valid. V2 is descriptive under A1 and does not enter case 0.
2. Step 2.1: L does not beat S\* = T5. All three conditions fail:
   - the mean (+0.0009) is less than 0.02;
   - the CI lower bound (-0.0075) is not above 0;
   - the seed 20261103 value (-0.0169) is not above 0.
3. Thus the case is **case 3, "no state advantage over the sender's own text"**. The later steps of 12.3 do not apply.

---

## 5. Mean NLL of each arm

| Arm | Audit | Run (`results.json`) | Difference | `table.md` |
|---|---|---|---|---|
| T0 | 0.432276 | 0.432276 | 0.0 | 0.4323 |
| T0c | 0.413191 | 0.413191 | 0.0 | 0.4132 |
| T1 | 0.435987 | 0.435987 | 5.6e-17 | 0.4360 |
| T2 | 0.408453 | 0.408453 | 5.6e-17 | 0.4085 |
| T5 | 0.402958 | 0.402958 | 5.6e-17 | 0.4030 |
| T6 | 0.405786 | 0.405786 | 5.6e-17 | 0.4058 |
| Source | 0.445940 | 0.445940 | 0.0 | 0.4459 |
| Q-only | 0.457052 | 0.457052 | 5.6e-17 | 0.4571 |
| T3 | 0.384033 | 0.384033 | 5.6e-17 | 0.3840 |
| T4 | 0.381697 | 0.381697 | 5.6e-17 | 0.3817 |
| L | 0.402062 | 0.402062 | 0.0 | 0.4021 |
| L-shuf | 0.452137 | 0.452137 | 5.6e-17 | 0.4521 |
| L-testshuf | 0.532180 | 0.532180 | 0.0 | 0.5322 |
| L-probe | 0.361684 | 0.361684 | 5.6e-17 | 0.3617 |
| probe-direct | 0.340680 | 0.340680 | 5.6e-17 | 0.3407 |
| sender-direct | 0.418906 | 0.418906 | 1.1e-16 | 0.4189 |
| base-rate | 0.531237 | 0.531237 | 1.1e-16 | 0.5312 |

Labels:

- Test y: my labels are equal to the saved `y` on 6,000 of 6,000 items. The test mean of y is 0.219.
- Train y: the mean of my train labels is 0.26475. This is equal to `base_rate_train` in `results.json`.
- c1 is equal to a1 in all train and test items.
- The test probabilities of the 8 seed-free text arms are the same in the 3 seeds. Design 10 permits this: only the readout folds change. The final fit on the full train split is the same when the CV selects the same L2 weight.

---

## 6. Secondary rows (design 12.4) and route loss (design 12.6)

| Row (A vs B) | Mean, audit | Mean, run | 95% CI, audit | 95% CI, run | Per-seed signs (audit / run) | A beats B (audit / run) |
|---|---|---|---|---|---|---|
| T4 vs S\* (T5) | +0.021261 | +0.021261 | [+0.0164, +0.0262] | [+0.0163, +0.0262] | + + + / + + + | yes / yes |
| T4 vs T3 | +0.002336 | +0.002336 | [-0.0048, +0.0093] | [-0.0047, +0.0093] | + + + / + + + | no / no |
| T4 vs T0c | +0.031494 | +0.031494 | [+0.0259, +0.0370] | [+0.0259, +0.0371] | + + + / + + + | yes / yes |
| T5 vs T2 | +0.005495 | +0.005495 | [-0.0011, +0.0119] | [-0.0011, +0.0121] | + + + / + + + | no / no |
| T6 vs T5 | -0.002827 | -0.002827 | [-0.0081, +0.0023] | [-0.0079, +0.0025] | - - - / - - - | no / no |
| L vs T6 | +0.003723 | +0.003723 | [-0.0043, +0.0118] | [-0.0045, +0.0120] | + + - / + + - | no / no |
| L vs T3 | -0.018029 | -0.018029 | [-0.0263, -0.0100] | [-0.0263, -0.0098] | - - - / - - - | no / no |
| L vs L-testshuf | +0.130118 | +0.130118 | [+0.1185, +0.1417] | [+0.1185, +0.1417] | + + + / + + + | yes / yes |
| L-probe vs T4 | +0.020013 | +0.020013 | [+0.0127, +0.0272] | [+0.0128, +0.0273] | + + + / + + + | yes / yes |

Route loss, NLL(arm) - NLL(probe direct) (my own bootstrap, seed 77720261023):

| Arm | Mean, audit | Mean, run | 95% CI, audit | 95% CI, run |
|---|---|---|---|---|
| T3 | +0.043353 | +0.043353 | [+0.0354, +0.0512] | [+0.0353, +0.0513] |
| T4 | +0.041017 | +0.041017 | [+0.0324, +0.0498] | [+0.0321, +0.0497] |
| L | +0.061382 | +0.061382 | [+0.0535, +0.0692] | [+0.0534, +0.0692] |
| L-probe | +0.021005 | +0.021005 | [+0.0136, +0.0285] | [+0.0136, +0.0284] |

All secondary and route-loss means agree to 1e-6. All CI ends agree within 0.003.

---

## 7. Every difference between the audit and the run

1. **Means.** The largest difference is 1.1e-16 (the order of float sums). This is inside the 1e-6 tolerance.
2. **CI ends.** The largest difference is 0.00024 in the rule rows and 0.00024 in the route-loss rows. The cause is the bootstrap seed. This is inside the 0.003 tolerance.
3. **c_round.** Audit 0.010549, run 0.010314 (difference 0.00024).
   - Cause: my own fit and my own folds. The preregistration does not fix the scale of the L2 term or the fold method. My L2 weights were 1e-3 in 5 of 6 fits and 1e-2 in 1 fit.
   - Effect on the case: none. Case 3 comes from step 2.1. c_round enters only in step 2.2. With either value, the c_round condition is false (-0.0273 is below 0.0105).
4. **y_SE (secondary label).** My y_SE differs from the saved `y_SE` on 18 of 6,000 test items.
   - Cause: ties. 118 test items have an SE equal to the train median (1.3592367). The rule is "strictly above the median". My entropy sum (in sorted answer order) gives these items a value 2.2e-16 above the median, so 18 of them become 1.
   - When I sum in draw order, my y_SE is equal to the saved `y_SE` on 6,000 of 6,000 items. My SE values are equal to the saved `se` values to 4.4e-16.
   - Effect: none on the case. No row of the reading rule uses y_SE. This is not an error of the run.
5. **Draft text in the frozen code.** The frozen `uconfirm.py` keeps its draft header. The docstring says "DRAFT harness (EXPLORATORY code, not frozen, not run on a model or on real data)", and `STATUS = 'DRAFT'`. Thus line 127 of `confirm.log` starts with "DRAFT uconfirm.py confirm".
   - The status in `results.json` and `table.md` is correct: "CONFIRMATORY (frozen run; the independent audit is still to do)".
   - Effect: none on any value. The file is frozen and must not change. The report must explain this text.
6. **Text readouts fit again (extra check).** The test NLLs differ by 0.003 or less (section 8). This check is descriptive and is not a pass condition.

There are no other differences.

---

## 8. Extra check: text readouts fit again from the saved states (descriptive)

The case depends on the text-arm probabilities, because S\* = T5. The run made these probabilities with its readout code, and the audit did not read that code. Thus I fit the readouts again from the saved receiver final states, with my own fit (the method of section 2, item 8, with the 960 state values as features). The inputs are `confirm/recv/<arm>_train.npz`, `confirm/recv/<arm>_test.npz` and `confirm/seed*/recv_T4_*.npz`. The train labels are my own labels (section 5).

| Arm | Test NLL, my refit | Test NLL, saved | Difference | Correlation of p | My L2 weight |
|---|---|---|---|---|---|
| T5 | 0.403036 | 0.402958 | +0.00008 | 0.9953 | 0.1 |
| T6 | 0.404681 | 0.405786 | -0.00110 | 0.9951 | 0.1 |
| T2 | 0.405580 | 0.408453 | -0.00287 | 0.9941 | 0.1 |
| T0c | 0.413779 | 0.413191 | +0.00059 | 0.9948 | 0.1 |
| T4, seed 20261101 | 0.379507 | 0.381330 | -0.00182 | 0.9958 | 0.1 |
| T4, seed 20261102 | 0.380395 | 0.381751 | -0.00136 | 0.9958 | 0.1 |
| T4, seed 20261103 | 0.380539 | 0.382010 | -0.00147 | 0.9957 | 0.1 |

Result:

- With my refit values, the best sender-output text arm is still T5.
- L against my refit T5: mean +0.0010, CI [-0.0081, +0.0100], per seed +0.0094, +0.0104, -0.0169. L does not beat it. Thus the refit also gives case 3.
- The saved text probabilities agree with an independent fit on the train split only. This check does not show that the two fits are identical: the L2 scale and the folds are different. Single-item probabilities differ by up to 0.19.

---

## 9. Notes for the report (not part of the verdict)

1. **Case 3 has a clear margin.** L fails all three conditions against S\*. L also does not reach the 0.02 mean against T6 (+0.0037), T2 (+0.0064) or T0c (+0.0111). Thus any S\* among T5, T6, T2 and T0c gives case 3. L reaches the 0.02 mean only against T0 (+0.0302) and T1 (+0.0339). These arms are not S\*.
2. **The T4 against S\* row (design 13, case 3).** T4 beats T5: mean +0.0213, CI [+0.0164, +0.0262], signs + + +. The mean is only 0.0013 above the 0.02 margin. The report must give this row. Under design 13, it may then say that a probe still reads more from the state than the sender's outputs show.
3. **The 2a rows are not the case.** T4 beats L (mean +0.0204, signs + + +), but the reading rule stops at step 2.1. The report must give case 3, not case 2a.
4. **L-probe beats T4 by a very small margin.** The mean is +0.020013. This is 0.000013 above the 0.02 margin. The per-seed value of seed 20261102 is +0.0042. Both the audit and the run give "beats", because the means agree to 1e-16. The report must show how near this row is to the margin.
5. **Amendment A1.** `V2_descriptive` is false (Source 0.4459 against T0 0.4323), as in Phase D. No statement of the report may call Source a ceiling.

---

## 10. How to repeat the audit

The scripts read the folder and write their outputs to the current directory. Run them in this order:

```bash
cd "<local>/Library/Application Support/Claude/side-session-notes/local_cd217fcd-565e-4198-8ba5-572cabf2f79d/uncertainty-confirm-audit-au"
R=research/latent_uncertainty_confirm_20261021
PY=.venv/bin/python
$PY -I au_recompute.py $R au_values.json
$PY -I au_compare.py $R au_values.json au_compare.json
$PY -I au_refit.py $R au_values.json au_refit.json
```

Run times on the laptop: `au_recompute.py` 3 s, `au_compare.py` a few seconds, `au_refit.py` about 40 s.

---

## 11. Files and hashes

Frozen files (`FREEZE.json`), sha256 on 2026-10-22, before the audit:

| File | sha256 now | Equal to `FREEZE.json` |
|---|---|---|
| `PREREGISTRATION.md` | `e4551e35ffe8622a7450964e345c9b73afb5d97abb378ef7f6e0451b4406f135` | yes |
| `uconfirm.py` | `cba5e677c6e98905632c2d5c5483b9155e84e13a3bb8282ddf11b5056d076d4c` | yes |
| `test_uconfirm.py` | `cd50a036019f336ddbf1a8d8fcc10f1ecd0567a86b980b06d33c47b6545da14f` | yes |
| `run_uconfirm.sh` | `fa76e46bd0098e150fb4cfdc23246fbdfbf55c5968f8dce25ff2297c840f9614` | yes |
| `fetch_data.py` | `fede2ed9d6f625ad5271c2f9d7a7d4ed5226740e08142a0692d7f59c8f961291` | yes |
| `dev/selection.json` | `838799334c06991f146e47dbf63fa089e8284743639c355aeaf6acb8a6f5def3` | yes |
| `items/items.json` | `e8dfd9c4e790fe3854b995d548faad41d2033a3684760be101195343191895f9` | yes |

Run outputs that the audit read:

| File | sha256 |
|---|---|
| `confirm/test_probs.npz` | `eeba84a3e152dc5ef754dd0ea1ad5bdeb71ef02a8497399405048c590ee1b0a9` |
| `confirm/results.json` | `77c6fdac3e385c6dff0a3b8a68992c1bcbb782b6aaa31cdf590f7c6236c98ce2` |
| `confirm/table.md` | `b88f864fa714983dfeac64609bfbb5bd9665a67770ed41a54edb950ddf52f5c3` |
| `confirm/hashes.jsonl` | `5ba142069a4a47ab302bf174faa16f665ff83954988f2123f7a6b9a0a2a7f675` |
| `confirm/seed20261101/probes.npz` | `ab9cd279d9f8f8ad942603ffd2a96041c9fe02bcf890ee2de7b3c9798299eaf8` |
| `confirm/seed20261102/probes.npz` | `3c9c3f38287cece459b214bf1396a14245b8056cce7f5768862052d4a0296d55` |
| `confirm/seed20261103/probes.npz` | `2a6a82521fc6a5f82586beebe46709748fa359f0f534dfcc261d8d499930264f` |
| `confirm/seed20261101/latent.npz` | `638f6acc5009d1d5ae94e9704d4b4cde66628563148c9cafee6ccf5b8607e81a` |
| `confirm/seed20261102/latent.npz` | `63f9c5ae4734a3570d596ebca5e12c6542558d54305f465ba2c442c98aeaf681` |
| `confirm/seed20261103/latent.npz` | `9240334bdfa4e7b1e62eebc27240f604436c0d332f859057c7673f47380f3f64` |
| `confirm/seed20261101/latent.json` | `754282aeac89cb7450433ae8e721e7e1607f7ab920d414b377c9b09d9087029f` |
| `confirm/seed20261102/latent.json` | `ce4c0c12c46a623e261cac240a882b0788ff7f77adb42ee8256f647bae7f774e` |
| `confirm/seed20261103/latent.json` | `bc91f0d78c3727f4c534d73bb295a7ad84a3d5e356772794a9fbe903b4cd811d` |

The audit also read `confirm/sender_train.json`, `confirm/sender_test.json`, the `n_cand` and `se` arrays of `confirm/sender_*.npz`, `confirm/recv/*.npz`, `confirm/seed*/recv_T4_*.npz` and `confirm/seed*/latent_*.json`.

Audit scripts and outputs. The originals are in the session scratchpad (`<tmp>/d9ad427c-194b-4d62-900b-83a4ef5cf293/scratchpad/`), which is temporary. Copies are in `<local>/Library/Application Support/Claude/side-session-notes/local_cd217fcd-565e-4198-8ba5-572cabf2f79d/uncertainty-confirm-audit-au/`.

| File | sha256 |
|---|---|
| `au_recompute.py` | `34d5bb4f7a20da779a1c5ec2db42b04b85e64b58f556fb024c7855ba2914f9b2` |
| `au_compare.py` | `dddb7a9f745adff3a2f9dd657ffd1fee924122e23808db19a7b40427f940133b` |
| `au_refit.py` | `f85b3f67f85156996f8ed703d2f929336fae34f434960a8c9caab095a1463795` |
| `au_values.json` | `625420e1cd9b38610b2848a01458ed49f13e7e68dc91a3dcee3ee72a0eeb61ef` |
| `au_compare.json` | `0bc53bb3c9346c14dcd4d10246b04ddd698f476c0d2743e2430f2dcf0698b4a8` |
| `au_refit.json` | `72ffc347ae77adbdf56c209eb0cb5aaa24f44043f1a63a03d1be70c16d76871a` |
