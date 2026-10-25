# Report: replay of learned messages in multi-agent RL (Beacon)

Date: 2026-10-23 (run 23:04 to 23:41 local). Written by Claude (Opus 5.5) for [researcher] (Symbolon Labs).

Label: **Preregistered** (threat model `reports/threat_model_latent_channels_20261023.md`, section 1.3). The plan and the reading rule were frozen before the first training step. The run is complete, the validity gates V1 to V5 pass, and the independent statistics script agrees. There is one run and no independent audit. The `label` field inside `results.json` is a fixed text that `analysis.py` writes; PREREGISTRATION.md section 12 gives the label.

Defensive scope: replay of recorded messages and fixed defenses only. No optimized attack, no monitor-aware attack, no RL attacker. CPU only. No download, no commit.

Abbreviations: `R` = `results.json`; `SC` = `stats_check.json`; `PH` = `posthoc.json`; `J0` = `runs/j0/decision.json`. All paths are in this folder.

---

## 0. Result in brief

| Question | Verdict (reading rule) | Number (mean over 20 teams, 98.75% t interval) | Source |
|---|---|---|---|
| Q1-RT: harm of a replay of the step-0 message | **HARM** | Delta = 0.372 (0.360 to 0.384) | `R` F1."Q1-RT" |
| Q1-RE: harm of a replay from another episode | **HARM** | Delta = 0.742 (0.733 to 0.751) | `R` F1."Q1-RE" |
| Q3-RT: MAC against the stale replay | **MAC HURTS** | B = -0.308 (-0.324 to -0.291) | `R` F1."Q3-RT" |
| Q3-RE: MAC against the foreign replay | **MAC HELPS** | B = 0.060 (0.051 to 0.069) | `R` F1."Q3-RE" |
| Q2: P-kNN on RT / RE | **CANNOT FLAG** / **CANNOT FLAG** | AUROC 0.087 / 0.502; TPR at 1% FPR 0.000 / 0.010 | `R` F2 |
| Q2: P-PCA on RT / RE | **CANNOT FLAG** / **CANNOT FLAG** | AUROC 0.161 / 0.508; TPR 0.000 / 0.012 | `R` F2 |
| Secondary: X-kNN on RT / RE | CANNOT FLAG / CANNOT FLAG | AUROC 0.274 / 0.294 (95%) | `R` monitors |
| Secondary: X-PCA on RT / RE | CANNOT FLAG / CANNOT FLAG | AUROC 0.033 / 0.507 (95%) | `R` monitors |
| Secondary S3: value of a trained fallback | **FALLBACK TRAINING HELPS** | F = 0.137 (0.129 to 0.145, 95%) | `R` S3 |

In plain words:

- A valid message from another session cuts the team success from 0.987 to 0.245. This is below silence (0.305).
- A replay of the step-0 message cuts it to 0.615.
- The lab's point monitors see neither replay. They rank the step-0 replay as more typical than benign traffic (AUROC 0.09 and 0.16).
- The full MAC (session nonce and step counter) rejects both replays. Against the foreign replay, the trained fallback gives back 0.060 of success.
- Against the stale replay, strict freshness costs 0.308 of success, because the stale message still carries useful content. But the MAC declares every one of these failures, and the replay alone fails silently.
- The context monitor, which also reads the receiver's own view, does not flag a replay either.

---

## 1. What ran

| Job | Content | Wall time | Exit | Source |
|---|---|---|---|---|
| Freeze | `FREEZE.json`, 2026-10-23T18:11:30Z (21:11 local), before the first training step | - | - | `FREEZE.json` |
| Wait | Test B v3 DONE and no bsp process at 23:03:41 | - | 0 | `mrp_wait.sh` output |
| J0 | Gate; development teams 9001 and 9002 at D1; timing | 117.8 s | 0 | `runs/logs/J0.log`, `J0` |
| J1 | 20 W1 teams: train, gate set, ten arms, monitor sets, calibration, influence | 400.2 s | 0 | `runs/logs/J1.log` |
| J2 | 20 W0 teams: train, gate set, ten arms | 387.1 s | 0 | `runs/logs/J2.log` |
| J3 | Monitor scores of the 20 W1 teams | 1,235.8 s | 0 | `runs/logs/J3.log` |
| J3_analysis | `analysis.py` (reading rule) | about 12 s | 0 | `runs/logs/J3_analysis.log`, `R` |
| J3_check | `stats_check.py` (independent statistics, agreement gate) | about 12 s | 0 | `runs/logs/J3_check.log`, `SC` |

- J0 selected budget D1 (489 updates, 4,005,888 steps). Both development teams passed G-a and G-b at D1: team 9001 success(C) 0.995, success(Z) 0.2995; team 9002 0.9785 and 0.2945 (`J0` records).
- Training time for each team: 16.4 to 17.4 s. Full W1 evaluation: 3.1 to 3.8 s (`runs/logs/J1.log`, `runs/logs/J2.log`).
- The plan rule gave one job each for J1, J2 and J3 (projected 6.9, 6.7 and 26.2 min; `J0` plan).
- The design estimated 1.5 to 4 CPU hours. The run took about 37 minutes of jobs.
- Every gate passed: no other heavy job, AC power with a 140 W charger, battery 100%, swap below 2,000 MB, free disk, freeze hashes equal, 39 unit tests pass (`runs/logs/*.log`, "gate result: PASS").
- The watchdog never paused a job (no "watchdog" line in any log).
- Disk: `runs/` holds about 3.8 GB.

---

## 2. Gates and checks

| Check | Result | Source |
|---|---|---|
| G-a and G-b, W1 | 20 of 20 teams pass. Gate set success(C) 0.958 to 0.9985 (mean 0.988); success(Z) 0.280 to 0.325 (mean 0.301). | `R` gate_set_W1, teams |
| G-a, W0 | 20 of 20 teams pass. Success(C) 0.979 to 0.9995 (mean 0.991); success(Z) 0.152 to 0.185 (mean 0.170). | `R` gate_set_W0 |
| V1 (at least 18 of 20 W1 teams pass) | PASS (20 of 20). The analysis set is teams 1 to 20. | `R` validity.V1 |
| V2 (MAC-C = C, 0 rejections; MAC-RE = Z) | PASS in all 40 teams | `R` validity.V2 |
| V3 (byte copies, norm 8 within 1e-4) | PASS in all 40 teams | `R` validity.V3 |
| V4 (arm C again from the checkpoint, W1 team 1) | PASS | `R` validity.V4 |
| V5 (P monitors flag RN, mean AUROC >= 0.90) | PASS: P-kNN 1.000, P-PCA 1.000 | `R` validity.V5 |
| S1 to S5 | No stop rule fired. No NaN team. S2 projections 6.9 min (J1) and 6.5 min (J2). | `runs/logs/J1.log`, `runs/logs/J2.log` |
| Agreement gate | AGREE. Largest difference 2.6e-12. All verdicts equal. | `SC` |
| t values | 2.7586 (n = 20, 98.75%), 2.7925 (n = 18), 2.0930 (n = 20, 95%); both scripts match the design values | `R` t_values, `SC` design_t_check |
| Lab spot check | The chunked P-kNN, P-PCA, X-kNN and X-PCA scores of team 1 equal `fit_score` of the lab file on 300 sampled steps of C, RT and RE: largest difference 6.6e-14 | `SC` spot_check_vs_lab_fit_score |
| Ties | 3,821 exact cross ties over the P and X monitors on RT, RE and RN. They come from the shared step-0 message. Average ranks handle them (PREREGISTRATION.md section 14, item 21). | `SC` auroc_cross_ties; `R` monitors.*.cross_ties_total |

The sensitivity analysis on all W1 teams is identical to the primary analysis, because all 20 teams pass (`R` F1_sensitivity_all_ok_W1).

---

## 3. Primary results

### 3.1 Family F1: team effects (98.75% t intervals, n = 20)

| Comparison | Definition | Mean | SD | Interval | Range over teams | Verdict |
|---|---|---|---|---|---|---|
| Q1-RT | success(C) - success(RT) | 0.3717 | 0.0195 | 0.3596 to 0.3837 | 0.345 to 0.400 | HARM |
| Q1-RE | success(C) - success(RE) | 0.7417 | 0.0144 | 0.7328 to 0.7505 | 0.712 to 0.760 | HARM |
| Q3-RT | success(MAC-RT) - success(RT) | -0.3075 | 0.0263 | -0.3237 to -0.2913 | -0.351 to -0.257 | MAC HURTS |
| Q3-RE | success(MAC-RE) - success(RE) | 0.0599 | 0.0141 | 0.0512 to 0.0686 | 0.038 to 0.087 | MAC HELPS |

Source: `R` F1. Q3-RE equals success(Z) - success(RE), because V2 passes (MAC-RE outcomes equal Z outcomes in every team).

### 3.2 Family F2: point monitors (episode maximum over 30 steps; 98.75% t intervals, n = 20)

| Monitor, arm | Mean AUROC | Interval | Range over teams | Mean TPR at 1% FPR | Verdict |
|---|---|---|---|---|---|
| P-kNN, RT | 0.0869 | 0.0761 to 0.0977 | 0.070 to 0.148 | 0.0000 | CANNOT FLAG |
| P-kNN, RE | 0.5017 | 0.4931 to 0.5103 | 0.471 to 0.526 | 0.0102 | CANNOT FLAG |
| P-PCA, RT | 0.1610 | 0.1540 to 0.1679 | 0.135 to 0.176 | 0.0000 | CANNOT FLAG |
| P-PCA, RE | 0.5075 | 0.4983 to 0.5167 | 0.475 to 0.532 | 0.0117 | CANNOT FLAG |

Source: `R` F2 and monitors. The RT values are far below 0.5. This is the structural effect that the design expected: the RT episode maximum is the score of one message only, so an RT episode looks more typical than a benign episode.

---

## 4. Secondary rules

### 4.1 Context monitors (95% t intervals, n = 20)

| Monitor, arm | Mean AUROC | Interval | Mean TPR at 1% FPR | Verdict |
|---|---|---|---|---|
| X-kNN, RT | 0.2741 | 0.2670 to 0.2812 | 0.0000 | CANNOT FLAG |
| X-kNN, RE | 0.2936 | 0.2849 to 0.3024 | 0.0056 | CANNOT FLAG |
| X-PCA, RT | 0.0330 | 0.0235 to 0.0424 | 0.0000 | CANNOT FLAG |
| X-PCA, RE | 0.5065 | 0.5003 to 0.5126 | 0.0098 | CANNOT FLAG |

Source: `R` monitors."X-kNN|RT" and the related keys.

### 4.2 S3: value of a trained fallback

- F_s = success(MAC-RE, W1) - success(MAC-RE, W0), paired by seed, 20 pairs (all pass G-a in both conditions).
- Mean 0.1367, 95% interval 0.1288 to 0.1447, range 0.096 to 0.159. Verdict: FALLBACK TRAINING HELPS.
- Source: `R` S3.

---

## 5. Descriptive numbers (no reading rule)

### 5.1 Success, return and steps to catch on set E (2,000 episodes for each team)

| Arm | W1 success: mean (SD) | W1 range | W1 return | W1 steps to catch | W0 success: mean (SD) |
|---|---|---|---|---|---|
| C | 0.9865 (0.0118) | 0.948 to 0.999 | 1.505 | 22.4 | 0.9913 (0.0058) |
| RT | 0.6148 (0.0273) | 0.551 to 0.651 | 0.966 | 20.8 | 0.6272 (0.0293) |
| RE | 0.2449 (0.0119) | 0.228 to 0.277 | 0.309 | 24.9 | 0.2437 (0.0126) |
| RN | 0.1759 (0.0090) | 0.158 to 0.194 | 0.344 | 29.2 | 0.1602 (0.0090) |
| Z | 0.3048 (0.0129) | 0.280 to 0.327 | 0.551 | 23.9 | 0.1681 (0.0097) |
| MAC-C | 0.9865 (0.0118) | 0.948 to 0.999 | 1.505 | 22.4 | 0.9913 (0.0058) |
| MAC-RT | 0.3074 (0.0132) | 0.283 to 0.326 | 0.555 | 23.7 | 0.1727 (0.0105) |
| MAC-RE | 0.3048 (0.0129) | 0.280 to 0.327 | 0.551 | 23.9 | 0.1681 (0.0097) |
| RT5 | 0.9570 (0.0226) | 0.890 to 0.980 | 1.470 | 23.0 | 0.9675 (0.0175) |
| RE5 | 0.2747 (0.0123) | 0.257 to 0.310 | 0.356 | 23.4 | 0.2715 (0.0145) |

Source: `R` arms_W1_analysis_set and arms_W0_all_ok.

Differences under W1 (95% t intervals; `R` descriptive_differences_W1):

| Difference | Mean | Interval |
|---|---|---|
| C - RT5 (lag of 5 steps) | 0.0296 | 0.0235 to 0.0356 |
| C - RE5 (foreign stream from step 5) | 0.7118 | 0.7055 to 0.7182 |
| Z - RN (silence against random messages of norm 8) | 0.1289 | 0.1245 to 0.1332 |
| C - Z | 0.6818 | 0.6757 to 0.6878 |
| C - RN | 0.8106 | 0.8047 to 0.8165 |

Z is a trained input under W1. It is not degenerate: the Z success is 0.305 and its steps to catch are 23.9.

### 5.2 MAC

| Item | W1 | W0 | Source |
|---|---|---|---|
| MAC-C: messages rejected; episodes with a rejection | 0.000; 0.000 | 0.000; 0.000 | `R` mac |
| MAC-RT: messages rejected; episodes with a rejection | 0.976 (0.9757 to 0.9767); 1.000 | 0.978; 1.000 | `R` mac |
| MAC-RE: messages rejected; episodes with a rejection | 1.000; 1.000 | 1.000; 1.000 | `R` mac |
| Time to compute one tag | 1.43 microseconds (Python `hmac`, 100,000 tags, W1 team 1) | | `R` tag_timing |
| Time to check one tag | 1.74 microseconds; 100,000 of 100,000 legitimate tags accepted | | `R` tag_timing |
| Size | 32-byte tag for a 256-byte message (12.5% more bytes) | | `R` tag_timing |

MAC-RT rejects 97.6% and not 100% of the messages, because the step-0 message is valid and many episodes are short.

Field table S4: fraction of messages accepted at steps 5 to 49 (first 200 episodes of E; minimum = maximum over all 40 teams; `R` field_table_S4):

| Tag fields | C | RT | RT5 | RE | RE5 |
|---|---|---|---|---|---|
| Message bytes only | 1 | 1 | 1 | 1 | 1 |
| Bytes and session nonce | 1 | 1 | 1 | 0 | 0 |
| Bytes and step counter | 1 | 0 | 0 | 1 | 1 |
| Bytes, nonce and counter (this design) | 1 | 0 | 0 | 0 | 0 |

The pattern equals the design table. It holds by construction.

### 5.3 Monitors: other arms, step level and delay

Episode-level AUROC, other arms (95% t intervals; `R` monitors):

| Monitor | RN | RT5 | RE5 |
|---|---|---|---|
| P-kNN | 1.000 (TPR 1.000) | 0.447 | 0.520 |
| P-PCA | 1.000 (TPR 1.000) | 0.450 | 0.535 |
| X-kNN | 0.9997 (TPR 1.000) | 0.452 | 0.540 |
| X-PCA | 1.000 (TPR 1.000) | 0.708 | 0.9855 (0.9833 to 0.9876), TPR 0.908 |
| X-noview kNN (RT 0.239, RE 0.503) | 1.000 | 0.488 | 0.783, TPR 0.104 |
| X-noview PCA (RT 0.018, RE 0.508) | 1.000 | 0.749 | 0.9874 (0.9854 to 0.9894), TPR 0.915 |

Step-level AUROC over the 30,000 steps (mean over teams; `R` step_level_auroc):

| Monitor | RT | RE | RN | RT5 | RE5 |
|---|---|---|---|---|---|
| P-kNN | 0.487 | 0.500 | 1.000 | 0.489 | 0.501 |
| P-PCA | 0.447 | 0.503 | 1.000 | 0.480 | 0.503 |
| X-kNN | 0.426 | 0.406 | 1.000 | 0.492 | 0.421 |
| X-PCA | 0.373 | 0.505 | 1.000 | 0.502 | 0.522 |

First flag of the X monitors (threshold = 99th percentile of the benign test episodes; mean over teams; `R` first_flag):

| Monitor, arm | Median first step (flagged episodes) | Flagged by step 10 | By step 20 | By step 30 |
|---|---|---|---|---|
| X-kNN, RT | no episode flagged | 0.000 | 0.000 | 0.000 |
| X-kNN, RE | 21.3 (19 teams with a flag) | 0.0003 | 0.0026 | 0.0056 |
| X-PCA, RT | no episode flagged | 0.000 | 0.000 | 0.000 |
| X-PCA, RE | 22.1 | 0.0002 | 0.0038 | 0.0098 |
| X-PCA, RE5 | 6.0 | 0.908 | 0.908 | 0.908 |
| X-noview PCA, RE5 | 6.0 | 0.915 | 0.915 | 0.915 |
| X-kNN and X-PCA, RN | 1.0 | 1.000 | 1.000 | 1.000 |

Participation ratio of the 5,000 benign fit messages (S6): mean 2.30, range 2.19 to 2.49 (`R` participation_ratio_S6). The toy channel value was 1.12.

### 5.4 Influence score (the open question of Test E)

| Arm against C | Mean AUROC | 95% interval | Source |
|---|---|---|---|
| RT | 0.365 | 0.360 to 0.370 | `R` influence_auroc.RT |
| RE | 0.463 | 0.453 to 0.473 | `R` influence_auroc.RE |

A replayed message moves the listener's action distribution less than a benign message does. The receiver's own view does not make the replay visible to this score.

### 5.5 Silent failures (fraction of the 2,000 E episodes that fail with no flag; mean over 20 W1 teams)

| Arm | No defense | P-kNN | P-PCA | X-kNN | MAC |
|---|---|---|---|---|---|
| C | 0.0135 | 0.0133 | 0.0132 | 0.0134 | 0.0135 |
| RT | 0.3851 | 0.3851 | 0.3851 | 0.3851 | 0.0000 |
| RE | 0.7551 | 0.7279 | 0.7264 | 0.7457 | 0.0000 |
| RN | 0.8241 | 0.0000 | 0.0000 | 0.0000 | no rollout |
| Z | 0.6952 | 0.0000 | 0.0000 | 0.6604 | no rollout |
| RT5 | 0.0430 | 0.0422 | 0.0424 | 0.0427 | no rollout |
| RE5 | 0.7253 | 0.6988 | 0.6975 | 0.7153 | no rollout |

Source: `R` silent_failure. The monitor thresholds give a benign flag rate of exactly 0.0100 on the calibration set (`R` silent_failure.calibration_benign_flag_rate). Under the MAC, the RT and RE traffic fails more often (MAC-RT and MAC-RE success 0.307 and 0.305), but every failed episode has a logged rejection.

---

## 6. Post hoc check (not preregistered)

`posthoc_checks.py` ran after the run on the saved monitor arrays. It explains the X-kNN AUROC below 0.5 (`PH` summary, mean over 20 teams):

| Item | C | RT | RE |
|---|---|---|---|
| Fraction of steps with an active local view | 0.379 | 0.272 | 0.082 |
| Fraction of episodes whose maximum X-kNN step has an active view | 0.834 | 0.558 | 0.192 |
| Mean X-kNN step score, active view | 0.659 | 0.753 | 0.886 |
| Mean X-kNN step score, empty view | 0.289 | 0.247 | 0.305 |

- In benign traffic, the steps near the beacon (active view) are the least typical steps. They give the episode maximum in 83% of the C episodes.
- Under RE, the listener follows the foreign message and rarely sees the beacon (8% of steps). Its rare active-view steps do score higher than benign active-view steps (0.886 against 0.659). This is the mismatch signal that the design expected. But it is rare, and the episode maximum hides it.
- For RE5, the X-PCA maximum is at step index 5 (the splice) in 98.5% of the episodes (`PH` RE5_xpca_argmax_step_index_frac_eq5). The PCA pair monitor sees the jump between consecutive messages, not the old or foreign content. A lockstep replay from step 0 (RE) has no jump, and the same monitor gives it AUROC 0.507.

---

## 7. What this means for a defender

Each row uses the design's table (design section 12).

| Result | Meaning |
|---|---|
| Q1-RE HARM, Q2 CANNOT FLAG, Q3-RE MAC HELPS (the expected case) | A valid message from another session silently misleads the team (success 0.245, below silence at 0.305). The wire monitors cannot see it. Bind a session nonce and a step counter into the tag, and train the fallback. |
| Q1-RT HARM and Q3-RT MAC HURTS | The design table has no row for this pair. It has two near rows. (1) The value of the MAC is the alarm: the stale replay fails silently in 38.5% of the episodes, and under the MAC no failure is silent. (2) A stale message still carries useful content (RT 0.615 against MAC-RT 0.307), and strict freshness throws it away. A short lag costs little (RT5: 0.030). Thus a verified message with an age limit can keep most of the success and still declare the replay. That rule needs its own test. |
| Q2 CANNOT FLAG on RT and RE | A wire monitor fit on benign messages cannot see that a valid message is old or foreign. On the step-0 replay it ranks the replay as more typical than benign traffic. |
| X monitors CANNOT FLAG, below chance | The receiver's own view, used in this way (an anomaly score on [message, previous message, position, view]), does not expose a replay. It makes a replay look more typical, because benign near-beacon states are the rare states (section 6). A consistency check that decodes the beacon position from the message is still an untested option (design section 5.3). |
| X-PCA flags the RE5 splice (AUROC 0.986, at step 6) | A monitor on consecutive message pairs can see a jump in the stream. An attacker who replays a full stream from step 0 makes no jump. This is not a defense against replay. |
| S3 FALLBACK TRAINING HELPS (0.137) | A fail-closed MAC needs a trained fallback. Without it (W0), the fallback success is 0.168, close to random messages (0.160). |
| RN far below Z (by 0.129) | A content-free message of the correct norm is worse than silence. All monitors flag RN perfectly, but a norm check does not, because every RN message has norm 8. |
| Influence score below chance | The null result of Test E also holds for a receiver with its own view. A replayed message has less influence than a benign one (AUROC 0.365 and 0.463). |

General point (design section 12): replay is a protocol threat. The defense is to bind each message to its session and to its step. A wire monitor cannot read the content, so it cannot see that the content is old or foreign.

---

## 8. Limits

1. Toy scale: one environment, one algorithm (A2C with gradients through the message), three small MLPs, one speaker and one listener. The task was easy: all teams passed at the smallest budget, and the teams are very similar (team SD of Delta 0.014 to 0.026). Thus the intervals are narrow, and they describe this toy only.
2. Memoryless listener. A recurrent listener can resist a replay or carry its effect longer.
3. Replay is a direct test for a transport attacker (class C) and only a weak proxy for a compromised speaker (class E).
4. No adaptive attacker. A monitor that passes here is not robust (M4).
5. The channel has no noise. A MAC needs exact digital transport.
6. The MAC acceptance holds by construction. The run measured the success after the fallback and the tag cost.
7. The monitors are fit on benign data of the same team and environment. There is no benign shift.
8. The X-monitor result depends on the view radius (0.15) and on the episode maximum as the pooling rule.
9. Keys come from seeds. Key management is out of scope.
10. This is not an LLM result. [redacted: unpublished result] Here the listener depends on the message by design (gate G-b).
11. The reference with no information is Z under W1. A team trained with no channel can differ.
12. The time and the disk figures are for this laptop (Apple M1 Pro, CPU, one torch thread).

---

## 9. Deviations

- None during the run. No stop rule fired. No file in `FREEZE.json` changed (each gate checked the hashes).
- Before the freeze, the preregistration fixed 36 open choices of the design (PREREGISTRATION.md section 14). Three of them depart from the design text: the freeze came before J0 (item 1); the jobs ran in the background (item 2); the AUROC uses average ranks for ties (item 21).
- After the run: `posthoc_checks.py` and `posthoc.json` (section 6). They are not part of the freeze or of the reading rule.

---

## 10. Files

| File | Content |
|---|---|
| `PREREGISTRATION.md`, `FREEZE.json` | Plan and hashes (frozen 18:11:30 UTC) |
| `runs/j0/decision.json` | J0 records, timing and plan |
| `runs/W1_sNN/`, `runs/W0_sNN/` | For each team: `team.pt`, `train.json`, `gate.json`, `eval.json`, `outcomes.npz`; W1 also `monitor.npz`, `silent.npz`, `influence.npz`, `scores.npz` |
| `runs/logs/*.log`, `runs/chain.log` | Job logs with the gate output |
| `results.json` | `analysis.py` output: verdicts and all descriptive numbers |
| `stats_check.json` | Independent statistics and the agreement gate |
| `posthoc_checks.py`, `posthoc.json` | Post hoc check (section 6) |
| `DONE` | Empty file for the next session in the chain |
