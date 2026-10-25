# CONFIRMATORY. A finer text format of the probe values against soft vectors (closed-book TriviaQA): report

- **Date:** 2026-10-24. Written by the Claude Code session that ran the test, for [researcher] (Symbolon Labs). Style: ASD-STE100.
- **Status:** CONFIRMATORY. The run followed `PREREGISTRATION.md`, frozen in `FREEZE_D.json` (2026-10-23 21:09:25, before the first model load) and `FREEZE.json` (2026-10-23 23:54:17, before the first fresh item went through a model). After the run, no frozen file is changed (sha256 check of every file in `FREEZE.json`).
- **Audit:** `audit/AUDIT.md`, an independent recompute by a separate agent with its own code (preregistration Part 8, item 5). **Verdict: audit passes.** Case C again. All point values agree within 1.1e-16. All CI bounds agree within 3.1e-4. All labels and all four "beats" decisions are equal. The audit took V1 to V3 from `results.json`; it checked the V4 pool counts itself.
- **Design:** `reports/format_gap_design_20261023.md`. **Parent run:** `ember/research/latent_uncertainty_confirm_20261021/` (sealed; read-only). After the run, all 419 files of the sealed folder have their `SEAL.json` sha256, and the folder has no new file.

## 1. Result

**Case C: "the gap is not in the number format".** The dev gate selected F\* = WD2 (the F3 word and the value to two decimals). On 7,829 fresh items, WD2 does not close the gap to the soft vectors, and its gain over F3 is not resolved.

| Primary row (fresh items) | Mean | 95% CI, two-level (rule) | 95% CI, item only | Label |
|---|---|---|---|---|
| gap(WD2, L-probe-R) = NLL(WD2) - NLL(L-probe-R) | +0.0387 | [+0.0322, +0.0456] | [+0.0329, +0.0445] | a gap remains |
| gap(F3, L-probe-R) | +0.0415 | [+0.0349, +0.0484] | [+0.0357, +0.0474] | a gap remains (no case R) |
| gain(WD2) = NLL(F3) - NLL(WD2) | +0.0028 | [-0.0002, +0.0059] | [-0.0001, +0.0058] | not resolved |

Source: `confirm/results.json` (key `rule.primary`) and `confirm/table.md`.

- Steps of the rule (preregistration 6.2): validity passes; F3 does not close (mean 0.0415 > 0.005), so not case R; WD2 does not close (mean 0.0387 > 0.005), so not case A; a gap remains (CI lower bound 0.0322 > 0), but WD2 does not gain over F3 (CI lower bound -0.0002 < 0), so not case B; thus **case C**.
- The share of the F3 gap that WD2 removes is 0.068 (descriptive; the rule gives this number for case B only).
- The audit repeated the gain CI with 20 more random seeds. The lower bound was below 0 each time (-0.000322 to -0.000152). Thus the "no gain" label is not an effect of one random draw.

**What the report may say (design 12, case C):** "No tested format reads the probe values better than F3. The soft-vector advantage is a medium effect at this receiver."

**Limit of that sentence.** As literal text, the first sentence is too wide. It is true for F\* (WD2), the one format that the rule tests. Two other candidates, which the dev gate did not select, have small gains over F3 with a CI above 0 in the descriptive rows: ICD7 +0.0032 [+0.0006, +0.0058] and ICD7D2 +0.0047 [+0.0010, +0.0085]. The preregistration does not allow a claim from a candidate other than F\*, and these rows are a best-of-five view. They are also much smaller than the gap that remains (ICD7D2 to L-probe-R: +0.0368 [+0.0303, +0.0437]).

**What the report must not say:** that the format cannot matter at other receivers.

**Effect on the main claim (design 12, row C):** case 3 of the sealed run does not change. One qualification applies: "Text carries the information. For a fixed set of values, this small receiver reads trained soft vectors better than any tested text format, by about 0.037 to 0.072 nats on fresh items. The gap is not in the number format." All seven text formats have a gap to L-probe-R with a CI lower bound above 0. The smallest lower bound is 0.0303 (ICD7D2).

## 2. Secondary rows (no case; `confirm/results.json`, key `rule.secondary`)

| Row | Mean | 95% CI (two-level) | Note |
|---|---|---|---|
| gap(WD2, L-probe) (joint readout) | +0.0310 | [+0.0222, +0.0403] | a gap remains |
| gap(F3, L-probe) (joint readout) | +0.0338 | [+0.0249, +0.0430] | a gap remains: the open row of the sealed run (+0.0200 there) replicates on fresh items, with a larger value |
| gap(L-probe, L-probe-R) | +0.0077 | [-0.0001, +0.0154] | the text readout rule reads the same soft vectors a little better than the joint readout; the CI touches 0 |
| gap(WD2, T0c) | -0.0381 | [-0.0436, -0.0326] | the values add information |
| gap(F3, T0c) | -0.0353 | [-0.0402, -0.0306] | the values add information |
| gap(F3, probe-direct) | +0.0501 | [+0.0424, +0.0576] | route loss |
| gap(WD2, probe-direct) | +0.0472 | [+0.0397, +0.0549] | route loss |
| gap(L-probe-R, probe-direct) | +0.0086 | [+0.0020, +0.0150] | route loss |
| gap(L-probe, probe-direct) | +0.0162 | [+0.0071, +0.0252] | route loss |
| gap(L, probe-direct) | +0.0608 | [+0.0515, +0.0713] | route loss |

**Other candidates (descriptive):**

| Format | gap(format, L-probe-R) | gain over F3 = NLL(F3) - NLL(format) |
|---|---|---|
| D2 | +0.0704 [+0.0623, +0.0788] | -0.0289 [-0.0336, -0.0242] |
| D3 | +0.0724 [+0.0643, +0.0808] | -0.0308 [-0.0357, -0.0259] |
| LOs (F°) | +0.0601 [+0.0524, +0.0681] | -0.0186 [-0.0232, -0.0139] |
| ICD7 | +0.0383 [+0.0320, +0.0452] | +0.0032 [+0.0006, +0.0058] |
| ICD7D2 | +0.0368 [+0.0303, +0.0437] | +0.0047 [+0.0010, +0.0085] |

The formats with digits only read much worse than F3 (by 0.019 to 0.031 nats). This repeats the Phase D view of design 2.2 on fresh items.

**Text against the latent message L (the "beats" rule of the sealed run):**

| A vs B | Mean NLL(B) - NLL(A) | 95% CI (item only) | Per seed | A beats B |
|---|---|---|---|---|
| F3 vs L | +0.0107 | [+0.0046, +0.0169] | +0.005, +0.008, +0.019 | no (mean below the 0.02 margin) |
| WD2 vs L | +0.0136 | [+0.0073, +0.0199] | +0.008, +0.011, +0.021 | no (mean below the 0.02 margin) |

- **Important for the main claim.** In the sealed run, T4 (= F3 here) beat L by +0.0204 (CI [+0.0134, +0.0274]). On the fresh items, F3 is still better than L in every seed, with a CI above 0, but by +0.0107. This is below the 0.02 margin. Thus the 10.3 check "text still beats the latent message on fresh items" does **not** pass by the "beats" rule. The case of this test does not use this row. Case 3 of the sealed run rests on "L does not beat S\*", which this test did not measure again. But a record that uses the words "T4 beats L" for this pair must now add that, on 7,829 fresh items, the difference was +0.011 nats, below the margin.

**Each primary row for each pool (descriptive; `rule.per_pool`):**

| Pool | n | gap(WD2, C\*) | gap(F3, C\*) | gain(WD2) |
|---|---|---|---|---|
| V (validation file) | 1,991 | +0.0360 [+0.0236, +0.0492] | +0.0348 [+0.0220, +0.0476] | -0.0012 [-0.0074, +0.0048] |
| S0 (train shard 0) | 1,838 | +0.0424 [+0.0288, +0.0559] | +0.0472 [+0.0334, +0.0611] | +0.0048 [-0.0010, +0.0109] |
| S1 (train shard 1) | 4,000 | +0.0383 [+0.0296, +0.0469] | +0.0422 [+0.0334, +0.0511] | +0.0039 [-0.0005, +0.0084] |

A gap remains in each pool. No pool shows a resolved gain.

**c_round on the fresh items (the recipe of design 2.1; mean of 3 seeds; `c_round_fresh`):** F3 +0.0122 (sealed run: 0.0103), D2 +0.0001, D3 0.0000, LOs -0.0004, ICD7 -0.0010, WD2 -0.0029, ICD7D2 -0.0093. WD2 keeps all the information of the probe values (no round-off), but it does not read better than F3 at this receiver.

**Token counts on the fresh items (seed 20261101; `tokens_fresh`):** mean tag tokens F3 3.97, ICD7 4.45, D2 6.0, D3 7.0, LOs 8.0, WD2 9.97, ICD7D2 10.45. Mean message tokens: T0c 51.4, F3 59.2, WD2 74.9.

## 3. Arms (fresh items, mean over seeds; `confirm/results.json`, key `arms`)

| Arm | Seeds | NLL | AUROC | Brier | ECE | NLL per seed |
|---|---|---|---|---|---|---|
| F3 | 3 | 0.3902 | 0.840 | 0.1210 | 0.0151 | 0.3902, 0.3902, 0.3902 |
| WD2 (F\*) | 3 | 0.3874 | 0.846 | 0.1202 | 0.0177 | 0.3874, 0.3873, 0.3874 |
| ICD7 | 3 | 0.3870 | 0.846 | 0.1211 | 0.0152 | 0.3872, 0.3870, 0.3868 |
| ICD7D2 | 3 | 0.3855 | 0.850 | 0.1206 | 0.0184 | 0.3855, 0.3855, 0.3855 |
| LOs (F°) | 3 | 0.4088 | 0.820 | 0.1281 | 0.0179 | 0.4087, 0.4090, 0.4086 |
| D2 | 3 | 0.4191 | 0.809 | 0.1321 | 0.0147 | 0.4188, 0.4193, 0.4190 |
| D3 | 3 | 0.4210 | 0.807 | 0.1328 | 0.0152 | 0.4211, 0.4211, 0.4209 |
| T0c | 3 | 0.4255 | 0.801 | 0.1345 | 0.0171 | 0.4255 (each seed) |
| L | 3 | 0.4009 | 0.837 | 0.1262 | 0.0305 | 0.3953, 0.3986, 0.4089 |
| L-probe (joint) | 6 | 0.3563 | 0.881 | 0.1111 | 0.0332 | 0.3441, 0.3646, 0.3555, 0.3586, 0.3669, 0.3484 |
| L-probe-R (C\*) | 6 | 0.3487 | 0.881 | 0.1084 | 0.0117 | 0.3416, 0.3519, 0.3498, 0.3503, 0.3523, 0.3462 |
| probe-direct | 3 | 0.3401 | 0.897 | 0.1080 | 0.0455 | 0.3401 (each seed) |

Base rate of y: train 0.2647, fresh items 0.2335.

## 4. Phase D' (dev gate; development data, not evidence; `dev/dev_result.json`, `dev/dev_selection.json`)

Job Dp ran on 2026-10-23 from 23:45:40 to 23:52:49 (427 s), after `marl_replay_20261023/DONE` existed and the machine checks passed.

| Format | Dev-test NLL | Dev-train NLL (out of fold) | Pooled dev NLL (3,000 items) | Mean tag tokens (max) |
|---|---|---|---|---|
| F3 | 0.40904 | 0.42285 | 0.41825 | 3.97 (4) |
| D2 | 0.43639 | 0.44530 | 0.44233 | 6.0 (6) |
| D3 | 0.43301 | 0.45176 | 0.44551 | 7.0 (7) |
| LOs | 0.42059 | 0.43066 | 0.42730 | 8.0 (8) |
| ICD7 | 0.40551 | 0.42402 | 0.41785 | 4.47 (5) |
| **WD2** | 0.40994 | 0.41085 | **0.41055** | 9.97 (10) |
| ICD7D2 | 0.40914 | 0.41482 | 0.41292 | 10.47 (11) |

- **G-F2 (reproduction):** dev-test NLL of F3 = 0.409036 (bar: 0.40904 ± 0.001). Pass. The F3 final states are bitwise equal to the Phase D states (maximum difference 0.0, on dev-train and dev-test).
- **Selection:** all six candidates are eligible (each tag has 11 receiver tokens or fewer). F\* = WD2 (no tie within 0.001). F° = LOs.
- **G-F1 (gain):** pooled NLL(F3) - pooled NLL(WD2) = 0.00770 (bar: 0.005). Pass. Item-bootstrap CI of this dev gain (descriptive): [+0.0016, +0.0138].
- **Decision:** "go". Plan M ran.
- **Note (descriptive, post hoc):** on dev-test alone, WD2 was a little worse than F3 (0.40994 against 0.40904). The pooled gain came from the dev-train items. The fresh-item gain (+0.0028, not resolved) is between these two dev values. The gate used the pooled value, as the frozen rule says.
- Receiver: the parameter hash before and after the passes is `fe5f2146...`; the weight-file sha256 is equal to the sealed run.

## 5. Validity (V1 to V4 pass; `confirm/results.json`, key `validity_detail`)

- **V1:** for seeds 20261101 to 20261103, the train probe values are equal to the sealed `confirm/seed*/probes.npz` (`fit`): maximum absolute difference 0.0, same NaN mask.
- **V2:** 16 weight-hash records (start and end of the 8 Phase C' model jobs), 3 files in each; every sha256 is equal to the sealed run. The receiver parameter hash before and after each of the 9 latent fits is `fe5f214635a1...28f4`.
- **V3:** all 9 latent fits end with finite losses; all fresh-item probabilities are finite.
- **V4:** 0 fresh questions are equal to a spent question; the pool counts (V 1,991, S0 1,838, S1 4,000) are equal to the frozen manifest and to the design.
- **Train side (design 15, risk 7):** the L-probe fits of seeds 1 to 3 give train probabilities equal to the sealed fits (maximum difference 0.0). The MPS fits repeated exactly.

## 6. Run record

| Job | Time (local) | Result |
|---|---|---|
| Prefreeze (`FREEZE_D.json`) | 2026-10-23 21:09:25 | 18 unit tests pass; 28 sealed inputs equal `SEAL.json` |
| Dp (Phase D') | 23:45:40 to 23:52:49 | rc 0; decision "go" |
| DL (download) | 23:53:23 to 23:53:53 | `rc.wikipedia/train-00001-of-00007.parquet`, 260,772,194 bytes, sha256 `55cd8290...def59` (equal to the Hub value); `data/FETCH_LOG.json` |
| I (fresh items) | 23:53:59 to 23:54:02 | V 1,991, S0 1,838, S1 4,000 (7,829); shard 1 has 8,841 usable rows (the design estimate was about 8,841) |
| F (`FREEZE.json`) | 23:54:17 | 18 tests pass; seal check passes |
| S1 to S4 (sender) | 23:54 to 04:18 (2026-10-24) | rc 0 each; 69, 58, 67 and 70 min; 2.02 s per item (`confirm/sender_times.json`) |
| Q1 to Q4 (seeds) | 04:18 to 08:40 | rc 0 each; 67, 63, 62 and 70 min |
| A (analysis) | 08:40 to 12:32 | rc 0; about 13 min of run time; the guard paused the job from 08:47:23 to 12:21:11 (below) |
| AU (independent recompute) | 2026-10-24, after A | audit passes (`audit/AUDIT.md`) |

- **Power pauses:** the charger reported 8 W (on AC power, battery 100%) at 08:41:04 (pause of 20 s) and at 08:47:23 (pause until 12:21:11, when the charger reported 140 W again). The guard (G27) paused and resumed job A as designed. No job reached its time limit. No job started again. The approval of APPROVALS.md, item B (restarts) was not necessary.
- **Swap:** the swap use after the seed jobs was 1,858 to 2,504 MB (limit 3,584 MB).
- **GPU time:** about 8.9 h (Dp 7 min, sender 264 min, seeds 262 min). The design estimate was about 9.2 h.

## 7. Deviations

The deviations of `PREREGISTRATION.md` Part 8 apply: one preregistration for D' and C', frozen before D'; no change to the wrappers of other folders; the download into this folder; the power guard; the audit by a separate agent. These are not new. There is no new deviation after the freeze. [researcher]'s relayed approval of 2026-10-23, about 21:30 (`APPROVALS.md`, item B) was recorded but not used.

## 8. Limits

1. One sender (Qwen3-1.7B), one receiver (SmolLM2-360M-Instruct), one task (closed-book TriviaQA). A result about how a receiver reads text applies to this receiver only.
2. The primary comparator L-probe-R and the six seeds came from a post hoc view of the sealed test arrays (design 2.3). The preregistration fixed them before any fresh item went through a model.
3. The 4,000 train items are the train items of the sealed run. The test items are new; the train side is not.
4. The fresh items come from three pools. The rows for each pool agree in sign and label.
5. WD2 uses about 10 tag tokens; F3 uses about 4; L-probe uses one soft vector for each slot. The longer text did not read better here.
6. The ICD7 and ICD7D2 rows are descriptive. A test of either format needs a new preregistration with fresh items.
7. Seed SDs: the L-probe-R per-seed NLL spans 0.3416 to 0.3523; the joint L-probe spans 0.3441 to 0.3669.

## 9. Files

| File | sha256 |
|---|---|
| `confirm/results.json` | `a597366bc4847ba424223c6826e14a5e5b3908e4818c02b9461f333f9645db88` |
| `confirm/table.md` | `8b457327d8dc1e0329b7d8c7bf4a10b6941b5236dab09c6054edb3fdf3989913` |
| `confirm/fresh_probs.npz` (per-item probabilities, y and pool) | `9c4ca926725eef5b682da6168210f5927be1d7ed08a884b136e57f115695ff10` |
| `dev/dev_selection.json` | `ac50917964efd3a75e8e6a03c4a09f0efe40f46f01ec645e88bed6a929825e02` |
| `dev/dev_result.json` | `d95d42c2b8a5af43b4b2c78f56d92ef3ced49e8bee1cccbb4ddc9913c46eb00c` |
| `fresh/fresh_items.json` | `44c02856bac8bdbd65a1d6b7f0cdb38463ca32f93cea467531fd1eddfe09d4ea` |
| `fresh/fresh_manifest.json` | `70531b13dbc1b7cda27ca71663b07ce14fa7eaf7a55e2525d3e00c69770031d2` |
| `FREEZE_D.json` | `34c761ffe8273d2e6bd6cf56f1dfd9c60dc2b39eccf1547c5afc13f00828c602` |
| `FREEZE.json` | `1a943b7bdae1f9b4a7e6c88d2ee7b3e91142fff6d9c0c3d41687e03bb45fbc22` |
| `audit/AUDIT.md` | `7847b737c208984d014d25f5ef1ab23348a6758b8b42116ce52e0b696a4fc65d` |
| `audit/au_recompute.py` | `189c10edf235914ccdd26a2a165c51e2de2c49a609bdc664ddf36a297fa4fe27` |
| `audit/au_recompute.json` | `be5fe980129c5d4762449c4d036599279ec0f2803fe64abb1d4e0c62296f2e55` |
| `data/FETCH_LOG.json` | `0cc5364939523cfa27df9a605183729bf9bc1451625cfa87621f2b9bb211861f` |
| `APPROVALS.md` | `d63ea86916805c009081694ea05ac58b22389fbee5615d474f35c25bce8a702b` |

The frozen files (`PREREGISTRATION.md`, `ugap.py`, `test_ugap.py`, `run_ugap.sh`, `guard.sh`, `chain.sh`) have the sha256 values of `FREEZE_D.json`. Logs: `logs/` (job, wrapper, guard and chain logs).
