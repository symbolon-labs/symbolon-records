# AUDIT (AU): format-gap confirmatory run, 2026-10-24

**Verdict: audit passes.** The audit code agrees with the harness on all point values (difference 1.1e-16 or less). All CI bounds agree within 0.00031. All labels are equal, the four "beats" rows are equal, and the case is equal: **case C, "the gap is not in the number format"**.

## 1. Scope and rules

- This audit applies Part 6 of `PREREGISTRATION.md` (6.1, 6.2, G16 to G22). The auditor read Part 6 before any result file.
- The audit code is `audit/au_recompute.py`. It does not import or copy code from `ugap.py` or `uconfirm.py`. The auditor did not read the statistics code of `ugap.py`.
- The script uses the CPU only. It loads no model. It does not change a file outside `audit/`. It reads nothing from `../latent_uncertainty_confirm_20261021/`.
- The script writes its own values to `audit/au_recompute.json` first. Only after that step does it read `confirm/results.json`.
- Run command (in the folder `format_gap_20261023`): `../../.venv/bin/python -I -B audit/au_recompute.py` (about 17 s).

## 2. Inputs and hashes

| File | sha256 |
|---|---|
| `confirm/fresh_probs.npz` | `9c4ca926725eef5b682da6168210f5927be1d7ed08a884b136e57f115695ff10` |
| `audit/au_recompute.py` | `189c10edf235914ccdd26a2a165c51e2de2c49a609bdc664ddf36a297fa4fe27` |
| `confirm/results.json` (compared, not used for values) | `a597366bc4847ba424223c6826e14a5e5b3908e4818c02b9461f333f9645db88` |

From `dev/dev_selection.json`: F\* = WD2, F° = LOs. C\* = L-probe-R. The npz has n = 7,829 items. L-probe and L-probe-R have 6 seeds (20261101 to 20261106). All other arms have 3 seeds (20261101 to 20261103).

## 3. Method

1. The script clips each probability to [1e-6, 1 - 1e-6]. It computes the item NLL for each arm and seed.
2. NLL_i(arm) is the mean over the seeds of that arm. gap(X, C) is the mean over the items of NLL_i(X) - NLL_i(C). gain(F\*) is gap(F3, F\*).
3. Two-level paired bootstrap (6.1): 10,000 draws. Each draw resamples the items with replacement, with the same items for all arms. For each arm, the draw resamples the seeds of that arm with replacement, independently. The statistic is the mean over the drawn items of the mean over the drawn seeds. The CI is the 2.5 and 97.5 percentiles (numpy linear).
4. The audit generator is numpy `Philox`, seed 991024. The harness uses `default_rng(20261023)`. Thus the audit draws are different from the harness draws.
5. The script computes the statistic in a weighted form (item counts and seed counts). A slow index-form loop on three draws gives the same values (difference 1.0e-15).
6. The "beats" rows (G21) use an item-only bootstrap of the seed-mean item differences (10,000 draws, Philox). Text seed k pairs with L seed k (k = 1 to 3).

## 4. Values: audit against harness

Point values are means of item NLL differences (nats). "AU CI" is the audit two-level 95% CI. "H CI" is the harness two-level 95% CI.

| Row | AU mean | H mean | max abs diff of mean | AU CI | H CI | max abs diff of CI bound |
|---|---|---|---|---|---|---|
| gap(F\*, L-probe-R) (primary) | +0.038691700 | +0.038691700 | 0 | [+0.032034, +0.045531] | [+0.032193, +0.045611] | 1.6e-4 |
| gap(F3, L-probe-R) (primary) | +0.041509434 | +0.041509434 | 0 | [+0.034819, +0.048356] | [+0.034938, +0.048394] | 1.2e-4 |
| gain(F\*) = gap(F3, WD2) (primary) | +0.002817734 | +0.002817734 | 0 | [-0.000296, +0.005840] | [-0.000176, +0.005928] | 1.2e-4 |
| gap(F\*, L-probe) (secondary) | +0.031030172 | +0.031030172 | 0 | [+0.021993, +0.040299] | [+0.022249, +0.040263] | 2.6e-4 |
| gap(F3, L-probe) (secondary) | +0.033847905 | +0.033847905 | 0 | [+0.024630, +0.043118] | [+0.024937, +0.043021] | 3.1e-4 |
| gain(F°) = gap(F3, LOs) (descriptive) | -0.018588677 | -0.018588677 | 0 | [-0.023196, -0.014008] | [-0.023245, -0.013876] | 1.3e-4 |
| gap(L-probe, L-probe-R) (descriptive) | +0.007661528 | +0.007661528 | 0 | [-0.000111, +0.015121] | [-0.000143, +0.015412] | 2.9e-4 |

Other agreements:

- Share of the F3 gap that F\* removes (G20): AU 0.067882, harness 0.067882 (difference 0).
- Mean item NLL of all 12 arms: maximum difference 1.1e-16.
- Maximum difference of any point value: 1.1e-16 (bar 1e-9). Maximum difference of any CI bound: 3.1e-4 (bar about 1e-3).

## 5. Labels and case (10.1, 10.2)

| Label | Rule | AU | Harness |
|---|---|---|---|
| F3 closes against C\* | mean <= 0.005 and CI upper < 0.010 | no (mean 0.0415) | no |
| F\* closes against C\* | same | no (mean 0.0387) | no |
| A gap remains between F\* and C\* | not closes and CI lower > 0 | yes (lower 0.0320) | yes |
| A gap remains between F3 and C\* | same | yes (lower 0.0348) | yes |
| F\* gains over F3 | CI lower of gain(F\*) > 0 | no (lower -0.000296) | no (lower -0.000176) |
| F\* against L-probe: closes / remains | as above | no / yes | no / yes |
| F3 against L-probe: closes / remains | as above | no / yes | no / yes |

Steps of 10.2:

0. Validity: V1, V2 and V3 are true in `confirm/results.json`. The audit takes them as given. For V4, the audit checked the pool counts in the npz: V 1,991, S0 1,838, S1 4,000 (equal to the expected counts). Also, the pool order in `fresh/fresh_items.json` is equal to the npz order, the manifest counts are equal to the expected counts, and no two fresh questions are equal after lowercase and whitespace collapse. The audit did not check the spent-question part of V4, because that check needs the sealed items. The run is valid.
1. F3 does not close against C\*. Not case R.
2. F\* does not close against C\*. Not case A.
3. A gap remains between F\* and C\*, but F\* does not gain over F3. Not case B.
4. A gap remains between F\* and C\*. **Case C: "the gap is not in the number format".**

AU case = C. Harness case = C. Equal.

## 6. The "beats" rows (G21, item-only CI)

The rule: the mean of NLL_i(worse) - NLL_i(better) is 0.02 or more, the CI lower bound is above 0, and each per-seed mean is above 0.

| Row | AU mean | AU CI (item only) | H CI (item only) | per-seed means (seeds 1 to 3) | AU beats | H beats |
|---|---|---|---|---|---|---|
| F3 beats L | +0.010750 | [+0.004530, +0.016775] | [+0.004617, +0.016917] | +0.00514, +0.00839, +0.01872 | no | no |
| F\* (WD2) beats L | +0.013568 | [+0.007268, +0.020080] | [+0.007269, +0.019881] | +0.00795, +0.01132, +0.02143 | no | no |
| L beats F3 | -0.010750 | [-0.016792, -0.004556] | [-0.016917, -0.004617] | all below 0 | no | no |
| L beats F\* | -0.013568 | [-0.019864, -0.007250] | [-0.019881, -0.007269] | all below 0 | no | no |

The per-seed means agree with the harness exactly (difference 0). Text is better than L on the fresh items: the CI lower bound is above 0 and each seed is above 0. But the mean is below the 0.02 margin. Thus "F3 beats L" and "F\* beats L" are false by the confirmatory rule. This is a secondary row with no case. Section 10.3 says that this row "checks that text still beats the latent message on fresh items". On these fresh items, the check does not pass at the 0.02 margin.

## 7. Extra checks (not part of the rule)

- **Monte Carlo spread.** The audit ran its bootstrap with 20 more Philox seeds (991124 to 991143). The CI lower bound of gain(F\*) was in [-0.000322, -0.000152] for all 20 seeds. It was below 0 every time. Thus the label "F\* does not gain over F3" does not depend on the draws. The harness bound (-0.000176) is in this range. All primary harness bounds are in or near the audit seed ranges.
- **G16 stream.** The audit wrote its own version of the G16 text: `default_rng(20261023)`, chunks of 250, item indices first, then the seed indices of each arm in sorted order of all 12 arm names, with `Generator.integers`. This version gives the harness CI bounds of all seven rows of section 4 to within 2.1e-15. Thus the harness bootstrap does what G16 says.
- **Note.** The CI lower bound of gap(L-probe, L-probe-R) is near 0. Its sign changes with the draws (range -0.000109 to +0.000293 over the 20 audit seeds). The rule gives this row no label, so this has no effect on the case.

## 8. Limits of this audit

- V1, V2 and V3 come from `confirm/results.json`. The audit did not recompute them, because they need the probe files, the weight hashes and the fit logs.
- The spent-question part of V4 is not checked. The audit did not read the sealed folder.
- The audit starts from `confirm/fresh_probs.npz`. It does not check the steps that made the probabilities (sender, receiver, readouts).

## 9. Verdict

**audit passes.** Reason: all point values agree to 1.1e-16 (bar 1e-9), all CI bounds agree within 3.1e-4 (bar about 1e-3), all labels and all four "beats" decisions are equal, and the case is equal (C). The V4 pool counts are correct.
