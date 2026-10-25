# EXPLORATORY. R.06 test T5b (factored code at full boundary width) and control C1b: report

Date: 2026-10-22. Post hoc diagnostic. The 56 fold IDs were in the training of the ancestral d0. All held-out IDs are inside a letter-digit block. The burned eight-ID bank was not used (gate 7).

## Result

| Count | Value | Thresholds | Source |
|---|---:|---|---|
| Fb, T5b fit-row joint correct (4 folds) | **672 / 672** (100%) | >= 605 | `RESULT_t5b.json` (`totals.fit_joint`) |
| Hb, T5b held-out joint correct (4 folds) | **57 / 224** (25.4%) | >= 168 passes; <= 56 supports H5-receiver | `RESULT_t5b.json` (`totals.heldout_joint`) |
| Nb, C1b held-out joint correct against true targets | **19 / 224** (8.5%) | <= 22 | `RESULT_c1b.json` (`totals.heldout_joint`) |

**Reading (frozen rule, PLAN.md section B, 4.5): inconclusive.**

1. Pass needs Hb >= 168. Hb = 57. No.
2. H5-receiver needs Fb >= 605 and Hb <= 56. Hb = 57, one row above the bar. No.
3. The strong form needs Fb < 605. Fb = 672. No.
4. Otherwise: inconclusive. The rule says: report the counts and do not move a threshold.

Thus T5b does not locate the barrier between the frozen d0 output path (H5-path) and the frozen receiver at this seam (H5-receiver). The T5 reading ("H5 is supported": the frozen interface with the d0 output path does not transfer factored codes) is unchanged.

## Counts for each fold

All counts are joint / constrained-sequential. Source: `run/t5b/fold{k}.json`, `run/c1b/fold{k}.json`.

| Fold | Fit rows | Held-out rows | T5b fit | T5b matched | C1b matched | C2b wrong, intended | C2b wrong, inserted | C3b random | C4b zero | C4b receiver only |
|---:|---:|---:|---|---|---|---|---|---|---|---|
| 0 | 164 | 60 | 164 / 164 | 25 / 26 | 6 / 6 | 0 / 0 | 25 / 26 | 0 / 0 | 0 / 0 | 0 / 0 |
| 1 | 168 | 56 | 168 / 168 | 12 / 16 | 7 / 7 | 0 / 0 | 12 / 16 | 0 / 1 | 0 / 0 | 0 / 0 |
| 2 | 168 | 56 | 168 / 168 | 13 / 13 | 4 / 4 | 0 / 0 | 13 / 13 | 0 / 0 | 0 / 0 | 0 / 0 |
| 3 | 172 | 52 | 172 / 172 | 7 / 7 | 2 / 6 | 0 / 0 | 7 / 7 | 1 / 0 | 4 / 0 | 4 / 0 |
| Sum | 672 | 224 | 672 / 672 | **57** / 62 | **19** / 23 | 0 / 0 | 57 / 62 | 1 / 1 | 4 / 0 | 4 / 0 |

## Descriptive values (section B, 4.5: no rule)

| Value | T5 | T5b | Difference |
|---|---:|---:|---:|
| Held-out joint (H, Hb) | 32 | 57 | +25 |
| Fit-row joint (F, Fb) | 672 | 672 | 0 |
| Control held-out joint (N, Nb) | 15 | 19 | +4 |

- **Complement share.** For each of the 56 held-out IDs: the norm of the 896-d offset divided by the norm of the boundary displacement, at site 3. Median 0.346 (range 0.293 to 0.419). Thus the fitted codes use the complement directions to a large degree.
- **C1b also fits its fit rows.** As in C1, the shifted-label fit reaches 100% teacher-forced accuracy on both tokens in every fold (final fit loss 0.0008 to 0.0012; T5b: 0.0008 to 0.0011).
- **Held-out answers follow the code.** In each fold, C2b "inserted" equals T5b "matched" (25, 12, 13, 7). Destructive controls stay at 0 to 4 of 224.
- **Per-fold spread.** The T5b matched count falls from fold 0 to fold 3 (25, 12, 13, 7). The folds have different held-out IDs, so this is not a trend.

## Gates and mechanics

Both jobs passed gates 1 to 7 (PLAN.md section B, 4.4, with the T5 amendments 1 to 3 through the byte copy of `t5.py`). Gate 6 (start identity) was exact in both jobs. At the end of each job, the receiver was restored exactly and its parameter hash was unchanged (6a3e1782...).

| Job | Start (UTC) | End (UTC) | Elapsed | Guard pauses | Freeze |
|---|---|---|---:|---:|---|
| T5b | 2026-10-22 14:57:04 | 16:25:41 | 5,318 s | 0 | 50b7d07096c0... |
| C1b | 2026-10-22 16:26:09 | 17:38:08 | 4,319 s | 0 | same |

CPU only, AC power at 140 W, `caffeinate -i -w`, power guard, appended logs (`run_t5b.log`, `run_c1b.log`). The smoke test passed in its first attempt (PLAN.md section D).

## What this shows and what it does not show

- With the full 960-d boundary width, a factored code still fits all fit rows. Held-out transfer rises from 32 to 57 of 224, but it stays far below the 75% bar of E3.
- By the frozen rule, the result is inconclusive. It does not separate the frozen d0 output path from the receiver.
- It does not test a per-site code, a trainable gate, norm scale or centers, cross-block IDs, or more than one seed, learning rate or complement basis (section B, 4.7).
- The jobs saved counts and codes (`fold{k}_code.pt`), but not the predictions of each row.
