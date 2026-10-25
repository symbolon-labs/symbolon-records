# EXPLORATORY. R.06 test T5 (factored oracle at the frozen interface) and control C1: report

Date: 2026-10-21. Post hoc diagnostic. The 56 fold IDs were in the training of the ancestral d0. All held-out IDs are inside a letter-digit block. The burned eight-ID novel bank was not used.

## Result

| Count | Value | Threshold | Source |
|---|---:|---|---|
| F, T5 fit-row joint correct (4 folds) | **672 / 672** (100%) | >= 605 | `RESULT_t5.json` (`totals.fit_joint`) |
| H, T5 held-out joint correct (4 folds) | **32 / 224** (14.3%) | >= 168 rejects H5; <= 56 supports H5 | `RESULT_t5.json` (`totals.heldout_joint`) |
| N, C1 held-out joint correct against true targets | **15 / 224** (6.7%) | <= 22 | `RESULT_c1.json` (`totals.heldout_joint`) |

**Reading (frozen rule, PLAN.md section 1):** F >= 605 and H <= 56. Thus "the interface accepts factored codes for the fit IDs, but they do not transfer. **H5 is supported.**" The N condition belongs only to the branch that rejects H5; N = 15 is also below 22.

For the majority cohort of R.06 failures, the failure is located at the reader interface: at receiver block 15, through the frozen d0 output path, a code built from one vector for each letter and one vector for each digit does not bind held-out IDs. T6 does not run, because its rule needs T5 to reject H5. The conditional follow-up T5b (output map also trainable) is not approved and did not run.

## Counts for each fold

All counts are joint / constrained-sequential. Source: `run/t5/fold{k}.json` and `run/c1/fold{k}.json`.

| Fold | Fit rows | Held-out rows | T5 fit | T5 matched | C1 matched | C2 wrong, intended | C2 wrong, inserted | C3 random | C4 zero | C4 receiver only |
|---:|---:|---:|---|---|---|---|---|---|---|---|
| 0 | 164 | 60 | 164 / 164 | 10 / 10 | 8 / 8 | 4 / 4 | 10 / 10 | 0 / 0 | 0 / 0 | 0 / 0 |
| 1 | 168 | 56 | 168 / 168 | 6 / 6 | 0 / 0 | 0 / 0 | 6 / 6 | 0 / 1 | 0 / 0 | 0 / 0 |
| 2 | 168 | 56 | 168 / 168 | 8 / 12 | 7 / 7 | 4 / 4 | 8 / 12 | 0 / 0 | 0 / 0 | 0 / 0 |
| 3 | 172 | 52 | 172 / 172 | 8 / 8 | 0 / 0 | 0 / 0 | 8 / 8 | 2 / 0 | 4 / 0 | 4 / 0 |
| Sum | 672 | 224 | 672 / 672 | **32** / 36 | **15** / 15 | 8 / 8 | 32 / 36 | 2 / 1 | 4 / 0 | 4 / 0 |

## Descriptive notes (not readings)

- **C1 also fits its fit rows.** The C1 fit targets are shifted labels, so the letter-digit map is not additive. Yet the teacher-forced fit accuracy at step 200 is 100% for both tokens in every fold (final fit loss 0.016 to 0.032; `run/c1/fold{k}.json`, `history`). T5 shows the same values (final loss 0.016 to 0.023). Thus, through this frozen interface, a sum of a letter vector and a digit vector can carry an arbitrary answer for each fit ID. The frozen path does not read the letter and the digit as separate parts of the code. This agrees with the reading.
- **Held-out answers follow the code.** For each fold, C2 "inserted" equals T5 "matched" (10, 6, 8, 8). When a held-out code works, the parts of another ID move the answer to that other ID in the same number of rows.
- **T5 is above its control, but far below the bars.** H = 32 against N = 15. Destructive controls stay at 0 to 4 of 224.
- **Equal numbers in other tests.** Two other tests in the records give the same counts: the code-construction test (composed codes 32/224 against 15/224 for its null, `latent_binding_code_construction_20260912/STAGE05_RESULT.md`) and the d0 step-0 held-out count of the fold record (32/224). These are different tests with different codes. No inference follows from the equal numbers.

## Gates and mechanics

Both jobs passed all gates under amendments 1 to 3 (PLAN.md section 5). Values in both jobs: native path against seam, max absolute logit 0.0 with the same argmax; routes, memory slot, outside sites and payload exact; direct-score check with the same argmax (difference 1.354e-4, descriptive). At the end of each job, the receiver was restored exactly, and its parameter hash was unchanged (6a3e1782...).

| Job | Start (UTC) | End (UTC) | Elapsed | Guard pauses | Freeze |
|---|---|---|---:|---:|---|
| T5 | 2026-10-21 16:52:26 | 18:25:36 | 5,591 s | 0 | ce382b1f8abd... |
| C1 | 2026-10-21 18:26:32 | 19:46:24 | 4,792 s | 0 | same |

Both jobs ran on the CPU only, on AC power at 140 W, with `caffeinate -i -w` and the power guard. No other heavy job was active (preflight in `run_t5.log` and `run_c1.log`). The smoke test needed six attempts; PLAN.md section 5 records each attempt and each amendment.

## What this shows and what it does not show

- It shows that, for these 56 in-block IDs, the frozen d0 output path into the frozen receiver at block 15 accepts factored codes for the IDs that it was fit on, but these codes do not carry held-out compositions.
- It does not separate the frozen d0 output path from the receiver itself. E3 (a factored code with a trained cross-attention reader after block 15) passed at 93.8%, so the receiver can read a factored code through a trained reader. T5b, with the output map trainable, would test this. It needs [researcher]'s word.
- It does not test cross-block IDs or the burned novel bank.
- One seed, one learning rate, one step budget, as in every R.06 test.
- The jobs saved counts and codes (`fold{k}_code.pt`), but not the predictions of each row. A list of which IDs transfer needs a new forward run.
