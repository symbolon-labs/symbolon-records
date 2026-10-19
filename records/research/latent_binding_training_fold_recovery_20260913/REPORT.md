# Four-fold alignment validation: NO-GO

Completed 2026-09-13. All four folds and the registered grid completed with valid
mechanics. The best positive step under the frozen selection rule is 100, but
both held-out metrics are worse than step 0. This procedure does not qualify
an all-56-ID refit. The user additionally directed that this be the last training
run before failure-mode analysis; no refit or further experiment was launched.

| Step | Joint / 224 | Constrained sequential / 224 |
|---|---:|---:|
| 0 | 32 | 32 |
| 50 | 23 | 23 |
| 100, registered selection | 22 | 25 |
| 200 | 17 | 20 |

Selection maximizes aggregate sequential count over positive steps, ties earlier,
then requires both metrics to strictly exceed baseline. Step 100 fails both
requirements. There is no fallback step, and none of the positive checkpoints
exceeds baseline on either metric. These are developmental counts, not a
significance claim or a novel-ID test.

| Fold (zero-based) | Fit / validation IDs | Step 0 J/S | Step 50 J/S | Step 100 J/S | Step 200 J/S |
|---|---:|---:|---:|---:|---:|
| 0 | 41 / 15 | 12/12 | 14/14 | 10/13 | 8/8 |
| 1 | 42 / 14 | 4/4 | 4/4 | 4/4 | 4/4 |
| 2 | 42 / 14 | 4/4 | 0/0 | 0/0 | 1/4 |
| 3 | 43 / 13 | 12/12 | 5/5 | 8/8 | 4/4 |

Each validation ID has four query contexts, held together. There are 56 IDs,
not 224 independent identities. The memories replace the selected ID slot in
a saved query-matched memory template. They are assembled development fixtures,
not fresh full-table source captures or evidence of full binding transfer.
All folds started from d0/theta0/fresh Adam; conditioning used only fit IDs.
Only the existing reader value matrix changed. No novel data or evaluation was
opened. The standing native novel result remains 822db91d's 4/4/4, still NO-GO.

## What failed

The independent saved-score review is in
`../latent_binding_fold_failure_modes_20260913/REPORT.md` and `RESULT.json`.
It is explicitly post hoc and changes no gate or selection rule.

At selected step 100, first-letter accuracy increases 56→60 and digit accuracy
conditional on the gold letter increases 36→41, but their successful overlap
shrinks 32→25. Ten baseline-correct rows regress (D4:1, G3:1, L7:4, O4:4).
Only three new sequential successes appear, O5 in green/blue/yellow; none is
joint-correct. Thus joint scoring gains zero rows and loses ten.

For those O5 rows, the correct prefix and its conditional digit both win, but
the joint score favors L4. Gold joint margins are -0.17604, -0.16944, -0.13425.
This is an observed scoring disagreement, not permission to change the readout.

Longer training does not consistently preserve correct omitted-ID behavior:
step 200 gains seven sequential rows but loses nineteen relative to baseline.
Fold 1's unchanged 4/56 is the same N1 ID at every point. Fold 2's return to
4/56 sequential at step 200 replaces the lost L7 with C1; equal totals hide
different identities. Only 11/56 held-out IDs have any sequential success at
any registered checkpoint.

Training loss decreases strongly in every fold. Last **pre-update** step-200
losses are 0.46124, 0.35505, 0.43490, 0.42638; teacher-forced letter/digit counts
are 164/125 of164, 168/146 of168, 168/128 of168, and172/140 of172. These are
training measurements before the last update, not terminal convergence claims.
Improving this fit coexists with worse held-out whole-ID performance.

## Mechanical and independent evidence

Freeze: `4e51101526401a01f9cb924c5d3f22cbf084f3cb427eabb040281e0eda17bc63`.
One CPU receiver job, session55372, launch06:48:26Z, terminal10:27:34Z.
Wrapper elapsed13147.945902177s, peak2960504KiB, exit0, no timeout/interruption
or forced kill, child reaped, USD0 external. Every fold was below9000s;
whole job below37800s. No model worker was restarted.

All eight historical familiar arms replayed exactly before fitting. Each fold
passed zero/perturbed native-export parity and finite-difference gradient
qualification. All16 checkpoints passed full native5/6-state contracts,
unchanged routes and nonquery positions, actual repeat equality, native/render
parity0.0 and matched direct scoring<=1e-4 with equal argmax. Maximum recorded
matched direct error is1.1444091796875e-5.

Available original and interrupted matched-parent candidates, scores, direct
diagnostics, theta, Adam and histories replayed exactly. The sealed failedpoint
fold1step100 diagnostic also replays exactly: matched error7.62939453125e-6;
the original cross-batch discrepancy .0001010894775390625 remains reproduced
and disclosed. This corrects comparison geometry without changing tolerance;
it does not retroactively pass the original incomplete job.

`ANALYSIS.json` independently reconstructs scalar predictions/counts/selection,
NumPy conditioning and all exported weights (maximum relative error0.0), and
replays saved reader contracts. `EXECUTION_AUDIT.json` in the supplemental tree
checks all optimizer step counters/moments/history prefixes and terminal records.
Stock/disabled/zero/restored arrays are exact; receiver hashes are unchanged;
runtime confirms no hooks or gradients. The wrapper's `restoration_verified:false`
means it does not perform that verification; runtime and arrays provide the proof.

Nine synthetic/historical tests pass. Fresh identity, main analysis replay,
failure-review replay and execution-audit replay are recorded in
`closing_checks.json`. Closing uses `close.py close`, followed by read-only
`close.py verify`; `seal.json` records the complete inventory. Older trees
remain unchanged. Replays of the same deterministic trajectory are not
independent efficacy replications.

## Stopping point

Stop after reporting and sealing this run. No all56 refit, optimizer extension,
checkpoint search, adapter, novel evaluation, FINAL, Arabic or generation.
The next session should first inspect the saved failure patterns and formulate
a bounded diagnostic of mapping/generalization failure. This run shows that
the registered stopping-point procedure fails; it does not establish identity
erasure, impossibility of transfer, or the causal superiority of another repair.
