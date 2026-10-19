# Convergence continuation: familiar PASS, novel 0/0/0 — the 4/4/4 was not budget-limited

Continuing the sealed training-only alignment from its step-200 checkpoint to a
fully-converged step-600 checkpoint **removes** the held-out whole-ID transfer:
novel 0/0/0 against 4/4/4 at step 200, while familiar stays 32/32/31.

| checkpoint | training loss (pre-update) | teacher-forced letters/digits | familiar o/d/c | novel o/d/c |
|---|---|---|---|---|
| d0 (no training) | - | - | 32/32/31 | 0/0/0 |
| sealed 822db91d, step 200 | 0.5465 | 224/224, 153/224 | 32/32/31 | **4/4/4** |
| this run, step 600 | 0.0954 | 224/224, **224/224** | 32/32/31 | **0/0/0** |

Candidate `ca7e755f2dceee34ca06a873111e92ae7d919d1ef737abfcd0c1740601a4dd22`.
Joint and constrained-sequential novel counts agree at 0. All novel destructive
intended counts are 0 (one random_71001 sequential hit, 0 joint). Familiar counts
and all paired/destructive controls PASS; novel count and paired gates FAIL.

## What ran

Exactly one change from the sealed best candidate: 400 further full-gradient
Adam steps on the identical 56 training IDs x 224 rows, same batch order and
size, same lr, same both-token full-vocabulary CE, same seed, same frozen
receiver, same 10 fixed bridge entries. The continuation starts from the sealed
theta and Adam moments; `weight(theta_start, coord)` reproduces the sealed
candidate weight exactly (relative error 0.0), and the full 11-entry start state
equals the sealed candidate.

Nothing else changed. No novel label, feature, oracle code or outcome entered
optimization, the loss, the stopping rule or checkpoint selection. The budget was
fixed before the run; there is exactly one terminal export.

## Mechanism: held-out margins move backwards after the peak

Per-ID novel gold margins (mean over the four contexts, nats):

| ID | d0 | step 200 | step 600 | 600 - 200 |
|---|---|---|---|---|
| A0 | -21.64 | -14.51 | -22.35 | -7.84 |
| A2 | -17.14 | -4.44 | -27.01 | -22.58 |
| C4 | -14.08 | -1.63 | -0.62 | +1.01 |
| I2 | -17.59 | -3.66 | -6.20 | -2.55 |
| K0 | -18.79 | -8.63 | -10.00 | -1.37 |
| K3 | -10.16 | -0.44 | -0.35 | +0.08 |
| M0 | -27.53 | -17.32 | -18.36 | -1.04 |
| M3 | -18.43 | **+0.25** | -0.44 | -0.68 |

Training moved every held-out ID up by +10 to +19 nats from d0 to step 200, and
then moved most of them back down. The single success (M3, +0.25) crossed below
zero at the converged checkpoint; the two closest failures (K3 -0.44, C4 -1.63)
improved only marginally. The exported weight moved 2041 in Frobenius norm from
the step-200 candidate. Fitting all 224 training rows exactly (letters and digits)
did not place any held-out whole-ID code inside its correct region.

So the sealed 4/4/4 was the observed peak of held-out whole-ID transfer along
this trajectory, reached before the training set was fit, not a truncated
optimum. More optimization of this objective is not the way to improve it.

## Complete mechanics

- Start reproduction bitwise (relative error 0.0) from the sealed theta, Adam
  moments and candidate; step 200 confirmed.
- Environment: historical d0 baselines replayed EXACTLY for all 8 arms in both
  pools with two actual calls each, and the sealed 822db91d candidate arms
  replayed EXACTLY (max abs 0.0), proving the same environment that produced the
  sealed scores.
- Zero, perturbed and terminal native-render teacher parity EXACT (error 0.0)
  with identical argmax; gradient finite-difference analytic vs central within
  the unchanged 0.1 relative tolerance at epsilon 0.01.
- All trained arms repeated EXACT; direct-score parity <= 1e-4 with the same
  argmax; query/key routing and saved 5/6-state reader contract EXACT; stock,
  disabled and zero controls EXACT; live receiver hash unchanged; no hooks and
  no receiver gradients at exit; full restoration to the d0 state.
- Independent NumPy weight reconstruction, scalar label/score joins and the
  paired bootstrap replay PASS. Five closing commands PASS (tests, runtime check,
  analysis, final contract, per-ID).
- One receiver-only job, launch 2026-09-11T09:58:09Z, exit 0, no timeout, child
  reaped, 8524.735650577s, peak rss 2907192 KiB, USD0 external. No optimizer
  existed anywhere except the declared continuation.

## Lineage and corrections

The first attempt at this experiment
(`latent_binding_convergence_20260911`, seal
`e79399a83d705c53a3a233aa1d3dabb9dfa258ea60540019ccb5fb4a32420d17`) is sealed
MECHANICS INVALID: it exited 1 during the sealed-candidate replay because the
bridge still held the d0 state; no fit ran and no candidate was produced. This
tree is the corrected successor with exactly one line changed (load the sealed
candidate state before the reference phase), disclosed in CORRECTION.md. The
corrected replay was verified arm-by-arm to reproduce the sealed arrays exactly
before the job was launched.

## Scope

These are eight held-out whole-ID combinations with prior design/evaluation
exposure, not pristine novel identities, and the task is native full-ID scoring,
not unrestricted generation. The result bounds this parameterization and this
supervision: it says held-out transfer peaks early along this trajectory. It does
not show that other code constructions, other training-ID-only validation
schemes, or other interfaces cannot transfer; and it does not change the
original d0 0/0/0 or any sealed tree. FINAL, Arabic and generation remain
reserved.
