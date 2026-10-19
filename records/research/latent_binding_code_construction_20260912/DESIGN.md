# Build the code from the component structure (DESIGN ONLY)

DESIGN ONLY: no harness frozen, no endpoint call, no fit, no candidate, no result.
This file pre-registers a *sequence with stop rules*; it is not run authorization.

EXECUTED 2026-09-12 — Stage0 partial reproduction (I2 -> I3 exact, C4 letter,
A0 not reproduced), then Stage0.5 LOO qualification FAILED (composed-LOO
letters 91/224 vs sham 224; shifted-label null 95; additive code reconstruction
30.5% in-sample / 51.3% LOO). The direction is CLOSED: Stage1/Stage2 are not
licensed, and the A0/I2/C4 sentence is not a foundation for further work. See
STAGE0_RESULT.md, STAGE05_RESULT.md, PREREGISTRATION.md,
STAGE05_PREREGISTRATION.md.

## Why this direction

Closed evidence that shapes it:

- Free56-dim value alignment (822db91d): novel4/4/4(M3only), NO-GO.
- Fixed ridge (ef4bd6e4):0/0/0. Soft-component22-dim CE (f61): familiar32/32/31,
  novel1/1/0. Convergence to step600: novel0/0/0 — more optimization of this
  objective is not the way (margins peak near step200 and fall).
- Exact-component oracle on f61 (golden one-hot codes at inference): familiar
  qualification FAILED (donor redirect8/32, random intended17/32); novel arms
  never opened. That intervention changed query code and payload together while
  d0base, keys, query states and prefixes stayed fixed.
- Training-only wire probe (56 IDs, no novel labels) decodes held-out whole IDs
  6/8 (letters6, digits6, joint6; errors A0->H3, A2->M3; shifted-label null0/8).
- In-session composition diagnostic (convergence session): composed letter/digit
  codes reproduced the correct first token for A0 (A4), I2 (I3) and C4 (C4)
  where the learned code did not.

**Premise risk (drives Stage0).** The composition diagnostic is recorded only as
a sentence in LATENT_BINDING_NEXT_SESSION_20260911.md and the convergence
INTERPRETATION; no script, arrays or recipe were sealed. Its claim must be
reconstructed from artifacts before any new fit or receiver job.

## Stage0 — deterministic reconstruction (read-only, no endpoint)

Use only sealed artifacts: the822d and f61 candidates, the sealed component
subspace qualification (C24x64, P24x22, Q960x64, B960x22, R), the probe
predictions, and the oracle rendering/intervention code. No receiver load, no
native answers, no new fit.

Enumerate exactly these candidate recipes for "composed letter/digit codes" and
record which (if any) reproduces the reported first-token result:

- R1 additive solve on the candidate's own training values: A = argmin ||Theta_train - Y A||; composed held-out code uses the known component pair.
- R2 sealed-subspace component codes: compose using the P/B path components with the probe's decoded held-out components (non-oracle path).
- R3 golden one-hot composition (reference only; the oracle already closed at inference).
- R4 the same three constructions rendered through the *unchanged* f61 export path (no intervention term) as a control.

Stage0 passes only if at least one non-oracle recipe reproduces, with exact
mechanics, the reported first-token behavior (A0 letter A, I2 letter I, C4
exact C4) within the frozen scorers. Otherwise stop and report the refutation;
no substitute experiment is launched from this design.

## Stage1 — qualification of the construction

Only after Stage0 passes. Convert the validated recipe into a *deployable,
no-novel-label* construction and qualify it on the 56 training IDs and the
familiar bank before any novel evaluation:

- The construction may use the training-only probe (and the sealed C/P/Q/B
  artifacts) at inference; it may not use held-out labels, held-out oracle
  codes, or novel features as training targets.
- Familiar original joint AND constrained-sequential must stay >=31; all
  destructive/canonical/zero controls unchanged at 0 (the oracle failed here, so
  this gate is the real test of the construction, not a formality).
- Independent render parity <=1e-4 same argmax, 5/6-state contract, routing,
  restoration and live-hash checks as in the sealed parents.
- If familiar qualification fails, stop: it is a qualification failure, not a
  novel-ID NO-GO and not evidence of receiver incapacity.

## Stage2 — one bounded native evaluation

Only on complete Stage1 PASS. One receiver-only job, fixed cap (>=3600s; pin the
exact cap before freeze), full128 both-token joint AND constrained-sequential,
original and changed arms, matched/wrong/random where applicable. Unchanged
gates: novel original>=31/31/28 is not expected from this line; report counts
and paired gates as they fall against the standing822d4/4/4. No generation.

## Selection discipline (mandatory)

Any recipe, threshold, rank or stop choice is made on 56-training-ID
leave-one-ID-out folds only. The eight held-out IDs evaluate the final fixed
artifact once; they never select a recipe, checkpoint or threshold. No lambda
sweep, no outcome-dependent extension, no post-outcome repair.

## Interpretation limits

Even a Stage2 novel pass is finite-bank transfer on eight ID combinations with
older design/evaluation exposure, not pristine novelty, not generation and not
Arabic. A failure closes this bounded construction, not all compositional or
component-structured methods. The exact-component oracle's familiar failure
already shows the answer is not a function of the selected value code alone;
Stage0/1 are designed to test the construction claim before that wall.

## Execution boundary

No endpoint job, source capture, new bank, training, optimizer change or
architecture change is licensed by this file. Stage0 is a read-only diagnostic
that may be built and run without further approval; Stage1 and Stage2 each
require their own frozen harness, tests and preregistration after Stage0's
outcome is known and reported. FINAL/Arabic/generation reserved; no reset,
cloud/browser, Rust, commit or push; USD0 external. All closed trees read-only.
