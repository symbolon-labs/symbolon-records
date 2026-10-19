# Convergence continuation (corrected successor; see CORRECTION.md)

Single-variable experiment. No new data, no new labels, no architecture change.

## Why

The best native candidate (822db91d, familiar 32/32/31, novel 4/4/4) came from an
optimization that was stopped mid-descent. Frozen-tree evidence:

- The terminal step-199 pre-update loss was 0.5465 and still falling about 4% per
  five steps; teacher-forced training digits were 153/224 and letters 224/224.
- Under a fixed 4-query seam test with each training ID's own code, only 37/56
  training IDs are fully correct, so the map is not even training-complete.
- The novel-ID gold margins improved monotonically with training: d0 to 822db91d
  moved every held-out ID's margin up by +9.7 to +18.7 nats (A0 +9.7, C4 +12.4,
  I2 +13.9, K0 +10.2, K3 +9.7, M0 +10.2, M3 +18.7). The closest held-out ID
  (K3) is only 0.435 nats short of correct.
- An exponential fit to the sealed trajectory puts the loss at ~0.004 and the
  training digit errors at ~4/224 by step 600, i.e. effectively converged.

## What changes

Exactly one thing: the training budget. Continue the identical optimization for
exactly 400 further full-gradient Adam steps (steps 200..599, total 600), from
the sealed step-200 state (theta and Adam moments) in the sealed tree.

Everything else is bit-identical and inherited: same 56 training IDs, same 224
rows, same batch order and size 8, same lr `0.05*sqrt(960/(64*56))`, same both-token
full-vocabulary CE, same seed 94117, same f32 export per forward, same frozen
receiver, same 10 fixed bridge entries. No novel label, feature, oracle code or
outcome enters optimization, the loss, stopping or checkpoint selection. The
stopping rule is a fixed step count decided before the run; there is exactly one
terminal export and no checkpoint selection.

## Mechanics required before the terminal evaluation counts

- The continuation start must reproduce the sealed candidate bitwise:
  `weight(theta_start, coord)` == sealed `reader.value.weight`, and the full
  11-entry state must equal the sealed candidate.
- The sealed Adam state must load with step 200, and its moments must be finite.
- Actual native replay of the sealed historical d0 baselines (all 8 arms, both
  pools, two calls each) must match the sealed arrays EXACTLY, and the sealed
  822db91d candidate arms must replay EXACTLY, proving the environment is the
  same one that produced the sealed scores.
- Zero/perturbed teacher parity and the finite-difference gradient check must
  pass with the start theta (unchanged tolerances: parity <= 1e-4 with the same
  argmax, relative gradient error <= 0.1 at epsilon 0.01).
- Terminal native-render parity <= 1e-4 with the same argmax; the 10 non-target
  entries, stock/disabled/zero controls, query/key routing, direct-score parity
  <= 1e-4, live receiver hash, no hooks and no receiver gradients.

## Gates

Unchanged. Familiar original/donor/changed minima 31/31/28 with all inherited
paired and destructive controls; only a full familiar PASS opens the novel arms.
Novel counts are reported joint and constrained-sequential with the same gates.
A familiar failure keeps the novel arms closed and is reported without tuning.

## Interpretation limits

A positive result would show that the sealed limited-transfer result was
budget-limited rather than representational, and would still be eight distinct
IDs with prior exposure, not pristine novel identities or unrestricted
generation. A negative result would show that completing this bounded
optimization does not improve held-out transfer for this parameterization; it
would not prove that no map, no budget or no interface can work. Either way the
original d0 0/0/0 and all sealed trees remain unchanged. FINAL, Arabic and
generation remain reserved; no reset, cloud, browser, Rust, commit or push;
USD0 external.
