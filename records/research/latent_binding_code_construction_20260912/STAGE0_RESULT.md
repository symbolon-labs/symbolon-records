# Stage 0 result: composition diagnostic partially reproduced — I2→I3 exact, A0 refuted

One receiver-only process, 2026-09-12, 6m43s, exit 0. Seven arms on the novel
original condition, no fit, no candidate, no gates, no new data. Artifacts:
`run/RESULT.json`, `run/<arm>.pt`, `run.log`, `PREREGISTRATION.md`.

## Mechanics (all exact, before any outcome was read)

| check | result |
|---|---|
| f61 / 822d sham payload == sealed prefill capture output[:,3:5] | 0.0 / 0.0 |
| 822d and f61 theta -> candidate weight reconstruction | 0.0 / 0.0 |
| f61 / 822d sham scores vs sealed trained_novel/correct.pt (token_lp, lp) | 0.0 / 0.0, predictions equal |
| direct-score parity | 3.814697265625e-06 <= 1e-4, same argmax |
| receiver hash before/after, hooks, gradients | unchanged / none / none |

Environment: tokenizers restored to the sealed 0.22.2 pin (today's binding
install had bumped it to 0.23.1, breaking `import transformers`); platform
identical; all sealed trees read-only; only this tree was written.

## Readout (32 novel rows; first-token rows / joint)

| arm | first-token rows | joint | note |
|---|---:|---:|---|
| f61 sham (learned) | 4 | 1 | M3 only |
| f61 R3 golden one-hot composition | 0 | 0 | catastrophic |
| f61 R1 fitted additive (22-dim) | 4 | 1 | C4 letter fixed |
| f61 R2 probe composition | 0 | 0 | |
| 822d sham (learned) | 4 | 4 | M3 only |
| **822d R1 fitted additive** | **8** | 0 | I2, C4 letters fixed |
| 822d R2 probe composition | 8 | 0 | same I2/C4 letters |

## The three cited IDs

| ID | learned (822d sham) | composed (822d R1 / R2) | claim | verdict |
|---|---|---|---|---|
| A0 | D1/D4 | B3 / G3 | A0 (A4) | **not reproduced** |
| I2 | G2/G3 | **I3 / I3** (all 4 rows) | I2 (I3) | **reproduced exactly** |
| C4 | D4 | C7 / C7 | C4 (C4) | first token correct, digit wrong |

The claim's substance — composed letter/digit codes reproduce the correct first
token where the learned code does not — holds for two of the three cited IDs
under the fitted additive construction in 822d's own code space. The A0 element
is refuted under every frozen recipe (golden one-hot, fitted additive, probe
composition). The one-line handoff claim therefore reflects either a
construction not captured by these three recipes or an inaccurate memory.

Composition is a trade, not a free gain: first-token rows double (4 -> 8) while
the single joint success (M3) is destroyed (4 -> 0).

## Consequences for the design (no Stage 1 licensed as written)

- Stage 0's pass rule required A0 letter A; it does not reproduce, so **Stage 0
  does not pass and Stage 1/2 are not licensed by DESIGN.md as written**.
- The reproducible construction is *fitted additive codes in the candidate's own
  code space* (R1/R2). Golden exact one-hot composition (R3) is catastrophic
  (0/32) and must not be carried into a successor.
- The candidate matters: f61's 22-dim space gives 4/32 first-token rows for R1
  and 0/32 for R3; 822d's 56-dim space gives 8/32.
- A revised design must fix the estimator (fitted solve on 56 ID-level features
  was used here) and any composition choice by 56-training-ID leave-one-ID-out
  folds, never on the eight; and it must state the expected trade against the
  learned joint count (4/32 at 822d).

No repair, no candidate, no optimizer, no new data, no threshold change.
FINAL/Arabic/generation untouched; USD0 external.
