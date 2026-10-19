# Stage 0.5 result: composition FAILS its training-ID-only qualification — direction closed

One receiver-only process, 2026-09-12, 1071.8 s, exit 0. Arms: sham, composed-LOO,
shifted-label null, over all 224 training rows (56 IDs x 4 colors), full
128-candidate scoring, every arm run twice with exact replay. No candidate fit,
no novel label read. Artifacts: `run/RESULT05.json`, `run/{sham,loo,null}.pt`,
`run05.log`, `STAGE05_PREREGISTRATION.md`.

## Mechanics

Receiver hash unchanged before/after (6a3e1782...), no hooks, no gradients;
`familiar_memory_eq_training_features` 0.0 (training features are exactly the
sealed familiar memories); both score runs per arm bit-identical.

## Readout (224 training rows)

| arm | letters | digits | joint |
|---|---:|---:|---:|
| sham (candidate unchanged) | **224/224** | 153/224 | 153/224 |
| composed-LOO | 91/224 | 62/224 | 32/224 |
| shifted-label null | 95/224 | 43/224 | 15/224 |

Frozen decision rule: FAIL. letters ratio 0.406 (< 0.90), joint vs null 2.13
(< 3), null letters ratio 0.424 (> 0.10). The null control is as good as the
real composition — the composed codes are not decoded as the intended ID.

## Why it fails (offline, no receiver)

- Additive code model A = lstsq(labels, native codes), fit on all 56 training
  IDs: **30.5% relative error** reconstructing the native codes. The candidate's
  56-dim codes are not additively generable even in-sample.
- Leave-one-ID-out: mean 51.3% relative error (median 45.3%).
- The LOO-composed code stays nearest to its own code for only 38/56 IDs, and
  its nearest *other* code shares its letter for just 18/56 — consistent with
  the observed wrong-letter readouts.
- Native codes are small (per-ID norm ~0.97) against values ~308, yet a ~0.3
  relative code error destroys the readout: the fitted codes are ID-specific,
  not compositional. This matches the sealed memorization-only findings.

## Consequence

Stage 0's two novel letter hits (I2 -> I3, C4 letter) are not evidence of a
workable mechanism: the same construction destroys 34 of 56 training IDs'
letters on the IDs it was fit for, and a wrong-label composition scores the
same. Per the frozen rule the composition direction is **closed**; the
A0/I2/C4 sentence is not a foundation for further work, and no Stage 1/2 will
be derived from it.

Novel-ID pure binding remains unproven, unchanged by this result. No repair,
no candidate, no optimizer, no new data, no existing gate changed.
FINAL/Arabic/generation untouched; USD0 external. All sealed trees read-only.
