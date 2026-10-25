# EXPLORATORY. R.06 localization, CPU tests T2 and T3: plan

Written 2026-10-20, before any computation in this folder. Parent plan: `../../../reports/r06_localization_plan_20261020.md` (sections 6 and 7). Requested by the parent session of this session group. These are post hoc diagnostics on a burned bank (the eight novel IDs) and on the fold bank (56 IDs that the ancestral d0 saw in training). No result here is a confirmatory or pristine claim.

Scope rules:

- CPU only. No model load on any device. No sender or receiver forward pass. No GPU.
- Read-only inputs. This folder is the only place this job writes.
- No T5, T6 or T7. They are model runs and need [researcher]'s word.
- No threshold, metric or seed changes after this file and `FREEZE.json` exist.

## Inputs (paths relative to `ember/research/`)

| Name | File | Content |
|---|---|---|
| Training IDs | `latent_binding_alignment_source_20260911/inputs.json` (`ids`) | 56 IDs, fixed order |
| Training sender states | `latent_binding_alignment_source_20260911/run/captures.pt` (`first`) | 56 x 2 x 2048, raw layer-13 states of the letter token and the digit token |
| Training slots (record) | `latent_binding_alignment_source_20260911/run/features.pt` (`wire`) | 56 x 64 |
| Novel and familiar stages | `latent_binding_identity_localization_20260910/stages.pt` | `sender_field`, `wire64`, `selected_3` with `meta` (`id`), 64 rows for each bank |
| Encoder weights | `latent_binding_training_alignment_20260911/run/candidate.pt` (`resampler.*`) | must equal the d0 entries below |
| d0 bridge | `latent_binding_training_fold_recovery_20260913/run/fold0/step0/candidate.pt` | `resampler.*`, `projector.weight`, `reader.value.weight` |
| d0 value base | `latent_binding_training_fold_recovery_20260913/coordinates.pt` (`[0]['base']`) | must equal the d0 `reader.value.weight` |
| Oracle fits | `latent_binding_constrained_code_extended_20260911/run/{novel,familiar}_fit.pt` (`initial`, `terminal`) | 8 x 64 each |

## Fixed definitions

Labels. For an ID such as "G3", the label row is 24 one-hot columns: 16 letter columns (A to P) and 8 digit columns (0 to 7). This is `target` in `latent_binding_training_feature_probe_20260911/probe.py`. The letter columns sum to 1, so the intercept is in the span. The design is intercept plus letter plus digit. Its rank is 22, because the 56 training IDs form two blocks (`latent_binding_training_fold_recovery_20260913/folds.json`).

Additive fit. For training matrix X (56 rows) and label matrix Y: coef = pinv(Y) X, with rcond = max(Y.shape) x float64 eps (as in the record). The prediction for any ID is its label row times coef.

Metrics on the 56 training IDs:

- rho_in(X) = ||Y coef - X||_F / ||X - mean(X)||_F. This is the record definition.
- rho_loo(X) = sqrt(sum_i ||x_i - y_i coef(-i)||^2) / ||X - mean(X)||_F, where coef(-i) is fit without ID i. Every leave-one-out fit keeps both blocks (`folds.json`, `loo_components_preserved` = 56).
- null(X): the median of rho_in over 1,000 random row permutations of Y (numpy `default_rng(20261020)`). Report it as a reference only. It does not enter a reading rule.
- Novel residual of ID z: ||y_z coef - z|| / sqrt(||X - mean(X)||_F^2 / 56). This is the record definition.

Probe. The record ridge probe: center X and Y on the fit rows, lambda = 0.001 x ||X_c||_F^2 / n, dual solve, letter = argmax of columns 0 to 15, digit = argmax of columns 16 to 23. The lambda rule is the record rule; for the actual slot it gives 0.0006289057653718944. The parent plan gives this value; the rule, not the number, applies to the other variants, as in the record ("the same fixed lambda rule"). Report the training-fit novel count and the leave-one-ID-out count (refit without each ID, lambda recomputed on the 55 IDs).

## T2. Encoder variants

For token states h (2 x 2048) of one ID, with LN = layer norm without affine parameters, K = `resampler.key.weight`, V = `resampler.value.weight`, q = row 0 of `resampler.queries`:

- k_t = normalize(K LN(h_t)); alpha = softmax over t of 2 x cos(q, k_t); u = sum_t alpha_t V LN(h_t); s = sqrt(mean(u^2)); slot = u / s.
- S_raw: the raw states, concatenated (4096). This is the record sender reference.
- S_LN: [LN(h_1), LN(h_2)], concatenated (4096).
- E_a: the actual slot.
- E_b: alpha replaced by its mean over the 56 training IDs; s computed as in E_a.
- E_c: s replaced by its mean over the 56 training IDs; alpha as in E_a.
- E_d: both replaced. E_d is linear in S_LN.
- For novel IDs, the replacements use the training means only.

Gates (stop and report on the first failure):

1. E_a for the 56 training IDs agrees with the record slots (`features.pt`), and E_a for the novel and familiar rows agrees with `wire64`: max absolute error at most 1e-5. The training-alignment and d0 encoder entries are equal.
2. rho_in(E_a) = 0.292580 and rho_in(S_raw) = 0.113833, each within 1e-4.
3. Probe on E_a: novel 6/8 with errors A0 to H3 and A2 to M3; leave-one-ID-out 44/56.

Reading rule (copied from the parent plan). d_nl = rho_in(E_a) - rho_in(E_d); d_lin = rho_in(E_d) - rho_in(S_LN).

- If d_nl >= 0.10 and d_nl >= 2 x d_lin: "H3, nonlinear" (the attention weights and the RMS scale cause most of the loss).
- If d_lin >= 0.10 and d_lin >= 2 x d_nl: "H3, linear" (the value matrix selects non-additive directions).
- Otherwise: "H3, mixed".
- Secondary: if the probe on E_d decodes A0 and A2 correctly and keeps the other six novel IDs correct, the nonlinear steps cause the two novel slot failures.

Report rho_in, rho_loo, null, novel residuals and probe counts for all six variants. Only the quantities above enter the reading.

## T3. Residual pull

- A = W_v P (64 x 64), from the d0 `reader.value.weight` and `projector.weight`, in float64.
- Gate 4: for all 128 saved novel and familiar rows, A x `wire64` agrees with `selected_3`: max absolute error divided by max absolute value at most 1e-5. The d0 value weight equals the `base` matrix.
- Gate 5: each oracle `initial` row equals the `selected_3` row of exactly one ID of its bank (max absolute difference 0). This gives the ID of each oracle row.
- Slots: training slots from the record (`features.pt`); one slot for each novel and familiar ID from `wire64` (rows with the same ID must be equal).
- f(X) = additive prediction from the 56 training slots; for a familiar ID, from the 55 other training IDs. r(X) = slot(X) - f(X).
- delta(X) = terminal(X) - initial(X).
- kappa(X) = cos(delta(X), -A r(X)).
- kappa_cross(X) = max over the other seven IDs Y of the same bank of cos(delta(X), -A r(Y)).

Reading rule (copied from the parent plan):

- Support for H4 (residual pull): kappa(X) >= 0.5 and kappa(X) > kappa_cross(X) for at least 6 of 8 novel IDs, and median kappa (novel) - median kappa (familiar) >= 0.3.
- No support: kappa(X) <= 0.2 for at least 6 of 8 novel IDs. This does not reject H4; T5 decides.
- Otherwise: inconclusive.

Report also, as description only: ||A f||, ||A r||, ||delta|| for each ID, and kappa for the six novel IDs whose slot the record probe decodes.

## Outputs

`t2_t3.py` (modes `selftest`, `freeze`, `run`), `FREEZE.json`, `RESULT.json`, `REPORT.md`. Every output carries the label EXPLORATORY.
