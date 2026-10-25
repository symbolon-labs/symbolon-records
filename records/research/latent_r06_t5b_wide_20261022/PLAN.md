# EXPLORATORY. R.06 test T5b (factored code at full boundary width) and control C1b: plan

Written 2026-10-22, before any T5b or C1b data exists. Source plan: `../../../reports/r06_t6_t5b_plans_20261021.md` (sections 2 and 4 are copied below word for word). T5 result: `../latent_r06_t5_factored_oracle_20261020/REPORT.md` (F = 672/672, H = 32/224, N = 15/224; reading "H5 is supported").

**Authorization.** [researcher], in this session, 2026-10-22: [redacted: private chat]

**Status of all outputs:** EXPLORATORY. Post hoc diagnostic. The fold IDs were in the training of the ancestral d0. The burned eight-ID bank is not used.

## A. Source plan, section 2 (copied word for word)

## 2. Rules for both plans

### 2.1 IDs, rows and folds

Both plans use only the 56 training IDs and their 224 rows (56 IDs x 4 colors) in `ember/research/latent_binding_training_fold_recovery_20260913/inputs.json`. Both plans use the four T5 folds from `folds.json` in the same folder, without change. Each ID is held out once.

| Fold | Fit IDs / rows | Held-out IDs / rows | Held-out IDs |
|---|---|---|---|
| 0 | 41 / 164 | 15 / 60 | A3 B6 C7 D4 E0 F7 G3 H2 I0 J7 L4 M2 N7 O5 P4 |
| 1 | 42 / 168 | 14 / 56 | A6 B0 D5 E2 F1 G6 H3 I3 J1 L5 M6 N1 O7 P5 |
| 2 | 42 / 168 | 14 / 56 | B2 C1 D7 E3 F4 G0 H6 I6 J4 K2 L7 N4 O1 P7 |
| 3 | 43 / 172 | 13 / 52 | B3 C5 D1 E6 F5 G2 H0 J5 K6 L1 N5 O4 P1 |
| Sum | 168 / 672 | 56 / 224 | |

I read these lists in this session from `folds.json` (`fit_rows`, `validation_rows`) and the row IDs of `inputs.json`. All 24 letter and digit labels occur in each fit set (parent plan, section 2).

### 2.2 IDs and splits that the plans do not touch

- **Burned 8-ID bank (A0, A2, C4, I2, K0, K3, M0, M3).** The `ember/AGENTS.md` rule forbids any trial on this bank. Neither plan loads a file that holds this bank: no `banks.novel` of `latent_binding_literal_id_swap_20260910/inputs.json`, no `latent_binding_identity_localization_20260910/stages.pt`, and no oracle `novel_fit.pt` or `familiar_fit.pt`. Neither plan has a descriptive novel-bank fit. I checked in this session that none of the eight IDs is in the 56 training IDs.
- **Cross-block IDs (64).** These are the E3 pool: 56 pristine novel-pool IDs and 8 DEV-OOB IDs (B7 C0 D3 E5 G4 J6 M1 O2), from `latent_anchor_program_20260930/DESIGN.md` section 10. Neither plan scores, fits or loads them.
- **FINAL splits.** FINAL-known also uses the 56 training IDs, but in other tables (seed 124001). Neither plan reads a file of `latent_anchor_program_20260930/` or `latent_anchor_E3_20261003/`. The only rows are the fold record's 224 rows.
- **Scope assert.** Each script asserts that every scored row, every fit target and every output name is one of the 56 training IDs. If the assert fails, the job stops before the first score.

### 2.3 Code, seed, threads and run protocol

- **Folders.** T6: `ember/research/latent_r06_t6_readout_<date>/`. T5b: `ember/research/latent_r06_t5b_wide_<date>/`. Here `<date>` is the date when [researcher] approves the run. The T5 folder stays read-only.
- **Reuse of T5 code.** Each new folder holds byte copies of `t5.py` and `seam.py`. A gate checks that their SHA-256 values are equal to the values in the T5 `FREEZE.json`. Thus both plans use the gate 3 version that T5 ran: amendment 1, plus amendment 2 if [researcher] approves it for T5. The `t5.py` of 02:30 has the same function names as the version that I read first; only `native_parity` reports two more descriptive values. The new script imports the copy of `t5.py` as a module. It does not call `t5.run`, `t5.smoke` or `t5.freeze`.
- **Receiver.** SmolLM2-360M-Instruct, revision a10cc1512eabd3dde888204e902eca88bddb4951, float32, eager attention, CPU only. Parameter hash 6a3e1782ebad58d1e9323d5eeaff191d112848b5f0058394a998d1ac54c492b3 before and after. No receiver parameter takes a gradient.
- **Seed and threads.** `Setup()` of `t5.py` calls `seed_all(60113)` of the channel module. This sets deterministic algorithms, 4 intra-op threads and 1 inter-op thread. Neither plan changes the thread count after this call.
- **Python.** `ember/.venv/bin/python -I -B`.
- **Protocol.** As in T5 PLAN.md section 4: synthetic tests, smoke test, freeze (`FREEZE.json` with the SHA-256 of the plan, the code and every input), then the jobs, one at a time.
- **Preflight and guard.** Copies of `preflight.sh` and `guard.sh` of the T5 folder. The preflight list also blocks on the text `latent_r06`, so that no other R.06 job runs at the same time. Run only on AC power at 60 W or more. Use `caffeinate -i -w <pid>` and `tee -a` for logs.
- **Failure.** If a job fails, report it as incomplete. Do not restart it. Do not select from partial folds without a new written decision.

## B. Source plan, section 4 (copied word for word)

## 4. T5b: factored code at full boundary width

Run T5b only if T5 supports H5 (section 1, rows 2 and 3).

### 4.1 Question and hypotheses

- **Question.** T5 fit a factored code in the 64-d value space. The frozen d0 output map sends this space to a 64-d subspace of the 960-d boundary state at block 15. Does a factored code bind held-out IDs if it can also use the other 896 boundary directions?
- **H5-path, frozen d0 output path.** The receiver reads a factored code at this seam, but not inside the 64-d column space of the d0 output map. Then T5b passes, and the frozen output path is the barrier. The E3 design agrees with this form: it trains its reader at block 15.
- **H5-receiver, frozen receiver at this seam.** The receiver at this seam (output of block 15, sites 3 and 4, one shared vector at both sites) has no factored layout that transfers, at any width. Then T5b fails like T5.

### 4.2 Design: what trains and what stays frozen

The parent plan says "repeat T5 with the output map also trainable". This plan uses an equivalent form with the T5 coordinates and learning rate. The reason follows.

- A trainable output map W (960 x 64) applied to a factored 64-d code gives W c0 + W a_L + W b_D. This is a factored code in the 960-d boundary space.
- A factored code has 25 free vectors (1 + 16 + 8). Because 25 <= 64, a trainable W can give any set of 25 boundary vectors. Thus the codes that a trainable W can reach are all factored 960-d boundary codes.
- T5b reaches the same set directly. It keeps the T5 code in the 64-d column space and adds a factored code in the 896-d orthogonal complement.
- The T5 coordinates are in boundary units. `coordinates.pt` holds the SVD of alpha x W_out, with alpha = gate x norm scale (`latent_binding_constrained_code_extended_20260911/runtime.py`, line 28). Thus one unit of `theta` moves the boundary state by one unit along one left singular vector. T5b uses the same unit for each complement direction. Thus the same learning rate applies to every direction, and nothing is tuned.

**Parameters (all float64, zero at the start).**

| Name | Shape | Space | Source |
|---|---|---|---|
| `theta_c`, `theta_a`, `theta_b` | 1 x 64, 16 x 64, 8 x 64 | 64-d column space of alpha x W_out (T5) | `t5.Factored` |
| `psi_c`, `psi_a`, `psi_b` | 1 x 896, 16 x 896, 8 x 896 | 896-d orthogonal complement | new |

Total: 1,600 + 22,400 = 24,000 trainable values.

**Frozen.** The start buffers `c0`, `a0`, `b0`; the output map W_out, gate, norm scale and the two query centers; the complement basis Q; the receiver. The receiver gets the same payload at sites 3 and 4, plus its frozen per-site center, as in the native path.

**Complement basis Q (960 x 896).** Compute it once, before the freeze, with a short script `complement.py`:

1. alpha = gate x norm scale of the d0 bridge (SHA-256 d0f16cb9705ba53daf6cc8f2bfad03a08efa96cbb4f77dd836c9533e948cba87).
2. U_full = left singular vectors of alpha x W_out (float64), from `torch.linalg.svd(..., full_matrices=True)`.
3. Q = columns 64 to 959 of U_full.
4. Save Q in `complement.pt`. Record its SHA-256 in `FREEZE.json`.

Adam is not invariant to a rotation, so the basis of the complement changes the optimizer path. The saved Q fixes the basis for all T5b and C1b folds.

**Code and render.**

- `Wide(Factored)`: a subclass of `t5.Factored` with the `psi` parameters and the buffer Q. `Wide.values(L, D)` returns a pair: the T5 value `Factored.values(L, D)` (64 values) and the offset (psi_c + psi_a[L] + psi_b[D]) times Q transposed (960 values).
- `render_wide(pair)` = `make_render(state)(value)` + float32(offset), with the same offset at sites 3 and 4.
- With psi = 0, the offset is exact zero, so `render_wide` gives the T5 render bit for bit.

**Steps (for each fold k, then C1b).**

1. Start: `c0`, `a0`, `b0` from `additive_init` of the fit-ID d0 codes, as in T5.
2. Fit: `t5.fit` without change. It calls `render(model.values(...))`, so `render_wide` and `Wide.values` work in it. Adam over theta and psi: learning rate 0.19364916731037085 (from `coordinates.pt`), betas (0.9, 0.999), eps 1e-8, no weight decay, no gradient clip, no early stop, no checkpoint selection. 200 steps. One step for each full pass over the fit rows, batches of 8, loss = sum of token cross-entropy / (2 x fit rows).
3. Score the arms of section 4.3 with `Setup.score` and `t5.counts`.
4. Save `fold{k}_code.pt` (theta, psi, c0, a0, b0) and `fold{k}.json` (with the fit history).

**Seed and threads.** `seed_all(60113)`, 4 intra-op threads, 1 inter-op thread. Random arm C3b: `torch.Generator().manual_seed(60113 + k)`.

**Reused functions.** From `t5.py`: `Setup` (`__init__`, `gates`, `native_parity`, `score`, `LD`, `tokens_of`, `close`), `Factored` (base class), `additive_init`, `make_render`, `fit`, `counts`, `shift_map`, `parts`, `sha`, `write_new`, `event`, `guard_files` (as a model for the T5b version). New: `Wide`, `render_wide`, `fold_fit_wide` (a copy of `Setup.fold_fit` with `Wide`), `run_wide` (a copy of `t5.run` with the arms below), `complement.py`.

### 4.3 Arms and controls

For each fold k:

- **Matched (T5b).** Held-out rows with their T5b code. Gives Hb_k (joint) and the sequential count.
- **Fit rows (T5b).** Fit rows with their T5b code. Gives Fb_k.
- **C2b, wrong parts.** For each held-out row, the code (value and offset) of the next held-out ID of fold k in sorted order (cyclic). Report the intended and inserted joint counts.
- **C3b, random.** For each held-out row, a Gaussian 960-d vector, scaled to the norm of (fitted payload minus center), added to the center at both sites. This differs from T5 C3, because the T5b code is a 960-d boundary code.
- **C4b, zero code.** Value 0 and offset 0, so the payload is the center. **Receiver only:** no payload.
- **C1b, shifted-label fit (minimum control; separate job).** The same fit, but the target of each fit row is the next training ID in sorted order of the 56 IDs (`shift_map`, cyclic). The code of each row is still built from the parts of its true ID. The start is the same as in T5b. Score the held-out rows with their C1b codes against the true targets. This gives Nb_k.

Hb, Fb and Nb are the sums over the four folds. No descriptive novel-bank fit runs (section 2.2).

### 4.4 Gates (before any fit)

Stop and report at the first failure.

1. **Branch.** `RESULT_t5.json` and `RESULT_c1.json` exist and have `completed: true`. The T5 rule on their totals gives "H5 supported" (F >= 605 and H <= 56) or "H5 supported, strong form" (F < 605).
2. **Code copies.** `t5.py` and `seam.py` of the T5b folder are equal (SHA-256) to the values in the T5 `FREEZE.json`.
3. **T5 mechanical gates 1 to 4.** `Setup.gates()` of the frozen T5 code, with the gate 3 version that T5 ran (section 2.3). Gate 5 runs at the end of each job.
4. **Coordinates.** `u` x diag(`s`) x `vh` of `coordinates.pt` equals alpha x W_out of d0 within 1e-9 (maximum absolute error).
5. **Complement.** Q transposed times Q equals the identity within 1e-12. `u` transposed times Q is zero within 1e-12. The SHA-256 of `complement.pt` equals the freeze value.
6. **Start identity.** For the 224 rows, `render_wide` of the fold-0 start code equals the T5 render of the same code (`torch.equal`). Call `t5.fit` on the fold-0 fit rows with 1 step and learning rate 0, so that no parameter changes. Its `history[0]` loss and teacher-forced counts must equal `history[0]` of the T5 `run/t5/fold0.json` (exact).
7. **Scope.** The assert of section 2.2.

Synthetic tests (`test_t5b.py`, before the smoke test):

- The offset lies in the complement.
- With psi = 0, `Wide.values` gives the `Factored` value and a zero offset.
- The values are additive in letter and digit (as `test_values_are_additive`).
- `t5.fit` moves psi.
- `render_wide` with a zero offset equals `make_render` exactly.

### 4.5 Reading rule

Counts: Hb is the T5b held-out joint count (of 224). Fb is the T5b fit-row joint count, summed over the folds (of 672; 90% is 605). Nb is the C1b held-out joint count (of 224).

1. If Fb >= 605, Hb >= 168 (75%) and Nb <= 22 (10%): T5b passes. A factored code binds held-out IDs at this seam when it can use the full boundary width. **The frozen d0 output path is the barrier (H5-path), not the receiver.**
2. If Fb >= 605 and Hb <= 56 (25%): factored codes fit the training IDs at full width, but they do not transfer. **The frozen receiver at this seam is the barrier (H5-receiver).**
3. If Fb < 605: a factored code cannot fit the training IDs at this seam, even at full width. **H5-receiver is supported in its strong form.**
4. Otherwise: inconclusive. Report the counts. Do not move a threshold.

The bars are the T5 bars. The 75% bar is the joint bar of E3.

**Descriptive only (no rule).** Hb - H and Fb - F. For each held-out ID, the share of the boundary code norm in the complement: norm of the offset divided by the norm of (payload minus center). Report the median.

### 4.6 Cost

| Part | Basis | CPU time |
|---|---|---:|
| `complement.py` (one SVD of 960 x 64) | | seconds |
| Smoke test: setup, gates, start identity, 3 Adam steps (fold 0), no code scored | 3.9 s setup; 38.4 s parity scores; 3 steps of about 3.7 h / 800 steps | about 3 min |
| T5b job, four folds of 200 steps | parent estimate for one T5 arm | about 3.7 h |
| T5b scores: 224 rows x 5 arms plus 672 fit rows = 1,792 rows | 0.64 s per row | 0.32 h |
| C1b job, four folds of 200 steps | parent estimate | about 3.7 h |
| C1b scores: 224 rows | 0.64 s per row | 0.04 h |
| **Total** | | **about 7.8 h** |

The extra 22,400 parameters add one 896 x 960 product for each batch. This is small against the receiver backward pass. The fit estimate is not a T5 measurement, because no T5 fit step had run when this plan was written. After T5 runs, the `seconds` values in the T5 fit history give the real step time. The T5b smoke test must measure the step time again and record it as a dated addition before the freeze.

### 4.7 Limits

- One seed, one learning rate and one complement basis.
- T5b keeps one shared vector at both sites, as the native path does. A per-site code is not tested (section 6, choice 4).
- The gate, the norm scale and the centers stay frozen.
- A pass shows that a factored code exists outside the d0 column space. It does not show that a trained bridge finds it.
- The fold IDs were in the training of the ancestral d0. All held-out IDs are inside a block. T5b tests in-block transfer only.

## C. Implementation notes (fixed before any data)

These notes fill in what sections A and B leave open. No rule or threshold of sections A and B changes.

1. **Preflight text match.** Section A asks the preflight to block on the text `latent_r06`. The shell wrapper of the launching command also holds this text (its folder path). Thus the preflight blocks on `latent_r06` only in Python processes (`python.*latent_r06`). It also blocks on `pilot.py`, `followup.py`, `uconfirm.py`, `phase0.py`, `harness.py`, `smcheck.py` and `latent_capacity`, as the T5 preflight does.
2. **Gate 3 version.** The byte copy of `t5.py` holds the T5 gates with amendments 1, 2 and 3 of the T5 PLAN.md. Thus gate 3 here is the same-machine native parity with the template sub-check descriptive, and gate 4 of T5 requires the same argmax, with the difference descriptive.
3. **Gate 6 data.** Gate 6 reads `run/t5/fold0.json` of the T5 folder (fit history only). It reads no held-out score.
4. **C3b norm.** "Fitted payload minus center" is the 960-d boundary displacement of the row's fitted T5b code; it is the same at both sites.
5. **Descriptive share.** For each held-out ID: the norm of float32(offset) divided by the norm of (payload minus center), at site 3. Report the median over the 56 held-out IDs.
6. **Files.** `t5b.py` (modes `smoke`, `freeze`, `run --arm t5b|c1b`), `complement.py`, `test_t5b.py`, `preflight.sh`, `guard.sh`, `launch.sh`, and the byte copies `t5.py` and `seam.py`.
7. **Cost.** The cost table of section B used an older estimate. The T5 jobs took 5,591 s (T5) and 4,792 s (C1) on this machine (`../latent_r06_t5_factored_oracle_20261020/REPORT.md`). The smoke test measures the T5b step time again; section D records it before the freeze.

## D. Dated additions

**2026-10-22, 14:56Z (UTC). Complement and smoke test PASSED; this is the smoke time required by section B (4.6) and section C (7).** `complement.py` wrote `complement.pt` (SHA-256 aac64dc8735dd62a8d81e4519fe7e8a00a7b2980cf2532c08de9b678d3576ed2): Q^T Q minus identity 3.3e-15, u^T Q 9.4e-14 (gate 5 limits 1e-12). Smoke attempt 1 ran after the preflight passed (AC 140 W, no blocking process). All gates passed: g1 branch (F 672, H 32, N 15: "H5 supported"); g2 code copies equal to the T5 freeze; g3 T5 mechanical gates (native against seam 0.0; direct-score difference 1.354e-4 descriptive, same argmax); g4 coordinates 6.7e-14; g5 complement; g6 start identity (render equal on 224 rows; with learning rate 0, loss 7.251722738146782 and teacher-forced counts [36, 24] equal the T5 fold-0 history exactly; no parameter changed); g7 scope (56 training IDs, no burned ID). Fit: 3 Adam steps on the 164 fold-0 fit rows took 4.94, 4.96 and 4.99 s. Scoring: 0.306 s per row for one scoring. No T5b or C1b code was scored. Output: `SMOKE.json`.

Estimate from this time: the T5b job takes about as long as the T5 job (5,591 s, about 1.6 h), and C1b about as long as C1 (4,792 s, about 1.3 h). No rule, threshold or step count changes after this addition.
