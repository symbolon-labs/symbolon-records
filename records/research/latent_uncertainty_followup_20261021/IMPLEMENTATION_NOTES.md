# Implementation notes for the uncertainty follow-up

Status: EXPLORATORY. Written 2026-10-20, 23:20, before the smoke test and before any follow-up data.
These notes record the choices that PLAN.md does not fix. PLAN.md does not change.
The code is `followup.py`. It starts as a copy of `../latent_uncertainty_pilot_20261020/pilot.py`.
The run wrapper is `run_followup.sh`. The pilot folder does not change.

## 1. What is the same as the pilot code

- Data generation, card text, sender prompts, sender batch 16, the block-21 and block-28 hooks,
  `round(9 x p) = floor(9 x p + 0.5)`.
- The T3 and T4 probes: L2 logistic fits on the standardized block-21 state, with cross-fit train
  messages and the L2 grid {1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10} (the grid after the pilot smoke test).
- The receiver input "Message: <message>\nWhat color is the box? Answer:", no chat template, no special
  tokens, the final norm output at the last token, and the zero-shot view.
- The readouts of the text arms and Card, the metrics, the bootstrap (10,000 draws) and the margin
  (0.02 nats).
- The L channel: the map init (bias = mean token embedding, weight std = embedding std / sqrt(2048)),
  AdamW (weight decay 0.01), batch 32, gradient norm clip 1.0, at most 40 epochs, patience 6, an inner
  validation part of 20%, and a final fit on the full train split with the selected epoch count.

## 2. What is different in the code

- Seeds: see PLAN.md section 3.1. The other seed is 20261021 in all places where the pilot used 20261020.
  The shuffle permutations use 20261022 (train) and 20261023 (test), as the pilot used SEED + 1 and
  SEED + 2.
- T5 uses the function of the T4 message, with the sender softmax in place of the probe output.
  The T5 digit of the answer color is the T2 digit, because both are round(9 x p) of the same value.
- T6 uses `numpy.random.default_rng([2026102103, split])`, with split 0 for train and 1 for test, and
  `rng.choice(4, size=8, p=p)` for each item in item order. The samples are in `t6_samples.npz`.
- The low-rank map is `Sequential(Linear(2048, r, bias=False), Linear(r, 960))`.
- Each latent configuration starts from `torch.manual_seed(20261021)`, as each pilot run started from
  its seed. Thus all L configurations start from the same map, and all L-lowrank configurations of one
  rank start from the same map.
- Stacked training: one generator (seed 20261021) gives the batch order of each epoch. In each step,
  the soft inputs of all active configurations go through the receiver in one pass (at most 13
  configurations in one pass). Each configuration has its own loss, its own gradient clip and its own
  AdamW step. A configuration stops after 6 epochs with no new best inner-validation NLL, or at 40
  epochs. The validation pass also uses the stack (64 items per configuration in one pass).
- Final fits: L, L-lowrank, L-shuf and L-lowrank-shuf train in one stack on the full train split.
  Each stops at its own selected epoch count. All four use the same batch order.
- The test passes: L and L-lowrank on the test states; L-testshuf and L-lowrank-testshuf on the permuted
  test states; L-shuf and L-lowrank-shuf on the permuted test states (as in the pilot).

## 3. Smoke test

- 64 train items and 32 test items, smoke seeds, at most 8 epochs, patience 3 (as in the pilot smoke test).
- `--check_stack`: two configurations (L at 1e-5, and L-lowrank r = 16 at 1e-4) train alone again on the
  same inner split. Their validation curves must agree with the stacked curves to 1e-3 nats.
- `--bench`: seconds per stacked train step for 1, 4, 8 and 13 configurations, for the time estimate.

## 4. Outputs

- `smoke/` and `run/`: items, sender outputs, probes, receiver states, `t6_samples.npz`, the latent
  grid and curves (`latent.json`), per-item test probabilities of every arm (`test_probs.npz`),
  `results.json` and `table.md`.
- `RESULTS.json` (a copy of `run/results.json`) and `REPORT.md` in this folder.
- Logs: `smoke.log`, `smoke_wrapper.log`, `run.log`, `run_wrapper.log`, all with `tee -a`.

## 5. Smoke test facts and the change before the run (2026-10-20, 23:15 to 23:19)

- Smoke test 1: 23:09:29 to 23:14:56, rc 0. All stages and all arms ran. The embedding-route identity
  check gave 0.0. The smoke verdict was "invalid" (Card NLL above T0 NLL with 64 train items). This is
  expected at this size, and it tests only the code.
- Stack check: the stacked and the separate validation curves agree to 1.4e-6 nats (L, 1e-5) and
  6.7e-7 nats (L-lowrank r = 16, 1e-4). The limit was 1e-3.
- Bench: 0.474 s for each step with 1 configuration; 1.670 s with 4; 3.384 s with 8; 5.184 s with 13.
  MPS memory 2.6 GiB to 13.5 GiB. Thus the stack gives only about 15% speed gain on this laptop.
- Because of this time fact, the PLAN.md addition of 23:15 changes the grids, the patience (3) and the
  epoch limit (20). The code defaults are now `--max_epochs 20 --patience 3`, `FULL_LRS = [1e-5, 3e-5, 1e-4]`
  and `LOW_LRS = [3e-4, 1e-3, 3e-3]`. The grid results are also written to `latent_grid.json` before
  the final fits.
- Smoke test 2 (latent and analysis stages only, on the cached smoke outputs of test 1): the first try
  at 23:15:57 stopped at the precondition check, because the R.06 job `t5.py smoke` was active. The
  retry ran from 23:16:38 to 23:19:10, rc 0. Stack check: 1.4e-6 and 6.1e-6 nats. The outputs of smoke
  test 1 for these stages are in `smoke/v1_*`.
- Full run: started 23:19:21 (pid 14413), with a time limit of 100 min. No other GPU or CPU job was
  active. Swap 0 MB. Charger 140 W.
