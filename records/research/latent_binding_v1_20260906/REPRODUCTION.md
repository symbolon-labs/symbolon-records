# Artifact map, verification, and frozen boundaries

This completed attempt is **MECHANICS/TRAINING NO-GO**: mechanical checks pass,
but the fixed bridge never reaches the 32-example overfit criterion. No
TRAIN/DEV/final data were created. The prior cross-family two-bit PASS is a
different sealed experiment and remains unchanged.

## Read-only checks

From `.`:

```bash
python3 research/latent_binding_v1_20260906/seal.py verify
python3 research/latent_binding_v1_20260906/audit.py
python3 research/latent_communication_dossier_20260906/verify.py --models
```

The first verifies this tree's complete payload inventory. The second derives
answers, predictions, accuracies, NLL, donor joins, zero/donor parity, terminal
reproduction, checkpoint selection and the failed gate from saved artifacts.
The third checks the sealed historical chain and model snapshots without
inference or downloads. A verification PASS means integrity, not binding PASS.

Do not rerun `job.py`, `run.py`, `final_checks.py`, `plot.py`, `seal.py create`,
or diagnostic `--save`/CSV options inside this completed tree. They write
artifacts and deliberately fail closed on existing destinations. A new
training attempt requires a separate directory and explicitly justified,
preregistered change; no automatic extension, prompt search, anchoring,
curriculum, or architecture retry is licensed by this report.

## Scientific authority and raw evidence

| Artifact | Role |
|---|---|
| `PREREGISTRATION.md`, `TEXT_PROTOCOL.md` | Pre-outcome task, controls, scoring, splits, optimizer, gates |
| `acquisition.json`, `initial_code/` | Initial six-file hashes, exact endpoint/tokenizer identities and code snapshot |
| `identity_pools.json` | Component-balanced known/novel whole-ID boundary |
| `data/tiny/examples.json` | Eight source/counterfactual tables and 32 query/target rows |
| `data/tiny/sender.pt`, `manifest.json` | Real source states/token IDs, source order, hashes and frozen sender parameter identity |
| `text_tiny/sender.json` | Actual sender prompts, tokens and all 128 two-token answer scores |
| `text_tiny/receiver.json`, `summary.json` | Actual/oracle text joins, repeated receiver scores, copy qualification |
| `run_tiny/mechanics.json`, `manifest.json` | Zero/disabled/stock parity, scoring seam, gradients, frozen receiver and training configuration |
| `run_tiny/curve.jsonl` | Every optimizer step, loss, gradient norm, gate and elapsed time |
| `run_tiny/eval_*/` | All 16 evaluations: metrics, per-example scores, actual control memories and diagnostics |
| `run_tiny/terminal_repeat/` | Actual seven-arm rerun, not reconstructed metrics |
| `run_tiny/best.pt`, `last.pt` | Step-240 and step-300 bridge states; neither licenses later stages |
| `run_tiny/decision.json`, `gates.json` | Frozen-rule NO-GO, exact rerun, endpoint/checkpoint identities |
| `tiny_job/state.json`, stage logs | Exact argv, start times, exit status, elapsed time and RSS high-water mark |

## Post-result analysis, distinct from preregistration

`audit.py`, `test_audit.py`, `diagnose.py`, `plot.py`, `final_checks.py` and
`seal.py` were added for reporting/verification while or after the frozen run
executed. They do not change its bridge, optimizer, scoring, or datasets.
`audit.json` and `final_checks.json` record independently rederived results;
`pretrain_tests.xml` and `final_tests.xml` preserve test outcomes. Diagnostics
for step 240 and terminal step 300 are explicitly descriptive, not new gates.
`training_curve.csv`, `control_curve.csv`, and `training_curves.svg` expose the
whole training trajectory. `REPORT.md` is the completed interpretation;
`SESSION_HANDOFF.md` records completion and superseded operational history.

The seal excludes only itself and disposable Python/pytest caches. Root
`LATENT_COMMUNICATION.md` and `AGENTS.md` are mutable navigation outside the
research seal. No artifact was pushed or uploaded; local integrity sealing is
not a remote backup. Preserve all failed predecessors: the exact bridge class
imports sealed earlier implementations whose hashes are part of the evidence.
