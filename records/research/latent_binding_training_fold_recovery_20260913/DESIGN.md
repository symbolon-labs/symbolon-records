# Recovery design: identical four-fold validation, separate execution

Status: DESIGN ONLY. No successor code frozen or model job launched.

The matched predecessor session 22258 terminated with code 143 before fold index 1 step 100, without completion/status records. Its unchanged 69-file inventory and structural checkpoint audit are recorded in ../latent_binding_fold_interruption_20260913/. Read-only host namespace inspection found no surviving research worker. Cause unknown; no efficacy selection from partial folds.

## Scientific contract

Retain predecessor prepared inputs byte-for-byte, folds 41/15,42/14,42/14,43/13, fit-only conditioning, fresh d0/theta0/Adam per fold, seed94117, original learning rate and full-gradient batching, 200 steps per fold, grid0/50/100/200, matched first8/batch8 direct parity <=1e-4 with same argmax. Retain historical familiar controls, all native/export/gradient/route checks, per-fold9000s and whole37800s caps. No novel data, all56refit, tuning, threshold or capacity change.

Use a fresh complete replay, not loading partial optimizer state into the old job. Every available matched-parent grid candidate, first/repeated score array, direct diagnostic and theta/Adam/history must compare exactly. Keep the older original-parent and sealed failedpoint diagnostic replay requirements too. Any discrepancy stops mechanically. Replaying the same deterministic trajectory does not create independent efficacy evidence.

Select only after all four folds and the complete grid: maximize aggregate heldout sequential count over positive steps, ties earlier; chosen step must strictly improve both joint and sequential over step0. No alternative step if chosen step fails joint. No selection from intermediate outputs.

## Execution correction to qualify before launch

New wrapper should handle SIGTERM/SIGINT by forwarding termination to its exact child, waiting/reaping, and writing an explicit interrupted completion record with exit status, signal, timeout=false, elapsed time and restoration status unknown unless runtime saved proof. Do not infer restoration from process death. Preserve ordinary timeout semantics and never retry automatically. A synthetic sleeping child must demonstrate signal forwarding, reaping, and a terminal record; normal exit and timeout paths must also be tested. Signal handling cannot guarantee a record if the entire namespace is killed or SIGKILL is used; disclose this limit.

Pin the full interrupted-parent inventory, retain existing model/package/source seals, and test exclusion, folds, loss weighting, export, selection, exact replay and wrapper lifecycle before freeze. Run a fresh frozen identity check before launch. One receiver-only worker; no overlap or automatic restart.

After valid terminal completion, independent scalar/NumPy counts and common-step selection audit, all saved contracts and parent replays, fresh identity/restoration checks, report and seal. An incomplete successor must get a separate incident record, never a successful-close seal. Only a qualifying complete result permits designing the separately frozen all56 refit, retaining familiar gates before any fixed diagnostic novel evaluation.
