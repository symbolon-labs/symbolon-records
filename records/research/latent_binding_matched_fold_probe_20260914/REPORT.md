# Matched training IDs preserve a substantial probe/native decoding gap

2026-09-14. All 24 registered offline probe conditions completed and qualified. No sender/receiver loaded, no model/bridge training, no new native answers and no novel data used. Exact fit-ID membership is now matched to the native four folds; this removes the earlier 55-other-ID LOO versus 41/42/42/43-ID fold supervision mismatch.

| Frozen representation | Heldout letters /56 | Heldout digits /56 | Whole IDs /56 | Persistent native failures decoded /45 |
|---|---:|---:|---:|---:|
| Two isolated sender states (4096 dimensions) | 56 | 56 | 56 | 45 |
| Original compressed slot (64 dimensions) | 44 | 47 | 38 | 27 |
| Original projected value (960 dimensions) | 43 | 35 | 24 | 16 |

Each ID is omitted entirely from its probe fit. The native comparator remains registered step100: 22 joint /25 sequential correct context rows of224, covering6/7 IDs in at least one context and4/4 IDs in every context. Probe predictions are one per ID, not224 independent trials. The45 persistent IDs have zero native success in the original four checkpoints/four contexts, but27 remain whole-ID decodable from the compressed slot under this fixed matched-fold rule. Thus the previously observed gap is not explained entirely by the probe having more training IDs.

This does **not** establish receiver-only causality. The probe uses isolated ID-field features; native scores involve context, a different objective/readout, and the fold-trained mapping at100. In particular, the projected probe here uses the **original frozen projected features**, not each fold's step100 projected values. These three probe conditions therefore do not trace one common step100 native forward pass through every stage. Full source recovery is evidence for this measured56-ID development bank, not all possible IDs or a new novel benchmark.

## Persistent cohorts and fold dependence

Whole-ID recovery counts:

| Cohort | IDs | Sender | Slot | Projected |
|---|---:|---:|---:|---:|
| Neither native component ever right | 28 | 28 | 12 | 10 |
| Only native prefix ever right | 11 | 11 | 9 | 4 |
| Only native conditional digit ever right | 6 | 6 | 6 | 2 |
| Both components right but never together | 0 | 0 | 0 | 0 |
| Ever native sequentially correct | 11 | 11 | 11 | 8 |

Whole-ID heldout counts by original fold:

| Fold | Heldout IDs | Sender | Slot | Projected |
|---|---:|---:|---:|---:|
| 0 | 15 | 15 | 13 | 6 |
| 1 | 14 | 14 | 7 | 8 |
| 2 | 14 | 14 | 10 | 4 |
| 3 | 13 | 13 | 8 | 6 |

All18 slot errors belong to the persistent cohort. Source recovery is perfect in every fold. The slot-to-projected reduction is an aggregate result, not a monotone decline on every ID/fold; fold1 projected recovery exceeds slot recovery. All individual predictions, errors and margins are retained in RESULT.json and independently checked in INDEPENDENT_AUDIT.json.

Earlier LOO slot recovery was44/56 and33/45 persistent; matched-fold recovery is38/56 and27/45. The sender two-state score stays56/56. Matching supervision therefore reduces the apparent slot decodability, while retaining a substantial gap. This is a comparison with recorded historical results, not a new LOO fit.

## Controls and interpretation limits

The registered within-fit one-position label shift yields zero heldout correct digits and zero whole IDs for all three representations. Heldout letters remain46/56 sender,36/56 slot and32/56 projected. This is a **structured shifted-label control, not a random-label chance baseline**: fit IDs are ordered, so104/168 overlapping fit occurrences retain their original prefix label, while none retains its digit or whole label. Do not use the strong shifted prefix score to claim unrestricted decoding or calculate a permutation p-value. The specified control was retained unchanged; no replacement null or threshold was selected after seeing results.

True-label fit recovery is168/168 whole-ID occurrences for sender and slot,163/168 projected. These are overlapping fit occurrences across folds, not168 distinct IDs. Assigned shifted-label fit recovery is168/168 sender and slot,148/168 projected. Projected fit/heldout degradation under the fixed ridge rule is consistent with an effect of feature geometry and regularization, but does not by itself identify the cause. Isotropic ridge is not invariant to general linear feature transformations. A worse projected probe therefore cannot establish information destruction at projection. Neither can slot probe failures establish total erasure after sender compression.

The strongest actionable conclusion is twofold: matched-fold source component identity is recoverable for every measured ID; a useful subset of persistent native failures is also recoverable after compression. There is still no evidence here that a particular receiver training intervention will recover them without sacrificing others.

## Execution and validation

Freeze37916541796ed52af67c29cb793a792cece3335c0d769f842eb4adb2358f38f0 pins code/plan, all consumed sealed source hashes, exact folds and versions. Seven qualification tests passed before the run: primal/dual agreement, independent Torch solution, heldout exclusion/fit-only centering, fixed shift, counts/margins, tie policy, and subprocess lifecycle checks (grouped into seven unittest cases).

One supervised job, session87675, exit0; no timeout/interruption, child reaped2026-09-14T00:33:51Z. Elapsed1.433918771s, peak709824KiB, four CPU threads, USD0 external. Wrapper restoration_verified=false is inapplicable here: no model was loaded or modified. There are no remaining jobs from this diagnostic.

All24 NumPy/Torch independent solutions agree within3.1429303604113557e-12 absolute score error, below the unchanged1e-8 gate, with exact component argmax agreement. No heldout component top-two gap is within1e-8. Saved-parameter forwards replay exactly; normal-equation residuals pass without refitting. An independent script recomputes predictions, gold margins, fit/heldout/cohort counts and native ID/context joins. Original source capture repeats and source/recovery slot/projected arrays are bit-identical. Saved native step100 score tensors repeat exactly and reproduce all224 comparison rows. Consumed source hashes are checked before and after execution.

## Next decision boundary

Do not spend model-training time on a receiver-only fix based on this result. The next useful bounded proposal is a **saved-array projection/readout transport audit**: test whether the known frozen projection can represent the already-fitted slot probe scores, distinguishing information availability from this ridge rule's geometry. Specify the algebra, tolerances, exact saved folds/parameters and failure criteria before execution; no probe refit, native run, or metric sweep. A separate comparison using fold-trained projected values would address the step100 stage mismatch, but would require a separately registered diagnostic and must not be conflated with this completed test.

No automatic follow-on experiment is launched. Best822d native novel4/4/4 remains NO-GO; originald0novel0/0/0, selected native fourfold failure, and D5 directional FAIL are unchanged. The broader scientific diagnosis remains unresolved. All predecessor sealed artifacts stay read-only.

After closure, safe commands are Python -B runtime.py verify/check and audit_saved.py verify with OMP/MKL/OPENBLAS_NUM_THREADS=4. They do not fit. Never run/freeze/save or edit this sealed tree. Diagnostic weights are analysis artifacts only, never an inference codebook or candidate.
