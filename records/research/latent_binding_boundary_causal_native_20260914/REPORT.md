# Removing this probe-readable subspace does not yield the expected native necessity result

Controlled native intervention completed with mechanical audit PASS. No fitting, optimizer, readout replacement, new candidate or novel evaluation. Frozen step100 native checkpoints; all224 heldout context rows in four original folds. Treatment removes the centered column space of the already-existing boundary readout; control perturbs one fixed orthogonal subspace by the same boundary displacement norm. Both reader sites change together.

**Primary result:** among27 persistent native failures that were decodable at the boundary, neither treatment nor control changes any prefix/digit correctness outcome. Treatment-minus-control mean gold log probability is **+0.0429181 nats per component**: prefix+0.0988717, gold-prefix-conditioned digit-0.0130356. Ten IDs have a negative mean contrast,17 positive. This is not the expected aggregate reduction in gold scoring from removing those directions. It is a descriptive finite intervention contrast, not a significance test or proof the native receiver ignores identity.

Relative to its own untouched baseline, treatment changes primary-cohort prefix logp by+0.0789995 and digit logp by-0.0101057; control changes them by-0.0198722 and+0.0029299. Positive treatment-minus-control therefore is not merely a reporting convention; average prefix gold scoring improves after this erasure, while the digit effect is slightly negative. No token correctness gain follows on this cohort.

## Native outcomes and preservation

These are full-vocabulary prefix predictions and **gold-prefix-conditioned** digit predictions; “both” is their teacher-forced overlap, not sequential whole-ID generation accuracy. All counts below are context rows, not independent IDs.

| Cohort / arm | Context rows | Prefix correct | Conditional digit correct | Both teacher-forced correct |
|---|---:|---:|---:|---:|
| Primary27IDs baseline |108|20|12|0|
| Primary27IDs treatment |108|20|12|0|
| Primary27IDs control |108|20|12|0|
| All56IDs baseline |224|56|41|22|
| All56IDs treatment |224|52|42|23|
| All56IDs control |224|56|40|22|

The treatment loses J5's correct prefix in all four colors. Both treatment and control gain D4's conditional digit in yellow. The control additionally loses O5's conditional digit in blue and B3's in blue. No other component correctness transitions occur. These small mixed changes are not a repair result or justification for deploying an erasure adapter.

Correctness stability is not output invariance. Within the primary cohort, treatment changes4prefix and2digit argmax predictions, while control changes5digit predictions; these are wrong-to-wrong changes. Across all rows treatment changes12prefix/7digit predictions; control0prefix/12digit. Native probabilities also move. The intervention therefore has causal finite effects, but does not provide a clean aggregate necessity result for the probe-readable directions on the preselected27IDs.

All56 equal-ID-weight mean treatment-minus-control gold component contrast is-0.00429391 nats (prefix-0.0135977, digit+0.00500991);27 IDs negative and29 positive. Every ID/context/arm prediction, gold logp and rival margin is retained in RESULT.json. Cohorts include the other18 persistent failures, the11 ever-successful IDs, and all original persistent categories; none was selected after seeing this native result.

## Why this is an interpretable test, and what it does not test

Saved-data prerequisites passed before model loading across448 color/site copies. Treatment leaves at most0.00884449 of the original centered probe-score norm (at least99.1155% erasure), control changes it by at most0.00524862 (0.5249%), and actual float32 displacement norms match within2.35345e-6 relative. Perturbation magnitude is at most0.519945% of boundary norm. These thresholds were fixed before the qualification run and were not adjusted.

The initial preflight wrongly assumed centered readout rank22 and stopped before saving interventions or loading a model. The saved label design has uncentered rank22 and centered rank21. A separate successor corrected this schema assumption using independent NumPy/Torch label-rank tests; its21-dimensional treatment/control and all other gates qualified. Both records are preserved. This is not a numerical-tolerance waiver. The older failed exact-score transport qualification also remains FAIL: here the transported matrix is simply an already-saved linear function whose actual score erasure/preservation was measured directly.

This intervention removes the signal available to **this particular fixed probe**, not every possible encoding of ID. A nonlinear receiver can use redundant information outside that subspace, use its directions differently, or be insensitive on already-wrong argmax decisions. Wrong-to-wrong/probability effects rule out saying “the receiver is completely unaffected.” One fixed orthogonal control and one finite erasure do not identify a unique causal pathway or demonstrate robust generalization across possible interventions. The test does not show that training the receiver to follow the probe would succeed.

## Execution and verification

Native freezece7348d69da657b4440e072d8a561458c9ca58f8aeecccd13dfc1e35bb0e294f. Qualified intervention parent seal12ec563e6b8ab7cf9afaac61a960f290bae333844a47ec8dd7efb265276158cc; failed rank-assumption predecessor seal505d456ec41be21a1e4b09a92406c990e31ec05c9f7db8679d78ee8d604f075f. All predecessor sources remain read-only.

Four native harness tests and fresh frozen model/source identity passed before launch. Receiver session15213 exited0, no timeout/interruption, child reaped2026-09-14T01:19:58Z. Elapsed88.930709886s, peak2950544KiB,CPU4intra/1interop,USD0external. No jobs remain. Wrapper restoration_verified=false is only the generic wrapper field; run/status.json, saved stock/restored outputs and independent analysis verify restoration_exact=true, equal model parameter hashes, no hooks/gradients and all four folds completed.

Every native baseline and repeat exactly matches the prior sealed full-vocabulary logits and native routing/memory/payload/output captures. Seam baselines and zero shams are exact. Both intervention repeats are exact. Stock/disabled/zero/restored outputs and hashes are exact. NumPy/Torch independent float64 gold-logp calculations agree within1.40166e-14; fresh saved-logit outcome replay PASS. Raw logits for all arms/repeats and row identities are saved. No threshold change or post-hoc arm selection.

## Decision and next evidence needed

Do not interpret boundary decodability as proof that the native decoder relies on this probe's directions, and do not train a repair on that assumption. This qualified causal perturbation gives mixed effects rather than the expected aggregate necessity result on the27persistent cohort. The earlier fact that27persistent IDs are readable at the receiver input remains valid, as do the underlying native failures.

A next protocol must distinguish redundant residual ID encoding from a mismatch between the probe and native decoder. That would require a separately registered residual-information/readout test or a more specific causal hypothesis, with supervision and preservation controls; do not automatically fit another probe or begin receiver training. No new native experiment follows this completed test. Best822d native novel4/4/4 remains NO-GO; originald0novel0/0/0 and originalD5directionalFAIL unchanged. No novel/FINAL/Arabic/reset/cloud/commit/push.

After sealing, read-only commands: Python-B analyze.py verify and runtime.py check with OMP/MKL/OPENBLAS_NUM_THREADS=4. Never run/freeze/save/edit the sealed tree.
