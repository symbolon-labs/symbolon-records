# No-fitting identity localization

Saved d0f16cb9 arrays only; native novel-ID verdict remains0/0/0 NO-GO. No endpoint execution, probe fitting, weight change or usage reset.

| Stage | Min relative separation | Within-ID L2 spread | Privileged Euclidean decode | Familiar NN/native agreement (Euclidean / cosine) |
|---|---:|---:|---:|---:|
| sender_field | 0.00559841 | 0 | 48/48 | 16/64 / 0/64 |
| wire64 | 0.0266101 | 0 | 48/48 | 40/64 / 40/64 |
| projected_memory | 0.0129795 | 0 | 48/48 | 40/64 / 40/64 |
| reader_value | 0.00501627 | 0 | 48/48 | 40/64 / 40/64 |
| selected_3 | 0.00501627 | 0 | 48/48 | 40/64 / 40/64 |
| output_map_3 | 0.0039718 | 0 | 48/48 | 40/64 / 40/64 |
| boundary_3 | 0.00272323 | 0 | 48/48 | 40/64 / 40/64 |
| selected_4 | 0.00501627 | 0 | 48/48 | 40/64 / 40/64 |
| output_map_4 | 0.0039718 | 0 | 48/48 | 40/64 / 40/64 |
| boundary_4 | 0.00302184 | 0 | 48/48 | 40/64 / 40/64 |

Exact cross-ID collisions: none in the measured stage codebooks.

Reference decoding uses one labeled novel-ID vector per ID, with evaluation outside its reference semantic table. This is privileged finite-codebook identification. Isolated same-ID representations repeat across contexts, so success is not learned unseen-ID generalization. The Euclidean classifier is exactly the affine rule2*x·c−||c||²; cosine supplies a fixed nonlinear normalization alternative. No classifier parameters were optimized. Shifted-label controls, all pair distances, ranks, predictions and per-ID agreement are saved in RESULT.json.

A distinct codebook shows that total identity erasure has not occurred on these eight IDs. It does not rule out partial loss, poor conditioning, inaccessible semantics, or failure on other IDs. If distinguishability persists through boundary_3/4, attributing failure specifically to the final output map is unsupported: identity distinctions survive that map and enter the frozen receiver. Native output usability and causal localization remain separate questions.

Familiar nearest-neighbor agreement is a metric-specific association. The prospectively fixed strong-agreement rule is56/64 overall and at least75% for every novel ID; every stage and both metrics are retained. A matching neighbor is not a causal mechanism, and a mismatch is not proof of information loss.

Reader.value, hard selection, output mapping and actual receiver-boundary reconstruction were checked against saved native captures; full raw deltas and query boundary states match exactly. Independent NumPy equations and classifiers, reverse bank extraction, state hashes and sealed parent inventories passed. No live endpoint hash is claimed for this saved-array-only run.

These eight IDs are absent from56 ancestral positive training IDs but have older evaluation/design-selection history. The novel bank is already burned; these diagnostics are post-outcome and cannot restore its held-out status. FINAL64 and Arabic remain unopened.

No bounded repair is selected or executed by this result. Separability can reject total erasure without identifying a causal repair stage. See PREREGISTRATION.md for the fixed method and RESULT.json/stages.pt for complete evidence.
