# Identity survives the measured channel; native use still fails

**The eight tested novel IDs remain distinguishable through the actual receiver
boundary. Total identity erasure at the64-dimensional slot is rejected on this
finite bank. A specific causal repair stage is not established.**

This is a successor diagnostic of frozen d0f16cb9, whose novel native score
remains0/32 original,0/32 shuffled donor and0/32 changed donor. It reads the
sealed novel and fresh-familiar native runs. No endpoint was loaded, no new
native answer was scored, no probe or bridge was fitted, and no reset was used.

| Representation | Dimension | Minimum ID separation / maximum vector norm | Reference ID decoding |
|---|---:|---:|---:|
| Isolated sender ID field |4096|0.00559841|48/48|
| ID slot |64|0.0266101|48/48|
| Projected reader memory |960|0.0129795|48/48|
| Reader value |64|0.00501627|48/48|
| Selected value, each query site |64|0.00501627|48/48|
| Final bridge output-map result, each site |960|0.00397180|48/48|
| Actual receiver boundary, site3 / site4 |960|0.00272323 /0.00302184|48/48 each|

Every stage has zero within-ID spread across the saved original/changed
occurrences, no cross-ID collision, and centered rank7 at both fixed float32
and float64 cutoffs. Thus the measured identity distinctions survive the output
map as well as compression. The native receiver nevertheless emits familiar
IDs. Calling this "identity dies in the final output map" would be incorrect.

The48/48 figures use one labeled reference per novel ID and exclude its semantic
table from evaluation. Squared-Euclidean nearest-reference classification is
equivalent to an affine classifier with scores2*x·c−||c||²; cosine-reference
classification also scores48/48 everywhere. Cyclically shifted reference labels
score0/48. No optimization occurs, but these are privileged labeled references.
Because isolated same-ID states repeat exactly, this proves finite-codebook
decodability, not learned transfer to an identity absent from the reference set.
A fitted probe on these same repeated vectors would not resolve that limitation.

Familiar-ID proximity explains only part of the observed output association.
The table below uses Euclidean references; cosine has the same40/64 agreement
at the wire and later stages. Each novel ID contributes eight original/changed
queries. Neither metric meets the prospectively fixed56/64 plus per-ID75% rule.

| Novel ID | Native prediction | Closest familiar wire | Closest familiar boundary, site3 |
|---|---|---|---|
|A0|G3|G3|G3|
|A2|G3|G3|G3|
|C4|G3|G3|G3|
|I2|G3|L1|L1|
|K0|G3|G3|C7|
|K3|C7|G3|C7|
|M0|B3|B3|B3|
|M3|D4|B3|B3|

This is an association test, not a causal nearest-neighbor model of the receiver.
I2 and M3 disagree at both displayed stages; K0/K3 change which stage agrees.
There is no selected best metric, stage or new threshold.

The actionable constraint is narrower than a repair prescription: **bottleneck
growth is not justified by total erasure on these IDs.** Surviving distinctions
that the native receiver maps to familiar answers are consistent with code
compatibility or downstream use failure. This inference does not exclude
partial information loss, poor conditioning, or missing structure needed to
generalize. A causal interface intervention would be a separate experiment;
no receiver training, new adapter, optimizer sweep or bounded fix is selected here.

The initial specification and primary analysis were frozen before diagnostic
values were computed. A post-result independent NumPy audit rechecks pairwise
geometry, ranks, classifiers, eligibility, controls and agreement without new
hypotheses or thresholds. Frozen value/selection/output equations reconstruct
the full saved raw deltas and both actual boundary states exactly; projection
reconstruction is also exact in Torch. NumPy relative errors are below the
fixed1e-5 bound. Reverse extraction, candidate restoration and parent seals pass;
all22 endpoint/tokenizer snapshot files were rehashed without loading endpoints.

Scope remains eight whole IDs absent from56 ancestral positive training IDs,
with older evaluation/design-selection history and a now-burned novel bank.
No pristine-ID or novel-component claim follows. FINAL64, Arabic and generation
remain unopened. See REPORT.md, RESULT.json and PREREGISTRATION.md for the full
stage results and fixed method.
