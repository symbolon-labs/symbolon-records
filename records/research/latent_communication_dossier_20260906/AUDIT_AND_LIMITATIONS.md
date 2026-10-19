# Adversarial audit, claim boundaries and unresolved questions

This is a record of checks and limits, not an attempt to explain every positive
result as a broader breakthrough. Authority:
[primary checks](../latent_channel_smollm_20260906/run_final/analysis.json),
[independent closure](../latent_channel_smollm_20260906/VERIFICATION.json),
[independent inputs](../latent_channel_smollm_20260906/postfreeze_input_audit.json).

## Claim ladder

| Property | Current evidence |
|---|---|
| Cross-model representational correspondence | Historical probe/map results; separate from this reader |
| Mapped morphological decodability | Historical result with stated recovery limits |
| Direct residual gender-coordinate control | Corrected tested interface NO-GO, particularly masculine/random separation |
| Learned same-family synthetic channel | V2 held-out PASS |
| Learned cross-family synthetic channel | Qwen→SmolLM2 held-out PASS |
| Content-dependent donor substitution | Exact observed probability-vector and symbol following |
| Natural semantic capability transfer | Not tested |
| Arabic specialist→generalist transfer | Not tested by the successful channel |
| Free generation or agentic/coding preservation | Not tested |
| Superiority to text | Not established; accuracy ties, text transport is smaller |

The successful bridge is not the failed specialist→linear-map→native-direction
design. The task, interface and training hypothesis changed with explicit user
authorization. That is a disclosed new branch, not a rescue of an old result.

## Leakage and correctness checks

| Alternative explanation | Evidence/check | Remaining scope |
|---|---|---|
| Receiver reads the original source | Latent receiver receives exact constant Q plus memory only | Strong text intentionally receives source information in its own separate arm |
| Class imbalance or default label | Exactly four labels per nonce, equal class counts; receiver-only/zero 25%, all correct classes 100% | Baseline predicts A; chance accuracy is not calibrated uncertainty |
| IDs/file names reveal labels | IDs are shuffled row indices and are not model inputs; source IDs independently re-tokenized | Historical V1 unnecessary metadata IDs were corrected pretraining and retained |
| Sequence length/padding leakage | All source length 22, latent query length 12; equal-length assertion, no padding | Fixed length/task template limits generality |
| Nonce memorization | TRAIN/DEV/FINAL and materialized prior nonces disjoint; all four symbols per nonce | Shared alphabet/template; fresh nuisance strings do not test unseen semantics |
| Batch ordering encodes labels | Seeded shuffled rows; native per-example transformer attention, no cross-batch normalization | Exact inference checks use fixed batches; arbitrary batch/device bit-exactness not claimed |
| Label tokenization error | Joint boundaries checked; exact leading-space versus bare chat token IDs separated | Current labels are one token; future multi-token tasks need new validation |
| Four-way renormalized loss masquerades as LM loss | Full-vocabulary CE/log-softmax in code; saved candidate probabilities come from full distribution | Most final output files retain only four label LPs plus global argmax, not full-vocab tensors |
| Endpoint fine-tuning | requires_grad=False, absent endpoint gradients, optimizer bridge-only; full parameter hashes unchanged | Remaining frozen receiver still participates in backward propagation to bridge |
| Disabled hook changes ordinary inference | Actual hook-removed stock vs disabled vs zero full-logit equality | Covers tested prompts/site/runtime; not a broad coding-preservation suite |
| Stale hidden cache or wrong checkpoint | Data/model hashes, source-ID retokenization, exact 32-source rerun, rebuilt memories, checkpoint selection rederived | Not all 1,024 sender forwards were repeated a second time |
| Shuffle join bug | Label-blind derangement, no self-donors, IDs/labels verified; all LP vectors equal donor correct vectors | Constant Q makes exact donor equivariance mechanically natural; useful decoding must also be high |
| Random-control bug | Independent seeded Gaussian slots, per-slot norm matching, exact memory recreation | Downstream projected/update energy differs from real memories |
| Final-based tuning | Freeze before final construction, selected DEV exact repeat, one final evaluation, exclusive outputs | Local timestamps are not externally notarized preregistration |
| Bootstrap samples are independent examples | Resample nonce groups jointly across four classes and all conditions | Finite synthetic task and one training seed; not population/per-model universality |

## Why this is content-specific, and why donor exactness alone is insufficient

Receiver-only Q is identical across records, so model behavior can change only
through the memory on the latent path. Correct memories decode all four symbols;
random/zero do not. Real memory swaps change the answer to the donor's symbol.
This establishes a content-sensitive channel under the fixed task.

However, exact shuffled/donor LP equality is also expected from a deterministic
receiver fed the same Q and same donor memory. The failed Llama run had that
equivariance too, while decoding only chance. The informative conjunction is:
high matching accuracy + high donor-symbol following + low original-target
accuracy under substitution + valid controls and held-out records. Exactness is
an integrity check, not by itself successful communication.

Shuffling is not a wrong-sign intervention here: four symbols have no registered
signed geometry. No opposite-class or same-class-only shuffle was evaluated in
this synthetic design. The required label-blind derangement can randomly supply
the same label. Do not relabel it as a guaranteed opposite-class control.

## Energy and state domination

Input transport norms are fixed to 8 per slot in every class. After the trained
projection, correct energy is greater than random; reader residual updates
differ by roughly an order of magnitude. Thus the phrase “norm-matched random”
must always specify **transport norm**. Correct-vs-random alone does not exclude
downstream energy as a contributing mechanism.

Shuffled real memories preserve the complete real-memory/reader-energy
distribution, yet original-target accuracy drops to chance and the donor
symbol wins. A global “any nonzero energy makes the model accurate” explanation
cannot explain that result. The bridge may nevertheless use class-dependent
directions and downstream amplitudes as a learned code. No amplitude-only or
projection-energy-matched intervention ablation isolated their roles.

Mean delta/native norm is 1.109, maximum 2.077; mean pre/post cosine .692,
minimum .425. This is substantial state replacement/modulation. Such magnitude
does not invalidate content-specific communication, but it rules out describing
the mechanism as a gentle nudge or assuming unrelated capabilities remain intact.

## Selection, replication and measurement uncertainty

The history includes V1 architecture failure, the V2 bounded-attention repair,
a Meta-Llama receiver failure, and a separate SmolLM2 qualification. Report all
of them. SmolLM2 was a qualified selected receiver, not a random sample from
model families. Same-family and SmolLM2 results used different fresh data, but
the same explicit initialization seeds/settings as controls. No dozens-model
search occurred, yet no model-family-wide success rate can be inferred either.

DEV was repeatedly inspected and legitimately selected the checkpoint. FINAL
was untouched until the freeze and evaluated once, with registered deterministic
repeats. Repeating saved inference is not restarting training from another
random initialization. Full training-seed replication, alternative hardware,
another independent dataset template and an independent research team were
not tested. “Independent audit” means a separately implemented arithmetic/input
check, not a blinded lab replication.

Zero observed errors produce degenerate empirical bootstrap intervals; they do
not prove population accuracy is exactly 100%. The paired gap is very large,
but the number of nonce groups, fixed alphabet and finite task remain relevant.
The specified 99% contrasts and 95% equivalence bands are the frozen operational
rule, not evidence of an exhaustive search-adjusted general theorem.

## Saved versus missing evidence

Saved: all four data partitions, source token IDs and full captured sender
hidden tensors, bridge best/last checkpoints, every evaluated tiny/full output
payload, candidate full-vocabulary LPs, unconstrained argmax tokens, memories,
donors, per-example reader diagnostics, optimizer-step curves, text messages/
scores, repeat tensors, freeze/manifests/logs and hashes.

Not saved as a universal archive: every full-vocabulary logit vector at every
final condition, receiver hidden states at every layer, every optimizer step's
weights, optimizer state for exact mid-training resume, a complete wheel lock/
binary environment image, power/thermal measurements, isolated bridge overhead,
or generated continuations. Full-logit parity was checked at execution and its
receipt is saved; four saved LPs cannot reconstruct the full vocabulary.

The underlying model files and some runtime dependencies live outside the
experiment tree in the local model cache/virtualenv. Revisions and hashes are
recorded; a Git commit does not capture the dirty research implementation,
ignored artifacts, external weights or all package binaries. The local inventory
does not prove an off-machine backup exists.

Older Arabic recovery has additional missing raw bridge arrays, full historical
intervention outputs and specialist model hashes. New documentation does not
repair those absences. It reports the narrower recovered-source evidence.

## Strongest remaining interpretation

A trainable two-bit encoder/decoder uses sender hidden states and the frozen
receiver's continuation computation to transmit one of four explicit symbols.
The sender's deep reasoning is unnecessary for this task in principle; no
embedding-only or sender-layer ablation establishes otherwise. The constant
query means the reader need not choose different source facts for different
questions. That interpretation is consistent with the intended first channel
gate and does not diminish its measured specificity—it bounds its meaning.

Do not claim thought transfer, emergent universal latent language, Arabic
specialist capability, cross-family universality, compression efficiency,
superiority to text, deployment readiness, novelty, or state-of-the-art. No
literature-priority review was performed for this documentation task.

## Exact decision and stopping boundary

**PASS — cross-family synthetic content-specific latent communication.**
No natural task, Arabic experiment, bottleneck/layer ablation, generation,
coding/agentic preservation test or Ember inference integration was run after
this gate. Those require separately scoped work, preregistration and fresh
data. The successful FINAL is burned and must not become the next DEV set.
