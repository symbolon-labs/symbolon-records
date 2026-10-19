# Latent channel v2 — bounded-attention repair

Registered before v2 model evaluation or optimization. V1 remains frozen, with
its failed 300-step experiment, outputs and diagnostic audits intact. This is
the second disclosed training configuration, not a retroactive V1 rescue.

## Evidence and hypothesis

V1's trained receiver attention was approximately [0.5,0,0,0.5,0,0,0,0]. The
selected slots encoded the constant `Signal` marker, while ignored slots retained
the actual symbol. Mean counterfactual context L2 was ~1e-7; gradients reaching
the semantic source token were ~4e-13. Hypothesis: unbounded learned Q/K scores
closed the semantic gradient path through attention saturation. Bounding scores
may preserve an optimizable information path. This is not yet a causal account
of V1's entire failure and does not establish that optimization will succeed.

## Exactly one repair

Replace q*k/sqrt(64) at both bridge attention operations with
`2 * dot(normalize(q), normalize(k))`, using L2 normalization eps=1e-8.
Scores lie in [-2,+2] up to floating-point roundoff. For n keys each softmax
weight is at least exp(-4)/n (a conservative bound). This prevents numerical
slot shutdown, without adding parameters, slots, layers, loss terms or gates.

Unchanged from V1: frozen Qwen3-1.7B -> Qwen3-0.6B, exact revisions below;
source block13 and receiver block13 (zero-based), K8 x width64, fixed-RMS
transport normalization, width1024 projector, one reader, trainable scalar
gate initialized .01, all projection biases absent. 590,337 trainable parameters,
0.099% of receiver. No prefix injection or residual-direction mapping.
Receiver disabled/zero parity remains mandatory.

Sender revision: 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e.
Receiver revision: c1899de289a04d12100db370d81485cdf75e47ca.
Float32 CPU, eager attention, 4 threads, deterministic algorithms.

## Task, data and controls

Task unchanged: sender sees eight nuisance digits and `Signal: X`; receiver
sees only `Hidden signal (A/B/C/D):`. Complete answer continuations are single
tokens ` A`, ` B`, ` C`, ` D`. Eight nonce groups x all four counterfactual
symbols = 32 tiny mechanical examples, with shuffled row order and opaque IDs.
Receiver-only is exactly25% by construction; no label-dependent lengths/padding.

New seeds: tiny11002 (8 groups), train22002 (256 groups), dev33002 (64 groups),
final44002 (256 groups). All v2 partitions nonce-disjoint, and all observed V1
nonces excluded. Final seed is reserved only: do not construct final until
checkpoint/architecture/scoring freeze after development. Tiny is deliberately
trained on; no tiny result is scientific evidence. Train/dev use fresh model
evaluations only after the preceding tiny gates pass.

Every evaluation includes receiver-only, correct, label-blind derangement
shuffle, two independently norm-matched random memories (seeds71001/71002),
zero, and lossless text relay. The original plain-completion text relay is
retained to isolate the attention repair; its V1 result was poor and it is
not a demonstrated strong text baseline. Before any comparative efficiency
claim, a separately registered instruction-following text baseline is needed.

## Fixed optimization and gate order

Full-vocabulary answer-token CE. AdamW lr=.003, weight_decay=0, clip_norm=1.
Tiny initialization/training seed51001, batch8, maximum300 steps, evaluation
every20. Stop at first check with correct>=95% (>=31/32), shuffle<=50%, both
random<=50%, and zero==receiver-only==25%. Require actual all-condition exact
inference rerun, full receiver parameter hash equality, finite nonzero bridge
gradients and no endpoint gradients. A low shuffle score alone is NOT a gate
PASS; correct must first be high. If this repair fails, stop this configuration
and diagnose before proposing any further configuration. Do not enlarge it.

Full training, if licensed: fresh initialization seed52001, train1024/dev256,
batch16, maximum600 steps, DEV every50, early stop after three checks without
>=.001 CE improvement. Select minimum DEV full-vocabulary CE, earlier ties.
No test-based selection. Starting full training must be preceded by a measured
local time/memory estimate; no cloud is needed if local computation suffices.

Final Gate3 rule inherited unchanged from V1: correct>=90%, each class>=85%,
receiver-only exactly25%, shuffle/random<=35%, zero logits identical to baseline.
Correct-minus-each non-text control paired99% group-bootstrap lower bound >+.50.
Shuffle/random minus baseline paired95% intervals inside [-.10,+.10]. 10,000
nonce-group bootstrap draws, seed81001. Fixed first32 exact rerun all controls.
No final materialization or evaluation until a separate checkpoint/code freeze.
No cross-family/natural/Arabic/generation/ablations until their earlier gates pass.

## Implementation and provenance

Reuse the sealed V1 dataset/training/inference orchestration through a Python
module, with explicitly registered v2 subclasses for bridge attention. No
parallel endpoint inference implementation and no edits to V1 or Ember.
Pin inherited channel.py SHA256:
e2de36c051ba503b441c2d84b959728cb09f8d064791012b5f673462039f6fbd.
Record both inherited-source and v2-extension hashes; inherited `code_sha256`
alone does not describe v2. Record model/tokenizer hashes, parameters, checkpoint,
seeds, split/capture hashes, timing, command, environment, and all attempts.
Preregistration is written before outcomes, but local hashes are integrity
evidence, not third-party timestamp attestations.

Prior external spend USD0; cumulative ceiling USD30. Remain local until tiny
mechanics/overfit/shuffle gates pass. No resource is currently provisioned.
