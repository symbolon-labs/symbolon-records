# Cross-family synthetic channel — preregistration

Written before cross-family model evaluation, data construction or optimization.
The sealed same-family V2 result is a Gate3 PASS; it is not cross-family evidence.
Its final44002 set is burned for development. No edits to V1/V2 or Ember code.

## Hypothesis and sole endpoint substitution

The identical learned-reader architecture can transmit the sender-only four-way
symbol to a frozen receiver from a different family. Sender remains
Qwen/Qwen3-1.7B revision70d244cc86ccca08cf5af4e1e306ecf908b1ad5e.
Receiver is meta-llama/Llama-3.2-1B-Instruct,
revision9213176726f574b556790deb65791e0c5aa438b6. Official access verified after
the user authenticated. Prior access was denied; TinyLlama metadata were examined
as an availability fallback only, with no weights downloaded or inference run.
TinyLlama is NOT used in this experiment.

Same sender block13 output (zero-based). Same deterministic middle-block rule:
floor(n_blocks/2)-1, hence receiver block7 for16 blocks (V2 was13 for28 blocks).
Receiver width2048 necessarily replaces1024 in the latent projector and reader
Q/K/V/output projections. These are dimension adaptations, not a layer search.
K8, width64, learned resampler queries, non-affine normalization, per-slot fixed
RMS (L2 norm8), bias-free projections, scalar gate initialized.01, both attention
scores2*cosine(q,k), eps1e-8, all remain unchanged. One reader only; no prefix.
918,017 trainable parameters; require<1% of actual receiver parameter count.
Both endpoint models/embeddings/LM heads/tokenizers frozen. Float32 CPU native
Transformers eager inference,4 threads/1 interop, deterministic algorithms.

## Identical task and fresh data

S = `Noise: <eight space-separated random digits>\nSignal: X`, X in A/B/C/D.
Q = `Hidden signal (A/B/C/D):`, identical for every receiver example.
Complete answer strings: ` A`, ` B`, ` C`, ` D`. Source22 tokens; receiver8
tokens under inspected tokenizers. Llama continuation IDs362/426/356/423, derived
from joint boundaries, not inherited Qwen IDs. No BOS is added to plain prompts,
matching the original add_special_tokens=False policy. No raw source/ID/label
enters the latent receiver query. No label-dependent length or padding.

Each nonce has all four counterfactual symbols. Fresh seeds and sizes:
tiny11003:8 groups/32 examples; TRAIN22003:256/1024; DEV33003:64/256;
FINAL44003:256/1024. Exclude all previously observed V1/V2 nonces before drawing
any new partition; enforce cross-partition nonce disjointness. Do not generate
FINAL until checkpoint/code/scoring freeze. TRAIN and DEV may be generated only
after tiny mechanics/overfit/shuffle gates pass. No final-based architecture or
checkpoint selection. No later task, Arabic or generation run is authorized here.

## Same controls and optimization

Every evaluation: receiver-only, correct, label-blind derangement shuffled
(seed61001), random norm-matched slots (71001 and71002), exact zero, original
lossless plain-text relay. Also run the same stronger actual sender -> one-letter
text -> receiver, oracle compact text, and missing-source chat control. Text
instructions remain byte-identical; each endpoint uses its native chat template.
Freeze date_string="06 Sep 2026" in Llama's template; enable_thinking=False
remains for Qwen. Chat A/B/C/D IDs32/33/34/35; validate joint boundaries.
No free generation. Cache only explicitly recorded invariant baseline/text
outputs during development, and the four unique compact text receiver messages.

Fresh bridge initialization; never load a V2 or tiny checkpoint into full training.
Keep initialization/optimizer seeds51001 (tiny) and52001 (full), and minibatch
seeds51002/52002, unchanged as experimental controls. Dataset seeds above are
fresh. Full-vocabulary CE; AdamW lr.003, weight_decay0, clip_norm1; no auxiliary
losses, dropout, scheduler or hyperparameter search.

Tiny: batch8, maximum300 steps, evaluate every20; first check with correct>=95%,
shuffle and both random<=50%, zero==baseline==25%, followed by actual exact
all-condition rerun, stock/disabled/zero parity, finite nonzero bridge gradients,
no endpoint gradients and unchanged endpoint hash. Low shuffle alone is not PASS.
If tiny fails, stop and diagnose; do not silently change architecture or proceed.

Full: batch16, maximum600 steps, DEV every50; early stop after three checks
without>=.001 CE improvement. Select minimum trained DEV full-vocabulary CE,
earlier ties. Recompute the selection from stored log-probability tensors;
actually rerun every selected DEV condition before final freeze. Require the
same DEV readiness point thresholds as V2, no fallback checkpoint.

## Unchanged heldout pass rule

Correct>=90%, each class>=85%; receiver-only exactly25%; shuffle/both random<=35%;
zero logits identical to baseline. Correct minus each non-text control paired99%
nonce-group bootstrap lower bound>+.50. Shuffle/random minus baseline paired95%
intervals wholly inside[-.10,+.10].10,000 group draws seed81001. Use all four
counterfactuals as one group, not four independent units. Require fixed first32
actual exact rerun with original full-set donor/random memories, model/code/split
hashes, reproduced memory tensors, and the mandatory text comparison.
Check swapped probabilities equal the donor's correct probabilities exactly;
donor-symbol following must match correct accuracy (and thus>=90% for PASS).
This is a construction/semantic check, not a relaxed accuracy threshold.

PASS establishes cross-family synthetic communication only. NO-GO preserves
the earlier same-family PASS and does not establish universal impossibility.
No post-final repair. Text need not lose for communication PASS; no efficiency
claim without matched timing evidence. Transmitted norms are matched, not
necessarily post-reader energy; inspect both and use shuffled energy controls.

## Compute, implementation, and integrity

Reuse pinned V1 inference/training orchestration and V2 bounded bridge operations
and statistics. New Python wrappers adapt receiver dimensions/site and provenance;
no parallel endpoint inference implementation or Rust modifications. Rehash prior
seals, exact model/tokenizer files (including any .model/.jinja), source modules,
runtime implementations, checkpoint, split, captures and all outputs. Record
commands, versions, layers, conditions, token IDs, seeds, elapsed time and hardware.

Local i5-1135G7,16GiB RAM, no CUDA. Approximately13GiB available before run.
Sender transient peak measured12.3GiB in V2. Receiver float32 weights~5GiB plus
runtime/autograd; expect<10GiB, measure actual tiny peak before scaling. Load
endpoints strictly sequentially. Initial budget: tiny mechanical training~15–25min,
full collection/training/checks~1–2h, confirmation/text~1h; estimates not latency
benchmarks. Stop on mechanics/resource failure, preserve partial attempts.
No cloud requested or provisioned; external/cloud spendUSD0, cumulative ceiling
approximatelyUSD30. Authentication tokens must never enter artifacts or logs.
Local hashes/timestamps are integrity evidence, not third-party attestations.
