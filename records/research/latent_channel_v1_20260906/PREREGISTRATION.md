# Learned latent channel v1 — preregistration

Frozen before model evaluation/training. Historical residual/prefix experiments
are closed and remain untouched. This is one new architecture, not an alpha or
layer search. Maximum external spend: USD 30; currently zero. No cloud before
local mechanics, tiny overfit, and shuffle gates pass.

## Hypothesis and scope

A frozen Qwen3-1.7B sender can transmit a random four-way symbol through eight
learned latent slots to a frozen Qwen3-0.6B receiver with one trained cross-attention
reader. This is a two-bit synthetic channel, not capability transfer, Arabic,
query-dependent lookup, or compression-efficiency superiority over text.

Models: Qwen/Qwen3-1.7B revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e;
Qwen/Qwen3-0.6B revision c1899de289a04d12100db370d81485cdf75e47ca. Load official
safetensors with installed Transformers, float32 on CPU, no quantization. Capture
sender zero-based block 13 output using a direct hook (not final normalized
hidden_states). Reader attaches after receiver zero-based block 13. Both have 28
blocks. Endpoint weights, tokenizers, embeddings and LM heads stay frozen.

## Data and leakage construction

Source: `Noise: <eight independent decimal digits separated by spaces>\nSignal: X`,
where X is A/B/C/D. Receiver query, identical for all examples:
`Hidden signal (A/B/C/D):`. Answers are the complete one-token continuations
` A`, ` B`, ` C`, ` D` (assert tokenizer identity). Every nonce group has all
four counterfactual signals. Nonce length, token length and padding are equal
across classes. IDs and labels never enter the model except X in sender source;
the receiver receives only the constant query and latent memory. Thus empirical
receiver-only forced-choice accuracy must be exactly 25%. Randomize row order.

Tiny: 8 nonce groups / 32 examples, generator seed 11001.
Train: 256 groups / 1024 examples, seed 22001.
Dev: 64 groups / 256 examples, seed 33001.
Final: 256 groups / 1024 examples, seed 44001. Reserve this seed now; do not
materialize or evaluate final until architecture, checkpoint and decision rule
are frozen. Assert no nonce overlaps across partitions; if overlap occurs,
deterministically reject already-used nonces without consulting outcomes.
Counterfactual groups are the bootstrap units. Tiny is mechanical only, and is
not reused for full training. DEV is reusable for stopping/checkpoint selection.
After final evaluation, its examples are burned. No test-selected revisions.

## Architecture (one configuration)

Sender states (length L, width 2048) -> non-affine LayerNorm -> bias-free K/V
projections to 64 -> 8 learned queries, one-head scaled dot-product attention ->
8x64 slots -> non-affine RMS normalization per slot (epsilon 1e-8) -> transport
-> bias-free 64-to-1024 projector -> one reader. Reader has non-affine query
LayerNorm and bias-free Q/K/V 1024-to-64 projections, one-head attention, and
bias-free 64-to-1024 output. Add context at all receiver query positions using
an unconstrained learned scalar gate initialized to 0.01. No other layer/site.
No auxiliary losses, dropout, trained classifier, or hidden reconstruction.
Slot normalization is an architectural scale control, not a learned class cue.
Zero memory yields exactly zero context (no projection biases). Disabled reader
returns the original tensor without arithmetic. Train only bridge parameters;
assert <1% of receiver parameters. Transport = 8*64*4 = 2048 float32 bytes;
expanded receiver memory = 32768 bytes, both reported. Text is much smaller for
this deliberately tiny task. Do not claim a compression advantage over text.

## Objective and fixed optimization

Full-vocabulary next-answer-token cross entropy, not a four-class replacement
head. Forced-choice accuracy uses the four complete answer log probabilities.
Also report unconstrained top-1 answer-token accuracy as a diagnostic, not a
license for generation. AdamW, lr=0.003, weight_decay=0, gradient clipping=1.
CPU float32, 4 threads, eager attention, deterministic algorithms, no dropout.
Tiny seed 51001, batch 8, max 300 optimizer steps; evaluate every 20 steps.
Stop at first evaluation satisfying Gates 1 and 2. Do not expand the step budget.
Full training starts fresh (seed 52001), batch 16, max 600 steps; DEV every 50.
Select minimum DEV full-vocabulary CE; ties favor earlier step. Stop after three
DEV checks without >=0.001 CE improvement. Any computing-only batch changes
require a documented pre-outcome amendment and equivalent effective batch size.

## Required conditions

Every evaluation includes receiver-only, correct, shuffled, random, zero, text.
Shuffled: label-blind fixed-seed random derangement of all examples; Q is constant.
Record donor identities and donor-label agreement. Random: independent Gaussian
slots, each rescaled to its matched correct slot norm; use two fixed seeds
71001 and 71002, each must pass. Zero: same memory shape filled with zeros.
Text: lossless relay of the exact source record followed by the receiver query,
no latent reader. This is an explicitly **oracle/lossless text handoff**, not
sender-generated summarization; copying observed bytes needs no sender inference.
Report that compute advantage honestly. It is a strong information-equivalent
baseline, not a claim about the sender's generative reliability.

Cache only deterministic frozen-source activations and frozen baseline/text
outputs, with input/model/code hashes. Correct/shuffled/random/zero are evaluated
through the trained reader. Record per-example logits/log probabilities, loss,
predictions, slot norms, expanded norms, attention entropy, context/residual norm,
gate and timings. Evaluate all controls on the tiny set as well as DEV/final.

## Gates and uncertainty

Gate 0: disabled hook vs stock logits bit-exact; zero memory bit-exact; actual
nonzero output change; finite nonzero gradients into resampler, projector,
reader and gate; no endpoint gradients; deterministic repeat equal. Validate
all relevant shape/token/data assertions and frozen parameter hashes.

Gate 1: tiny correct accuracy >=95% (at least 31/32).
Gate 2: tiny shuffled and each random <=50%; zero == receiver-only ==25%.
If max steps fail, stop F: MECHANICS / TRAINING NO-GO. Diagnose using saved
gradients, gate/attention/norms and train curves; do not enlarge architecture.

Gate 3 final: correct >=90%, per-class >=85%; receiver-only exactly25%; shuffled
and each random <=35%; zero predictions/logits exactly match receiver-only.
Correct-minus-each-control paired 99% cluster-bootstrap accuracy lower bound
must exceed +0.50 (including zero, excluding text). Shuffled/random minus
receiver-only 95% cluster-bootstrap CI must lie within [-0.10,+0.10].
Use 10,000 bootstrap draws, seed 81001; groups of four counterfactual records.
Exact deterministic rerun of fixed first 32 examples, all conditions, must
match; frozen endpoint/checkpoint hashes unchanged. Full controls fail if only
nonzero energy or class bias explains success. Inspect per-class, donor-following,
norm-only predictability and cross-attention diagnostics. Text need not lose for
a channel PASS, but must be reported. One final evaluation, no rescue afterward.

Only after Gate3 may ablations or cross-family work be preregistered. Cross-family,
natural tasks, Arabic and generation are not licensed by a tiny overfit PASS.

## Resource and artifact discipline

Run sender extraction separately and unload before receiver training. Estimate
CPU resident weights ~6.9GB sender / ~2.4GB receiver plus runtime/activations.
Check memory and runtime with tiny before full-scale decisions. Cloud, if justified,
needs a separate cost/runtime/hardware plan, shutdown and verified artifact copy.
Keep all attempts, errors and amendments. Save model/tokenizer file hashes,
environment/package versions, source hashes, git status/commit, seeds, split and
checkpoint hashes, command lines, training curves, controls and total elapsed time.
No Ember source changes. Final verdict follows the user's A–F ladder; do not
turn a mechanical PASS into a communication result.
