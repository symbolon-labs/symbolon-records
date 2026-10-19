# Cross-family synthetic channel: F — MECHANICS / TRAINING NO-GO

The fixed Qwen3-1.7B → Llama-3.2-1B-Instruct configuration failed the 32-example
overfit gate. No held-out cross-family result exists. The earlier same-family
PASS is preserved. No architecture retry, full training, final evaluation,
natural-language task, Arabic task, ablation, or generation was run.

An important confound remains: the frozen receiver also failed the prescribed
oracle text-handoff sanity check. This result cannot isolate family mismatch or
the latent architecture from receiver/output elicitation and optimization.

## Hypothesis and chronological boundary

The sealed same-family V2 experiment established its held-out synthetic channel:
correct100%, baseline25%, shuffled25.49%, two random controls25.00/25.10%,
zero25%, actual sender-text100%. Earlier V1's unbounded-attention training failed;
V2's bounded-attention repair was separately registered. Those outcomes are not
cross-family evidence. All163 sealed V2 files were independently rehashed and
left unchanged. The old Arabic residual branch was not reopened.

Here the sole experimental endpoint substitution was the receiver. Necessary
width adaptation and the identical middle-block selection rule were registered
before inference. Official Llama access became available after user authentication;
the downloaded weight file matched its upstream LFS digest. TinyLlama metadata
had been inspected during the earlier access problem, but no TinyLlama inference
or training occurred.

## Architecture and implementation

```text
sender-only record S
  → frozen Qwen3-1.7B block13 states [22 × 2048]
  → learned bounded-attention resampler
  → 8 × 64 float32 slots, L2 norm8 per slot
  → bias-free 64 → 2048 projection
  → one bounded cross-attention reader after Llama block7
       query = receiver residual; key/value = latent memory
       h' = h + learned_scalar × reader_output
  → remaining frozen Llama blocks and frozen LM head
  → full-vocabulary answer-token cross-entropy / forced choice
```

Both bridge attention operations use2*cosine(q,k), eps1e-8. The attention width
is64; gate initialization is.01. No embedding prefix, endpoint fine-tuning,
hidden-state reconstruction loss, or Ember runtime modification was introduced.
Only918,017 bridge parameters are trainable:0.074284% of the receiver's
1,235,814,400 parameters. All embeddings, native blocks, LM heads and tokenizers
remain frozen. Source extraction is zero-based block13; receiver insertion is
zero-based block7, floor(16/2)-1, not a searched layer.

The transport is2,048 bytes/example. Original captured states are180,224 bytes,
an88:1 compression ratio. Receiver-expanded memory is65,536 bytes. The task
contains only2 bits of answer information; this bottleneck is not bit-efficient.

Exact endpoints:

- Qwen/Qwen3-1.7B, revision70d244cc86ccca08cf5af4e1e306ecf908b1ad5e.
- meta-llama/Llama-3.2-1B-Instruct, revision9213176726f574b556790deb65791e0c5aa438b6.

Native Transformers eager, float32 CPU, no quantization; Python3.14.7,
torch2.13.0+cu130, transformers5.10.2, numpy2.4.6;4 intra-op/1 inter-op threads.
The code imports pinned historical inference/training and bounded operations;
new wrappers adapt dimensions/site and provenance. Final-stage code was written
and mechanically tested but was not licensed for model execution.

## Data, scoring and preregistered rule

S is `Noise: <eight random digits separated by spaces>\nSignal: X`, where
X∈A/B/C/D. Q is always `Hidden signal (A/B/C/D):`. Every nonce has all four
counterfactual symbols. Q contains no source, example ID, label, or nonce and is
identical on every example, making its balanced accuracy exactly25%.

Tiny seed11003 yielded8 fresh nonces/32 examples, disjoint from all observed
V1/V2 nonces. TRAIN22003/1024, DEV33003/256, FINAL44003/1024 were registered.
TRAIN/DEV were instantiated only in structural unit tests, not captured or
scored; this literal early-generation deviation is recorded in AUDIT_NOTES.md.
No full-training/DEV artifacts exist. FINAL was never instantiated or evaluated.

Plain query8 tokens; sender source22 tokens; plain text relay31 receiver tokens.
Full continuations ` A`, ` B`, ` C`, ` D` each have one token under Llama,
IDs362/426/356/423. Joint prompt/continuation token boundaries were verified.
Scoring uses full-vocabulary log probabilities, not four-way renormalized loss.
No free generation was run. Unconstrained next-token IDs are diagnostics only.

Tiny rule: fresh initialization51001; minibatches51002; AdamW lr.003,
weight_decay0, clip1, batch8; at most300 steps, checks every20. Require correct
≥95%, shuffle and both random≤50%, zero==baseline==25%, parity, finite nonzero
bridge gradients, unchanged frozen endpoints, and actual exact rerun. Low
shuffle alone is not a Gate2 PASS. No setting was changed after outcomes.

All300 steps were executed. The raw inherited decision's gate2=true means only
that destructive controls were low; semantic_gates.json correctly makes gate2
false because correct overfit failed. The minimum-CE checkpoint was step60,
correct25%, CE1.392983. It was not substituted for the terminal checkpoint to
rescue the result. Highest observed evaluated correct accuracy was28.125% at80.

## Results: training-set mechanical gate only

| Condition | Terminal accuracy | Per-class recall A/B/C/D |
|---|---:|---|
| Receiver-only |25.000%|100/0/0/0%|
| Correct latent |25.000%|100/0/0/0%|
| Shuffled latent |25.000%|100/0/0/0%|
| Random71001 |18.750%|12.5/0/62.5/0%|
| Random71002 |28.125%|50/12.5/50/0%|
| Zero latent |25.000%|100/0/0/0%|
| Original plain text relay |21.875%|87.5/0/0/0%|
| Actual sender → compact text → receiver |25.000%|0/0/0/100%|
| Oracle compact text → receiver |25.000%|0/0/0/100%|
| Missing-source native-chat receiver |25.000%|0/0/0/100%|

The actual sender's text classification was100% on all classes. Llama selected
D for all four unique compact messages, including oracle messages. Thus the
text sanity check failed; do not characterize this as “text works, latent fails.”
Both endpoints used native chat templates with unchanged instruction bodies;
Llama's date field was fixed to06 Sep2026. Chat label IDs32/33/34/35 and joint
boundaries passed. No prompt search or alternative label scoring followed.

All16 evaluations and300 optimizer records are retained in run_tiny and
TINY_ANALYSIS.json. Terminal correct CE5.52348 vs baseline5.30682 and shuffled
5.49297. Correct's mean toward-answer margin change vs baseline was−3.00813
nats, paired99% interval[−3.12880,−2.88422]. Correct−shuffled margin change was
−.04734,[−.14022,+.03749]. Correct−baseline and correct−shuffle accuracy gains
were both0, intervals[0,0]. These are descriptive bootstraps of8 training nonce
groups,10,000 draws seed81001, not uncertainty about held-out generalization.

## Integrity and failure diagnostics

Stock/disabled/zero full-vocabulary parity passed before optimization and at
the trained terminal checkpoint. Initial intervention changed logits and every
bridge parameter had finite nonzero gradients. Frozen endpoint gradients were
absent. The receiver parameter hash before/after training and terminal rerun was
2adc7a7195502293baf783cf01cc849c60952a2f451fb5f6cb3ab3941dc52d99.

All terminal outputs were actually rerun, without baseline/text caches, and
matched exactly for every example and all7 original conditions. Shuffled
probability vectors exactly equaled the corresponding donor's correct vectors;
donor-symbol following remained25%. This proves routing equivariance, not
successful decoding. Latent tensors and donor indices were independently rebuilt
from source captures and the saved checkpoint and matched stored tensors.

Correct/shuffled/random transported slot norms were8 within2e-6, all classes
matched; zero was exactly0. Shuffling preserves the full activation-energy
distribution. At the terminal correct checkpoint, reader entropy2.079438 was
near log(8); attention weights were approximately.125. Mean delta/residual norm
ratio was2.00146 and mean cos(h,h')=.47875: strong perturbation without recovery.
Random controls matched transport norms, not resulting reader energy; their
mean delta/residual ratios were.23873/.19998. No norm-based success is claimed.

Pre-clipping gradient norm median425.913, maximum14,721.599; clipping remained1.
The learned scalar crossed zero, range[−.001903,+.019230]. This is evidence of
unstable optimization, not a causal diagnosis of why it failed. Uniform reader
attention alone also does not prove the resampler lost all information.

Strongest remaining alternatives: fixed receiver prompt/answer elicitation
(oracle text also failed), optimization instability, and an unsuitable single
interface placement. None was tested by changing this experiment. These results
do not prove that Llama cannot read symbols, that family mismatch is intrinsically
fatal, or that all learned latent readers fail.

## Provenance, compute and limitations

acquisition.json records upstream-verified receiver weight hash
1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f.
data/tiny/manifest.json inventories sender shards and tokenizer hashes;
run_tiny/manifest.json records the receiver files, optimizer, environment and
Ember commit ae550f0fbcf9dea1a5507f5695b077d2cc77d02b. Extension receipts pin
new code, inherited code, split seeds, prompt/preregistration and layer choices.
No research edits were made to Rust, frozen historical trees, or paper outputs.

The first shell launch failed before execution because .venv was empty. It was
restored using recorded versions; the sealed Qwen implementation hash and all
163 historical files matched. All currently installed packages were inventoried;
unrecorded prior dependency versions cannot be asserted identical. Seven tests
passed both after restoration and after confirmation-code implementation.
See ENVIRONMENT_RECOVERY.md, environment_restored.json and the separate XML
receipts. Local seals are integrity records, not external preregistration proof.

Local CPU only, i5-1135G7/16GiB; endpoints loaded sequentially. Tiny capture and
training queue1307.67s, terminal audit80.31s, text diagnostic71.74s: summed
experiment stages1459.72s (24.33min), excluding installation/download/audit work.
Peak child RSS11.68GiB is the queue-wide cumulative peak, not a receiver-only
measurement. Median optimizer step2.563s. External/cloud spend:USD0.

Actual terminal rerun over32 examples: baseline6.17s, correct6.35s, shuffled6.49s,
random6.67/6.79s, zero6.97s, plain text21.03s. These are unpaired single-run
condition timings, not efficiency estimates. Strong-text sender34.86s on32;
receiver4.95s for four unique compact messages plus one missing-source control,
with explicit caching. Do not compare that amortized cache cost to uncached
latent inference. Strong-text prompts61 sender tokens and58 receiver tokens;
transmitted symbol1 UTF-8 byte versus2,048 latent bytes. No efficiency advantage.

## Decision

Gate0 mechanics: PASS. Gate1 tiny overfit: NO-GO. Gate2 semantic shuffle gate:
not established. Held-out synthetic cross-family, natural transfer, Arabic,
capability preservation, ablations and generation: NOT LICENSED / NOT RUN.

Final verdict: **F — MECHANICS / TRAINING NO-GO**, specifically a training
NO-GO for this frozen cross-family configuration. Stop. No rescue run is queued.
