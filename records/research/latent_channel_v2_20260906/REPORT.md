# Learned latent channel V2

**A — LATENT COMMUNICATION CHANNEL ESTABLISHED. Gate3 PASS.**

Scope: frozen Qwen3-1.7B -> frozen Qwen3-0.6B, a held-out synthetic four-symbol
channel. Correct latent100%; receiver-only25%; shuffled25.49%; random25%/25.10%;
zero25%. Strong actual sender-text and oracle-text both100%: **no accuracy
advantage over text**. No cross-family, natural-task, Arabic or free-generation
claim. The frozen final decision and all checks are in
[analysis.json](run_final/analysis.json).

## Hypothesis and prior failures

The closed Arabic residual/prefix branch demonstrated representational
correspondence and causal steering, but not robust content-specific
bidirectional control. Its corrected receiver-native experiment remains NO-GO
([report](../latent_bridge_redo_cached_20260906/REPORT.md)). That is a negative
result for the tested interface, not a proof that latent communication is
impossible. No further residual-direction map/layer/alpha search was run here.

The first explicit-reader configuration, V1, also failed: 300 tiny-set steps,
correct25%, controls25%. Its attention collapsed onto slots encoding a constant
source marker, with effectively absent gradients to the semantic token. The
failed implementation, all outcomes and audit remain sealed
([V1 report](../latent_channel_v1_20260906/REPORT.md)).

V2 is the second disclosed architecture configuration. The sole architectural
repair bounds both bridge attention scores to `2*cosine(q,k)`, eps1e-8. No extra
parameters, layers, slots, losses or gain search. This targets the observed
attention-saturation failure. Success would support this practical repair, not
prove a complete causal explanation of V1's optimization failure.

## Architecture and endpoints

```text
S: Noise: <eight random digits> / Signal: X
    frozen Qwen3-1.7B, block13 output (22 x 2048)
    learned 8-query bounded-attention resampler, width64
    fixed-RMS normalization: each latent slot has L2 norm8
    ------- 8 x 64 float32 latent channel -------
    learned bias-free 64 -> 1024 projection
Q: Hidden signal (A/B/C/D):
    frozen Qwen3-0.6B through block13
    one learned width64 cross-attention reader + scalar gate
    frozen remaining receiver blocks and original LM head
    full-vocabulary answer-token probabilities
```

Both block indices are zero-based13, after block output. The reader operates on
all eight query positions at this one site. Its Q/K/V projections have width64;
output projection returns1024. Non-affine LayerNorm precedes resampler and reader
queries. All projections are bias-free. Gate initialized .01. No prefix tokens,
external classifier head, endpoint fine-tuning, dropout or auxiliary objective.

Trainable parameters: **590,337 / 596,049,920 receiver parameters (0.099%)**.
The optimizer receives only the bridge object. Endpoint embeddings, normal
transformer blocks, normalizations, LM heads and tokenizers remain frozen.

Sender revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`;
receiver revision `c1899de289a04d12100db370d81485cdf75e47ca`.
Official safetensors loaded float32; no quantization; native PyTorch/Transformers
eager CPU inference. No Ember Rust or generation-semantics modifications.

Channel2048 bytes/example; captured source180,224 bytes/example;88:1 activation
compression. Projected receiver memory32,768 bytes. The actual task carries only
two bits, so this is not efficient source coding: one text character is smaller.
K8 is the first tested bottleneck, not a demonstrated minimum.

## Task and leakage controls

The receiver query is identical for every example and contains no source, ID,
label or nonce. Every nonce appears in all four counterfactual classes. Thus
Q alone has exactly25% accuracy; the source-only Signal field determines Y.
All sources have22 tokens, queries8, with no label-dependent padding/length.
Full continuations ` A`, ` B`, ` C`, ` D` are single tokens362/425/356/422;
joint continuation boundaries are checked, not assumed.

Tiny seed11002:32 examples/8 nonce groups. TRAIN22002:1024/256.
DEV33002:256/64. FINAL44002:1024/256. All nonce-disjoint, with observed V1
nonces excluded. Labels A–D deliberately repeat; this tests a four-symbol
alphabet across fresh nuisance records, not novel alphabet identities or
compositional generalization. Opaque shuffled IDs are artifact joins only.

Controls: receiver-only, correct memory, label-blind derangement shuffle, two
independently norm-matched Gaussian memories (71001/71002), exact zero, original
plain text, actual sender-selected compact text, oracle compact text, and chat
receiver-only. Shuffled examples can share the target label by chance; they are
not adversarially chosen to be wrong. Stronger text uses a separately frozen
official non-thinking chat format. All conditions are forced-choice, not free
generation. Unique compact text receiver prompts are explicitly cached.

## Training and preregistered decision

Full-vocabulary CE; AdamW LR.003, weight_decay0, gradient clip1. Tiny seed51001,
batch8, max300, checks20. Tiny first passed at160: correct100%, shuffle31.25%,
random25%/18.75%, zero/baseline25%; all-condition exact rerun passed. Strong
sender-text and oracle-text100%. These are trained-on mechanical evidence only.

Full run starts from fresh initialization seed52001, batch16, max600, DEV50.
Early stop after three checks without >=.001 CE improvement; independently
select minimum recorded trained DEV CE, earlier ties. It stopped at300, which
also had minimum DEV CE. No final-based selection or fallback checkpoint.

| Step | DEV correct | DEV CE | DEV shuffled | DEV random1 / random2 |
|---|---:|---:|---:|---:|
| 0 |25%|5.296581|25%|25% /25%|
| 50 |41.80%|1.353939|24.22%|25% /25%|
| 100 |93.75%|.394837|24.61%|25% /25%|
| 150 |100%|.0001034|23.83%|25% /25%|
| 200 |100%|.0036443|23.83%|25% /25%|
| 250 |100%|.0017360|23.83%|25.78% /25%|
| 300 |100%|.0000596|23.83%|25.78% /25%|

All300 training steps, losses, gradient norms and gate values:
[training curve](run_full/curve.jsonl). Every DEV check is retained, including
failures and nonmonotonic loss. Selected checkpoint actually reproduced all
seven DEV conditions exactly. Strong DEV text sender/oracle both100%.

Final PASS requires correct>=90%, every class>=85%, receiver-only25%, randomized
controls<=35%, zero exact; correct-minus-each non-text control paired99% CI
lower bound>+.50; randomized-minus-baseline paired95% CIs inside[-.10,+.10].
10,000 nonce-group bootstrap draws seed81001. Require fixed-first32 actual exact
rerun, unchanged endpoint/code/tokenizer hashes, valid donor/random construction.
The exact protocol and freeze are in [PREREGISTRATION.md](PREREGISTRATION.md),
[FINAL_EXECUTION.md](FINAL_EXECUTION.md), and
[FINAL_FREEZE.json](run_full/FINAL_FREEZE.json).

## Final results, uncertainty and decision

The single frozen final experiment completed with **PASS** on every statistical
and mechanical criterion. All1,024 examples/256 groups were evaluated. No test
example was used for checkpoint selection; the final set is now burned for
development. No fallback model or changed decision threshold was used.

| Core condition | Final accuracy | A / B / C / D recall |
|---|---:|---|
| Receiver-only |25%|100 /0 /0 /0%|
| Correct latent |100%|100 /100 /100 /100%|
| Shuffled latent |25.49%|27.73 /23.83 /24.22 /26.17%|
| Random71001 |25%|98.83 /0 /0 /1.17%|
| Random71002 |25.10%|98.83 /0 /0 /1.56%|
| Zero latent |25%|100 /0 /0 /0%|
| Original plain text |1.07%|4.30 /0 /0 /0%|
| Actual sender -> compact text -> receiver |100%|100 /100 /100 /100%|
| Oracle compact text -> receiver |100%|100 /100 /100 /100%|
| Receiver chat without source |25%|100 /0 /0 /0%|

Source: [final core records](run_final/evaluation/per_example.json). Actual
first32 receiver rerun is bit-exact; full-vocabulary stock/disabled/zero parity
and the complete receiver parameter hash match. Correct unconstrained next-token
argmax also100%; this remains a next-token diagnostic, not a generation claim.

An independent post-outcome audit finds every shuffled answer-log-probability
vector exactly equals the corresponding donor's correct-latent vector, including
unconstrained top-token identity. Donor-symbol following100%. Independent paired
99% accuracy-gain intervals: baseline/zero[75,75]pp; shuffled[70.80,78.03]pp;
random1[74.41,75.59]pp; random2[74.22,75.59]pp. These use256 nonce groups, not1024
independent observations. The independently implemented intervals agree exactly
with the now-completed frozen final analysis ([audit](independent_core_audit.json)).

Randomized-control minus baseline paired95% intervals: shuffled[-2.34,+3.32]pp;
random1[-.49,+.49]pp; random2[-.39,+.59]pp. All lie within the registered±10pp
equivalence band. Correct has256/256 successes in each class. Bootstrap intervals
collapse at100% when no errors are observed; that does **not** establish perfect
population accuracy or eliminate finite-sample uncertainty. The large paired
contrasts, fresh heldout data and content swaps support the stated narrow result.

| Condition | Mean correct-vs-best-other margin | Median margin | Full-vocabulary CE |
|---|---:|---:|---:|
| Receiver-only |−2.290|−4.305|5.296556|
| Correct latent |15.288|12.975|.000061|
| Shuffled latent |−8.778|−15.288|12.739655|
| Random1 |−3.760|−6.743|6.327990|
| Random2 |−3.783|−6.793|6.421299|
| Zero |−2.290|−4.305|5.296556|
| Original plain text |−1.257|−1.403|4.889210|
| Actual sender-text / oracle text |2.235|2.169|.226043|
| Receiver chat only |−1.119|−1.365|1.858869|

Margins and CE are nats. All per-example probabilities, class-specific intervals
and paired margin intervals are retained in the primary analysis. Cross-prompt
CE differences are not used to claim communication efficiency.

Final freeze was recorded2026-09-06T03:42:51Z before construction/capture.
Local hashes/timestamps provide integrity evidence, not third-party attestation.
An independent input audit re-tokenized every captured source in all four
partitions, verified source token IDs, exact label-continuation boundaries,
counterfactual balance, nonce disjointness and all11 frozen component hashes
([input audit](postfreeze_input_audit.json)).

## Norm and reader diagnostics

Transport slot norms range7.99999857–8.00000095, with mean8.0 in every class.
Both Gaussian controls match these norms within1e-5; zero memory is exactlyzero.
Expanded memory is not energy-matched to random: correct per-class mean norms
are60.80/68.28/57.31/57.32, random means~21.4. This is a learned projection
response, not a label-dependent input length or transmitted-norm shortcut.

Final correct reader entropy mean1.5813 nats (uniform8-way=2.0794); minimum
attention probability across the observed correct condition=.005947. Learned
gate=.034073. Mean delta/residual norm ratio2.0636 (range1.0713–3.7305), mean
cosine(h,h+delta)=.53266. Shuffled memories preserve the exact same distribution
of these diagnostics. Random ratios average.2775/.2822 and cosines.9563/.9548.
Thus randoms alone do not eliminate a receiver-energy explanation; the matched
shuffled distribution and exact donor-following are the decisive content checks.
Large state changes remain a limitation, not a gentle-control claim.
See [reader diagnostics](final_reader_diagnostics.json).

## Reproduction and provenance

Actual all-condition DEV rerun: exact on256 examples. Actual final receiver
rerun: exact on fixed first32, using original full-set donors/random memories.
Actual fresh sender extraction: all32 saved hidden states reproduce exactly,
maximum absolute difference0 ([sender verification](sender_reproduction/verification.json)).
Source IDs were independently re-tokenized for every split. All latent tensors
were re-derived from frozen source captures and checkpoint, not trusted solely
from execution-generated manifests. Model/tokenizer files and frozen code hashes
were independently re-read. Every text join, sender-selected symbol and receiver
lookup was validated against stored records. No silent inference cache is used;
the stronger text receiver's four unique messages are explicitly cached.

Critical SHA256 identities:

- Selected bridge: `722e535d0dd396104a8dc330a94aa98c7c10fc7cb529abd6ccdd722ae7a9ff93`.
- V2 extension: `ee773ac60f7068f3e350ce60120f5f5bed268beece1e9fd7793a48fcb57eaf29`.
- Inherited V1 source: `e2de36c051ba503b441c2d84b959728cb09f8d064791012b5f673462039f6fbd`.
- Final freeze: `6e7743d5dc0e6fdcdffa6bb2fa44860ae3840b81084a1aabe1ecbf4b0ec83509`.
- Final split: `e5cb26974701e24271f5fa1544c75375bce77168a2119deea1c3adbcf8c3ce83`.
- Final source capture: `97ebe5a23841cdb7c1541cc79f2b5f536a5c166595bfb7515080b8a00553569f`.
- Receiver parameters, before/after: `2b7e3b5c1515ecd63a512b9c7464ce1e91ca4b766a6af1b93dd93b793f79d0d1`.
- Sender rerun parameters, before/after: `deb5e69d2fcd24d139ce2927f8375c2c445867b196bc179df4688d98990e0ec4`.

Endpoint shard/tokenizer hashes and revisions are fully enumerated in the freeze;
the inherited source hash alone does not describe V2. Runtime: Python3.14.7,
PyTorch2.13.0+cu130 (CPU only), Transformers5.10.2, deterministic algorithms,
4 threads/1 interop. Ember checkout `ae550f0fbcf9dea1a5507f5695b077d2cc77d02b`,
with pre-existing dirty Rust/research files preserved; no Ember code is used for
this training prototype. The installed Qwen runtime source is also frozen.
All17 contract tests pass ([JUnit record](tests.xml)); new audit/profile helpers
also passed syntax checks and their real-data runtime assertions. No Rust code
was changed, so no Rust fmt/clippy claim is made for the pre-existing dirty tree.
Implementation: [bounded reader/resampler](channel_v2.py),
[inherited orchestration](../latent_channel_v1_20260906/channel.py),
[trained bridge checkpoint](run_full/best.pt). Existing run commands intentionally
refuse overwrite; do not rerun training/evaluation into these frozen directories.

## Limitations to carry into the final interpretation

The frozen sender may merely preserve its last token, while the bridge learns
symbol classification and a four-code output interface. This is compatible with
a real minimal digital channel; it does not establish transfer of sender
reasoning, natural semantics, Arabic capability or general-purpose communication.
Because Q is constant, the reader need not learn query-dependent retrieval.
Neither sender reasoning nor the necessity of the receiver's deeper computation
has been isolated by ablation. These limitations restrict the claim; they do not
give the receiver the varying symbol without the sender channel.

Reader changes are large: mean delta/native-residual norm ratio~2.06, mean
cosine(h,h+delta)~.533. Random transport slots have the same input norms but induce
smaller receiver deltas (~.28 mean ratio). Shuffled memories preserve the correct
distribution of reader energy, so their semantic contrast is critical. Do not
claim gentle modulation or receiver-delta-energy-matched random controls.

Strong text solves FINAL perfectly. The poor original plain-completion
text result is a prompt-format failure, not evidence of a latent advantage.
No controlled deployment latency comparison, full-training-seed replication, minimal-K
ablation, cross-family, natural-task, Arabic or free-generation result exists.
Bit-exact disabled/zero checks cover this experimental input, not a broad coding,
tool-use or English capability suite.

## Compute

Local i5-1135G7 CPU,4 torch threads,16GiB host; sequential model processes.
Measured sender transient peak~12.3GiB. Full training including DEV/checks took
1903.29s. TRAIN capture685.67s, DEV capture176.52s; DEV actual rerun194.34s;
strong DEV text371.81s. Final sender capture570.49s; core receiver scoring and
checks821.75s; stronger final text1444.11s (sender forwards1423.95s; five unique
receiver forwards1.85s, excluding their exact rerun). The sender reproduction
took46.19s. These phase timings include differing amounts of load/hash/setup;
the stage-level accounting is preserved separately in job state files.

Warm bridge component medians at batch8: resampler.185ms/example,
projector.016ms, reader.115ms, combined.344ms. Five warmups and20 repetitions
over the fixed first32 source states and an actual stock receiver state; includes
the implementation's diagnostic work, excludes endpoint inference, loading,
serialization and transport ([profile](bridge_profile.json)). This is not an
end-to-end latency measurement. Correct final receiver inference took90.77s
for1,024 examples, baseline92.77s, plain text193.40s; their sequential difference
is not an overhead estimate.

Latent prompts:22 sender tokens/8 receiver tokens. Strong text prompts:61/35,
including the fixed chat scaffold; transmitted symbol1 UTF-8 byte versus2048
latent bytes. Text receiver caching is explicit and must not be confused with
1,024 measured receiver forwards. Prompt lengths, batching and caching differ;
these measurements establish neither latent latency superiority nor text latency
superiority under a matched deployment protocol.

Recorded V2 job wall-time subtotal:6,835.98s (1.899h), including the sender
reproduction and component profile. Adding V1's sealed reported838.13s subtotal
and approximately374s sender download gives~2.236h recorded local job/download
wall time for V1+V2. This excludes some audits/tests/orchestration, the initial
failed pre-training exporter, and the older Arabic branch; it is not an exact
grand total or processor CPU-hour count. No overlapping parent wait or nested
condition timing is double-counted ([ledger](compute_ledger.json)).
Thermal variability and different batching/cache policies prohibit a speed claim.
External/cloud spend **USD0**, ceiling approximatelyUSD30. No cloud resources
were created. Electricity and assistant-service billing are not measured here.

## Decision and next boundary

**A — LATENT COMMUNICATION CHANNEL ESTABLISHED**, limited to this held-out
same-family synthetic two-bit task. Changing sender-only symbol content changes
the frozen receiver's selected symbol; destroying or mismatching that content
removes target accuracy. Strong text is equally accurate and has a much smaller
message. The latent channel is functional, not shown advantageous.

No K/layer ablations, cross-family model, natural task, Arabic task, general-
capability suite or free-generation experiment was run. The next scientific gate
is the same synthetic task across model families, with a separately frozen
bridge and fresh heldout data. No later gate is licensed by this result alone.
The final local integrity inventory is [seal.json](seal.json); it rehashes both
endpoint snapshots, all V1 sealed files, and the completed V2 artifact payload.
