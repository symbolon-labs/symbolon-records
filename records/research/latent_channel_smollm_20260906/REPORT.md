# Qwen → SmolLM2 cross-family synthetic latent channel

**A — LATENT COMMUNICATION CHANNEL ESTABLISHED. Cross-family Gate 4 PASS.**

Frozen Qwen3-1.7B → frozen SmolLM2-360M-Instruct, same held-out four-symbol
task and bounded bridge: correct latent 100%; baseline/zero 25%; shuffled
26.37%; random 25.49%/25.68%. Donor swaps follow the donor symbol 100%, with
exactly matching donor probability vectors. Every registered statistical and
mechanical check passed, as did independent post-primary audits.

Actual sender-text and oracle-text also scored 100%: **no accuracy advantage
over strong text**. This establishes a cross-family synthetic two-bit channel,
not natural-language capability transfer, Arabic, or free generation.

## Hypothesis and chronology

A bounded learned resampler/reader can communicate a sender-only four-way
symbol to a frozen non-Qwen receiver. Endpoint weights remain frozen.

The earlier Arabic residual/prefix branch failed content-specific robust
bidirectional control. It is not reopened. Explicit-reader V1 failed tiny
training; bounded-attention V2 then passed the same-family synthetic gate
(Qwen3-1.7B → Qwen3-0.6B: final correct100%, baseline25%, shuffled25.49%,
random25%/25.10%, zero25%; strong actual text100%).

The first cross-family attempt, Qwen → Llama-3.2-1B-Instruct, failed tiny
training (terminal25%, best evaluated28.125%). Llama also failed the exact
oracle-text sanity task at25%, including singleton/order/scoring checks.
That failure cannot isolate family mismatch from receiver elicitation or
optimization. It remains sealed, not erased by another receiver's outcome.

The user then separately authorized receiver qualification. One new candidate,
SmolLM2-360M-Instruct, passed all four exact oracle messages and donor-following
checks with frozen weights and reproducible scoring. No prompt tuning occurred.
Qualification alone is not latent-channel evidence. The user authorized this
fresh bridge run after qualification. All four historical artifact seals were
independently rehashed before this run.

## Exact architecture

```text
S = Noise: <eight random digits> / Signal: X
  frozen Qwen3-1.7B, block13 output [22 × 2048]
  learned 8-query resampler, width64, attention 2*cosine(q,k)
  fixed per-slot L2 norm8
  ======== 8 × 64 float32 latent channel ========
  bias-free learned 64 → 960 projector
Q = Hidden signal (A/B/C/D):
  frozen SmolLM2-360M-Instruct through block15
  one width64 cross-attention reader + learned scalar gate
  remaining frozen blocks and original LM head
  full-vocabulary answer-token probabilities
```

Block indices are zero-based. Receiver insertion follows the unchanged middle
rule floor(32/2)-1; there was no layer search. Both bridge attentions use bounded
2*cosine(q,k), eps1e-8, non-affine query normalization, and bias-free projections.
Gate initialization .01. Reader modifies all receiver-query positions at one
site. Trainable bridge parameters569,857 / receiver361,821,120 (0.1575%).
No prefix, external classifier, endpoint fine-tuning, auxiliary loss, dropout,
or Ember Rust/inference change. Native eager float32 CPU execution.

Sender Qwen/Qwen3-1.7B revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
Receiver HuggingFaceTB/SmolLM2-360M-Instruct revision
`a10cc1512eabd3dde888204e902eca88bddb4951`.
Receiver weight SHA256
`e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e`.
Full model/tokenizer identities are recorded in acquisition.json and manifests.

Transport2,048 bytes/example; sender capture180,224 bytes/example;88:1
activation compression; expanded receiver memory30,720 bytes. The task contains
only two information bits. This is not efficient source coding: a text character
is much smaller. K8 is fixed, not a demonstrated minimum.

## Data and scoring

Every nonce occurs in all four counterfactual classes A–D. Q is identical for
every example and carries no source, nonce, ID, label, or variable padding.
Therefore a deterministic receiver seeing only Q obtains exactly25%.
The sender-only Signal field determines the answer. This tests a fixed alphabet
across fresh nuisance records, not new symbol identities or compositional tasks.

Tiny11004:32 examples/8 groups; TRAIN22004:1024/256; DEV33004:256/64;
FINAL44004:1024/256. All partitions exclude each other and materialized prior
V1/V2/Llama nonces. Only tiny was instantiated before tiny gates passed.
FINAL construction requires actual selected-DEV reproduction and an immutable
checkpoint/code/model/scoring freeze.

Source length22 tokens; plain receiver Q12 tokens. Full continuations
` A`, ` B`, ` C`, ` D` are one token each (330/389/340/422); joint tokenization
boundaries are asserted. Training uses full-vocabulary cross-entropy, not
four-label-renormalized loss. All evaluations are forced-choice; no free
generation. Unconstrained next-token argmax is only a diagnostic.

Controls: receiver-only, correct, label-blind derangement shuffle61001,
per-slot norm-matched Gaussian memories71001/71002, zero, original plain text,
actual sender-selected compact text, oracle compact text, and missing-source
chat. Shuffling can match labels by chance. Strong text uses the unchanged
instruction bodies with native non-thinking chat templates and fixed date.
Its four unique receiver messages are explicitly cached, not treated as
independent repeated receiver computations.

## Preregistered training and decision

Fresh tiny initialization51001/minibatches51002; fresh full initialization52001/
minibatches52002. AdamW lr.003, weight_decay0, clip1. Tiny batch8, max300,
checks20. Full batch16, max600, DEV50, early stop after three checks without
>=.001 CE improvement. Select minimum trained DEV CE, earliest tie, independently
recomputed from stored tensors. No final-based repair or fallback checkpoint.

Tiny requires correct>=95%, shuffled/random<=50%, zero/base25%, actual exact
rerun, parity, finite nonzero bridge gradients, absent endpoint gradients and
unchanged receiver hash. Tiny passed at160: correct100%, shuffled15.625%,
random/base/zero25%, all-condition exact rerun; actual sender/oracle text100%.
These are mechanical training-set results only.

Selected DEV must reproduce exactly and meet correct>=90%, each class>=85%,
baseline/zero25%, randomized<=35%, and the mandatory oracle-text check.
FINAL repeats those point thresholds and requires paired99% nonce-group
bootstrap lower bounds for correct-minus-each non-text control >+.50.
Randomized-minus-baseline paired95% intervals must lie wholly within±.10.
10,000 group draws, seed81001. Require exact donor probability following,
reconstructed memories/random norms, zero/stock parity, actual32 receiver rerun
using original full-set memories, actual32 sender capture rerun, frozen hashes,
and text comparison. No result is PASS unless every required check passes.

## Results, diagnostics, compute and final decision

Full training stopped at400 by the registered rule; step400 also had the
minimum DEV loss. Receiver final parameter SHA256 matches its initial and
independently qualified hash:
`6a3e1782ebad58d1e9323d5eeaff191d112848b5f0058394a998d1ac54c492b3`.
Selected bridge file SHA256:
`e75da6bfc6dc180b973696de763f944b83396e4d966e783c91b00343c9cd4170`.

| Step | DEV correct | DEV CE | DEV shuffled | DEV random1 / random2 |
|---|---:|---:|---:|---:|
|0|25%|5.327746|25%|25% /25%|
|50|33.20%|1.390052|26.17%|25% /25%|
|100|38.28%|.730314|24.61%|25.39% /25%|
|150|99.61%|.057845|25%|25.39% /24.22%|
|200|81.25%|1.006632|24.22%|23.83% /24.61%|
|250|100%|.00022711|24.61%|25.78% /23.83%|
|300|100%|.00003787|24.61%|25.78% /23.44%|
|350|100%|.00002890|24.61%|25.78% /23.44%|
|400|100%|.00002347|24.61%|25.78% /23.44%|

The step200 regression is retained, not smoothed away. These are repeatedly
inspected DEV outcomes, not the final evidence. Complete raw training steps,
gradient norms and scalar-gate values are in run_full/curve.jsonl; every DEV
evaluation is retained. The plotted curve and CSV contain all observed checks.

The selected checkpoint reproduced all seven DEV conditions exactly using
actual inference, with unchanged receiver parameters and stock/disabled/zero
full-vocabulary parity. Strong DEV actual sender-text and oracle-text were100%;
missing-source chat25%. The final freeze was written at
2026-09-06T07:12:19Z, before final construction/capture began at07:12:20Z.
These local records are integrity evidence, not external timestamp attestations.

The one-shot FINAL evaluation completed on1,024 examples/256 groups. The
formal decision passed all mandatory text, sender rerun and integrity checks.
Actual first32 receiver rerun
matched bit-exactly using the original full-set control memories. Stock,
bridge-disabled and zero-memory full-vocabulary logits matched exactly.

| Core condition | FINAL accuracy | A / B / C / D recall |
|---|---:|---|
|Receiver-only|25%|100 /0 /0 /0%|
|Correct latent|100%|100 /100 /100 /100%|
|Shuffled latent|26.37%|25.39 /26.17 /26.95 /26.95%|
|Random71001|25.49%|92.19 /0 /4.69 /5.08%|
|Random71002|25.68%|93.36 /0 /7.81 /1.56%|
|Zero latent|25%|100 /0 /0 /0%|
|Original plain text|70.90%|100 /16.41 /86.33 /80.86%|
|Actual sender → compact text → receiver|100%|100 /100 /100 /100%|
|Oracle compact text → receiver|100%|100 /100 /100 /100%|
|Receiver chat without source|25%|100 /0 /0 /0%|

### Paired uncertainty and semantic specificity

Independent paired 99% confidence intervals for correct-minus-control accuracy
gain, in percentage points: baseline/zero [+75,+75]; shuffled
[+70.117,+77.149]; random1 [+72.754,+76.270]; random2 [+72.754,+75.879].
Every lower bound exceeds the preregistered +50pp threshold. Shuffled accuracy
is 26.367% (95% CI [23.730,29.102]); both random controls remain near chance.
Randomized-minus-baseline paired 95% intervals: shuffled [−1.270,+4.102]pp;
random1 [−.781,+1.855]pp; random2 [−.488,+1.855]pp, all within the registered
±10pp equivalence band.

All 256 examples in each class are correct under matching latent. The bootstrap
interval at 100% collapses because no errors were observed; this is not a
population guarantee. Resampling uses 256 nonce groups containing all four
counterfactual labels, not 1,024 independent draws. Complete per-class and margin
uncertainty is retained in [analysis.json](run_final/analysis.json).

Every shuffled log-probability vector and unconstrained token equals the matching
donor's correct output exactly. Donor-symbol following is 100%; accuracy against
the recipient's original symbol drops to chance. This is stronger evidence than
an output change or correct-over-baseline gain. The shuffle preserves the entire
real memory/reader-energy distribution while breaking its association with the
recipient label. Random controls largely revert to the receiver's A default.
Zero memory is exactly receiver-only, not merely equal in aggregate accuracy.

| Condition | Mean toward-answer margin | Median margin | Full-vocabulary CE |
|---|---:|---:|---:|
|Receiver-only / zero|−1.5764|−2.3757|5.327631|
|Correct latent|12.3578|12.5619|.00002344|
|Shuffled latent|−8.7807|−16.1630|12.042940|
|Random1|−1.8263|−3.1495|6.147821|
|Random2|−1.8233|−2.9891|6.096479|
|Original plain text|.3079|.1194|3.462420|
|Actual sender-text / oracle text|1.1520|.9356|1.291071|
|Receiver chat without source|−.8487|−1.2764|2.915272|

Margins and losses are nats. Different prompt formats make cross-prompt CE an
inappropriate efficiency comparison. Correct unconstrained next-token argmax
is also100%, but no continuation was decoded: this is not a generation result.

### Norms and reader response

Correct transport slot norms range 7.99999905–8.00000095, mean8 in every class;
random norms match within1e-5, zero is exactlyzero. Expanded correct-memory
mean norms by class are54.38/71.35/49.50/46.49; random means are approximately
22.1. Thus random is norm-matched at the **transport**, not after the learned
projection or at the residual update. Correct-versus-random alone cannot rule
out this downstream energy difference. The distribution-preserving shuffle and
exact donor-following result are the stronger specificity controls.

Correct reader attention entropy averages1.9303 nats (uniform8-way=2.0794);
the minimum observed attention probability is.003569, maximum.659624. Learned
gate=.062198. The update is substantial: mean delta/residual norm ratio1.1090,
range.3767–2.0767; mean cosine(h,h')=.69156, range.42491–.93621. Random updates
average only.1121/.1086 of the residual. Zero update is exactzero. Correct and
shuffled aggregate diagnostics are identical because shuffling permutes memories.
No claim of gentle modulation, minimal sufficient capacity, or capability
preservation follows. Full distributions/per-class diagnostics are in
[final_reader_diagnostics.json](final_reader_diagnostics.json).

### Determinism, provenance and leakage audit

Actual selected-DEV all-condition rerun and final first32 all-condition receiver
rerun are bit-exact. Final first32 native sender captures are bit-exact with
maximum absolute difference0; sender parameter hash before/after is
`deb5e69d2fcd24d139ce2927f8375c2c445867b196bc179df4688d98990e0ec4`.
Receiver hashes match both training and the independent prior qualification.
All memories and random/shuffle donors were rebuilt exactly from the selected
checkpoint and saved source captures. No inference cache substitutes for these
actual reruns. Determinism here means fixed implementation/hardware/batches;
this is not an independent training-seed replication or platform-invariance claim.

The independent input audit re-tokenized every source in all four splits,
verified captured IDs, fixed lengths, joint label boundaries, constant receiver
query, class balance, unique artifact IDs, and exclusion of every materialized
prior nonce. Receiver inference receives only Q plus memory; IDs/labels/nonces
are artifact joins or loss targets, never receiver input. Training order is
seed-shuffled; no batch-shared normalization or cross-example attention exists.
Final construction followed freeze, and the final set is now burned for future
development. Alphabet identities deliberately repeat; only nonce records are
held out. See [input audit](postfreeze_input_audit.json).

Model files, tokenizers, exact revisions, frozen source/runtime components,
checkpoint and data hashes were independently re-derived at closure. All389
historical files in the V1/V2/Llama/qualification seals remain unchanged.
Independent bootstrap calculations agree exactly with the primary analysis;
see [VERIFICATION.json](VERIFICATION.json) and
[independent_core_audit.json](independent_core_audit.json).
Seven pretraining structural/guard tests passed. No Rust code changed, so this
run did not rerun unrelated Rust fmt/clippy/tests. The pre-existing dirty worktree
was preserved. No push or commit was made. Local seal.json is an integrity
inventory, not an externally timestamped preregistration.

### Compute and text comparison

No cloud provisioning: external spend USD0, ceiling approximatelyUSD30.
Local i5-1135G7/16GiB, no CUDA,4 intra-op/1 inter-op threads, sequential endpoint
loads. Python3.14.7, torch2.13.0+cu130, transformers5.10.2, numpy2.4.6.

Measured sequential job wall time for this bridge run is6,962.67s
(116.04min), excluding qualification/prior runs, setup, between-job gaps,
independent audits and reporting. Tiny capture/train365.39s; TRAIN/DEV source
capture832.42s; full training including DEV checks1,942.07s. Confirmation
included DEV repeat161.92s, DEV text433.43s, final capture584.82s, final core
scoring/repeat750.12s, final strong text1,762.11s, sender repeat42.67s.
The cumulative child peak was12,756,428KiB (12.17GiB); do not attribute this
queue-wide high-water mark to the receiver alone. Electricity cost unmetered.
The preregistered confirmation estimate15–30min was too low: actual62.63min.

Final source capture manifest elapsed580.33s includes loading/hashing/capture;
correct receiver-plus-reader condition91.31s/1,024 examples (~89.2ms/example,
batch8). Receiver-only99.11s, shuffled67.41s, zero72.44s, plain-text241.30s.
These sequential, thermally variable timings are not paired benchmarks; the
faster later conditions are not evidence of negative bridge overhead.
Resampler/projector/reader overhead was not separately isolated. It is included
in capture-independent preparation and receiver timings; do not infer it by
subtracting unmatched conditions or claim an end-to-end deployment latency.

Strong text sender inference1,741.57s over1,024 examples, receiver unique-message
inference1.788s for four compact messages, excluding its deterministic rerun.
Sender chat61 tokens, receiver compact chat56 tokens; relay content is one
symbol/one byte/one answer token. Latent Q12 tokens and memory2,048 bytes;
original plain-text relay35 receiver tokens. Strong text sender scores the
four alternatives and relays its own selected symbol, not a metadata label;
oracle is separately parsed from source. Neither text nor latent uses free
generation. Cached receiver text work cannot be compared as1,024 independent
end-to-end calls. Latent and strong text tie in accuracy; no efficiency advantage
has been established, and text is clearly smaller as a transport for this task.

### Failure analysis, limits and decision

The strongest remaining interpretation is a learned
two-bit codebook conveyed through hidden states. It would not establish richer
semantics, query-dependent retrieval, Arabic specialization, coding preservation,
or a latency/accuracy advantage over text. Those stages are not run here.
The sender sees an explicit Signal character; no embedding-only or sender-layer
ablation was run. Thus this does not show that deep sender computation or its
language specialization is necessary. The endpoints also share the broad
decoder-only Transformer design; distinct pretrained families do not mean
fundamentally unrelated computational architectures.

The architecture's success on this receiver does not rescue the failed Llama
configuration, prove the original residual-coordinate theory, or show that
family mismatch never matters. Receiver qualification was a disclosed selection
stage, not an untouched random receiver sample. Only one new qualified candidate
and one fresh tiny/full bridge configuration were trained here. Same-family,
failed cross-family and this successful cross-family work all remain reported.
The transient step200 DEV regression also limits optimization-robustness claims.

**PASS: content-specific cross-family synthetic latent communication.**
Every required destructive control remains at chance and donor substitution
changes the predicted symbol to the donor's. No task/architecture tuning on
FINAL occurred. No bottleneck/layer/gate ablations, natural task, Arabic task,
coding/agentic preservation suite or downstream generation was run. The next
scientific gate is a separately preregistered cross-family natural semantic task
with sender-only information; it requires new data and authorization, not reuse
of this final set. Exact verdict: **A — LATENT COMMUNICATION CHANNEL ESTABLISHED**.
