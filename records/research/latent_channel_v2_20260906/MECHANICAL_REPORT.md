# Bounded-attention repair — local mechanical gate

**Gates0–2 PASS. Held-out channel NOT YET ESTABLISHED.**

V1 remains a frozen NO-GO. V2 is a second, disclosed configuration addressing
the specific attention-saturation failure found in V1, not a reinterpretation
of that failed experiment. No larger model, adapter, layer sweep or alpha sweep.

## Hypothesis and implementation

V1's reader put all attention on two answer-independent slots. Their resamplers
attended the constant `Signal` marker before the varying symbol. Other slots
retained symbol information, but the receiver ignored them. V2 replaces both
bridge attention scores with `2*cosine(q,k)` (normalization eps1e-8), bounding
scores and preventing numerically zero attention probabilities. This maintains
a trainable information path; whether it improves held-out communication remains
to be measured.

```
Frozen Qwen3-1.7B source states, block13, 22x2048
 -> trained bounded-attention resampler, 8 queries
 -> fixed-RMS latent channel, 8x64 float32 (2048 bytes)
 -> trained 64-to-1024 projection
 -> one trained bounded cross-attention reader, receiver block13
 -> scalar-gated context + native residual
 -> frozen Qwen3-0.6B continuation / original LM head
```

590,337 trainable parameters, 0.099% of 596,049,920 receiver parameters.
Endpoint models, embeddings, tokenizers and LM heads frozen. Both sites are
zero-based block13. No Ember code changes; inherited endpoint inference and
orchestration remain those of the pinned V1 Python implementation.

## Data, optimization, decision rule

Fresh tiny seed11002: eight distinct nuisance-digit records, each with all four
counterfactual symbols A/B/C/D. Source contains `Signal: X`; receiver sees only
the constant `Hidden signal (A/B/C/D):`. IDs are opaque shuffled row indices;
source token count is22 for every class; query count8; full continuations each
one token. No raw source is passed to the latent receiver.

Receiver-only must be exactly25% by construction. Every symbol has eight
examples. Tiny nonces are disjoint from V1; TRAIN/DEV/final seeds are separately
registered. No tiny result is a scientific held-out estimate.

Full-vocabulary answer-token CE, AdamW lr.003, weight_decay0, clip1, batch8,
seed51001, maximum300 steps, evaluate every20. Same optimization as V1; only
attention scoring and fresh nuisance data changed. Stop at first check with
correct>=31/32, shuffle and both random controls<=50%, zero==baseline==25%,
then require an actual all-condition exact rerun and frozen-weight hashes.

## Observed results

The gate passed at step160, not before. Training was non-monotonic: correct
accuracy reached78.125% at80, fell to25% at120, recovered to62.5% at140, then
100% at160. All checkpoints' evaluations are retained; no stability or
hyperparameter-robustness claim is made from this single repaired run.

| Condition | Accuracy | A/B/C/D recall | Full-vocabulary CE |
|---|---:|---|---:|
| Receiver-only | 25% | 100/0/0/0% | 5.296556 |
| Correct latent | 100% | 100/100/100/100% | 0.000988 |
| Shuffled latent | 31.25% | 50/25/25/25% | 9.709389 |
| Random71001 | 25% | 62.5/0/0/37.5% | 5.695503 |
| Random71002 | 18.75% | 37.5/0/0/37.5% | 6.049250 |
| Zero latent | 25% | 100/0/0/0% | 5.296556 |
| Original plain text relay | 3.125% | 12.5/0/0/0% | 4.753271 |

Correct also had100% unconstrained next-token argmax accuracy at this checkpoint.
This is still a next-token diagnostic, not a free-generation or held-out claim.
Shuffled donors are a label-blind derangement, not adversarially selected to
be wrong. Random slots are independently norm-matched. Core measurements:
[step160](run_tiny/eval_0160/metrics.json),
[per-example records](run_tiny/eval_0160/per_example.json),
[complete training curve](run_tiny/curve.jsonl).

## Text baseline repair (separate control, unchanged latent experiment)

The poor plain-completion relay is not a competent text baseline. Before running
a stronger control, one official non-thinking chat format was preregistered.
The actual frozen sender chooses a symbol using forced-choice scoring and passes
that one-character text to the frozen receiver; no free generation is used.
An oracle text control parses the same symbol directly from the visible record,
without consulting hidden gold metadata. Receiver-only chat withholds the record.

| Strong text control | Tiny accuracy |
|---|---:|
| Frozen sender symbol selection | 100% |
| Sender-selected text -> frozen receiver | 100% |
| Oracle compact text -> frozen receiver | 100% |
| Receiver chat without source | 25% |

All classes scored100% in both text handoffs. The four unique compact receiver
prompts were evaluated and exactly rerun, then explicitly cached and joined to
each example. Sender inference ran on every distinct source. This is a genuine
model-mediated text condition plus an oracle upper reference. No claim that
latent communication is better than text is supported by these results.
[Protocol](TEXT_BASELINE_PREREGISTRATION.md), [results](text_tiny/summary.json),
[per-example joins](text_tiny/per_example.json).

## Verification, uncertainty and limitations

Three unit tests passed: attention lower bound/extreme-score gradient behavior,
zero/norm/parameter/gradient contracts, and split/counterfactual leakage checks.
Real-model Gate0 passed: stock/disabled/zero logits bit-exact, nonzero memory
changes logits, finite nonzero bridge gradients, no endpoint gradients, exact
nonzero repeat. Step160 all-condition rerun matched log probabilities and
unconstrained token IDs exactly. Full in-memory receiver parameter hash before
and after training matched:
`2b7e3b5c1515ecd63a512b9c7464ce1e91ca4b766a6af1b93dd93b793f79d0d1`.

Joint prompt+continuation tokenization was independently checked against the
actual scored single-token answers for source, latent query and plain text.
V1's82 sealed files still match their hashes. These checks rule out several
mechanical failures, not tiny-set memorization. No training-from-initialization
rerun or independent seed replication has been completed.

No held-out bootstrap inference is claimed from this trained-on set. The final
decision uses the preregistered nonce-group paired intervals and all-class gates.
The most important remaining alternative explanation is tiny-set overfitting;
the strongest methodological limitation is the minimal two-bit copy task.
Cross-family, natural transfer, Arabic and generation are not yet licensed.

## Provenance and compute

Model revisions remain Qwen3-1.7B
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` and Qwen3-0.6B
`c1899de289a04d12100db370d81485cdf75e47ca`, float32 CPU, no quantization.
The inherited source hash alone is insufficient to describe V2; both hashes
are recorded in [extension provenance](train_tiny_extension.json):

- V1 inherited channel: `e2de36c051ba503b441c2d84b959728cb09f8d064791012b5f673462039f6fbd`.
- V2 extension: `ee773ac60f7068f3e350ce60120f5f5bed268beece1e9fd7793a48fcb57eaf29`.
- V2 protocol: `eb1be3758008f3debddcc7cf007d5fa7ba4dba215027166eefb7b1f21094246e`.
- Last checkpoint: `62f64ac46d096b303407cf5f480f5c3b424fe72b159e50e34d9392a223712b46`.

Tiny training plus checks took447.96s; sender collection approximately23.75s.
Strong-text timing is recorded separately, including actual sender inference
and the explicitly cached receiver stage. No isolated deployment latency or
text-efficiency advantage is claimed. Local CPU hardware/resource estimates
are in [LOCAL_RESOURCE_PLAN.md](LOCAL_RESOURCE_PLAN.md).

**Cumulative external/cloud spend: USD0, ceiling approximatelyUSD30.**

## Decision and ongoing work

**Gates0–2 PASS; proceed to previously registered TRAIN/DEV.**
The goal remains active. A local sequential job is collecting TRAIN1024, then
DEV256, then starts fresh full training (seed52001, not the tiny checkpoint).
It does not automatically construct or evaluate final data. Final seed44002
remains reserved and untouched. No channel-established verdict is licensed yet.
