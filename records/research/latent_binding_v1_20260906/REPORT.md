# Randomized binding V1 — experiment record

Status: **COMPLETE — MECHANICS/TRAINING NO-GO (mechanics PASS, training NO-GO).**
This is the post-result report. The decision rule and
initial implementation were frozen separately before inference. See
`PREREGISTRATION.md`, `TEXT_PROTOCOL.md`, `acquisition.json`, and `initial_code/`.

## Question and inherited evidence

Can the already successful heterogeneous two-bit channel transmit a queried
binding between independently varying identities and colors, including whole
identifiers not encountered in bridge-training sources or targets?

The sealed predecessor is `../latent_channel_smollm_20260906/`. Its scope is
sender-only four-symbol communication, not binding or natural-language
capability transfer. That result and its unsuccessful antecedents remain
unchanged; `../latent_communication_dossier_20260906/` documents the chronology.
The initial binding preflight independently reverified historical seals and
both model snapshots. Neither earlier Arabic failures nor the two-bit PASS
licenses a binding claim.

## Architecture — unchanged

```text
Four sender-only bindings (19 tokens; receiver query absent)
    ↓ frozen Qwen3-1.7B, block 13 output [19, 2048]
Exact inherited learned-query bounded-attention resampler
    ↓ [8, 64] latent memory, slot RMS 1
Exact inherited bias-free 64 → 960 projection
    ↓ one bounded cross-attention reader after receiver block 15
Frozen SmolLM2-360M-Instruct continuation
    ↑ receiver input: “Which ID is <color>?” only
128 globally fixed ID candidates, both token log probabilities summed
```

Model revisions are Qwen `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` and
SmolLM2 `a10cc1512eabd3dde888204e902eca88bddb4951`. Layer indices are zero-based.
The bridge is the identical imported Python class, freshly initialized; no
prior bridge checkpoint is loaded. Both bridge attention operations use
`2*cosine(q,k)`; the single scalar reader gate starts at 0.01. There are
569,857 trainable parameters. Endpoint parameters, embeddings, LM heads,
tokenizers, and normal transformer blocks remain frozen.

Transport is 2,048 bytes/example in float32. The captured source tensor is
155,648 bytes/example, a 76:1 tensor-byte compression ratio. The receiver-side
expanded memory is 30,720 bytes. These are tensor sizes, not wire-protocol,
end-to-end memory, latency, or efficiency advantages over text.

No identity lane, lexical anchors, curriculum, extra bridge depth, wider
bottleneck, different insertion site, or new inference engine is introduced.
The task mechanically needs two-token answer scoring; the native forward
wrapper exposes the last two logits. This is tested against direct scoring,
not assumed equivalent. All execution is PyTorch native eager float32 on CPU;
Ember Rust inference is not modified.

## Data and absence of receiver-side information

The identifier universe is A–P × 0–7: 128 two-character strings. A fixed
component-balanced partition reserves 64 whole strings for the novel-ID arm.
Both halves contain every letter and digit; novel whole IDs are not novel
tokens and are not claimed absent from endpoint pretraining.

Every source has four distinct IDs, one per color (red, green, blue, yellow),
with randomized row order. Each source is queried for all four colors. The
receiver sees only its five-token query. It never receives the source table,
its four IDs as a shortlist, the example ID, or the target. The scoring
candidate bank is global and unchanged across examples. Candidate first-token
prefixes are ordinary likelihood conditioning, not gold information supplied
to inference; every candidate receives the same scoring treatment.

Only the tiny split was materialized: seed 121001, eight tables,
32 queries, and 32 distinct known-pool IDs. It is deliberately for overfit,
not generalization. Later registered splits are TRAIN 122001 (256 tables),
DEV 123001 (64), FINAL-known 124001 (64), and FINAL-novel 125001 (128).
TRAIN/DEV require the tiny gate; final data require selected-checkpoint and
decision-rule freeze. In each later balanced 64-table round, each pool ID is
queried once per color. Thus a deterministic query-only predictor has at most
1/64 accuracy, possibly zero if its preferred ID is outside that arm's pool.
Uniform guessing across the global bank is 1/128. Those two reference rates
must not be conflated.

## Controls and interpretation

- Correct memory: matching source, with the query absent from sender input.
- Shuffled memory: another table, literally identical receiver query. Report
  both accuracy against the original target and following of the donor answer.
- Wrong binding: a separately encoded counterfactual source with identical IDs
  and row order, but cyclically reassigned colors. Every queried answer changes.
  This separates identity availability from retrieving the correct binding.
- Random memory: two independent Gaussian seeds, each slot norm-matched and
  shared across a table's four queries. Postprojection/residual energy is not
  claimed matched; response norms are diagnostic outputs.
- Zero memory and receiver-only: actual native paths, requiring exact parity.
- Actual sender→text: sender scores its answer from the real source and query;
  receiver gets that predicted ID in a fixed compact copy instruction.
- Oracle text: copy the correct ID under the same receiver instruction. This
  is an output-task qualification control, not the actual sender's result.

No free generation is performed. Text scores also sum both full-vocabulary
token log probabilities across the same 128 IDs. No prompt variants are tried.

## Frozen gates and selection

First, the receiver must copy oracle IDs at ≥90% overall and ≥75% per color.
Failure stops before optimization. Then require stock/disabled/zero full-logit
parity, active effect, deterministic repeat, direct-scoring agreement within
1e-4 with identical argmax, candidate-order invariance, finite nonzero bridge
gradients, and no endpoint gradients.

Tiny training: AdamW learning rate 0.003, weight decay 0, clip 1, batch 8,
300 steps maximum, evaluate every 20, seeds 51001/51002. Correct must reach
≥95%; shuffled donor following ≥95%; original-target shuffled/random ≤25%;
baseline ≤12.5%; zero parity and an actual terminal rerun must hold. Tiny
counterfactual performance is descriptive because those sources are untrained;
it is mandatory at DEV/final. There is no step extension after failure.

Only a tiny PASS licenses fresh full training and construction of TRAIN/DEV.
The full optimizer, budget, checkpoint-selection and final readiness rules are
already specified in the preregistration. Only after DEV readiness may either
final dataset be created. Both final arms must independently pass accuracy,
donor-counterfactual, destructive-control and paired uncertainty requirements.
Known-ID performance cannot substitute for novel-ID performance.

## Measurements and audit trail

The driver is `job.py`; current state and per-stage logs are under `tiny_job/`.
Source tensors, original/counterfactual inputs, hashes and metadata are under
`data/tiny/`. Text prompts, token IDs, per-candidate two-token scores and joins
are under `text_tiny/`. The completed `run_tiny/` holds
mechanics, every-step learning curve, all evaluation arms, reader diagnostics,
checkpoint hashes and actual terminal repeat.

`audit.py` independently parses tables, recomputes predictions from saved
scores, rebuilds donor joins, checks source/payload/code hashes and verifies
control parity. It does not import the experiment implementation or run models.
Its status describes artifact consistency, never scientific success.

## Results, uncertainty, failure analysis, and decision

Text and mechanical qualification completed before optimization:

| Tiny qualification, 32 queries | Accuracy |
|---|---:|
| Actual Qwen sender answer | 31/32 (96.875%) |
| Actual sender → compact text → receiver | 31/32 (96.875%) |
| Oracle ID → compact text → receiver | 32/32 (100%) |

Oracle copy is 8/8 for each queried color. The sender's sole wrong answer was
K2 → O2 (`tiny-000019`); the receiver copied the sender's erroneous ID rather
than repairing it. These are forced-choice tiny-set measurements, not
held-out estimates or generation results. The sender has source plus query
in the text arm; latent source extraction has the source only.

Both native scoring checks passed: maximum direct-versus-prefix score gaps
1.1444091796875e-5 (sender) and 3.814697265625e-6 (receiver), below the fixed
1e-4 tolerance, with identical argmax and exact same-path reruns. All distinct
receiver compact messages were actually rescored and matched exactly.

The bridge mechanical check also passed: stock/disabled/zero full logits are
bit-exact, enabling the fresh bridge changes logits and repeats exactly,
candidate ordering is invariant, and all nine bridge parameter tensors have
finite nonzero gradients while endpoint gradients remain absent. Direct
latent-path scoring differs by at most 3.814697265625e-6. This licenses the
tiny training attempt, not a communication claim.

### Fixed 300-step training attempt

The tiny accuracy gate failed. No evaluated checkpoint reached the required
≥95% (at least 31/32). The best mean full-ID training NLL and highest recorded
accuracy both occurred at step 240: 14/32. Terminal step 300 regressed to 2/32.
There was no architecture retry, optimizer change, added training, or later
split construction.

| Condition / target | Step 240, best NLL | Step 300, terminal |
|---|---:|---:|
| Receiver-only | 1/32 (3.125%) | 1/32 (3.125%) |
| Correct latent | 14/32 (43.75%) | 2/32 (6.25%) |
| Shuffled, original target | 0/32 (0%) | 1/32 (3.125%) |
| Wrong binding, original target | 13/32 (40.625%) | 2/32 (6.25%) |
| Random 71001 | 1/32 (3.125%) | 1/32 (3.125%) |
| Random 71002 | 1/32 (3.125%) | 1/32 (3.125%) |
| Zero latent | 1/32 (3.125%) | 1/32 (3.125%) |
| Shuffled, donor answer | 14/32 (43.75%) | 2/32 (6.25%) |
| Wrong binding, changed answer | 4/32 (12.5%) | 1/32 (3.125%) |

Both-token correct-answer NLL: 1.840829 at step 240; 3.854716 at step 300.
Per-color correct accuracy (red, green, blue, yellow): best 4/8, 3/8, 3/8, 4/8;
terminal 2/8, 0/8, 0/8, 0/8. The text qualification above used the same tiny
examples and achieved 31/32 actual handoff, 32/32 oracle copy. It remains
explicitly forced-choice and a different, native-chat text interface.

[Complete training curves](training_curves.svg) show every evaluated step,
with no smoothing or omitted regressions. Machine-readable records:
[all optimizer steps](training_curve.csv), [all control evaluations](control_curve.csv).
Raw per-example scores and reader measurements remain in every `run_tiny/eval_*`.

### Failure analysis and strongest alternatives

The best checkpoint partially recovers IDs, but does not reliably transmit
the changed binding. Only 18/32 predictions name any ID in the corresponding
table, of which 14 are the queried answer. Thus identity recovery is itself
incomplete; this is not a clean case where identity is solved and only
retrieval fails. Changing colors among identical IDs leaves 23/32 predictions
unchanged at step 240 and 30/32 unchanged at step 300, although every target
changes. Changed-answer following is only 4/32 and 1/32, respectively.

At step 240, same-ID original/counterfactual memories have mean cosine
0.999594. Within-table slot cosine averages 0.998898; query-position reader
entropy is 2.077326 nats versus maximum ln(8)=2.079442. These are consistent
with weak slot specialization and weak sensitivity to changed bindings, but
geometry does not establish the causal failure mechanism. Small latent
differences can in principle matter; actual counterfactual answers, not the
cosines, are the decisive unfavorable observation.

The reader is not simply inactive. At the best checkpoint its query-position
update norm averages 0.891× native residual norm, ranging 0.439–1.843×;
mean cosine between pre/post-reader residuals is 0.766. Terminal corresponding
means are 0.757× and 0.805. Large updates and low loss are not communication.
Random memories have matched transport norms, not matched downstream update
energies. Correct slot norms remain 8 within approximately 1e-6; a norm channel
is not a supported account of the limited training performance.

The strongest remaining alternative is optimization failure under this one
fixed initialization, 300-step budget, and full-vocabulary two-token objective,
rather than a fundamental impossibility of the frozen architecture. The
trajectory improves and then regresses substantially. No second seed, learning
rate search, longer training, or architectural repair was run. This task also
requires more answer-token identity recovery than the earlier two-bit task.
Native-chat copy qualification does not by itself isolate the difficulty of
the plain-query latent output interface. The experiment therefore does **not**
uniquely diagnose lexical grounding, query-conditioned retrieval, slot collapse,
or prompt/optimization effects.

### Uncertainty and unlicensed stages

These are exact descriptive counts on eight deliberately trained-on tables,
with four correlated queries per table. They are not held-out accuracy
estimates; checkpoint selection further precludes treating the peak as an
unbiased generalization result. No generalization claim or population CI is
offered. The preregistered paired final bootstrap was not run because the
experiment never qualified for final data. It cannot rescue a failed tiny gate.

TRAIN, DEV, FINAL-known, FINAL-novel, relational reasoning, natural comprehension,
Arabic, downstream generation, and capability preservation were **not run**.
The reserved whole-ID pool is still untested. Do not label this a novel-ID
generalization NO-GO: it stopped earlier, at tiny training.

### Decision

**MECHANICS/TRAINING NO-GO for binding V1: mechanics PASS, tiny training NO-GO.**
Every terminal two-token score vector actually reproduced bit-exactly in all
seven arms. Receiver parameter hashes match before and after all 300 steps;
sender hashes matched before and after capture/text scoring. The independent
audit rederived the NO-GO and selected step 240 from raw correct-label scores.
The preceding two-bit cross-family PASS remains valid within its original
scope. Novel variable-binding communication has not been established.

## Provenance verification and preservation

`final_checks.json`: PASS for integrity/testing, explicitly not scientific
success. Nine tests passed (four original mechanical/data/scoring tests, five
independent audit-adversarial tests). All 17 recorded evaluations, including
the actual terminal repeat, pass independent parsing, two-token score sums,
prediction/accuracy/NLL derivation, donor derangement checks, exact donor-score
correspondence and zero parity. Initial six-file code snapshots remain exact.
The curve SVG contains all 64 plotted measurements and is valid XML.

The historical verifier rechecked 569 source entries, all 22 endpoint snapshot
files, and the separate historical documentation seal. No historical payload
changed. Git HEAD remains `ae550f0fbcf9dea1a5507f5695b077d2cc77d02b`, and the
existing dirty worktree status is unchanged. No Rust code, historical data,
paper output, model weight, tokenizer, or endpoint inference implementation
was changed. Bridge/scoring code and preregistration were frozen before the
first binding outcomes; post-result audit/report scripts are identified as
such, not retroactively called preregistered.

Critical identities independently matched:

| Artifact | SHA-256 |
|---|---|
| Frozen sender parameters | `deb5e69d2fcd24d139ce2927f8375c2c445867b196bc179df4688d98990e0ec4` |
| Frozen receiver parameters | `6a3e1782ebad58d1e9323d5eeaff191d112848b5f0058394a998d1ac54c492b3` |
| Best bridge, step 240 | `cb38ecb8fbe7418822281796acc426e7687280a72206e89d976805b0e75bfa65` |
| Terminal bridge, step 300 | `6368a614c33245cc4bef07f7d9fa41766265c7b9f59a24d8bc58b49657fc3aef` |

The full model/tokenizer file inventories are in `acquisition.json`; original
and counterfactual source IDs/tensors and their independently checked hashes
are under `data/tiny/`. Prompts and all candidate token IDs/scores are preserved.
Runtime versions are recorded independently in `final_checks.json`.
See [verification and artifact map](REPRODUCTION.md). This local tree is sealed
after documentation; it is not remotely backed up or pushed.

## Compute

Local Intel i5-1135G7, approximately 16 GiB RAM, no CUDA device, four intra-op
threads and one inter-op thread. Endpoints were loaded sequentially.

| Completed subprocess | Elapsed seconds | Minutes |
|---|---:|---:|
| Original/counterfactual sender capture | 55.764 | 0.929 |
| Sender text scoring and parity/hash checks | 1,062.255 | 17.704 |
| Receiver text scoring, repeat, parity/hash checks | 405.503 | 6.758 |
| Tiny mechanics, 300 steps, all evaluations and terminal repeat | 4,604.523 | 76.742 |
| Driver total | 6,128.054 | 102.134 |

The child-process peak RSS high-water mark was 11.175 GiB; the value is
cumulative across sequential children, not a separate per-stage peak. The
300 optimizer steps themselves sum to 257.137 seconds; full 128-ID control
scoring dominates the training-stage time. The original tiny estimate of
20–60 minutes was low: the complete stage took 76.742 minutes. Initial
preflight/tests and post-job documentation/verification are outside driver time.

Actual sender text task scoring took 752.922 seconds for 32 queries, excluding
its parity/hash/loading overhead. Receiver text scoring took 190.259 seconds
for 33 distinct actual/oracle compact messages, excluding its repeat and
parity/loading overhead. Those messages are explicitly deduplicated; 33 is
not 64 independently executed handoff legs. Sender text prompts have 54 tokens,
receiver copy prompts 52, and the ID payload has two answer tokens. Latent
source/query lengths are 19/5, and its float32 payload is 2,048 bytes/table,
shared across the four queries. These native-forward scoring timings are not
optimized generation latency or a controlled deployment benchmark; accuracy
also differs sharply. No latent efficiency advantage is claimed.

No cloud was provisioned; external compute spend is $0 against the approximately
$30 ceiling. Local energy and assistant-service usage were not metered. No
job remains active. Full-stage compute was not purchased or launched because
the tiny gate failed.
