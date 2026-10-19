# Learned latent channel v1 — tiny-gate result

**F — MECHANICS / TRAINING NO-GO**

One preregistered architecture, one training seed, 300 optimizer
steps, and no paid compute. Maximum observed tiny-set accuracy:
25.00%. Gate0 passed; Gate1 (>=31/32) failed.
Gate2 control ceiling was satisfied, but
chance-level shuffle is not evidence that useful accuracy was destroyed unless
the correct channel first works. No held-out claim follows from these 32 items.
The raw `decision.json` field `gate2=true` denotes only its <=50% control
ceiling, **not a semantic shuffle-gate PASS**. Likewise `best_dev_ce` refers
to tiny training-set evaluation here, not a DEV result. These generic field
names do not license a later gate.

## Hypothesis and architecture

Can a trained reader let a frozen receiver recover a sender-only random symbol?
The closed residual/prefix design was not retried. Prior evidence is summarized
in [the completed native-control report](../latent_bridge_redo_cached_20260906/REPORT.md).
That prior NO-GO did not establish universal receiver impossibility.

```
sender-only random record
  -> frozen Qwen3-1.7B, block 13 output (22 x 2048)
  -> trainable 8-query attention resampler (width 64)
  -> fixed-RMS 8 x 64 latent transport; NO textual intermediate
  -> trainable 64 -> 1024 projection
  -> one trainable width-64 cross-attention reader, scalar gate
  -> frozen Qwen3-0.6B block 13 output + gated context
  -> unchanged remaining blocks, final norm and LM head
  -> answer-token cross entropy / forced-choice A, B, C, D
```

Both block indices are zero-based. Bridge: **590,337**
trainable parameters / 596,049,920 receiver parameters
= **0.0990%**. All endpoint parameters frozen;
full in-memory receiver parameter hashes before/after training match. No Ember
source changes, endpoint fine-tuning, new linear map, or residual-coordinate sweep.

Official endpoint sources: [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)
and [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B). Exact revisions:

- Sender: `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
- Receiver: `c1899de289a04d12100db370d81485cdf75e47ca`.
- Official safetensors loaded as float32, no quantization. All local file hashes
  and tokenizer identities are in the manifests and independent audit.

## Data and exact rule

Sender sees eight irrelevant random digits and `Signal: X`, X in A/B/C/D.
Receiver sees only `Hidden signal (A/B/C/D):`, identically for every example.
Every nonce appears with all four counterfactual symbols. Therefore Q contains
zero target information and a deterministic receiver-only classifier scores
exactly 25%. Rows are shuffled; token length is exactly 22 for all sender inputs.
Labels and IDs are not inputs to the bridge or receiver. Only the sender sees X.

Tiny seed 11001: eight nonce groups, 32 training examples. Train/dev seeds
22001/33001 and reserved final seed 44001 are fixed in
[PREREGISTRATION.md](PREREGISTRATION.md). Train/dev model evaluations and final
materialization have not run at this reporting stage. Final remains untouched.
Four shared output symbols are the task alphabet, not reusable record identities;
this would test a two-bit channel, not arbitrary-string generalization.

Training: full-vocabulary CE, AdamW lr .003, weight decay 0, batch 8,
gradient clipping 1, seed 51001, max 300 steps, evaluation every 20. Stop only
at correct >=95% and shuffled/each random <=50%; zero == baseline ==25%.
No learning-rate, layer, width, gate, or magnitude search followed the outcomes.

## Measurements and controls

Last step 300; accuracies are forced-choice, not free generation.
CE is the full-vocabulary answer-token loss, not a four-class renormalized loss.

| Condition | Accuracy | Full-vocabulary CE | A/B/C/D recall (%) |
|---|---:|---:|---|
| receiver_only | 25.00% | 5.2966 | 100/0/0/0 |
| correct | 25.00% | 1.3867 | 100/0/0/0 |
| shuffled | 25.00% | 1.3867 | 100/0/0/0 |
| random_71001 | 25.00% | 5.5937 | 100/0/0/0 |
| random_71002 | 25.00% | 5.6089 | 100/0/0/0 |
| zero | 25.00% | 5.2966 | 100/0/0/0 |
| text | 0.00% | 4.7288 | 0/0/0/0 |

Shuffling is a label-blind derangement; donor IDs/labels are recorded. Random
memories use two independent seeds and match every correct slot's L2 norm.
ZERO actually runs the reader with zero memory; disabled baseline skips it.
Zero-memory outputs match baseline exactly, before and after training.

Paired 99% cluster-bootstrap intervals (10,000 draws, eight nonce groups):
these describe the trained-on mechanical set and are **not held-out evidence**.

| Correct minus condition | Accuracy difference | 99% interval |
|---|---:|---|
| receiver_only | +0.00 pp | [+0.00, +0.00] pp |
| shuffled | +0.00 pp | [+0.00, +0.00] pp |
| random_71001 | +0.00 pp | [+0.00, +0.00] pp |
| random_71002 | +0.00 pp | [+0.00, +0.00] pp |
| zero | +0.00 pp | [+0.00, +0.00] pp |
| text | +25.00 pp | [+25.00, +25.00] pp |

Training curves: [optimizer CSV](training_curve.csv),
[all-condition evaluation CSV](evaluation_curve.csv),
[accuracy plot](training_accuracy.svg), and [raw curve](run_tiny/curve.jsonl).
All evaluated checkpoints are represented; the report does not hide losing checks.

## Receiver and channel diagnostics

- Final gate: 0.025016.
- Median correct context/native-residual norm ratio: 0.572022.
- Median pre/post residual cosine: 0.872052.
- Mean reader entropy: 0.693147 nats (maximum ln8=2.079442).
- Mean resampler attention on the only varying source token:
  0.09181985 (uniform reference 1/22=.04545455).
- Resampler entropy: 1.919689 / maximum
  3.091042 nats.
- Mean off-diagonal slot cosine:
  0.96111137.
- Mean sender final-state difference for same-nonce/different-symbol pairs:
  58.058818; mean corresponding whole-memory difference:
  0.174937.

All earlier source rows are bit-identical within counterfactual groups, as causal
attention requires. The final sender row changes with the hidden symbol. Thus
the task really presents sender-only information. Fixed per-slot norm is 8;
norms, gradients, per-example reader diagnostics and expanded-memory norms are
saved in [the independent diagnostic audit](tiny_audit/diagnostics.json).
Attention concentration/slot similarity are diagnostics, not independently
established causal failure mechanisms. No attention ablation was run.

### Located failure: reader selection collapsed onto the constant marker

Inspection of the **actual trained reader** found last-query attention split
approximately 50/50 between slots 0 and 3; the other six slots receive numerically
zero attention. Those two slots attend almost exclusively to source token 19,
the constant `Signal` marker, **before** the answer-bearing token. They are
therefore effectively answer-independent. Other, ignored slots still change
with the sender's symbol. The composed attention-routing diagnostic assigns
only 7.94e-09 mass to
the answer-bearing source token, versus approximately 1 to the marker.

More directly, changing the symbol while holding the nuisance digits fixed
changes the actual receiver context by only
1.05e-07 mean L2
(8.59e-07 maximum). The actual checkpoint
log probabilities were reproduced during this inspection. Source-side key/query
gradient norms have fallen to approximately 1e-10/7e-10, while output and gate
gradients remain nonzero. This identifies an effectively closed semantic path
in this checkpoint, consistent with attention saturation and prior-only training.
It does **not** prove which initialization or optimizer choice caused that
collapse, or that all learned readers fail. Full tensors and routing details:
[attention audit](attention_audit/summary.json).

## Text handoff, bytes and compute

TEXT is an **oracle lossless copy** of the source record, followed by the same
receiver query. It is not sender-generated summarization. It avoids sender-model
compute, so charging it a sender forward would be unfair. Its poor observed
score means this particular plain completion prompt is not a validated text
communication baseline; no latent-over-text advantage is claimed. A stronger
instruction/chat text baseline was not tuned on these outcomes.

Transport: K=8, width64, float32 = **2048 bytes/example**; after receiver projection
32768 bytes. Uncompressed sender states: 22*2048*4 = 180224 bytes, giving an
88x representation-byte reduction before projection, or 5.5x after projection.
This is still vastly larger than the two useful bits or a one-character text
message; no efficiency advantage is established.

Receiver query: 8 tokens; lossless text query:
31 tokens. Initial 32-example inference timings:
baseline 3.32s, correct reader
3.41s, text 9.69s.
These are sequential CPU timings on a thermally variable laptop, not matched
latency benchmarks. Later baseline/text outputs were cached, explicitly flagged.
Precise isolated bridge overhead and end-to-end deployment latency are unmeasured.

Recorded sender captures (including the metadata-correction repeat): 54.66s;
tiny run: 738.67s; independent diagnostic rerun:
38.53s; attention inspection: 6.27s;
sum of these timed components: 838.13s.
This excludes download, unit tests, the initial failed pre-training exporter
attempt and orchestration. Downloading the new sender took approximately 374s.
Hardware: Intel i5-1135G7, CPU-only, 4 PyTorch threads, 16GB physical RAM.
External/cloud spend: **USD 0**, budget ceiling USD30. No cloud resources launched.

## Correctness, provenance and failures retained

Gate0: actual stock/disabled/zero full logits bit-exact; nonzero reader changed
logits; all bridge parameter gradients finite and nonzero; endpoints had no
gradients. Three unit tests and Python compilation passed. Actual trained
all-condition rerun matched stored outputs bit-for-bit. Full training was not
rerun from initialization; do not confuse inference repeatability with that test.

Independent model/tokenizer hash verification: **PASS**.
Preregistration SHA256: `24e0696c72610edc880275ba27c26f6c5392e48b49e3ff903226d6e38b4497eb`.
Training implementation SHA256: `e2de36c051ba503b441c2d84b959728cb09f8d064791012b5f673462039f6fbd`.
Last checkpoint SHA256: `274efcd2a98b1dbafbe87fcb65ada08a656f8d8b8ab12d9327ef080b13636a4a`.
Complete identities are in [run manifest](run_tiny/manifest.json),
[sender manifest](data/tiny/manifest.json), and the independent audit.

Two pre-optimizer diagnostics/exporter errors and the metadata-only ID correction
are documented in [MECHANICAL_CORRECTIONS.md](MECHANICAL_CORRECTIONS.md).
Original files remain. Independently regenerated sender inputs and activations
matched bit-for-bit after the ID correction. No failed training seed was discarded.

## Interpretation and decision

The fixed tiny overfit gate failed. Stop before scaling or cloud: no working communication channel was established.
The strongest remaining alternative to architectural incapacity is an
optimization/task-interface failure in this particular one-configuration
prototype. Missing gate gradients and accidental endpoint training are ruled
out by direct checks; broader learnability is not ruled out. Loss reduction,
large interventions and representational differences do not establish communication.

Held-out synthetic: **NOT RUN**. Cross-family: **NOT LICENSED**. Natural tasks:
**NOT LICENSED**. Arabic: **NOT LICENSED**. Generation: **NOT LICENSED**. Ablations:
**NOT RUN**. The final-evaluation implementation exists but has not been executed.
Do not report a scientific channel NO-GO on an untouched held-out test, or infer
that all learned readers fail, from this mechanical training result.
