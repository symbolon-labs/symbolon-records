"""Generate a conservative tiny-gate report from stored evidence and seal it."""
import csv
import json
import statistics
from pathlib import Path

from channel import ROOT, CONDITIONS, sha, write_json


def main():
    run = ROOT / "run_tiny"
    decision = json.loads((run / "decision.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    audit = json.loads((ROOT / "tiny_audit" / "diagnostics.json").read_text())
    routing = json.loads((ROOT / "attention_audit" / "summary.json").read_text())
    last = run / f"eval_{decision['step']:04d}"
    metrics = json.loads((last / "metrics.json").read_text())
    records = json.loads((last / "per_example.json").read_text())
    curve = [json.loads(x) for x in (run / "curve.jsonl").read_text().splitlines()]
    evaluations = [(int(p.parent.name.split("_")[1]), json.loads(p.read_text()))
                   for p in sorted(run.glob("eval_*/metrics.json"))]
    with open(ROOT / "training_curve.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curve[0]))
        w.writeheader()
        w.writerows(curve)
    with open(ROOT / "evaluation_curve.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "condition", "accuracy", "full_vocab_ce", "gate"])
        for step, m in evaluations:
            for c in CONDITIONS:
                w.writerow([step, c, m["conditions"][c]["accuracy"], m["conditions"][c]["full_vocab_ce"], m["gate"]])
    # Repo-native SVG plot, directly from stored measurements; no generated image.
    series = []
    colors = {"correct": "#0072b2", "shuffled": "#d55e00", "random_71001": "#009e73"}
    for c, color in colors.items():
        points = " ".join(f"{50+step/300*650:.2f},{240-m['conditions'][c]['accuracy']*200:.2f}" for step, m in evaluations)
        series.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>')
    (ROOT / "training_accuracy.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="300" viewBox="0 0 760 300">'
        '<rect width="760" height="300" fill="white"/><g font-family="sans-serif" font-size="12">'
        '<text x="50" y="20">Tiny mechanical set: forced-choice accuracy (not held-out evidence)</text>'
        '<path d="M50 40 V240 H700" fill="none" stroke="black"/>'
        '<text x="14" y="45">100%</text><text x="20" y="245">0%</text>'
        '<text x="50" y="265">0</text><text x="670" y="265">300 steps</text>'
        '<text x="60" y="288" fill="#0072b2">Correct</text><text x="160" y="288" fill="#d55e00">Shuffled</text>'
        '<text x="270" y="288" fill="#009e73">Random seed 71001 (overlapping lines may hide each other)</text>'
        + ''.join(series) + '</g></svg>\n')
    condition_table = []
    for c in CONDITIONS:
        m = metrics["conditions"][c]
        condition_table.append(f"| {c} | {100*m['accuracy']:.2f}% | {m['full_vocab_ce']:.4f} | "
             + "/".join(f"{100*x:.0f}" for x in m["per_class"]) + " |")
    contrasts = audit["paired_accuracy_diagnostics_not_scientific_evidence"]
    contrast_table = [f"| {c} | {100*v['correct_minus_control']:+.2f} pp | "
                      f"[{100*v['ci99'][0]:+.2f}, {100*v['ci99'][1]:+.2f}] pp |"
                      for c, v in contrasts.items()]
    last_mass = [x for row in audit["resampler_attention_last_source_token"] for x in row]
    ratios = [r["conditions"]["correct"]["diagnostics"]["delta_residual_ratio"] for r in records]
    cosines = [r["conditions"]["correct"]["diagnostics"]["cosine"] for r in records]
    entropies = [r["conditions"]["correct"]["diagnostics"]["reader_entropy"] for r in records]
    source_norms = audit["sender_final_row_counterfactual_distances"]
    slot_distances = audit["slot_counterfactual_distances"]
    max_accuracy = max(m["conditions"]["correct"]["accuracy"] for _, m in evaluations)
    passed = decision["gate1"] and decision["gate2"]
    verdict = "MECHANICAL GATES PASS — HELD-OUT CHANNEL NOT YET ESTABLISHED" if passed else "F — MECHANICS / TRAINING NO-GO"
    source_s = sum(json.loads(p.read_text())["elapsed_s"] for p in (ROOT / "data").glob("*/manifest.json"))
    timed_s = source_s + decision["elapsed_s"] + audit["elapsed_s"] + routing["elapsed_s"]
    text_timing = json.loads((run / "eval_0000" / "metrics.json").read_text())["times"]
    report = f'''# Learned latent channel v1 — tiny-gate result

**{verdict}**

One preregistered architecture, one training seed, {decision['step']} optimizer
steps, and no paid compute. Maximum observed tiny-set accuracy:
{max_accuracy:.2%}. Gate0 passed; Gate1 (>=31/32) {'passed' if decision['gate1'] else 'failed'}.
Gate2 control ceiling {'was satisfied' if decision['gate2'] else 'failed'}, but
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

Both block indices are zero-based. Bridge: **{manifest['trainable_parameters']:,}**
trainable parameters / {manifest['receiver_parameters']:,} receiver parameters
= **{100*manifest['parameter_fraction']:.4f}%**. All endpoint parameters frozen;
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

Last step {decision['step']}; accuracies are forced-choice, not free generation.
CE is the full-vocabulary answer-token loss, not a four-class renormalized loss.

| Condition | Accuracy | Full-vocabulary CE | A/B/C/D recall (%) |
|---|---:|---:|---|
{chr(10).join(condition_table)}

Shuffling is a label-blind derangement; donor IDs/labels are recorded. Random
memories use two independent seeds and match every correct slot's L2 norm.
ZERO actually runs the reader with zero memory; disabled baseline skips it.
Zero-memory outputs match baseline exactly, before and after training.

Paired 99% cluster-bootstrap intervals (10,000 draws, eight nonce groups):
these describe the trained-on mechanical set and are **not held-out evidence**.

| Correct minus condition | Accuracy difference | 99% interval |
|---|---:|---|
{chr(10).join(contrast_table)}

Training curves: [optimizer CSV](training_curve.csv),
[all-condition evaluation CSV](evaluation_curve.csv),
[accuracy plot](training_accuracy.svg), and [raw curve](run_tiny/curve.jsonl).
All evaluated checkpoints are represented; the report does not hide losing checks.

## Receiver and channel diagnostics

- Final gate: {metrics['gate']:.6f}.
- Median correct context/native-residual norm ratio: {statistics.median(ratios):.6f}.
- Median pre/post residual cosine: {statistics.median(cosines):.6f}.
- Mean reader entropy: {statistics.mean(entropies):.6f} nats (maximum ln8=2.079442).
- Mean resampler attention on the only varying source token:
  {statistics.mean(last_mass):.8f} (uniform reference 1/22=.04545455).
- Resampler entropy: {audit['resampler_entropy']:.6f} / maximum
  {audit['resampler_max_entropy']:.6f} nats.
- Mean off-diagonal slot cosine:
  {audit['slot_pair_cosine_mean_off_diagonal']:.8f}.
- Mean sender final-state difference for same-nonce/different-symbol pairs:
  {statistics.mean(source_norms):.6f}; mean corresponding whole-memory difference:
  {statistics.mean(slot_distances):.6f}.

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
only {routing['attention_composition_source_mass_last_query'][-1]:.3g} mass to
the answer-bearing source token, versus approximately 1 to the marker.

More directly, changing the symbol while holding the nuisance digits fixed
changes the actual receiver context by only
{routing['counterfactual_context_L2_mean']:.3g} mean L2
({routing['counterfactual_context_L2_max']:.3g} maximum). The actual checkpoint
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

Receiver query: {metrics['receiver_query_tokens']} tokens; lossless text query:
{metrics['text_receiver_tokens']} tokens. Initial 32-example inference timings:
baseline {text_timing['receiver_only']['s']:.2f}s, correct reader
{text_timing['correct']['s']:.2f}s, text {text_timing['text']['s']:.2f}s.
These are sequential CPU timings on a thermally variable laptop, not matched
latency benchmarks. Later baseline/text outputs were cached, explicitly flagged.
Precise isolated bridge overhead and end-to-end deployment latency are unmeasured.

Recorded sender captures (including the metadata-correction repeat): {source_s:.2f}s;
tiny run: {decision['elapsed_s']:.2f}s; independent diagnostic rerun:
{audit['elapsed_s']:.2f}s; attention inspection: {routing['elapsed_s']:.2f}s;
sum of these timed components: {timed_s:.2f}s.
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

Independent model/tokenizer hash verification: **{audit['independent_hash_verification']}**.
Preregistration SHA256: `{sha(ROOT / 'PREREGISTRATION.md')}`.
Training implementation SHA256: `{manifest['code_sha256']}`.
Last checkpoint SHA256: `{decision['last_checkpoint_sha256']}`.
Complete identities are in [run manifest](run_tiny/manifest.json),
[sender manifest](data/tiny/manifest.json), and the independent audit.

Two pre-optimizer diagnostics/exporter errors and the metadata-only ID correction
are documented in [MECHANICAL_CORRECTIONS.md](MECHANICAL_CORRECTIONS.md).
Original files remain. Independently regenerated sender inputs and activations
matched bit-for-bit after the ID correction. No failed training seed was discarded.

## Interpretation and decision

{'The tiny gate passed; it licenses held-out development, not a channel claim.' if passed else 'The fixed tiny overfit gate failed. Stop before scaling or cloud: no working communication channel was established.'}
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
'''
    (ROOT / "REPORT.md").write_text(report)
    # A seal is content integrity, not independent attestation of wall-clock time.
    files = {str(p.relative_to(ROOT)): sha(p) for p in sorted(ROOT.rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts and p.name != "seal.json"}
    write_json(ROOT / "seal.json", {"files": files, "verdict": verdict, "external_spend_usd": 0})
    assert all(sha(ROOT / path) == digest for path, digest in files.items())
    print(json.dumps({"verdict": verdict, "sealed_files": len(files), "report": str(ROOT / "REPORT.md")}))


if __name__ == "__main__":
    main()
