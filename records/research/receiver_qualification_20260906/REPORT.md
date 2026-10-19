# PASS — SmolLM2 receiver qualified on the exact oracle-text task

This closes receiver qualification only. Same-family synthetic latent
communication remains established; the earlier Qwen→Llama fixed configuration
remains a tiny-training NO-GO; cross-family latent communication remains
**UNPROVEN**. No bridge was trained or modified in this qualification experiment.

## Hypothesis and fixed decision

Before attempting another cross-family bridge, require a non-Qwen frozen
receiver to recover the four oracle-supplied symbols under the exact previously
used compact-text task. PREREGISTRATION.md was written before new inference.
Only one new candidate was selected, SmolLM2-360M-Instruct; Qwen served as a known
positive harness control and Llama as a known negative/reproducibility diagnostic.
No prompt, answer, architecture, precision, or model search followed outcomes.

The candidate is a smaller independent pretrained family, not another Qwen.
Its official checkpoint is HuggingFaceTB/SmolLM2-360M-Instruct, revision
a10cc1512eabd3dde888204e902eca88bddb4951. Native metadata: model_type=llama,
32 blocks, hidden width960, 361,821,120 frozen parameters. Selection was based
on small local resource requirements and instruction-model availability, not
observed task performance. [Official model card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct).

## Exact task, finite domain, and controls

User body is byte-identical to the sealed text baseline:

```text
Copy the Signal value. Reply with exactly one letter: A, B, C, or D.

Signal: X
```

X takes A/B/C/D. Each endpoint uses its native chat template, generation-prompt
marker, enable_thinking=False, and fixed date06 Sep2026 where applicable.
Answer strings are exact bare A/B/C/D. For SmolLM2 they are single token IDs
49/50/51/52; joint prompt/continuation boundaries were verified. No alternative
whitespace, label wording, few-shot examples or parsing rules were tried.

There are only FOUR unique oracle cases. They were evaluated exhaustively.
Repeat/order/singleton checks do not create independent cases or statistical
sample size. No synthetic TRAIN/DEV/FINAL set was constructed or consumed.
No bootstrap CI is appropriate for duplicated versions of this finite task.

Cyclic donor substitution transmits B/C/D/A for original targets A/B/C/D.
A qualified receiver must follow all four donor symbols and get all four
original targets wrong. The missing-source message uses exactly
`The Signal value was withheld.` after the same instruction; its identical
input, paired with balanced targets, has25% accuracy by construction.

The preregistered qualification rule requires4/4 oracle cases, positive margin
for each symbol,4/4 donor following and0/4 original-target accuracy, missing
input25%, deterministic repeats, order/singleton consistency, correct scoring,
and unchanged endpoint identities. This preserves the old >=90% oracle-text
sanity requirement; with four unique cases it requires all four to be correct.

## Results

| Receiver | Oracle | Donor following | Donor vs original target | Missing input | Verdict |
|---|---:|---:|---:|---:|---|
| Qwen3-0.6B, positive control |4/4|4/4|0/4|25%|PASS|
| Llama-3.2-1B-Instruct, diagnostic |1/4|1/4|1/4|25%|NO-GO|
| SmolLM2-360M-Instruct, candidate |4/4|4/4|0/4|25%|PASS|

SmolLM2 correct-vs-best-wrong log-probability margins for A/B/C/D were
1.064722,0.806402,0.713643,2.023290 nats. The unconstrained next-token argmax
also equaled the intended symbol for each oracle case. That is a next-token
diagnostic, not a free-generation result. Missing-source forced choice selected
A, while its unconstrained next token was different; the chance result is not
evidence of calibrated uncertainty.

## Mechanics and independent verification

All endpoint runs used native Transformers eager, float32 CPU, no quantization,
no hooks and no optimizer; all endpoint parameters had requires_grad=False and
no gradients. Sealed text_baseline.score was reused without numerical changes.
Its full-vocabulary log probabilities were independently recovered from saved
native final-position logits. The same native transformer was used throughout;
no parallel inference engine was created.

SmolLM2 same-batch rerun and missing-source rerun were bit-exact. Reverse-order,
permuted-order, and donor-reordered probability vectors were also exactly equal
after matching examples. Singleton-vs-batch maximum label log-probability
difference9.54e-6, below preregistered1e-4, with identical predictions. Native
full-sequence-logit vs last-logit-only maximum full-vocabulary difference3.43e-5,
below1e-4, with identical argmax. No tolerance was changed after results.

Qwen's positive control reproduced the sealed historical outputs exactly and
passed all checks. Llama also reproduced its historical all-D behavior exactly.
Llama still selected D on separate singleton examples and reordered batches;
its full-logit seam passed. These checks rule out the tested batch-order and
last-logit-selection explanations for this failure, not every possible runtime
or elicitation issue. The native Llama implementation was the same for Llama
and SmolLM2; their weights/configurations/tokenizers and templates differ.

verify_results.py independently reconstructed decisions from saved log
probabilities and logits, checked native rendered prompts and token boundaries,
verified donor joins, recomputed file identities, and re-fetched the candidate's
pinned upstream LFS hash. The weight SHA256 is
e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e.
VERIFICATION.json records the requirement-by-requirement audit and the unchanged
163-file V2 and96-file failed-cross-family historical seals.

## Implementation interruption and provenance

The first candidate launch stopped BEFORE model loading/inference because
Hub1.30's offline snapshot resolver demanded unrelated ONNX exports. It did not
produce an oracle outcome. Its started receipt and original code are preserved.
The path-only repair selected the already-downloaded native runtime filenames;
new results went to smollm_candidate_snapshot_path_fix. qualify_initial.py
matches all pre-repair execution code hashes. See PATH_FIX.md. This was not an
accuracy-driven retry or a change to model behavior.

Each completed run records model revision, all root model/tokenizer hashes,
native runtime source hash, code/preregistration hashes, model configuration,
input/answer token IDs, prompt bytes/hashes, seed93001, frozen parameter hashes,
timings and memory. Python3.14.7, torch2.13.0+cu130, transformers5.10.2,
numpy2.4.6;4 CPU threads/1 inter-op; deterministic algorithms. Three qualification
unit tests passed before execution and again after the resolver repair.
No Rust or historical experiment content was changed.

## Compute and scope

Completed native stages, including their recorded verification overhead:
Qwen19.89s, Llama53.24s, SmolLM2 15.06s; summed88.19s. Candidate peak RSS
2,876,028KiB (about2.74GiB). Download roughly76s,723,674,912 weight bytes;
download and final evidence-audit time are not part of the88.19s. Local CPU only,
external/cloud spendUSD0. No model job remains intentionally running.

This shows the receiver can read each explicit oracle symbol under this exact
native-chat format. It does not establish latent reading, held-out communication,
Arabic transfer, general instruction competence, or generation. In particular,
the oracle chat input is not the bridge's plain hidden-signal query; qualification
removes one necessary readiness confound, not all future elicitation, placement,
or optimization risks. No comparison of bridge efficiency is available here.

## Decision and next boundary

**RECEIVER QUALIFIED: SmolLM2-360M-Instruct at the pinned revision.**
The receiver-readiness gate is satisfied before any new cross-family bridge
training. A subsequent bridge experiment would need a separate preregistration,
fresh development data, a bridge initialized from scratch, and the original
tiny-overfit/shuffle gates before held-out evaluation. No such run was launched
here. No latent architecture change was needed to establish receiver readiness.
