# Receiver oracle-text qualification

## Boundary and hypothesis

Written before new model inference. Prior same-family V2 is PASS; the fixed
Qwen3-1.7B → Llama-3.2-1B-Instruct tiny bridge is NO-GO. Cross-family latent
communication is unresolved. This experiment qualifies a receiver, not a bridge.
No optimizer, latent architecture edit, hidden intervention, natural task,
Arabic, generation, or bridge FINAL data is licensed here. Historical trees stay
sealed. The preceding research turn made progress: it produced the failed tiny
gate, exact terminal rerun, and failed oracle-text diagnostic.

Hypothesis: a small non-Qwen receiver can recover each oracle-supplied symbol
under the existing exact compact-text task. Qualification is required before
any future bridge training, but does not itself license or prove communication.

## Fixed models, no model-size escalation or prompt search

Harness positive control: Qwen/Qwen3-0.6B,
c1899de289a04d12100db370d81485cdf75e47ca, previously100% on this oracle task.
Known negative/reproducibility diagnostic: meta-llama/Llama-3.2-1B-Instruct,
9213176726f574b556790deb65791e0c5aa438b6, previously25% (all D).
One new candidate: HuggingFaceTB/SmolLM2-360M-Instruct,
a10cc1512eabd3dde888204e902eca88bddb4951. It is smaller than the failed
receiver, a separate pretrained family from Qwen, with native Llama architecture.
The official model card and repository metadata were inspected for size and
availability only. No candidate inference outcomes were consulted.
If the candidate fails, record the failure; do not silently add more models,
change prompt wording, answer strings, precision or architecture.

## Exact task and finite domain

Reuse the sealed receiver instruction byte-for-byte:
`Copy the Signal value. Reply with exactly one letter: A, B, C, or D.\n\n`
Append `Signal: X`, X=A/B/C/D. Native endpoint chat template, one user message,
add_generation_prompt=True, enable_thinking=False, fixed
date_string="06 Sep 2026". Answers are exact bare strings A/B/C/D, no added
whitespace or punctuation. The missing-source control appends exactly:
`The Signal value was withheld.`

This is the identical oracle compact-text task, not a prompt variant. There are
only FOUR unique oracle inputs. Evaluate exhaustively; repeated or permuted
copies are mechanical checks, not fresh semantic examples, sample-size gains,
or held-out evidence. No fake bootstrap from duplicated prompts. No train/dev/
final synthetic split is constructed. For missing input, identical Q paired with
balanced A/B/C/D must yield25% accuracy by construction.

## Scoring, controls, deterministic checks

Reuse sealed text_baseline.score for these single-token label vocabularies;
require joint prompt+answer tokenization == prompt tokens + answer tokens.
Fail closed on unsupported continuation tokenization, not silently truncate.
Full-vocabulary log probabilities only; native frozen LM head, no generation.
No hook is installed. Native eager float32, torch deterministic,4/1 threads,
seed93001, no quantization. Record exact config, models, tokenizer files and
runtime source hashes and versions before/after. Recompute upstream weight
hashes independently for the new download. No credentials in artifacts.

Primary run: batch4 oracle inputs in A/B/C/D order plus batch1 missing source.
Repeat exactly at the same batch size, requiring bit-exact log probabilities.
Reorder to D/C/B/A and C/A/D/B, unpermute outputs, and check probability
agreement (max absolute label-log-probability difference<=1e-4), same predictions.
Run each oracle separately (batch1); require the same predictions and <=1e-4
probability difference vs batch4. This checks for cross-example leakage and
batch-dependent scoring. These tolerances are not accuracy relaxations.

Independent scoring seam check: native forward with logits_to_keep=0 and select
the final prompt position; compare FULL-VOCAB logits to logits_to_keep=1,
tolerance1e-4 and identical top token. Compare selected-label probabilities
against the sealed scorer. No parallel transformer implementation is introduced.

Counterfactual donor control: cyclically replace each oracle symbol with the
next symbol A→B→C→D→A. Run those four messages explicitly, require predictions
follow the transmitted donor (100%) and hence are wrong for original targets
(0%). Missing-source accuracy must equal25% over balanced targets. Record
per-symbol logit margin, full-vocabulary CE, unconstrained next-token ID,
input IDs, answer IDs, prompt hashes, timing, endpoint parameter hash.

## Decision rule

QUALIFIED only if oracle accuracy is4/4 (the >=90% legacy text-sanity criterion
on four unique cases), each class correct, positive correct-vs-best-wrong margin,
donor-following4/4 and original-target0/4, missing-source25%, all deterministic/
batching/scoring checks pass, and endpoint weights/identities remain unchanged.
The positive-control Qwen must qualify; otherwise mark HARNESS INVALID and do
not accept new candidate outcomes as qualification. Reproduce stored Qwen and
Llama oracle outputs at identical batches exactly where their runtime identity
matches; a mismatch must be diagnosed, not smoothed over.

A qualified non-Qwen receiver satisfies this receiver-readiness goal ONLY.
Cross-family latent communication remains unproven. A failed candidate does not
establish model-family impossibility. Report all attempted models, including
the already-known Llama failure. Do not rewrite prior verdicts.

## Compute

Local CPU only. New model safetensors723,674,912 bytes; float32 weights roughly
1.45GB plus inference overhead, comfortably below16GiB host capacity. Load one
endpoint at a time. Total native checks estimated5–10min including verification;
download duration uncertain. No GPU/VRAM or cloud required; external spendUSD0,
project external ceiling approximatelyUSD30 remains. Save logs/receipts; no idle
model job after completion. Fresh qualification directory, exclusive outputs.
