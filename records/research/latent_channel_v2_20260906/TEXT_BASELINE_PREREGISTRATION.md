# Stronger same-task text baseline — fixed before its evaluation

The V1/V2 plain-completion lossless relay performs poorly. It cannot support
a claim that a learned latent channel is better than text. Add one explicitly
instruction-formatted baseline, retaining the old baseline and all its results.
This changes ONLY the text-control formatting, not the latent task, query,
architecture, training, splits or pass thresholds. No prompt grid or selection.

Use the existing pinned frozen Qwen3 endpoints and official tokenizer chat
template with `enable_thinking=False`, `add_generation_prompt=True`. Only
forced-choice scoring of full continuations A/B/C/D; no free generation.

Sender user message, literal:
`Copy the value of the Signal field in the record below. Reply with exactly one letter: A, B, C, or D.\n\n{source}`

Receiver user message, literal:
`Copy the Signal value. Reply with exactly one letter: A, B, C, or D.\n\nSignal: {letter}`

Conditions:
1. TEXT-SENDER: sender chooses its highest-probability single-letter continuation;
   pass that literal letter as compact text to the frozen receiver. Record both
   stages' full-label log probabilities and tokens, not just sender accuracy.
2. TEXT-ORACLE: use the letter parsed from the sender-visible record (not from
   external gold metadata) in the same receiver message. This is a lossless
   two-bit text relay upper reference, requiring no sender-model inference.
3. RECEIVER-CHAT-ONLY: same receiver instruction, with `The Signal value was
   withheld.` instead of the field. Must score exactly25% on balanced records.

The receiver has only four distinct compact messages plus one missing-field
message. Evaluate each unique message once, cache by exact rendered prompt,
then join results to example IDs. No label-dependent cache omissions. This is
an explicit deterministic text-response cache, not an inference-latency benchmark.
Every source record still gets its own actual sender forward. Cache hashes,
models, exact prompts, continuations, token IDs and join logic are recorded.

First run on reusable tiny mechanical data after the tiny training process has
ended (pass or failure), to diagnose text interface correctness. No architecture
selection or changes based on these text outcomes. Text-oracle>=90% is the
predeclared sanity criterion for a competent text interface; if it fails, retain
the failure and do not claim a latent/text advantage. Text-sender is reported
honestly regardless of whether its sender stage succeeds.

Repeat the identical text baseline on DEV/final only when those splits are
licensed. Final execution must be included in the final freeze; no changes
after final inspection. Paired uncertainty will compare these joined predictions
against latent conditions on the same nonce groups. The stronger text baseline
does not alter the latent channel's core correct-vs-control pass rule.

Budget: local CPU only, cumulative external spend USD0; no resources provisioned.
