# Exact inherited strong text baseline

Use the sealed V2 text_baseline.py scoring/orchestration and the exact receiver
qualification render policy. Sender instruction:
`Copy the value of the Signal field in the record below. Reply with exactly one letter: A, B, C, or D.\n\n`
Receiver instruction:
`Copy the Signal value. Reply with exactly one letter: A, B, C, or D.\n\n`
Source appended to sender. Receiver gets only `Signal: X`, using actual sender
forced-choice X, or oracle parsed X; missing control is exactly
`The Signal value was withheld.` Native templates, generation marker,
enable_thinking=False, date_string="06 Sep 2026". Bare A/B/C/D continuations,
full-vocabulary scores and exact joint token boundaries. No generation.
Every sender example scored; four unique receiver oracle messages cached
explicitly, not treated as independent samples. Actual deterministic receiver
rereads, model/provenance hashes, frozen endpoints and all outputs recorded.
No post-outcome prompt or answer-format tuning. Tiny text after the tiny bridge
gate (or diagnostic after its failure), DEV before final freeze, FINAL after
frozen final materialization. Oracle text sanity>=90% remains required before
scaling. Full task/bridge rules in PREREGISTRATION.md.
