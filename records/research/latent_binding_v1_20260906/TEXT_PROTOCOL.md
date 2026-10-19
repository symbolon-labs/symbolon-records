# Fixed binding text comparison

This file and executed source are hashed before any model outcome. No prompt
variants or post-outcome label-format tuning are permitted in this experiment.

Actual sender user body (source is the real four-binding table):

```text
Read the bindings and answer the question. Reply with exactly one ID and nothing else.

<source>

Which ID is <color>?
```

Receiver compact handoff user body:

```text
Copy the answer ID. Reply with exactly the ID and nothing else.

Answer ID: <sender-predicted-ID>
```

Oracle replaces the predicted ID with the answer parsed from the actual source
for the queried color. Native chat templates, add_generation_prompt=True,
enable_thinking=False,date_string="06 Sep 2026". Bare two-token IDs as
continuations; score all128, sum both full-vocabulary LPs, no normalization.
No free generation or source-specific candidate list. Actual sender scores
every query from the real source; compact receiver messages may be deduplicated
by exact ID and the cache is explicitly counted. Outcome fields are separate:
sender task accuracy, actual full text handoff, oracle copy accuracy.

Both native scorers compare the prefix-reuse result against all128 direct
teacher-forced candidate scores on the first tiny prompt, tolerance1e-4 and
identical argmax, with an exact fast-path repeat. The receiver compact messages
also actually rerun. This parity check is mechanical, not an extra task result.
If oracle copy fails, do not train around an unqualified receiver output task.
This task-specific qualification is separate from the earlier four-letter test.
