# Novel whole-ID gate: actual text comparison

**NOVEL WHOLE-ID BINDING NO-GO; TEXT COMPARISON COMPLETE.** Native comparator is the same frozen d0f16cb9 candidate on
one burned novel DEV191003 trial, eight fresh semantic tables/32queries with eight
training-unseen whole IDs. Native original0/32,
shuffled donor0/32, changed donor
0/32; minima31/31/28 plus all
controls and paired requirements. Text completion cannot rescue native failure.
No fit, tuning, architecture/readout/capacity/threshold change occurred.

| Condition | Native | Sender ID correct | Actual receiver text | Oracle receiver text |
|---|---:|---:|---:|---:|
| original | 0 | 32 | 32 | 32 |
| changed | 0 | 29 | 29 | 32 |

All64 source/query pairs were actually scored afresh, all128 two-token IDs,
full-vocabulary log-softmax/float32 sum/argmax. Actual sender ID string went to
frozen receiver copy prompt; oracle copying is separate. No historical scores
or perfect messages replace actual text. The sender sees S+Q/chat/unprefixed
IDs; latent sees S-only/plain query/leading-space IDs. This access/template
mismatch prevents an equal-access or efficiency superiority interpretation.

| Condition | Latent-minus-text mean |95% CI | Positive fraction | Eight table means |
|---|---|---|---|---|
| original | -1.000000 | [-1.0, -1.0] | 0.000000 | [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0] |
| changed | -0.906250 | [-0.96875, -0.8125] | 0.000000 | [-1.0, -1.0, -0.75, -1.0, -0.75, -1.0, -0.75, -1.0] |

Paired eight-table bootstrap,10000 draws, historical DEV seed95002. Four queries
stay together; familiar-ID dependence and tiny-bank size limit population claims.
No superiority claim when intervals include zero. Text has no accuracy gate.

| Condition/arm | Mean NLL | Mean margin | Median margin |
|---|---:|---:|---:|
| original: sender | 0.000354 | 15.840333 | 16.854497 |
| original: actual_text | 0.794897 | 4.887475 | 4.761464 |
| original: oracle_text | 0.794897 | 4.887475 | 4.761464 |
| changed: sender | 0.205330 | 14.574629 | 16.114854 |
| changed: actual_text | 1.532154 | 3.742513 | 4.716199 |
| changed: oracle_text | 0.794897 | 4.887475 | 4.761464 |

| Condition | Example | Query | Target | Sender | Actual receiver |
|---|---|---|---|---|---|
| changed | context-novel-000000 | Which ID is yellow? | A0 | A2 | A2 |
| changed | context-novel-000017 | Which ID is yellow? | A2 | C4 | C4 |
| changed | context-novel-000025 | Which ID is red? | M0 | K0 | K0 |

Per-color/table/ID and all per-example metrics retained in summary.json. Sender
batch4 saves each chunk and repeats first4 exactly; receiver batch8 repeats all
actual/oracle messages exactly. Both endpoints pass native direct-score<=1e-4,
exact argmax/replay, input/token/message joins and live unchanged-state hashes.
No gradients/hooks/training. Independent audits rehash22snapshot files and the
native parent seal. Eight tests/seven fresh closing checks retained at closure.

| Endpoint | Contexts | Prompt tokens | Scoring s | Job wall s | Peak GiB | Parity max error |
|---|---:|---:|---:|---:|---:|---:|
| sender | 64 | 54 | 1099.986 | 1372.844 | 12.158 | 2.28881836e-05 |
| receiver | 8 | 52 | 31.631 | 137.439 | 2.748 | 5.24520874e-06 |

Total endpoint wall1510.283s/25.17min, no timeout/retry, no workers at
seal. User-authorized no text wall-time kill; native remains cap2400s. USD0new
external, historic~USD30unreconciled. No cloud/browser/Rust/commit/push.
Freeze7fb4c4a273a7229431625c07524b40635bd4a831c9c1bfa9411e7035f3b1d922; independent summaryb78cf0ce39b22882f3b74f9237a9b7cccd11b6a5e4f2d66684c6e1becfcd6683.

Passing establishes small training-unseen positive whole-ID combination
validation on this new context bank. These eight IDs have older-candidate
evaluation/design-selection history; they are not pristine tests or novel
characters. All128candidates/negative CE/components were exposed; endpoint
pretraining novelty is unknown. The original64FINALpool remains reserved.
Native failure rejects this fixed candidate at the novel whole-ID gate regardless
of text success. No second trial,tuning,new adapter or threshold change follows.
Arabic specialist-to-generalist transfer,freegeneration and capability
preservation remain unproven and do not launch automatically.
