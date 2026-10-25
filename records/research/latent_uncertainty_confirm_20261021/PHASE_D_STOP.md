# EXPLORATORY. Phase D stopped at gate G-D3 (2026-10-21, 15:51 local)

Written by the parent session after run R2a. This is a scratch-mode stop on development items. It is not a result about the claim (design section 17, rule 6).

## What ran ([researcher]'s approval, 2026-10-21 about 13:42: [redacted: private chat])

| Run | Time | Result |
|---|---|---|
| D1 download | 13:43 | 2 files, sizes and sha256 equal to fetch_data.py (`data/FETCH_LOG.json`) |
| R0 smoke | 13:44 to 13:52 | parser 0 of 20 view items failed; batched equals single; R1 projection 1.70 h |
| R1 sender, gates | 13:53 to 15:29 | G-D1 pass (dev accuracy 0.2347); G-D4 pass (0.9937); parser failure rate 0.0283 |
| R2a probes, text | 15:29 to 15:51 | G-D2 pass (probe AUROC 0.889, logistic); **G-D3 FAIL** |
| R2b latent, select | not run | Phase D stops when a gate fails (design 9.2; APPROVAL_REQUEST.md section F, step 5) |

## G-D3

Rule: "the Source arm has a lower dev-test NLL than T0". Values (`dev/text.json`, `variants.*.eval.nll`): Source 0.4632, T0 0.4429. Source is also below T0 on AUROC (0.749 against 0.778).

Other dev-test values, for the decision only (scratch mode; no rule uses them): Q-only 0.4772; T2 best format 0.4293; T3 best 0.3927; T4 best 0.4090; T5 best 0.4376; T6 best 0.4316.

## State after the stop

- No job runs. The power guard is stopped.
- Swap use after R2a: about 2,973 MB (limit in run_uconfirm.sh: 2,048 MB). A new run cannot start until it is lower.
- The design gives no fallback for a G-D3 failure. The next step needs [researcher]'s decision.

## Amendment A1 (2026-10-21, about 15:55 local, after the Phase D data of R2a)

[researcher]'s words, in the parent session: [redacted: private chat]

- Reason: on the dev-test items, NLL(Source) = 0.4632 and NLL(T0) = 0.4429 (`dev/text.json`). The receiver (SmolLM2-360M-Instruct) does not use the 120-word evidence window to check the answer. The text arms with numbers show that the receiver reads text: the best T2, T3 and T4 formats have 0.4293, 0.3927 and 0.4090.
- Change: G-D3 is descriptive. It is computed and reported, but it does not stop Phase D. The validity check V2 (NLL(Source) <= NLL(T0)) is also descriptive in Phase C. It is reported as `V2_descriptive`, and it does not enter case 0. The Source arm stays in every run as a descriptive row.
- Cost: the claim loses the evidence-reading ceiling. No reading-rule statement may call Source a ceiling.
- This change came after the dev data of R2a. The preregistration must say so. No confirmatory item was touched.
- Code: `uconfirm.py` (the gate record of stage text, `require_gates` of stages latent and select, the validity dict). The R2a outputs do not change.
- Order: R2b runs after the R.06 T5 and C1 jobs. R2b probably needs more than 120 min; it runs in two parts with the resume of the harness.

## Operations note (2026-10-21, 16:05 local, parent session)

- The unit tests were run after amendment A1. One test still expected the old V2 rule (a bad Source arm gives case 0). The test now expects the amended rule (Source is descriptive). All 36 tests pass.
- The swap limit of `run_uconfirm.sh` changed from 2,048 MB to 3,584 MB. Reason: after R2a, the swap use stayed at about 2,573 MB with no job active, and `memory_pressure` showed 93% free memory. The swap holds old pages, not current memory pressure. This changes a machine check only, not a rule of the design.
- R2b runs now, before T5, because T5 waits for [researcher]'s answer about its gate 4. [researcher] approved R2b and its run in two parts.
