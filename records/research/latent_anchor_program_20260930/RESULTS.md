# Results log (scratch mode)

## 2026-10-01: E0, phase A, temperature screen

**E0 (done).** Splits v2, captures and text arms built; port validated against sealed V1
artifacts (`E0_data/VALIDATION.json`, all checks pass, lab.py 36eaec6f).
Text baseline (sender answer, then receiver copy): DEV-known 94.6%, DEV-oob-pure 96.9%,
DEV-inblock-pure 98.4%; oracle copy 100% everywhere.

**Phase A (8 runs, unchanged bridge 8x64, seed 1, lr 3e-4 to 1e-2, E1 and E2).** The bridge
does not fit the task. Best training-subset accuracy 6.7%, best DEV-known 6.7%, DEV-oob-pure
0% in every run (one E2 checkpoint at lr 1e-2 hit 4/64, not reproduced at the final step).
Training CE falls only from 4.85 to about 3.7 in 2,000 steps; lr 1e-2 collapses the reader gate
to 0. H1 and H2 cannot be evaluated: there is no fitted model to transfer. E2 symbols changed
nothing measurable.

**Temperature screen (6 runs, 1,000 steps).** Raising the attention scale from 2 to 10 (fixed or
learnable) sharpens attention as intended: resampler entropy 2.93 -> ~1.0, reader entropy
2.08 -> ~1.4. Fit does not improve: training accuracy <= 6.7%, DEV-oob-pure 0%.

**Reading.** Near-uniform attention was a real symptom but not the binding constraint. In all
14 runs the reader gate stalls near 0.05 and training CE stays near 4, so the reader's update at
receiver block 15 barely moves the answer distribution. The full 60-run E1 grid was not launched:
it would vary seed, lr and width around a model that does not fit.

**Next (not started).** A receiver-side ceiling test: replace the sender path with an oracle
memory (learned per-ID vectors placed in slots tagged by color) and train only the projector and
reader on TRAIN. If that cannot fit either, the bottleneck is the single gated reader at block 15
and the fix is on the receiver side (larger gate init, multiple readers or sites, or the sealed
typed-slot reader of candidate d0, which did reach 32/32 on familiar IDs). If it fits, the
bottleneck is sender-to-memory, and E1/E2 resume with that reader.

## 2026-10-02: receiver-side ceiling (oracle memory)

Gold symbolic memory: one slot per source line, learned color + letter + digit embeddings, no
sender. Only the oracle embeddings, projector and reader train (k=4, sealed reader, gate 0.01).

| Run | DEV-known | DEV-oob-pure | DEV-inblock-pure | Changed following (oob) |
|---|---:|---:|---:|---:|
| lr 1e-3, best step 1000 | 100% | 42.2% | 37.5% | 32.8% |
| lr 3e-3, best step 800 | 100% | 67.2% | 54.7% | 67.2% |

**Reading.** The single gated reader at block 15 can drive the receiver: it fits trained IDs on
fresh tables perfectly and transfers partway to unseen out-of-block IDs. So the phase-A failure
is on the sender-to-memory side (the learned-query resampler), not the reader. Training takes
off only after about 300 steps, so slow early curves are not evidence of failure on their own.
Next: per-line sender slots (R1), which keeps the oracle's structure but reads values from the
frozen sender's ID-token and color-token states.

### Gate initialization (oracle memory)

| Run | Best step | DEV-known | DEV-oob-pure | DEV-inblock-pure | Changed following (oob) |
|---|---:|---:|---:|---:|---:|
| lr 1e-3, gate 0.01 | 1000 | 100% | 42.2% | 37.5% | 32.8% |
| lr 1e-3, gate 1.0 | 300 | 100% | 65.6% | 43.8% | 64.1% |
| lr 3e-3, gate 0.01 | 800 | 100% | 67.2% | 54.7% | 67.2% |
| lr 3e-3, gate 1.0 | 800 | 100% | **98.4%** | 81.2% | **98.4%** |

Starting the reader gate at 1.0 instead of the sealed 0.01 lets a compositional memory carry
almost perfectly to unseen out-of-block IDs, with the binding intact (changed-answer following
98%). In every sealed run and in phase A the gate stalled near 0.05. The per-line sender-slot run
at gate 0.01 was stuck near 5% after 600 steps and was stopped in favour of the gate-1.0 queue
`gate1` (line slots, resampler bridge, symbol channel).

## 2026-10-02: sender paths at gate 1.0 (queue `gate1`, seed 1, best step by DEV-oob-pure)

| Sender path | lr | DEV-known | DEV-oob-pure | DEV-inblock-pure | Changed following (oob) |
|---|---:|---:|---:|---:|---:|
| Oracle memory (reference) | 3e-3 | 100% | 98.4% | 81.2% | 98.4% |
| Per-line sender slots (R1) | 3e-3 | 99.1% | 15.6% | 29.7% | 18.8% |
| Per-line sender slots (R1) | 1e-3 | 95.1% | 25.0% | 34.4% | 25.0% |
| Learned-query resampler (E1) | 3e-3 | 15.6% | 0% | 1.6% | 0% |
| Learned-query resampler (E1) | 1e-3 | 17.9% | 1.6% | 1.6% | 1.6% |
| Resampler + symbols (E2) | 3e-3 | 17.4% | 1.6% | 0% | 0% |

**Reading.** Two brakes, removed in turn. The 0.01 gate stops everything from fitting. With
gate 1.0, the learned-query resampler is the second brake: the same sender states fit trained IDs
perfectly once each line gets its own slot. With per-line slots, binding of trained IDs is solved
(DEV-known 95-100%). What remains is transfer to unseen IDs: 16-25% out-of-block versus 98% for
the oracle's explicit letter and digit embeddings, and accuracy on unseen IDs falls as DEV-known
approaches 100% (overfitting to trained ID states). Next queue `lines2`: factorized letter/digit
maps, and sender-side symbols on the per-line slots (E2-line).

## 2026-10-02: variants on per-line slots, and a sender probe

| Variant (gate 1.0, lr 3e-3) | Seed | DEV-known | DEV-oob-pure | Hybrid: own / donor / positional |
|---|---:|---:|---:|---|
| Factorized letter/digit maps (R2) | 1 | 87.9% | 1.6% | n/a |
| Line slots + symbols (E2L) | 1 | 98.7% | 29.7% | 0 / 28.1% / 3.1% |
| Line slots + symbols (E2L) | 2 | 98.2% | 23.4% | 0 / 21.9% / 3.1% |
| Factorized + symbols | 1 | 92.4% | 0% | 0 / 0 / 0 |
| Symbols carry identity, sender gives color only (symid) | 1 | 81.7% | 7.8% | 0 / 1.6% / 1.6% |

Symbols added next to a continuous identity term are ignored (the hybrid arm follows the donor's
continuous states, not the symbols). Removing the continuous identity term does not help, because
the color input is itself entangled with identity (below).

**Sender probe (ridge, trained on TRAIN lines, CPU, no receiver).** Linear decodability of each
line's components from the layer-normed block-13 state at each position, tested on DEV-oob-pure
(unseen out-of-block IDs):

| Position | Letter | Digit | Color |
|---|---:|---:|---:|
| letter token | 100% | 3.1% | 39.1% |
| digit token | 100% | 96.9% | 35.9% |
| ' =' token | 100% | 98.4% | 45.3% |
| color token | 100% | 76.6% | 100% |

The sender represents unseen out-of-block identities linearly and compositionally: a probe fit on
56 in-block IDs reads their letters and digits almost perfectly. The color token also carries the
line's letter perfectly, which explains the symid failure. So the bottleneck is not sender
information. Continuous maps let the receiver memorize whole-ID features, while the oracle
succeeded because its memory is factored by construction. Next queue `factored`: linear heads on
sender states produce soft letter, digit and color distributions, and slots are their weighted
embeddings (pure latent, and with auxiliary head supervision from the sender's own source text).

**Identity forcing on line slots (queue `lines3`) failed.** Symbol-only identity: DEV-oob-pure
7.8% and 9.4% (seeds 1, 2). Identity dropout 0.5: 12.5% and 9.4%. Both fit DEV-known 69-99%.

## 2026-10-03: sender-derived factored memory (queue `factored`) - best result so far

Linear heads on frozen sender states produce soft letter and digit distributions (from each
line's ' =' token) and a soft color distribution (from its color token). Each slot is the
probability-weighted sum of learned letter, digit and color embeddings. At test time the memory
is computed only from sender hidden states. "Aux" adds cross-entropy on the heads against the
sender's own source-text components during training.

| Run | Best step | DEV-known | DEV-oob-pure | DEV-inblock-pure | Changed following | Donor following | Shuffled / random / zero / receiver-only |
|---|---:|---:|---:|---:|---:|---:|---|
| Pure latent, seed 1 | 0 | 1.8% | 0% | 0% | 0% | 0% | - |
| Aux, seed 1 | 400 | 99.1% | 60.9% | 71.9% | 64.1% | 62.3% | 1.6% / 0% / 0% / 0% |
| Aux, seed 2 | 2000 | 99.1% | 73.4% | 59.4% | 78.1% | 73.8% | 3.1% / 0% / 0% / 0% |

DEV-known collision rows: 98-99% (no collision penalty on trained IDs).

**Reading against the revised hypotheses (DEV only, not a confirmatory claim).**
- H1 (unchanged continuous bridge transfers): falsified. Best DEV-oob-pure 1.6% across every
  resampler run, with or without gate 1.0, temperature or symbols.
- H2 (sender-side factored channel, receiver still binds): gain over matched E1 is 59-72 points
  and controls stay at or below 3.1%, but changed-answer following is 64% and 78%, below the 80%
  bar. Inconclusive by the preregistered rule, close on seed 2.
- The pure-latent version never trained (reader entropy stuck at ln 4): the softmax heads need the
  auxiliary signal to start.

**Caveat on what this channel is.** Each slot carries soft distributions over 16 letters, 8
digits and 4 colors, trained toward the source text. It is a learned, sender-computed, factored
symbolic code, much closer to a compressed text message than to an unconstrained latent. That is
the honest framing for any write-up.

**Operational note.** From about 23:10 UTC on 2026-10-02, jobs launched in the background were
throttled to roughly a tenth of normal speed (GPU busy, process idle); the identical code ran at
full speed in the foreground. Runs launched from the app's Terminal panel run at full speed.
The factored aux seed-2 run was restarted three times for this reason; only the terminal run
(logs/factored5.log) completed and is reported.

**Next options.** (1) More seeds and a small lr/step grid for factored-aux, selecting on
DEV-oob-pure only. (2) A head-accuracy diagnostic on DEV-oob-pure: the probe reads components at
98-100%, yet transfer is 61-73% against the oracle's 98%, so the loss is either in the heads or
in co-adapted embeddings. (3) If (1) reaches the H2 bar on DEV, preregister E3 on the 56 pristine
novel IDs with that single configuration.

## 2026-10-03: factored-memory grid and frozen ridge heads

DEV-oob-pure joint at the best step (selected on DEV-oob-pure), all with gate 1.0, k=4:

| Variant | lr | Seeds 1-5 | Mean |
|---|---:|---|---:|
| Learned heads + aux | 3e-3 | 60.9, 73.4, 67.2, 76.6, 79.7 | 71.6% |
| Learned heads + aux | 1e-3 | 53.1, 73.4, 68.8, 67.2, 51.6 | 62.8% |
| Frozen ridge heads (hard one-hot) | 3e-3 | 81.2, 75.0, 92.2 (seeds 1-3) | 82.8% |

Best single run: ridge heads seed 3, DEV-oob-pure 92.2%, changed following 92.2%, donor following
91.8%, DEV-known 100%. Controls stayed at or below about 3% wherever measured.

**Diagnostic.** Learned heads on DEV-oob-pure read letters at 94% and digits at 91-92%, so the
answer line is fully correct in the heads only 84-86% of the time; ridge heads (fit in closed form
on TRAIN lines) read 100% / 98.4%. Hard one-hot evaluation of learned-head checkpoints helped one
seed (76.6 -> 84.4%) and hurt another (67.2 -> 62.5%). With near-perfect ridge symbols, end-to-end
still ranges 75-92% across seeds, so the remaining variance is in how well the receiver composes
unseen letter-digit pairs. The single oracle run (98.4%) may have been a favourable seed; four
oracle replicates (queue `oracle_seeds`) were queued and then paused at the user's request before
any finished.

**H2 on DEV.** Ridge heads clear every H2 bar on seed 3 and miss the 80% following bar narrowly on
seeds 1-2 (76.6%, 78.1%). Learned heads + aux seed 5 (lr 3e-3) also clears it (changed following
84.4%). Seed variance is large, so E3 must fix one configuration and one seed in advance.

**Paused 2026-10-03.** Resume from the Terminal panel:
`caffeinate -dims ../../.venv/bin/python -B sweep.py oracle_seeds > logs/oracle_seeds.log 2>&1`

## 2026-10-03: oracle replicates (resumed after pause)

DEV-oob-pure joint at the best step, gate 1.0, lr 3e-3, k=4:

| Memory | Seeds | Mean (sd) | Changed following mean |
|---|---|---:|---:|
| Oracle, gold symbols | 98.4, 76.6, 71.9, 87.5, 90.6 | 85.0% (9.6) | 82.8% |
| Ridge heads on sender states | 81.2, 75.0, 92.2 | 82.8% (7.1) | 82.3% |
| Learned heads + aux on sender states | 60.9, 73.4, 67.2, 76.6, 79.7 | 71.6% (6.7) | 74.4% |

The max control (shuffled, random, zero, receiver-only) is at or below 4.7% in every run; the
4.7% is the 3 of 64 rows whose donor answer equals the own answer. DEV-known is 99-100%
everywhere.

**Reading.** The single 98.4% oracle run was a favourable seed. Even with perfect symbols the
receiver composes unseen out-of-block pairs at 72-98% depending on seed. The sender-derived
ridge-head channel (82.8%) is statistically indistinguishable from that ceiling on 3 vs 5 seeds,
so the sender side is essentially solved; the remaining gap is in the receiver's compositional
read, and it is mostly seed variance. Learned heads are about 11 points below because their
heads misread 6-9% of unseen components.

**Implication for E3.** A confirmatory run must fix one configuration and one seed in advance.
Ridge heads at seed 3 is the best DEV point (92.2%), but picking the best of three seeds on DEV
overstates the expected FINAL score; the honest expectation is the 3-seed mean, about 83%, with
a 75% bar (H3) passed by 2 of 3 seeds. Reducing seed variance in the receiver read (for example
an ensemble of seeds, or more training diversity) would make E3 far more informative than
picking a seed.

## 2026-10-03: seed ensembles (no training; saved best-step checkpoints, hard heads)

Ensemble = mean of candidate probabilities across members (log-mean-exp of the 128-candidate
scores).

| Ensemble | Split | Joint | Changed following | Donor following | Shuffled |
|---|---|---:|---:|---:|---:|
| Ridge heads, seeds 1-3 | DEV-known | 100% | 100% | 100% | 0% |
| Ridge heads, seeds 1-3 | DEV-oob-pure | **96.9%** | **100%** | 96.7% | 4.7% |
| Ridge heads, seeds 1-3 | DEV-inblock-pure | 85.9% | 87.5% | 86.9% | 3.1% |
| Oracle, seeds 2-5 | DEV-oob-pure | 90.6% | 89.1% | 90.2% | 4.7% |
| Oracle, seeds 2-5 | DEV-inblock-pure | 98.4% | 93.8% | 98.4% | 4.7% |

Shuffled 4.7% = the 3/64 rows whose donor answer equals the own answer (expected floor).

**Reading.** A three-seed ensemble of the sender-derived ridge-head channel transfers to unseen
out-of-block IDs at 96.9% with perfect changed-answer following, matching the text baseline
(96.9%). Ensembling removes most of the per-seed receiver variance.

**Caveat (selection).** Each member's checkpoint step was chosen on DEV-oob-pure, so these DEV
numbers are optimistic; DEV-inblock-pure, which only entered as a tie-breaker, is the less-selected
estimate (85.9%). The clean test is E3 on the 56 pristine FINAL IDs.

**Proposed E3 configuration (not run).** Ridge heads fit on TRAIN lines (lambda 10), hard one-hot
symbols, factored memory k=4 w=64, reader gate init 1.0, AdamW lr 3e-3, batch 32, 2,000 steps,
seeds 1, 2, 3, each member's step chosen on DEV-oob-pure, ensemble by log-mean-exp. Decision on
FINAL-novel-pure (56 IDs, 224 rows) per H3: joint >= 75%; shuffled, random, zero, receiver-only
<= 10%; changed following >= 75%. Report collision and non-collision rows separately (FINAL tables
repeat letters and digits; DEV-oob-pure never does). Single-model results (each seed) reported
alongside the ensemble.

## 2026-10-03: E3 confirmatory result - PASS

Sealed tree: `../latent_anchor_E3_20261003/` (REPORT.md, RESULT.json, AUDIT.json, SEAL.json).
FINAL-novel-pure (56 never-seen IDs, 224 rows), primary 3-seed ridge-head factored ensemble:
joint 93.8% (95% CI 90.6-96.4), changed-answer following 95.1%, every control <= 1.3%.
Members 72.3 / 76.8 / 83.5%. Text handoff 97.3%. Collision rows 95.0% vs 92.7% without.
Secondary free-form per-line slots ensemble: 14.3% (FAIL). FINAL-mixed novel rows 92.0% vs 17.9%.
Independent audit identical. The program's open question is answered for symbolic IDs: a factored
sender-computed code transfers; a free-form continuous latent does not.

## 2026-10-03: E5 - free-form continuous slots under factoring pressure (scratch, DEV only)

Hypothesis, written before any E5 result: adding training-time linear probes on each continuous
slot (predict the line's letter, digit and color; probes are not in the memory path) lifts
free-form transfer on DEV-oob-pure from about 25% (E3 secondary arm on DEV) to >= 50%.
Falsified if every run stays below 35%. Also isolates the input: identity from the ' =' token
(probe-decodable at 98-100%) instead of the mean of the two ID-token states.
Queue `e5`: aux weight 1 and 10 (seeds 1, 2), ' =' input without and with aux weight 1 (seed 1).
Note: DEV-oob-pure is burned for development only; no new confirmatory ID space exists yet.

### E5 results (DEV-oob-pure joint at the best step; lr 1e-3, gate 1.0, k=4)

| Variant | Seed 1 | Seed 2 | Slot probe accuracy on unseen OOB (letter/digit/color) |
|---|---:|---:|---|
| Continuous line slots, no probes (E3 secondary, for reference) | 25.0% | 23.4% | n/a |
| + slot probes, weight 1 | 82.8% | 68.8% | 97-100% |
| + slot probes, weight 10 | 81.2% | 75.0% | 100% |
| ' =' input, no probes | 15.6% | - | n/a |
| ' =' input + probes, weight 1 | 48.4% | - | 92-100% |

Ensembles (log-mean-exp, no training): weight-10 pair 82.8% on DEV-oob-pure (changed 87.5%,
in-block 89.1%); all four probe runs 89.1% (changed 92.2%, in-block 85.9%, DEV-known 100%,
shuffled 0%). For comparison, the E3 ridge-head ensemble scored 96.9% / 85.9% on the same DEV splits.

**Reading. H5 supported on DEV.** Training-time factoring pressure turns a free-form continuous
channel that does not compose (about 25%) into one that does (69-83% single, 89% ensembled). The
memory path stays fully continuous: no heads, symbols or argmax at test time; the probes only shape
training. The slots end up linearly encoding unseen IDs' components almost perfectly, and the
receiver reads them compositionally. Input choice is not the driver: the ' =' input without
probes stays at 16%. Weight 10 replicates more tightly than weight 1 (75-81% vs 69-83%), and
weight 1 overfits late on seed 2 (68.8% best, 51.6% final).

**Caveats.** DEV only and DEV-selected best steps. The probes are supervised with the sender's own
source-text components, so the latent is free-form in form but trained toward a factored target.
A confirmatory test needs a new pristine ID space: the 56 FINAL IDs were used up by E3.

## 2026-10-05: EXP 0, cross-model geometry map (scratch; `exp0_geometry.py`, `runs/exp0_geometry.json`)

No bridge training. Qwen3-1.7B (depths 7/14/21/28; 14 = E3 tap) and SmolLM2-360M (depths 0/8/16/24; 16 = E3
reader point) read the same raw tables. Units: letter / digit / color fields, aligned by character span. Maps and
probes fit on train tables (+ twins), evaluated on DEV OOB tables (IDs never in training). Units touching token 0
are dropped (no-BOS attention-sink outliers); per-dimension z-score. A first run without those two fixes gave
meaningless error/CKA (outlier-dominated) and a wrong binding story (receiver "has no binding code"); superseded.

DEV OOB, probe accuracy (receiver-trained probe on receiver states / on orthogonally rotated sender states /
sender-trained probe on sender states). Chance: letter 6%, digit 12.5%.

| S->R depth | ortho err | linear err | identity (rot) | order (rot) | binding letter r / rot / s | binding digit r / rot / s |
|---|---:|---:|---:|---:|---|---|
| 7->8   | 0.39 | 0.18 | 100% | 97% | 95 / 89 / 100 | 92 / 75 / 94 |
| 7->24  | 0.42 | 0.19 | 100% | 95% | 99 / 92 / 100 | 94 / 73 / 94 |
| 14->16 | 0.54 | 0.26 | 100% | 100% | 99 / 83 / 97 | 88 / 64 / 85 |
| 21->24 | 0.32 | 0.19 | 100% | 100% | 99 / 88 / 98 | 94 / 82 / 98 |
| 28->24 | 0.39 | 0.20 | 100% | 97% | 99 / 74 / 99 | 94 / 72 / 95 |

Property (color) survives at 100% everywhere; number 91-100%. Receiver depth 0 (embeddings) has binding at
chance, as it should (sanity check).

**Reading.** Both models encode binding linearly at the value token, even for fresh IDs. One rotation carries
property, identity and order almost intact; binding is the lossiest part (letter 74-92%, digit 64-82%), so the
ordering semantic > identity > binding holds, but mildly. The E3 tap (depth 14) is the worst of the four sender
depths for binding; depth 7 and depth 21 are better. Next (EXP 1): use the rotation itself as the bridge,
no training, on the binding battery, to see whether the free-form failure was geometry or the learned bridge.

## 2026-10-05: receiver competence check (before EXP 1)

Sender and receiver tokenize the tables identically (19 tokens, same pieces), so a token-level splice is possible.
But SmolLM2-360M reading the table and query as text (SENDER_INSTRUCTION + table + query, chat template,
128-candidate scoring; `E0_data/smol_text_competence.py`) is at chance among the four table IDs:

| split | original | changed table |
|---|---:|---:|
| dev_known (224) | 29.9% | 30.4% |
| dev_oob_pure (64) | 25.0% | 25.0% |
| dev_oob_mixed (128) | 28.1% | 28.9% |

**Reading.** The receiver cannot do the lookup itself. In every bridge in this program the trained reader and
bridge did the binding, and the receiver mainly emitted the ID. That explains the pattern: free-form latents fail
on fresh IDs (the bridge must invent composition), factored codes pass (they hand over composition ready-made),
and a geometry-only bridge (EXP 1 as proposed) is capped at SmolLM's own ~25-30%. EXP 1 is not run on this pair.
A real test of "does geometry carry binding" needs a receiver that solves the text task (for example
SmolLM2-1.7B-Instruct, cross-family, if it scores well on text).
