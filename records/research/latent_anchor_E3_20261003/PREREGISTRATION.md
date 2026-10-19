# E3 preregistration: confirmatory test on the 56 pristine novel IDs

Written 2026-10-03, before any FINAL split was built, captured or scored. Authorized by the
researcher ([redacted: private chat]). Development history: `../latent_anchor_program_20260930/RESULTS.md`.
This is the only confirmatory run of the program. No retries, no re-selection, no threshold
changes after FREEZE.json is written.

## Question

Does a channel computed from frozen Qwen3-1.7B hidden states let a frozen SmolLM2-360M-Instruct
receiver answer "Which ID is <color>?" for tables made entirely of ID strings it never saw in
training, including letter-digit pairs that break the training block structure?

## Arms (all checkpoints chosen on development data only and frozen by hash)

**Primary: factored-symbol channel, 3-seed ensemble.** For each source line, closed-form ridge
heads (fit on TRAIN lines, lambda 10) read the letter and digit from the sender's ' =' token state
and the color from the color token state, all at block 13. Hard one-hot head outputs index learned
letter, digit and color embeddings; slot = sum, RMS-normalised; 4 slots. Projector 64 to 960 and
one bounded cross-attention reader after receiver block 15, gate initialised at 1.0. AdamW lr 3e-3,
batch 32, 2,000 steps, seeds 1, 2, 3; each member is its best-step checkpoint on DEV-oob-pure.
Ensemble: log-mean-exp of the three members' 128-candidate scores.

**Secondary: free-form continuous channel, 3-seed ensemble.** Per-line sender slots: slot k =
W_id * mean(LN h[5k], LN h[5k+1]) + W_col * LN h[5k+3], RMS-normalised, no symbols and no heads.
Same receiver side, gate 1.0, AdamW lr 1e-3, batch 32, 2,000 steps, seeds 1, 2, 3, best step on
DEV-oob-pure, same ensembling. On DEV this family reached at most 25-30% on unseen out-of-block IDs.
It is included to test, rather than assume, whether a free-form latent transfers.

Single-member results are reported for both arms. They do not enter the decision.

## Data (built only after FREEZE.json)

Pools: 56 trained known IDs; 56 pristine novel IDs (novel pool minus the 8 DEV-OOB IDs).
The `used` table set is threaded through TRAIN and every DEV split first, so no FINAL table
repeats a development table.

| Split | Construction | Tables | Rows | Seed |
|---|---|---:|---:|---:|
| FINAL-novel-pure | 1 balanced round over the 56 novel IDs | 56 | 224 | 125001 |
| FINAL-mixed | one novel ID per table, 3 trained companions; each (novel ID, color) once | 224 | 896 | 125002 |
| FINAL-known | 1 balanced round over the 56 trained IDs | 56 | 224 | 124001 |

Every table has a wrong-binding twin (same IDs and line order, colors cyclically shifted, every
answer changes). Sender states are Qwen3-1.7B block-13 captures, as in E0.

## Controls (per arm, per split)

correct; shuffled (another table's memory; derangement seed 61001); wrong binding; zero memory;
two random memories norm-matched per slot (seeds 71001, 71002); receiver only (no memory).
Text arms: actual sender-text handoff and oracle copy, unchanged E0 protocol.

## Primary decision (H3), on FINAL-novel-pure, primary ensemble

PASS only if all hold:
1. joint 128-candidate accuracy >= 75%;
2. changed-answer following under wrong binding >= 75%;
3. receiver-only, zero, random_71001 and random_71002 joint accuracy <= 10% each;
4. shuffled joint accuracy <= 10%, computed on rows whose donor answer differs from the own answer.

Otherwise FAIL. No partial credit and no fallback arm.

## Reported without entering the decision

- Secondary arm on the same criteria, labelled PASS/FAIL descriptively.
- FINAL-mixed novel-answer rows and FINAL-known; donor following.
- Accuracy split by collision rows (answer shares its letter or digit with another line); DEV
  never contained collisions among unseen IDs.
- Constrained-sequential accuracy, in-table rate.
- Table-grouped bootstrap 95% intervals (10,000 resamples of tables, seed 125099) for joint
  accuracy and changed-answer following.
- Single-member results; text-arm baselines.

## Analysis integrity

`e3.py` runs freeze, build, capture, score and analyze. `audit.py` independently recomputes every
decision quantity from the saved per-row candidate scores without importing `e3.py`. SEAL.json
hashes every file at closure.

## What a PASS licenses

That this particular channel, sender-computed factored letter/digit/color codes read by a trained
reader in a frozen receiver, communicates color-to-ID bindings for never-seen symbolic IDs. It is
a learned symbolic code, close to a compressed text message, not a free-form latent; the
secondary arm addresses the free-form question separately. No efficiency, natural-language or
Arabic claim follows.

## Development reference (recorded before freeze; the pipeline smoke test on DEV only)

| Arm | DEV-oob-pure ensemble | Members | Changed following |
|---|---:|---|---:|
| Primary (ridge heads) | 96.9% | 81.2, 75.0, 92.2 | 100% |
| Secondary (free-form line slots) | 28.1% | 25.0, 23.4, 23.4 | 26.6% |

Each member's step was selected on DEV-oob-pure, so these DEV figures are optimistic. Expected
FINAL range for the primary ensemble, judged from the less-selected DEV-inblock-pure (85.9%) and
the single-member mean (82.8%): roughly 80-95%. The secondary arm is expected to fail.
