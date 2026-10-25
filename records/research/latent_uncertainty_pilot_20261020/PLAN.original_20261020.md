# Pilot: does a latent channel carry uncertainty that text loses?

Status: EXPLORATORY pilot. One seed, synthetic task, development scale. Not confirmatory.
Written 2026-10-20, before any data of this pilot exists. [researcher] approved this run in "Yo gin":
[redacted: private chat]
[redacted: private chat]

Motivation: arXiv 2606.20662 ("Confidence Laundering in Agent Systems: Why Uncertainty Needs a
Latent Carrier") argues that agent handoffs lose uncertainty and that a latent carrier keeps it.
Its abstract reports no experiment. This pilot is a first test with frozen models.

## Task (synthetic, exact ground truth)

Each item is a box with a hidden true color c* from {red, blue, green, yellow} (uniform).
Five witness reports describe it. Each report is correct with probability q = 0.65, and else names
one of the three other colors uniformly. The sender reads the card with the five reports. The
receiver never sees the card; it sees only the sender's message.

The exact Bayes posterior P(c | reports) is known: the product over reports of q (if the report
names c) or (1 - q)/3 (if not), normalized over the four colors. It is the best possible answer
distribution, and the vote split makes the right confidence vary from item to item.

Splits: 2,000 train items and 1,000 test items, from separate seeds. Box IDs are random 3-character
codes; train and test share no ID.

## Models (frozen, pinned snapshots)

- Sender: Qwen3-1.7B, snapshot 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e.
- Receiver: SmolLM2-360M-Instruct, snapshot a10cc1512eabd3dde888204e902eca88bddb4951.
- Sender state: the residual stream after block 21 (3/4 depth, fixed now), at the last token of the
  answer prompt. Block 28 (last) is recorded as a secondary view only.

## Arms (the message that reaches the receiver)

All text messages use one template: "The box is <color>." plus, where an arm says so,
" Confidence: <d> out of 9." or a per-color list.

- T0, answer only: the sender's answer (argmax of its four color logits).
- T1, verbal confidence: answer plus the digit the sender states when asked how sure it is (0-9).
- T2, token confidence: answer plus round(9 x the sender's own probability of its answer).
- T3, probe scalar in text (control 1): answer plus round(9 x p), where p is a logistic probe on the
  sender state (train split) that predicts whether the sender's answer is correct.
- T4, probe distribution in text (control 2): a 4-way logistic probe on the sender state (train
  split) predicts the true color; the message lists each color with round(9 x p_c).
- L, latent: the sender state goes through a trained linear map into one soft input vector of the
  receiver, before the same question. The map and the readout train on the train split.
- L-shuf (control): L with the sender states shuffled across items.
- Card (ceiling): the receiver reads the whole card itself.
- Bayes (reference, no receiver): the exact posterior.

## Readout (same for every arm)

The receiver processes its message and the question "What color is the box? Answer:". A linear
readout (960 -> 4) on the receiver's final hidden state, trained on the train split, gives the
receiver's color distribution. Every arm gets the same readout capacity. The latent arm also trains
its input map (the channel itself). Secondary view: the receiver's own zero-shot logits over the
four color tokens, with no readout.

## Metrics (test split)

Primary: mean negative log-likelihood (NLL, nats) of the true color. Secondary: Brier score,
accuracy, expected calibration error (15 bins), and the gap to the Bayes NLL.
Uncertainty: paired bootstrap over test items (10,000 draws), 95% CI of each NLL difference.

## Reading rule (fixed now)

1. Latent carries uncertainty that text loses: L has a lower NLL than the best of T0, T1 and T2 by
   at least 0.02 nats, with the 95% CI of the difference above 0, AND L is also lower than T3 and
   T4 by the same rule. Then the latent route holds something that a short text message, even one
   written from the same probe, does not.
2. Better estimator, not a latent carrier: L beats T0-T2, but L is not better than T3 or T4 (CI
   includes 0, or T3/T4 win). Then the uncertainty is in the sender's states, but text can carry it
   once a probe reads it out.
3. No signal: L does not beat T0-T2. Then the latent route adds nothing here.
4. Sanity: L-shuf must be near the "no information" level (NLL near ln 4 = 1.386 or worse), and
   Card must be at least as good as T0. If either fails, report the pilot as invalid.

Nothing in this file changes after the data exists. Any later change is a new, dated section.
