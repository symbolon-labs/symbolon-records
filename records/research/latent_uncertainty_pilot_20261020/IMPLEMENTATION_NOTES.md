# Implementation notes for the uncertainty pilot

Status: EXPLORATORY. Written 2026-10-20, before the smoke test and before any pilot data.
These notes record the choices that PLAN.md does not fix. PLAN.md does not change.
The code is `pilot.py`. The run wrapper is `run_pilot.sh`.

## 1. Data

- Train seed 2026102001. Test seed 2026102002.
- The smoke test uses other seeds (2026102091 and 2026102092). Thus the smoke test does not see pilot items.
- Box IDs have 3 characters from A-Z and 0-9. The test generator rejects each ID that the train split uses.
- Card text (the same text for the sender and for the Card arm of the receiver):

  ```
  Box K7Q. Five witnesses saw this box. Each witness is right 65% of the time.
  Witness 1 says red.
  Witness 2 says blue.
  ...
  Witness 5 says red.
  ```

## 2. Sender (Qwen3-1.7B, bf16 on MPS)

- Answer prompt: the Qwen3 chat template with thinking off. The user turn is the card, a blank line, and
  "Which color is the box most likely? Answer with one word: red, blue, green, or yellow."
  The assistant turn starts with the text "The box is". The last token of the answer prompt is " is".
- Answer (T0): the argmax over the logits of the tokens " red", " blue", " green", " yellow" at that position.
- Token confidence (T2): p is the softmax over these four logits, at the answer color.
- Verbal confidence (T1): a second prompt. It contains the first user turn, the sender answer as an assistant
  turn ("The box is red."), and a new user turn: "How sure are you that your answer is right? Reply with one
  digit from 0 (not sure at all) to 9 (completely sure)." The digit is the argmax over the ten digit tokens
  at the first assistant token (thinking off).
- "The state after block k" is the output of decoder layer k (1-indexed), from a forward hook, at the last
  token of the answer prompt. Block 21 is the main state. Block 28 is before the final norm.
- round(9 x p) is floor(9 x p + 0.5).

## 3. Messages

- T0: "The box is red."
- T1, T2, T3: "The box is red. Confidence: 7 out of 9."
- T4: "The box is red. Confidence: red 7 out of 9, blue 1 out of 9, green 1 out of 9, yellow 0 out of 9."
  The color after "The box is" is the sender answer in all arms. The list always has the order red, blue,
  green, yellow.
- The T3 and T4 probes are L2 logistic regressions on the block-21 state (standardized with train statistics).
  The L2 weight comes from 5-fold cross-validation NLL on the train split.
- Train-split messages for T3 and T4 use out-of-fold probe outputs (cross-fitting). Thus the receiver readout
  sees probe numbers of the same quality as at test. Test messages use the probe fit on the full train split.

## 4. Receiver (SmolLM2-360M-Instruct, fp32 on MPS)

- No chat template, and no special tokens. Input text: "Message: <message>\nWhat color is the box? Answer:".
- The fixed prefix "Message:" is the same in all arms. It keeps the first position (the attention sink)
  the same for all arms, so that the soft vector of the L arm is not at position 0.
- Card arm: the card text is the message.
- L arm: the token embeddings of "Message:", then one soft vector, then the token embeddings of
  "\nWhat color is the box? Answer:".
- Final hidden state: the output of the final norm at the last token (":").
- Zero-shot view: the softmax over the receiver logits of " red", " blue", " green", " yellow" at that position.

## 5. Readout

- T0-T4 and Card: an L2 multinomial logistic regression (960 -> 4, with bias) on the standardized final
  hidden states. The L2 weight comes from the grid {1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1} by 5-fold
  cross-validation NLL on the train split. Then the readout is fit again on the full train split.
- L: the map (2048 -> 960, with bias) and the readout (960 -> 4, with bias) train together with AdamW
  (weight decay 0.01, batch 32, gradient norm clip 1.0, cross-entropy on the true color).
  The input is the block-21 state, standardized with train statistics.
  The map starts with bias = mean receiver token embedding and weight std = (embedding std) / sqrt(2048).
- L hyperparameters: the learning rate comes from {3e-5, 1e-4, 3e-4, 1e-3}, and the epoch count
  (at most 40, patience 6) comes from the NLL on an inner validation part (20% of the train split).
  Then L trains again on the full train split with these values. The test split is not used for any choice.
- The readout capacity is the same in all arms (one linear layer 960 -> 4). The fit method of L is different,
  because its map must train through the receiver.
- L-shuf: the same procedure, learning rate and epoch count as L. The train states and the test states are
  each permuted across items. Secondary view: the trained L with only the test states permuted.

## 6. Metrics and reading rule

- NLL in nats. Brier is the sum over the four colors of (p - y)^2. ECE uses 15 equal bins of the top
  probability on [0, 1].
- Bootstrap: 10,000 paired draws of test items (seed 20261020). The 95% CI is the 2.5 and 97.5 percentiles
  of the mean NLL difference.
- "L beats X" (rule cases 1 to 3): the mean of NLL(X) - NLL(L) is at least 0.02 nats, and the lower
  bound of its 95% CI is above 0.
- "The best of T0, T1 and T2" is the arm with the lowest test NLL point estimate.
- Case 1: L beats the best of T0-T2, L beats T3, and L beats T4.
- Case 2: L beats the best of T0-T2, but L does not beat T3 or does not beat T4. This includes the result
  where L is lower with a CI above 0 but by less than 0.02 nats.
- Case 3: L does not beat the best of T0-T2.
- Sanity (case 4): L-shuf passes if its NLL is at least ln 4 - 0.02 = 1.366. Card passes if its NLL is
  not more than the NLL of T0 (point estimates). If one check fails, the pilot is invalid.

## 7. Outputs

- `smoke/` and `run/`: items, sender outputs, receiver states, L training logs, per-item test
  probabilities of every arm (`test_probs.npz`), and `results.json`.
- `RESULTS.json` (a copy of `run/results.json`) and `REPORT.md` in this folder.

## 8. Change after the smoke test, before any pilot data (2026-10-20, 22:05)

- The smoke test (64 train, 32 test, smoke seeds) ran without error. All arms gave sane numbers.
- In the smoke test, the cross-validation chose the largest L2 weight (1.0) for every probe and readout.
  This is the edge of the grid. Thus the grid is now {1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10} for all
  probes and readouts. No other choice changed.
- Secondary-view fact from the smoke test: with T0 messages, the zero-shot receiver puts " blue" first
  for every item. The zero-shot view is a weak view of this receiver.

## 9. Bayes-9 reference rows (2026-10-20, 22:13, before any pilot result)

- The parent session asked for the dated PLAN.md section "Addition 2026-10-20, 22:13". It adds the
  reference rows "Bayes-9" (literal) and "Bayes-9-cal" (calibrated). See that section for the definitions.
- The full run (pid 2594) started at 22:05 with the code before this change. After it ends, `pilot.py`
  runs again on the cached outputs. Only the analysis stage runs, and it adds these rows.
- From the generating model only: the exact population cost of the 1/9 rounding for a calibrated reader
  is 0.0042 nats (Bayes NLL 0.3493, Bayes-9-cal NLL 0.3535).
