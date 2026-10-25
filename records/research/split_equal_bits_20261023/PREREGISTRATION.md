# PREREGISTRATION: split learning at equal bits, activations against a symbolic message (CIFAR-10)

**Status: PREREGISTRATION, frozen by `FREEZE.json` before the first training step.** After the freeze, a change to this file, to `sebits.py`, to `test_sebits.py` or to `run_sebits.sh` stops every job (each job checks the sha256 values).

- Written: 2026-10-23, by a Claude Code session for [researcher] (Symbolon Labs). Style: ASD-STE100.
- Design: `reports/split_equal_bits_design_20261023.md` (below: "the design"). This file copies the rule of the design and fixes every choice that the design left open (Part 9). Part 10 lists each difference from the design.
- Mode: J0 is scratch mode. J1 to J6 are confirmatory mode (`ember/AGENT.md`, "Research modes").
- Purpose: defensive measurement. The test measures how much of a client's input each message format lets a trained decoder rebuild. It uses a public data set and the lab's own small models. It does not attack a real system.

---

# Part 1. Summary

- **Question.** In split learning, a client sends one message to a server for each image. At an equal cap of bits, compare an activation message (RAW, BOT) with a symbolic message (TOPK, LAB) on two axes: test accuracy, and input leakage to a trained decoder (PSNR).
- **Primary pair.** SYM = TOPK, ACT = BOT, at cut C2 and B* = 64 bits.
- **Primary rule.** Design section 8, copied in Part 7. Cases S, E, X, T, A, U and 0.
- **Seeds.** Run seeds 20261111, 20261112 and 20261113. Split seed 20261110. Bootstrap seed 20261024. Scratch seed 20261100 (J0 only).
- **Jobs.** J0 (smoke), J1 to J5 (training, messages, decoders), J6 (analysis). Each job runs for at most 120 min of running time. A job with a projected time above 110 min runs in parts (Part 8.3).
- **Not in this run.** Cut C1 (J7, J8a, J8b) and the end-to-end variant (J9). [researcher] did not approve them.

---

# Part 2. Data

## 2.1 Archive

- File: `cifar-10-binary.tar.gz` from `https://cave.cs.toronto.edu/kriz/cifar-10-binary.tar.gz`. The old address `https://www.cs.toronto.edu/~kriz/cifar.html` sends a permanent redirect (301) to `cave.cs.toronto.edu`.
- The dataset page (read 2026-10-23) lists 162 MB and the md5 `c32a1d4ab5d03f1284b67883e8d87530` for the binary version. It states no license. It asks users to cite Krizhevsky (2009), "Learning Multiple Layers of Features from Tiny Images". It does not mention the 2020 withdrawal of 80 Million Tiny Images. Treat the data as research-only data. Publish no rebuilt image beyond small figure panels.
- The download goes into the new empty folder `data/download/`. The data step (`sebits.py data <archive>`) does these steps:
  1. It computes the sha256 and the md5 of the archive. If the md5 is not the md5 of the page, it stops.
  2. It reads the six members `cifar-10-batches-bin/data_batch_1.bin` to `data_batch_5.bin` and `test_batch.bin` with `tarfile`, in memory. It extracts no file to disk. Each member must be a file of 30,730,000 bytes.
  3. It reads each record (1 label byte, then 1,024 R, 1,024 G and 1,024 B bytes, row-major) with numpy. It checks 5,000 train images and 1,000 test images for each class.
  4. It writes `data/cifar10_train.npz` (x, y), `data/cifar10_test_images.npz` (x) and `data/cifar10_test_labels.npz` (y). No file is a pickle.
  5. It writes the archive record (URL, bytes, sha256, md5) and the sha256 of the three npz files into the `data` field of `FREEZE.json`. This is the only change to `FREEZE.json` after the freeze, and it occurs one time.
- Every later job checks the sha256 of each npz file that it opens.

## 2.2 Splits (split seed 20261110)

- `numpy.random.default_rng(20261110)`. For each class 0 to 9, in this order: a permutation of the train indices of the class. The first 70% (3,500) go to D_task, the next 10% (500) to D_dev, the last 20% (1,000) to D_aux. Each split is then sorted by index.
- Then, from the same generator, in this order: one flip bit (probability 0.5) for each image of D_aux, of D_dev and of the test split.
- Then the membership subsets, from the same generator: 10,000 random positions in D_task (the first 5,000 are the train members, the next 5,000 the test members); 5,000 random positions in D_aux (train non-members); 5,000 random positions in the test split (test non-members). These subsets are not stratified by class, so that no subset needs a test label.

| Split | Size | Use |
|---|---|---|
| D_task | 35,000 | Train N and every arm. The "members". Quantizer cells and RAW channel ranks. |
| D_dev | 5,000 | Selection, temperatures, g_fill, decoder and probe epoch selection, checks V1, V2 and V6 |
| D_aux | 10,000 | Decoders and probes. Never used to train N or an arm. |
| Test | 10,000 | Test messages, predictions and reconstructions (J2 to J5); all test metrics (J6 only) |

## 2.3 Normalization and augmentation

- Normalization: the mean and the population standard deviation of each channel of the D_task pixels in [0, 1].
- Augmentation (N and every arm): a zero pad of 4 pixels in pixel space, a crop at a uniform offset from 0 to 8 in each axis, and a horizontal flip with probability 0.5. Then the normalization. The random values come from a CPU generator of the unit.
- No augmentation for messages on D_dev, D_aux and test, for decoders and for probes. The decoder data are D_aux and its horizontal flips (Part 5.2).

## 2.4 Test-label boundary

- J0 opens the train file only. In J0, D_dev takes the role of the test split.
- J1 opens the train file only.
- J2 to J5 open the train file and the test images. They write test messages, test predictions and test reconstructions. They compute no test metric.
- Only J6 opens `cifar10_test_labels.npz`. The code raises an error if a job other than J6 asks for test labels.

---

# Part 3. Models and training

## 3.1 Network N (one for each run seed)

- Three stages. Each stage has two 3×3 conv layers (no bias), each with BN and ReLU, and one 2×2 max pool. Channels 32, 64, 128. Then global average pool and a linear layer 128→10.
- Conv parameters: stage 1 10,080; stage 2 55,296; stage 3 221,184; head 1,290 (a unit test checks them).
- SGD, Nesterov momentum 0.9, weight decay 5e-4 on all parameters, batch 128, 30 epochs of 273 steps (the last part batch is dropped).
- Learning rate for each step: a linear warm-up from 0.05/546 to 0.05 over the first 546 steps (2 epochs), then a cosine decay from 0.05 to 0 over the other steps.
- V6: the D_dev accuracy of N must be 0.80 or more.

## 3.2 Frozen client

- After N trains, stages 1 and 2 are frozen and stay in eval mode (BN running statistics fixed). Every arm reads h2 (64×8×8) from this client, without gradient.
- V3: the sha256 of the parameters and buffers of stages 1 and 2 is recorded after N trains. It is checked before and after each unit that trains an arm or a decoder, and in the message and selection units.

## 3.3 Arms at cut C2

| Arm | Client part after the frozen client | Message | Server part | Trained parts and learning rates |
|---|---|---|---|---|
| RAW (c, s, q) | Average pool of the c ranked channels of h2 to s×s; quantile cells of q bits | c·s²·q bits | Nearest upsample to 8×8, zeros for the dropped channels, stage 3 and head of N | Server (from N): 3e-4 |
| BOT (c', s', q) | 1×1 conv 64→c' (with bias), average pool to s'×s', tanh, uniform q-bit levels with a straight-through gradient | c'·s'²·q bits | 1×1 conv c'→64 (no bias), BN, ReLU, nearest upsample to 8×8, stage 3 and head of N | Encoder and adapter (new): 1e-3; server (from N): 3e-4 |
| TOPK | Head: 1×1 conv 64→16 (no bias), BN, ReLU, flatten (1,024), linear 1,024→256, ReLU, linear 256→10; then a TOPK format (Part 4.3) | at most 148 bits | Rebuilt logits / T | Head (new): 1e-3 |
| LAB | The same head; the message is the first class of a stable descending sort of its logits | 4 bits | The label is the prediction; the class distribution is softmax(log(row) / T), where row is the add-one smoothed D_dev confusion row of the label | None beyond the head |

- Arm training: AdamW (betas 0.9 and 0.999, eps 1e-8), weight decay 1e-4 on all trained parameters, batch 128, 15 epochs of 273 steps, a cosine decay from the base rate to 0 for each step (no warm-up), the augmentation of Part 2.3, cross-entropy with the D_task labels.
- The TOPK head trains on its float logits. The format is applied after the training.
- RAW in training: the server reads the decoded levels of the quantizer (the same values as decode(unpack(pack(encode(h2))))). BOT in training: the straight-through value t + (t_q − t).detach().
- Temperature: after the training, each arm (LAB and LAB-full included) gets one softmax temperature T from the grid 0.50, 0.55, ..., 3.00 (51 values). T minimizes the D_dev NLL (float64). If two values give the same NLL, the lower T is kept. For LAB and LAB-full, the logits are the log of the confusion row.

## 3.4 Reference rows

| Row | What it is | Bits | Notes |
|---|---|---|---|
| LAB-full | The first class of the logits of N, run on the client | 4 | D_dev confusion row of N; T from D_dev |
| TOPK-full | A TOPK format of the logits of N at 64 bits | at most 64 | Format and g_fill selected on D_dev in seed 1 (as Part 6); seeds 2 and 3 use the seed-1 format and g_fill multiplier; cells and T are fit again in each seed |
| TOPK-float | The 10 logits of the TOPK head as float32 | 320 | T from D_dev; decoder in seed 1 only |
| BOT-float | A BOT with the selected B* configuration, trained with no quantizer; float32 values | 32 for each value | Seed 1 only; accuracy and NLL only, no decoder |
| RAW-native | h2 at full size, 8-bit quantile cells for each channel: (64, 8, 8) | 32,768 | The server is stage 3 and head of N with no further training (a standard split of N); T from D_dev |
| F0 | No message: the mean image of the decoder training targets | 0 | Part 5.4 |
| FY | The true label (oracle): the mean image of each class of the decoder training targets | n/a | Part 5.4 |

## 3.5 Candidates at each budget

RAW (c, s, q): channels ranked by the population variance of their global average on D_task (no labels; no augmentation; a tie goes to the lower channel index).

| B | Candidate 1 | Candidate 2 | Candidate 3 |
|---|---|---|---|
| 16 | (16, 1, 1) | (4, 1, 4) | (2, 1, 8) |
| 64 | (64, 1, 1) | (16, 1, 4) | (8, 1, 8) |
| 256 | (64, 1, 4) | (64, 2, 1) | (32, 1, 8) |
| 1,024 | (64, 2, 4) | (64, 4, 1) | (16, 4, 4) |

BOT (c', s', q):

| B | Candidate 1 | Candidate 2 | Candidate 3 |
|---|---|---|---|
| 16 | (16, 1, 1) | (4, 1, 4) | (4, 2, 1) |
| 64 | (16, 1, 4) | (4, 2, 4) | (16, 2, 1) |
| 256 | (64, 1, 4) | (16, 2, 4) | (16, 4, 1) |
| 1,024 | (16, 4, 4) | (64, 2, 4) | (16, 8, 1) |

TOPK formats: Part 4.3.

---

# Part 4. How bits are counted

## 4.1 Rules (design 5.1)

1. **Unit.** Bits for each image, client to server, at test time.
2. **What is counted.** The bit string that `pack` writes for one image. The server, the decoder and the probes read only the output of `unpack` on that string, decoded to numbers.
3. **Fixed-length codes only.** Each arm writes the same number of bits for every image. No entropy code. No code across images.
4. **Cap.** An arm at budget B writes at most B bits. RAW and BOT write exactly B bits. LAB writes 4 bits. TOPK writes the bits of its format (at most 148).
5. **Free parts (0 bits).** The architectures, the weights, the quantizer cells and levels, the RAW channel list, the class order, the TOPK g_fill value, the LAB confusion rows and the temperatures.
6. **Class index codes.** An ordered list of k different classes from 10 has a lexicographic rank among the 10!/(10−k)! lists. The rank is written in ceil(log2(10!/(10−k)!)) bits: k = 1 to 5 gives 4, 7, 10, 13 and 15 bits. A unit test checks all lists for k = 1 to 5.
7. **Value codes.** Each value goes through a fixed scalar quantizer with 2^q cells and costs q bits.
   - RAW values and TOPK gaps: quantile cells. The 2^q − 1 edges are the quantiles j/2^q (j = 1 to 2^q − 1; numpy linear method) of the D_task values of that slot. Cell j holds the values v with edge(j−1) < v ≤ edge(j). Thus a value equal to an edge goes to the lower cell, and exact zeros of a ReLU decode to 0. Each cell decodes to the mean of the D_task values in it. An empty cell decodes to the midpoint of its two edges. A slot with no D_task value (an F-all class that is the first class of every D_task image; not possible with a trained head) gets edges and levels of 0. No labels.
   - RAW slots: one set of cells for each sent channel, fit on the pooled values of that channel at all grid positions.
   - BOT values: uniform levels −1 + 2j/(2^q − 1), j = 0 to 2^q − 1. The index is j = round((t + 1)/2 · (2^q − 1)) (round half to even), clamped to the range.
   - Float values (TOPK-float, BOT-float): the 32 bits of the IEEE float32 value.
8. **Not counted.** Server-to-client traffic and training gradients.

## 4.2 Bit string layout

- Each field is written with its width, most significant bit first. Fields follow in this order:
  - RAW and BOT: channel-major (in the sent channel order), then grid row, then grid column.
  - F-top(k, p): the rank of the ordered list, then the k − 1 gap cells in rank order 2 to k.
  - F-all(p): the first class (4 bits), then the 9 gap cells of the other classes in class order.
  - LAB: the class (4 bits). TOPK-float: the 10 logits in class order.
- V4 in every job: for every message, the bit string has the declared length, unpack(pack(m)) = m, and each field is in its range. When a saved message is read again, pack(unpack(b)) = b.
- Saved messages: `numpy.packbits` of the bit strings, with the bit count.

## 4.3 TOPK formats

- The client sorts its 10 logits with a stable descending sort: z(1) ≥ z(2) ≥ ... ≥ z(10). A tie goes to the lower class index. A gap is z(1) − z(j).
- **F-top(k, p):** the ordered list of the top k classes and the k − 1 gaps z(1) − z(j), j = 2 to k, at p bits each. Cells: one set for each rank j, fit on the D_task logits of the trained head (no augmentation).
- **F-all(p):** the first class (4 bits) and the 9 gaps of the other classes in class order. Cells: one set for each class c, fit on the D_task images where c is not the first class.
- **Rebuilt logits:** the first class 0; a sent class −(decoded gap); a class outside the F-top list −g_fill. The server's NLL uses the rebuilt logits / T. The server's prediction is the first class in the message.
- **g_fill:** m times the largest decoded level of the rank-k cells, with m in (1, 1.5, 2, 3). For each m, T is fit (Part 3.3). The pair (m, T) with the lowest D_dev NLL is kept. A tie goes to the first m in this list.

| B | Candidates (bits) |
|---|---|
| 16 | F-top(2, 9) = 16; F-top(3, 3) = 16; F-top(4, 1) = 16; F-all(1) = 13 |
| 64 | F-all(6) = 58; F-top(4, 16) = 61; F-top(5, 12) = 63 |
| 256 and 1,024 | F-all(16) = 148 |

TOPK at 1,024 bits is the same message as TOPK at 256 bits (the same format, the same cells and the same decoder).

## 4.4 Exact bits of each arm

| Arm | B = 16 | B = 64 | B = 256 | B = 1,024 | Reference |
|---|---|---|---|---|---|
| RAW | 16 | 64 | 256 | 1,024 | RAW-native: 32,768 |
| BOT | 16 | 64 | 256 | 1,024 | BOT-float at B*: 32 for each value |
| TOPK | 13 or 16 | 58, 61 or 63 | 148 | 148 | TOPK-float: 320 |
| LAB | 4 | 4 | 4 | 4 | LAB-full: 4 |

A unit test checks every value of this table.

---

# Part 5. Leakage measurement

## 5.1 Decoder input

- The decoder reads the output of `unpack`, decoded to numbers:
  - RAW and BOT with s = 1: a vector of c values. With s = 2 or 4: a c×s×s map. With s = 8 (BOT (16, 8, 1), RAW-native): a c×8×8 map.
  - TOPK, TOPK-full: 20 numbers, the 10 rebuilt logits (before T) and a mask of the 10 sent classes. TOPK-float: the 10 float logits and a mask of 10 ones.
  - LAB, LAB-full: a one-hot vector of 10.
- Each input element is standardized with the mean and the population standard deviation of that element over the 20,000 decoder training inputs. An element with a standard deviation below 1e-6 is set to 0.

## 5.2 Decoders

- **Decoder-S.** Input layer: a vector goes through a linear layer to 256×4×4 (no activation). A 2×2 or 4×4 map goes through a 2× or 1× nearest upsample to 4×4 and a 1×1 conv to 256 channels. An 8×8 map goes through a 1×1 conv to 256 channels and enters the 8×8 stage.
- Trunk: four stages at 4×4, 8×8, 16×16 and 32×32 with 256, 128, 64 and 32 channels. Each stage has two 3×3 conv layers (no bias) with BN and ReLU. A 2× nearest upsample comes before each stage after the first. A last 3×3 conv (with bias) gives 3 channels, then a sigmoid. The trunk has about 1.76 M parameters (a unit test checks it).
- **Decoder-L.** The same layout with twice the channels (512, 256, 128, 64).
- **Training.** Pairs: the 10,000 D_aux images with their messages, and the 10,000 horizontally flipped D_aux images with the messages of the flipped images. Adam (learning rate 1e-3, default betas, no weight decay), a cosine decay to 0 for each step, batch 128, 20 epochs of 156 steps. Loss: pixel MSE in [0, 1]. At epochs 2, 4, ..., 20, the D_dev MSE (float output) is computed; the epoch with the lowest value is kept (a tie goes to the earlier epoch).
- **Decoder seeds.** Each decoder has its own seed: a hash of the run seed and the decoder name (`sub_seed`).
- **Outputs.** The test reconstructions are stored as uint8 (round(255·x)). J2 to J5 compute no test metric.

## 5.3 Metrics (J6, on test)

- **Primary: PSNR** for each image: 10·log10(1 / MSE), with the MSE over the 3×32×32 values in [0, 1] of the uint8 reconstruction. Then the mean over images.
- **SSIM:** 11×11 Gaussian window, σ = 1.5, K1 = 0.01, K2 = 0.03, data range 1, on each RGB channel over the 22×22 valid positions, then the mean (float64). A unit test compares it with a direct loop.
- **R²_pix** = 1 − MSE(arm) / MSE(F0), with the MSE averaged over images and seeds.
- **Excess leakage** = PSNR(arm) − PSNR(FY).

## 5.4 Floors and closed forms

- The decoder training targets are the 20,000 images of Part 5.2 (D_aux and its flips). They are the base of every floor:
  - F0: the mean image of the targets.
  - FY: the mean image of the targets of each true class, applied to the true test label.
  - V1 closed form: the mean image of the targets of each LAB message label, applied to the D_dev LAB message.
- Each floor image is rounded to uint8 before its PSNR, as the reconstructions are.

## 5.5 Attribute and membership probes (J6, descriptive)

- Arms: RAW@64, BOT@64, TOPK@64, LAB, LAB-full and TOPK-full, in each seed.
- Probe input: the decoded message features of Part 5.1, flattened, standardized for each feature on the probe training set.
- Probe: MLP input → 256 → 256 → classes (ReLU), Adam 1e-3, batch 128 (the last part batch is kept), 30 epochs. The epoch with the highest D_dev accuracy is kept (a tie goes to the earlier epoch). Seed: a hash of the run seed and the probe name.
- **Flip.** The message of the flipped image when the flip bit is 1, else the message of the image. The probe trains on D_aux, selects on D_dev, and is scored on test. Chance is 0.5.
- **Brightness.** The mean luma (0.299 R + 0.587 G + 0.114 B) in [0, 1], in three bins with the D_aux terciles as edges. The label-only probe reads the one-hot true label. The brightness excess is the per-image difference of the two probes' correctness.
- **Membership.** Input: the message features and the one-hot true label. Train: the 5,000 train members (D_task) and the 5,000 train non-members (D_aux). Test: the 5,000 test members (D_task) and the 5,000 test non-members (test split). No epoch selection (D_dev has no members): the epoch-30 probe is used. Score: the difference of the two logits. Metric: AUROC by ranks (mid-ranks for ties).

---

# Part 6. Selection on development data (J1, seed 20261111)

- **RAW and BOT at each budget.** Train the three candidates. Keep the candidate with the highest D_dev accuracy. Among the candidates with a D_dev accuracy of at least (best − 0.2 points), keep the one with the lowest D_dev NLL (at its own T). A tie goes to the first candidate in the table order.
- **TOPK at each budget.** Keep the format with the lowest D_dev NLL (with its best g_fill and T). A tie goes to the first format in the table order. Top-1 accuracy is the same for all formats.
- **TOPK-full at 64 bits.** The same rule on the logits of N.
- **Leakage has no part in the selection.**
- **Seed 1.** The selected candidates of J1 are the seed-1 arm models. Seeds 2 and 3 train only the selected RAW and BOT configurations, and use the selected TOPK and TOPK-full formats with their g_fill multipliers. In seeds 2 and 3, the quantizer cells, the RAW channel ranks, the temperatures and the LAB rows are fit again on the data of that seed's N and head.
- `selection.json` is an output of J1. It reads only D_task and D_dev.

---

# Part 7. Reading rule (copied from design section 8; fixed before any test metric)

## 7.1 Definitions

- For test image i and seed s: c(A, s, i) is 1 if arm A is correct and 0 if not. P(A, s, i) is the PSNR of the Decoder-S output (the first Decoder-S of the arm in that seed). A bar means the mean over the 3 seeds.
- Δacc = 100 × mean over i of [c̄(SYM, i) − c̄(ACT, i)], in points. A negative value favors ACT.
- Δleak = mean over i of [P̄(ACT, i) − P̄(SYM, i)], in dB. A positive value means that ACT leaks more.
- Δacc_s and Δleak_s: the same means for seed s alone.
- CI: a paired bootstrap over test images, 10,000 draws, seed 20261024. The draws are `numpy.random.default_rng(20261024).integers(0, 10000, size=(250, 10000))`, 40 times. Every comparison uses the same draws. The 95% CI is the 2.5 and 97.5 percentiles (numpy linear method) of the mean difference.

## 7.2 Primary pair

SYM = TOPK and ACT = BOT, at cut C2 and B* = 64.

## 7.3 Validity checks

The result is case 0 (invalid) if a check fails.

| Check | Condition | Data |
|---|---|---|
| V1 decoder | Decoder-S on LAB gets at least PSNR(closed form) − 0.2 dB, in each seed | D_dev |
| V2 content-free | A Decoder-S trained on shuffled message-image pairs (BOT and TOPK at B*, seed 1) gets at most PSNR(F0) + 0.2 dB | D_dev |
| V3 frozen client | The sha256 of stages 1 and 2 is the same before and after each arm and each decoder | All jobs |
| V4 bits | Every message has its declared length, and unpack(pack(m)) == m | All jobs |
| V5 finish | All 3 seeds of all primary arms and decoders finish with finite losses | All jobs |
| V6 base model | The D_dev accuracy of N is at least 0.80 in each seed | J1, J4, J5 |

- V2 details: the 20,000 training messages are permuted with a fixed permutation from the run seed; the images stay in order. The epoch selection uses a shuffled D_dev pairing. The V2 value is the PSNR on the true D_dev pairs.
- V5 details: a non-finite epoch loss stops the job (check failure). J6 also checks that every planned decoder output exists.

## 7.4 Outcomes on each axis

Accuracy:

- **A+ (SYM keeps accuracy):** the CI lower bound of Δacc is above −1.0.
- **A− (ACT is more accurate):** Δacc ≤ −1.0, the CI upper bound is below 0, and Δacc_s < 0 in each seed.
- **A?:** all other results.

Leakage:

- **L+ (SYM leaks less):** Δleak ≥ 0.5 dB, the CI lower bound is above 0, and Δleak_s > 0 in each seed.
- **L= (equal):** the whole CI of Δleak is inside (−0.5, +0.5) dB.
- **L− (ACT leaks less):** Δleak ≤ −0.5 dB, the CI upper bound is below 0, and Δleak_s < 0 in each seed.
- **L?:** all other results. If more than one of L+, L= and L− holds (possible only if the percentile CI does not contain the mean), the outcome is L?.

## 7.5 Cases

| | L+ | L= | L− | L? |
|---|---|---|---|---|
| **A+** | S: symbolic dominates | E: equal | X: activation leaks less, accuracy kept | U |
| **A−** | T: trade-off | A: activation dominates | A: activation dominates | U |
| **A?** | U | U | U | U |

- **U (unresolved).** The report names the failed condition and makes no format claim.
- **T (trade-off).** The report also gives the exchange rate Δleak / |Δacc|, in dB for each point of accuracy.

## 7.6 Decoder dependence (J3, seed 1)

- J3 trains Decoder-L for TOPK@64, BOT@64, RAW@64, LAB and RAW-native, and two more Decoder-S seeds for TOPK@64 and BOT@64.
- The primary case gets the label "decoder-dependent" (written into the case label of `results.json` and `table.md`) in two conditions. Condition 1: the sign of the seed-1 Δleak with Decoder-L differs from the sign of the seed-1 Δleak with Decoder-S, for the primary pair. Condition 2: the two values differ by more than 1 dB. With that label, the report may not say "leaks less" without the label.
- The standard deviation of the test PSNR over the three Decoder-S seeds of TOPK@64 and of BOT@64 (seed 1) is descriptive.

## 7.7 Secondary rows (labeled "secondary"; no case claim)

1. TOPK against RAW at B*.
2. TOPK against BOT, and TOPK against RAW, at B = 16, 256 and 1,024.
3. LAB against TOPK at each budget: the leakage cost of the extra TOPK bits (Δleak = P̄(TOPK) − P̄(LAB)), and the NLL gain. For NLL, TOPK "beats" LAB if NLL(LAB) − NLL(TOPK) ≥ 0.02 nats, its CI lower bound is above 0, and it is above 0 in each seed.
4. LAB-full and TOPK-full against BOT at B*.
5. The SSIM version of the primary pair, with a margin of 0.02 in place of 0.5 dB (and the same accuracy margin).
6. Attribute and membership rows for each arm of Part 5.5, with CIs.
7. (Cut C1 and J9 rows: not in this run.)
8. Descriptive: c_quant = NLL(TOPK at B) − NLL(TOPK-float) on test; BOT-float against BOT; RAW-native; floors F0 and FY; R²_pix; client and server MACs of each arm; a plot of accuracy against PSNR for all arms and budgets (`results/acc_psnr.svg`).

Nothing in this rule changes after a test metric exists. A later change is a new, dated section.

---

# Part 8. Jobs, stop rules and operations

## 8.1 Jobs

| Job | Seed | Content | Test data |
|---|---|---|---|
| Data | none | The archive check and the npz files (Part 2.1) | Writes the test files; reads no label for a metric |
| J0 (scratch) | 20261100 | Unit tests. One full N (30 epochs). The RAW cells of all candidates. One epoch of the TOPK head, of RAW (16,1,4), (64,2,1), (16,4,4), of BOT (16,1,4), (16,8,1) and of BOT-float. TOPK and TOPK-full format selection. Messages. One epoch of Decoder-S for a vector, a 2×2, a 4×4 and an 8×8 input, of one shuffled-pair Decoder-S and of one Decoder-L. One epoch of each probe type. Writes `times.json`, `job_plan.json` and `j0_result.json`. | Not read (D_dev has the test role) |
| J1 | 20261111 | N; RAW cells; TOPK head and TOPK-float; TOPK formats and LAB; TOPK-full, LAB-full and RAW-native; 12 RAW and 12 BOT candidates; `selection.json`; BOT-float | Not read |
| J2 | 20261111 | Messages of all arms; 18 Decoder-S (RAW ×4, BOT ×4, TOPK ×3, LAB, LAB-full, TOPK-full, TOPK-float, RAW-native, 2 shuffled-pair controls); V1 and V2 | Images only; no metric |
| J3 | 20261111 | 5 Decoder-L; 4 more Decoder-S (TOPK@64 and BOT@64, decoder seeds 2 and 3) | Images only; no metric |
| J4 | 20261112 | N; RAW cells; head; formats; reference rows; 4 RAW and 4 BOT (selected); messages; 15 Decoder-S (as J2 without TOPK-float and the 2 controls); V1 | Images only; no metric |
| J5 | 20261113 | As J4 | Images only; no metric |
| J6 (CPU) | all | All test metrics, the probes, the bootstrap, the rule, `results/results.json`, `results/table.md`, `results/acc_psnr.svg` | The only job that reads test labels |

## 8.2 Stop rules (design 10.3)

- **J0.** If N takes more than 20 min, or the projected time of one Decoder-S is more than 6 min, stop and report to [researcher]. If the J0 model has a D_dev accuracy below 0.80, stop and report to [researcher]. (The code was frozen before J0, so a recipe change needs a dated amendment and [researcher]'s word.)
- **J1, J4, J5.** If V6 fails, stop. A change to N or its recipe needs a dated amendment and [researcher]'s word.
- **Any job.** If V3, V4 or V5 fails, or the code raises an error, the job stops and the event is recorded. The work stops, and the session reports to [researcher] and waits. If a rerun of the same frozen code cannot repair the event, the result is case 0.
- **Re-freeze.** A code change after the freeze needs a dated amendment in this file and [researcher]'s word. Then `sebits.py freeze <label>` keeps the old file as `FREEZE_<label>.json`, copies the data record and writes a new `FREEZE.json`.
- **J6 and V5.** If a planned decoder output is absent, J6 writes case 0 and the list of absent outputs, and computes no comparison.
- **Test labels.** No job reads test labels except J6.

## 8.3 Job parts

- J0 measures the time of each unit and projects the time of each job (`times.json`).
- If a job's projected time is above 110 min, the job runs in consecutive parts of at most 110 min each, in the list order of its units (`job_plan.json`). A part runs with `--part i`.
- Each unit writes a "done" marker. A rerun skips finished units. Each unit seeds its own generators, so the result does not depend on the part boundaries.
- A part does not start if a unit of an earlier part has no "done" marker, or if `job_plan.json` differs from the file that J0 wrote (its sha256 is in `j0_result.json`). A job that resumes after N exists checks V6 again before any unit.

## 8.4 Operations

1. Wait until `ember/research/format_gap_20261023/DONE` exists and no process of that folder runs.
2. Before the download and before each job: no other heavy lab job runs; the swap use is at most 3,584 MB; `pmset -g ac` gives 60 W or more; the Mac draws from AC power and the battery is above 50%. `run_sebits.sh` checks these points.
3. `run_sebits.sh` runs each job with `caffeinate -i -w <pid>` and `tee -a`. It pauses the job when the Mac goes on battery or the battery falls to 50% or less, and resumes it when the power returns. Its time limit is 120 min of running time.
4. Python runs with `-I -B`. The archive is read with `tarfile` and numpy only.
5. After J6: an independent check of the primary numbers from the saved files (`AUDIT.md`), a seal (`SEAL.json`), the report (`REPORT.md`), then the empty file `DONE`.

---

# Part 9. Choices that the design left open, fixed here

1. Download address `https://cave.cs.toronto.edu/kriz/cifar-10-binary.tar.gz`; md5 check against the dataset page; the archive is read in memory.
2. Split method: 70%, 10% and 20% of each class's permutation, in class order 0 to 9; the order of the generator calls of Part 2.2.
3. Membership subsets are random positions, not stratified by class (no subset needs a test label).
4. Normalization statistics: population standard deviation, on D_task pixels.
5. The zero pad of the augmentation is in pixel space, before the normalization.
6. The last part batch is dropped in the training of N, arms and decoders. It is kept in the probes.
7. Conv layers before a BN have no bias. The BOT encoder conv has a bias. The decoder's last conv has a bias.
8. Weight decay applies to all trained parameters (N and arms).
9. N warm-up: linear for each step, from 0.05/546 to 0.05; cosine decay for each step after it.
10. Arm schedule: cosine decay for each step from the base rate to 0, no warm-up.
11. The frozen client is in eval mode during every arm training. Server BN layers are in train mode during arm training.
12. RAW channel rank: population variance of the global average on unaugmented D_task; a tie goes to the lower index.
13. RAW cells: one set for each sent channel, pooled over grid positions; cell rule edge(j−1) < v ≤ edge(j); an empty cell decodes to the midpoint of its edges.
14. Dropped RAW channels enter the server as 0.
15. BOT index rounding: round half to even; clamp to the range.
16. BOT-float is a separate model with the selected B* configuration and no quantizer, not the quantized model run without its quantizer.
17. The TOPK head trains on float logits; the format is applied after the training.
18. Sort ties go to the lower class index. The prediction of TOPK, TOPK-full, LAB and LAB-full is the first class in the message. The prediction of RAW, BOT, TOPK-float, BOT-float and RAW-native is the first class of a stable descending sort of the server logits.
19. g_fill and T are selected together by the D_dev NLL; a tie goes to the first multiplier.
20. Temperature grid: 51 values from 0.50 to 3.00; NLL in float64; a tie goes to the lower T.
21. LAB and LAB-full: the distribution is the add-one smoothed D_dev confusion row of the sent label, with a temperature from the D_dev grid (the design gives each arm one temperature).
22. RAW-native uses the server of N with no training. Its cells are 8-bit quantile cells for each channel over all 64 positions.
23. In seeds 2 and 3, the format and g_fill multiplier of TOPK and TOPK-full come from seed 1. Cells, ranks, temperatures and confusion rows are fit again.
24. Selection tie rule: candidates within 0.2 points of the best D_dev accuracy, then the lowest D_dev NLL, then the table order.
25. Decoder input: Part 5.1, with per-element standardization on the 20,000 training inputs.
26. Decoder input layer: no activation after the linear layer or the 1×1 conv. An 8×8 map enters the 8×8 stage.
27. Decoder optimizer: Adam with default betas and no weight decay; cosine decay for each step.
28. Decoder epoch selection uses the float D_dev MSE. All reported PSNR and SSIM values (D_dev and test) use the uint8 reconstructions.
29. Floors F0 and FY and the V1 closed form use the 20,000 decoder training targets; each floor image is rounded to uint8.
30. V2 shuffle: a fixed permutation of the 20,000 messages; epoch selection on a shuffled D_dev pairing; the value on true D_dev pairs.
31. All generators come from `sub_seed(run_seed, tag)`: the first 4 bytes of sha256("run_seed/tag"), big-endian.
32. Probes: Part 5.5, including no epoch selection for the membership probe.
33. Flip-probe messages: assembled from the saved messages of each image and of its flipped copy.
34. Brightness bins: edges are the D_aux terciles (numpy linear method); a value equal to an edge goes to the upper bin.
35. Membership score and AUROC: Part 5.5. The AUROC CI uses the bootstrap draws of Part 7.1 over the 10,000 test-side probe examples; for each draw, the mean AUROC of the three seeds.
36. Decoder dependence compares seed-1 Decoder-L with seed-1 Decoder-S (Part 7.6).
37. TOPK at 1,024 bits = TOPK at 256 bits.
38. MACs are arithmetic from the layer sizes (conv: C_in·C_out·k²·H·W; linear: in·out); bias, BN, ReLU and pool are omitted.
39. Device: MPS for J0 to J5 with `PYTORCH_ENABLE_MPS_FALLBACK=0`; CPU for J6. MPS results are not bit-exact between runs.
40. The independent check after J6 shares no code with `sebits.py`.

---

# Part 10. Differences from the design

| # | Design | This preregistration | Reason |
|---|---|---|---|
| D1 | Freeze after J0; J0 can change the recipe | Freeze before J0 (before the first training step). A J0 stop rule ends the work and goes to [researcher]. | The order of the parent session's task |
| D2 | Probes in J2, J4 and J5 | All probes in J6 | The label-only probe and the membership probe need test labels. Only J6 may read them. |
| D3 | Split a job's decoder list if J0 projects more than 110 min | Split any job's unit list in the same way | Same rule, applied to every job, so that each part stays under 120 min |
| D4 | The archive sha256 in `FREEZE.json` | The data step writes the archive record into `FREEZE.json` one time, after the freeze | The download comes after the freeze |
| D5 | Optional J7 to J9 | Not run | No approval |
| D6 | "Probes" are not described for membership epoch selection | The membership probe has no epoch selection | D_dev has no members |
| D7 | J0 runs the unit tests | J0 runs the unit tests, which include an end-to-end CPU test of J0 to J6 on tiny synthetic data (about 1.5 min) | Pre-freeze review item 8 (Part 13) |

---

# Part 11. Files

- `sebits.py`: data, models, arms, pack and unpack, decoders, probes, analysis, jobs. `test_sebits.py`: unit tests (synthetic data only). `run_sebits.sh`: the wrapper.
- `FREEZE.json`: sha256 of the four frozen files, the test counts, and (after the data step) the data record.
- Outputs: `times.json`, `job_plan.json`, `j0_result.json` (J0); `selection.json` (J1); `results/s<seed>/` (models, messages, decoders, checks, times); `results/results.json`, `results/table.md`, `results/acc_psnr.svg` (J6); `logs/`.

---

# Part 12. Approval record

[researcher]'s words, 2026-10-23 at about 20:30 local, in the parent session (copied from the task brief of this session): [redacted: private chat]

This approval covers the CIFAR-10 download (binary version, about 162 MB), job J0, and the primary grid J1 to J6 after the freeze. It does not cover cut C1 (J7, J8a, J8b) or the end-to-end variant (J9).

---

# Part 13. Pre-freeze review

- A separate reviewer (a code-review subagent of this session) read the design, `sebits.py` and `test_sebits.py` before the freeze. It ran the unit tests on the CPU only. It found no blocker and 8 minor items.
- All 8 items were fixed before the freeze:
  1. The order "freeze, data, J0" is now explicit, with a re-freeze procedure (Part 8.2).
  2. A part does not start if an earlier part is not done; `job_plan.json` is locked by its sha256; V6 is checked when a job resumes (Part 8.3).
  3. J6 writes case 0 when a planned decoder output is absent (Part 8.2).
  4. Every `torch.load` has a `map_location`, so the CPU analysis does not use the GPU.
  5. Saved RAW-native messages are read in smaller chunks (lower peak memory).
  6. The "decoder-dependent" label is attached to the primary case (Part 7.6).
  7. LAB and LAB-full get a temperature, as the design says (Part 3.3).
  8. A new end-to-end CPU test runs J0 to J6 on tiny synthetic data. It found two defects, which were fixed before the freeze: a crash on an empty F-all slot (Part 4.1, rule 7) and an unclear error for a part number above the plan.
- After the fixes: 44 unit tests pass on the CPU.
