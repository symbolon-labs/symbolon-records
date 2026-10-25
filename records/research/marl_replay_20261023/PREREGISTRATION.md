# Preregistration: replay of learned messages in multi-agent RL (Beacon)

Date: 2026-10-23. Written by Claude (Opus 5.5) for [researcher] (Symbolon Labs).

Status: frozen when `FREEZE.json` exists. `FREEZE.json` is written before the first training step. The label of the result is in section 12.

Source design: `reports/marl_replay_design_20261023.md` (the "design"). This file copies the design and fixes each choice that the design left open. Section 14 lists these choices. If this file and the design disagree, this file applies, and section 14 gives the reason.

## 1. Approval and scope

- [researcher]'s words, 2026-10-23 at about 20:30 local, in the parent session: [redacted: private chat]
- This approval covers the CPU jobs J0 to J3 of this test. It covers no download and no GPU job.
- Scope: a defensive measurement. The test contains no optimized attack, no monitor-aware attack and no RL attacker.
- CPU only. One heavy job at a time on the laptop. No download, no commit, no push and no public post.
- No sealed tree changes. The test imports `latent_sabotage_harness_20261011/detector_suite.py` read-only (with `python -B`, so no bytecode file is written there).
- The burned-bank rule of `ember/AGENTS.md` does not apply: this test reads no saved array of any earlier run.
- Start condition: the jobs wait until `ember/research/bridge_safety_pair17_20261023/DONE` exists and no process of that folder runs (`mrp_wait.sh`).

## 2. Questions

- Q1. Does a replayed valid message, from another time step (RT) or from another episode (RE), change the team's success?
- Q2. Can the lab's kNN and PCA wire monitors, fit on benign messages, flag the replay?
- Q3. Does a MAC bound to a session nonce and a step counter stop the replay, and what success remains after the fixed zero fallback?

## 3. Environment: Beacon (design section 2)

| Item | Value |
|---|---|
| Arena | Unit square [0, 1] x [0, 1], no obstacles |
| Beacon start | Uniform in [0.1, 0.9]^2; heading uniform in [0, 2 pi) |
| Beacon motion | At each step: heading += 0.3 x N(0, 1); move 0.015 along the new heading; reflect at the walls (the normal component of the heading changes sign). The motion does not depend on the agents. |
| Listener start | Uniform in the arena, at a distance of at least 0.3 from the beacon (rejection sample) |
| Listener actions | 9: direction k x 45 degrees for k = 0 to 7 (k = 0 is +x), and 8 = stay. Step length 0.04 (diagonals also 0.04). Positions are clipped to the arena. |
| Catch | Distance listener to beacon at most 0.06 after the listener moves and before the beacon moves. The episode ends with success. |
| Episode length | At most 50 steps |
| Speaker observation | (x, y, cos theta, sin theta) of the beacon at the start of the step |
| Listener observation | Own position (2); local view (3): ((beacon - position) / 0.15, 1) if the distance is at most 0.15, else (0, 0, 0); delivered message (64). Total 69. |
| Message | m = 8 u / norm(u), u = the 64 speaker outputs. Norm 8 (RMS 1). |
| Team reward | r_t = 1 at a catch, plus (d_{t-1} - d_t). d_{t-1} is the distance at the start of step t. d_t is the distance after the full step (listener move and beacon move). At a catch, d_t is the catch distance. |

One step: the speaker sends m_t; the channel delivers a message (the attack arms replace m_t; the MAC arms check the tag first); the listener acts on its position, its view and the delivered message; the listener moves; the catch check; the beacon moves.

## 4. Team and training (design section 3)

| Network | Input | Layers | Output | Parameters |
|---|---|---|---|---|
| Speaker | 4 | 64, 64, tanh | 64, scaled to norm 8 | 8,640 |
| Listener | 69 | 64, 64, tanh | 9 logits | 9,225 |
| Critic | 9 (beacon state, own position, local view) | 64, 64, tanh | 1 | 4,865 |

- The listener has no memory.
- A2C with a centralized critic. The policy loss flows through the message into the speaker. The channel has no noise. The speaker has no own loss.
- 512 parallel environments; rollouts of 16 steps (8,192 steps for each update); gamma 0.99; GAE lambda 0.95; entropy coefficient 0.01; value coefficient 0.5; Adam, learning rate 3e-4, eps 1e-5; global gradient clip 0.5; advantage normalization for each update.
- Device: CPU, `torch.use_deterministic_algorithms(True)`, 1 thread.
- Checkpoint: the final weights. No checkpoint selection on any evaluation set.
- W1 (primary): each training episode is a blackout episode with probability 0.15. In a blackout episode the listener receives the zero vector at every step.
- W0 (secondary): no blackout.
- Budgets: D1 = 489 updates (4,005,888 steps); D2 = 977 updates (8,003,584); D3 = 1,954 updates (16,007,168).
- Main teams: 20 W1 teams and 20 W0 teams, seeds 1 to 20 in each condition. Development teams: 9001 and 9002 (W1).

## 5. Arms (design section 4)

The attacker sits on the transport. It records messages and tags and resends them. It has no key and computes no new message.

| Code | Delivered message at step t |
|---|---|
| C | m_t |
| RT | m_t at t = 0; m_0 at every t >= 1 |
| RE | m^d_t: step t of the donor stream d(e), in lockstep from t = 0 |
| RN | A new vector at each step, uniform on the sphere of norm 8 |
| Z | The zero vector |
| MAC-C | C traffic through the MAC |
| MAC-RT | RT traffic through the MAC (rejected message: zero vector) |
| MAC-RE | RE traffic through the MAC |
| RT5 | m_t for t < 5; m_{t-5} for t >= 5 (secondary) |
| RE5 | m_t for t < 5; m^d_t for t >= 5 (secondary) |

Donor pool: 2,000 recorded benign streams of 50 steps (seed 900000 + s). RT and RE messages carry the valid tag of their own step and session.

## 6. Defenses (design section 5)

### 6.1 MAC

- tau_t = HMAC-SHA256(K, "beacon-v1" || sender_id || N_e || uint32(t) || bytes(m_t)).
- K = SHA-256("marl-replay|key|" + team seed). sender_id = b"S0". N_e = the first 16 bytes of SHA-256("marl-replay|nonce|<set seed>|<episode index>"). uint32(t) is little-endian. bytes(m_t) = the 64 floats as float32, little-endian (256 bytes).
- Receiver: compute the expected tag from its own N_e, its own t and the received bytes; deliver the message only if the tags are equal (constant-time compare); else deliver the zero vector and log a rejection; accept at most one message for each pair (N_e, t).
- Field table S4: the same check with four field sets (bytes; bytes and nonce; bytes and counter; all three).
- Cost: the mean time to compute and to check one tag over 100,000 tags (W1 team 1).

### 6.2 Point monitors (primary for Q2)

- P-kNN: squared Euclidean distance to the 5th nearest of 5,000 benign fit messages. P-PCA: squared reconstruction error after a PCA with 16 components on the same 5,000 messages.
- Method of `fit_score` in `H/detector_suite.py`, one slot of 64 dimensions. The kNN uses chunks and the expansion |x|^2 - 2 x.f + |f|^2 in float64. Unit test 4 checks agreement with `fit_score` below 1e-9.
- Fit set: 1,000 benign fit episodes of 30 steps (arm C, termination off); 5,000 of the 30,000 step messages, sampled with seed 810000 + s.
- Episode score: the maximum over the 30 steps of a monitor rollout.

### 6.3 Context monitors (secondary)

- x_t = [delivered m_t, delivered m_{t-1}, own position, local view] (133 dimensions). At t = 0 the previous message is the step-0 message.
- Scale: subtract the fit mean; divide each of the four blocks by the square root of its total variance on the fit vectors (population variance).
- X-kNN (k = 5) and X-PCA (q = 16), fit on the same 5,000 subsampled steps. Episode score: the maximum over 30 steps.
- X-noview (descriptive): the same feature without the local view, with kNN and with PCA.

### 6.4 Influence score (secondary, descriptive)

- S_rand at step t = KL(p || q). p: the listener's action distribution with the delivered message. q: the mean action distribution with 5 random messages of norm 8.
- Computed on the monitor test set (30 steps, termination off) for arms C, RT and RE. The 5 references are new at each step (seed 840000 + s) and the same in the three arms.
- Episode score: the mean over the 30 steps. AUROC of RT and of RE against C. No reading rule.

## 7. Metrics (design section 6)

- Task rollouts: evaluation set E, 2,000 episodes, all ten arms. Success rate (primary), return, steps to catch (mean length of the caught episodes).
- Unit of inference: the team.
- Monitor rollouts: termination off, exactly 30 steps, 1,000 episodes for each arm (C, RT, RE, RN, RT5, RE5) of the monitor test set (seed 820000 + s).
- AUROC for each team: the rank formula of `auroc` in `H/detector_suite.py`, with average ranks for ties (section 14, item 21). TPR at 1% FPR: `tpr_at_fpr` of the same file (threshold = 99th percentile of the 1,000 benign episode scores).
- Step-level AUROC over the 30,000 steps (descriptive), all six monitors.
- X monitors (and X-noview): the first step (1 to 30) where the step score is above the episode threshold. Median over the flagged episodes; fractions of all episodes flagged by step 10, 20 and 30.
- Participation ratio of the 5,000 benign fit messages (S6).
- MAC: fraction of received messages rejected (steps that occurred) and fraction of episodes with a rejection.
- Silent failure: an episode fails and no defense flags it. Defenses: none; P-kNN; P-PCA; X-kNN; MAC. Monitor episode score = the maximum over the steps that occurred. Threshold = 99th percentile of the scores of 1,000 benign task episodes of the calibration set (seed 830000 + s). Rows: C, RT, RE, RN, Z, RT5, RE5. The MAC column uses the MAC-C, MAC-RT and MAC-RE rollouts (a flag = any rejection). It is empty for RN, Z, RT5 and RE5, because the design has no MAC rollout for them.

## 8. Seeds and independence (design section 7)

| Stream | Seed |
|---|---|
| Network weights (speaker, listener, critic in this order) | s |
| Training environment (starts, beacon noise, blackout, action uniforms) | 100000 + s |
| Gate set (2,000 task episodes; arms C and Z) | 400000 + s |
| Gate set action uniforms | 1,400,000 + s |
| Evaluation set E (2,000 task episodes) | 500000 + s |
| Action uniforms for E | 510000 + s |
| RN messages on E | 600000 + s |
| Donor map on E (permutation of the 2,000 donor streams) | 700000 + s |
| Monitor fit set (1,000 episodes of 30 steps) | 800000 + s |
| Monitor fit set action uniforms | 1,800,000 + s |
| Fit subsample (5,000 of 30,000) | 810000 + s |
| Monitor test set (1,000 episodes of 30 steps) | 820000 + s |
| Monitor test set action uniforms | 1,820,000 + s |
| RN messages on the monitor test set | 1,600,000 + s |
| Donor map on the monitor test set (first 1,000 of a permutation of 2,000) | 1,700,000 + s |
| Calibration set (1,000 task episodes) | 830000 + s |
| Calibration set action uniforms | 1,830,000 + s |
| Influence references | 840000 + s |
| Donor pool (2,000 benign streams of 50 steps) | 900000 + s |

- Common random numbers: inside one team, all arms of one set use the same starts, beacon paths and action uniforms. The beacon path does not depend on the listener, so each set is a fixed table.
- Each set generator draws in this order: beacon starts, headings, listener starts (rejection rounds in row order), then the heading noise (n x T).
- Training draws, for each step: the action uniforms; then the heading noise; then new starts for the rows that ended (row order); then one blackout uniform for each new episode. The blackout uniform is drawn in W0 too, but it does not apply.
- Unit test 7 checks that no seed value is shared between teams (seeds 1 to 20, 9001, 9002).
- W0 team s uses the same seeds as W1 team s.
- Development teams see only the gate arms C and Z.

## 9. Gates and stop rules (design section 8)

### 9.1 Development phase (job J0)

- Train the two development teams to D1. Evaluate G-a and G-b on the gate set. If both teams pass, select D1.
- Else continue the same runs to D2, then to D3. A continued run is identical to a new run at the larger budget, because the learning rate is constant and the run is deterministic.
- Select the first budget at which both teams pass G-a and G-b.
- If no budget passes, stop. Report to [researcher]. Do not change the environment. Do not start a main run.
- If the selected budget is D3, stop and ask [researcher] before the main run (design section 11).
- J0 cap: 60 minutes. Before each larger budget, if the time used plus the projected time is more than 60 minutes, stop.

### 9.2 Team gates (gate set, arms C and Z)

- G-a: success(C) >= 0.80.
- G-b: success(C) - success(Z) >= 0.20. W0 teams use G-a only.

### 9.3 Validity gates

- V1: at least 18 of 20 W1 teams pass G-a and G-b. If not, Q1 to Q3 are VOID. The analysis set is the W1 teams that pass. A sensitivity analysis uses all W1 teams that trained without NaN.
- V2: in every team (W1 and W0), MAC-C has 0 rejections and its per-episode success and length equal those of C; MAC-RE success and length equal those of Z. If not, the run is VOID.
- V3: every RT, RE, RT5 and RE5 message equals the sent message (tensor equality; a sample also by the 256-byte string). Every non-zero message norm is 8 within 1e-4. If not, the run is VOID.
- V4: W1 team 1: arm C again, from the saved checkpoint in a new network object, gives the same success, length and return. If not, the run is VOID.
- V5: P-kNN and P-PCA flag RN with a mean AUROC >= 0.90 over the analysis set. A monitor that fails V5 gets VOID for Q2.

### 9.4 Job stop rules

- S1: if a unit test fails, the job does not start (`mrp_gate.sh`).
- S2: after 3 teams of a job, if the projected job time is more than 110 minutes, stop after the current team.
- S3: if a loss, a gradient norm or a message contains NaN, stop that team. Count it as a failed team. Add no replacement team.
- S4: if the Mac leaves AC power, if the battery falls to 50% or less, or if the charger gives less than 60 W, pause the job (SIGSTOP). Continue at AC power, battery >= 55% and charger >= 60 W (`mrp_watchdog.sh`).
- S5: after the freeze, add no teams, no seeds and no arms.
- Time caps: a job stops itself before a team that would end after 115 minutes. The watchdog ends a job after 125 minutes.
- A stop rule or an error ends the work. The session then reports to [researcher] and waits.

## 10. Reading rule (design section 9)

- Family F1 (4 comparisons): Q1-RT, Q1-RE, Q3-RT, Q3-RE.
- Family F2 (4 comparisons): P-kNN and P-PCA, each on RT and on RE.
- Each comparison: a two-sided 98.75% Student t interval over the n analysis teams (t = 2.759 at n = 20; 2.793 at n = 18).

Q1: Delta_s = success_s(C) - success_s(arm).

| Verdict | Condition |
|---|---|
| HARM | Mean Delta >= 0.10 and lower bound > 0.05 |
| NO HARM | Upper bound < 0.05 |
| UNCLEAR | Any other result |

Q3: B_s = success_s(MAC arm) - success_s(the same arm without the MAC). For RE this is success_s(MAC-RE) - success_s(RE), which equals success_s(Z) - success_s(RE) when V2 passes.

| Verdict | Condition |
|---|---|
| MAC HELPS | Mean B >= 0.05 and lower bound > 0 |
| MAC HURTS | Mean B <= -0.05 and upper bound < 0 |
| NO CLEAR EFFECT | Any other result |

Q2: episode-level AUROC_s and TPR_s at 1% FPR.

| Verdict | Condition |
|---|---|
| FLAGS | Mean AUROC >= 0.90, lower bound > 0.80, and mean TPR >= 0.50 |
| CANNOT FLAG | Upper bound of AUROC < 0.70, and mean TPR < 0.05 |
| RANK ONLY | Any other result |

Secondary rules (two-sided 95% t intervals, t = 2.093 at n = 20):

- X-kNN and X-PCA on RT and on RE: the Q2 verdicts.
- S3: F_s = success_s(MAC-RE, W1) - success_s(MAC-RE, W0), paired by s, over the teams that pass G-a in both conditions. FALLBACK TRAINING HELPS if mean F >= 0.05 and lower bound > 0. Else NO CLEAR EFFECT.
- All other numbers are descriptive. Report every number, also when a rule gives UNCLEAR or RANK ONLY.

## 11. Jobs

| Job | Content | Command (from the folder, through `mrp_launch.sh`) |
|---|---|---|
| J0 | Unit tests (gate); development phase; timing | `jobs.py j0` |
| J1 | 20 W1 teams: train, gate set, ten arms on E, monitor sets, calibration set, silent-failure rows, influence, V4 on team 1 | `jobs.py teams --cond W1` |
| J2 | 20 W0 teams: train, gate set, ten arms on E | `jobs.py teams --cond W0` |
| J3 | Monitor scores; `analysis.py`; `stats_check.py` (agreement gate) | `jobs.py score`, then the two scripts |

- Plan rule (fixed before J0): J0 measures the training time of a development team at the selected budget, the gate time, the full W1 evaluation time on an untrained network (seed 9999; only the wall times are kept), and the monitor scoring time on random arrays of the real shapes. For J1, J2 and J3, the number of teams in each job is the largest k in {20, 10, 5, 4, 2, 1} with k x (minutes for each team) x 1.25 <= 100. `jobs.py plan` prints the job list, and `mrp_chain.sh` runs it in order. Each job starts only if the job before it exits 0.
- Each job runs in the background under `caffeinate -i -w <pid>`, with its log through `tee -a` in `runs/logs/<job>.log` and the watchdog.
- Disk: about 1.2 GB of monitor rollouts and about 2 to 4 GB of silent-failure rows for the 20 W1 teams. The gate requires 10 GB free.

## 12. Status label of the result

- If the run is complete, V1 to V4 pass and the agreement gate gives AGREE: "Preregistered" (threat model section 1.3): a frozen plan, a fixed reading rule, one run, no independent audit.
- If V1 fails: "Preregistered, gate failed". Q1 to Q3 are VOID. The numbers are reported as descriptive.
- If V2, V3 or V4 fails: the run is VOID (an error in the code). Report and wait for [researcher].
- If the work stops early: no result label. DONE gets one line with the reason.
- These parts hold by construction (design section 0): the MAC rejection rates; MAC-RE equals Z. The report says so.

## 13. Predictions (written before the run)

From the design (sections 0 and 12), not from a result:

- Q1-RE: HARM (close to certain, because G-b requires that the team needs the channel).
- Q2 P monitors on RE: CANNOT FLAG or RANK ONLY, near chance. On RT: AUROC can be below 0.5, because the RT episode maximum is over one message only.
- Q3-RE: MAC HELPS under W1 if the trained fallback beats the foreign message.
- Q1-RT, Q3-RT, the X monitors and S3: open.

## 14. Fixed choices (the design left these open)

1. FREEZE.json is written before J0, not after J0 (design 15.1). Reason: the task says "Write FREEZE.json before the first training step". J0 selects the budget by the fixed rule of section 9.1.
2. The jobs run in the background, not from the Terminal panel (design 11 and 15.1). Reason: a timing check on 2026-10-23 gave the same speed (0.813 s background, 0.822 s foreground, one-thread MLP benchmark in the session scratchpad). Test B v3 runs the same way.
3. J0 continues one run per development team through D1, D2 and D3 (section 9.1).
4. D3 selected: stop and ask [researcher].
5. Initialization: orthogonal weights, gain sqrt(2) for the hidden layers, 0.01 for the policy head, 1.0 for the critic head and the speaker head; biases 0. One generator (seed s) initializes speaker, listener and critic in this order.
6. One Adam optimizer for the three networks. The gradient clip is over all parameters.
7. Loss = policy loss + 0.5 x mean((V - R)^2) - 0.01 x entropy, with R = GAE advantage + V.
8. Truncation after 50 steps: the reward of the last step gets gamma x V(state after the step) (bootstrap). A catch is terminal (no bootstrap).
9. Shaping distance d_t: after the full step; at a catch, the catch distance (section 3).
10. Beacon: the noise changes the heading before the move; a wall reflection changes the sign of the normal component; the speaker sees the heading of the last move (the start heading at t = 0).
11. Inclusive limits: catch at distance <= 0.06; view at distance <= 0.15; start at distance >= 0.3.
12. Action index: direction k x 45 degrees for k = 0 to 7, 8 = stay.
13. Training actions: inverse-CDF samples with uniforms from the training stream. Draw order as in section 8.
14. Evaluation actions: the stochastic policy, by inverse-CDF samples with the shared uniforms. No greedy policy.
15. Streams that the design table does not name: the action uniforms of the gate, fit, test and calibration sets (set seed + 1,000,000); RN and the donor map on the monitor test set (1,600,000 + s and 1,700,000 + s).
16. E donor map: d(e) = perm[e], where perm is a permutation of the 2,000 donor streams. Monitor test set: the first 1,000 entries of a second permutation.
17. Key, nonce and field encodings as in section 6.1.
18. The MAC arms filter the recorded wire stream. The receiver check does not depend on the listener, so the delivered stream is fixed before the rollout.
19. Influence score on the monitor test set, arms C, RT and RE, 5 new references at each step, the same in the three arms.
20. X-noview with both kNN and PCA (descriptive).
21. AUROC with average ranks for ties. Reason: an RT episode and its paired C episode share the step-0 message (and, for the X features, the full step-0 vector). Thus their episode maxima can be exactly equal. The lab `auroc` gives such ties arbitrary ranks. Without ties both formulas agree (unit test 16). The report gives the number of ties.
22. Block scaling uses the population variance (ddof 0).
23. Silent-failure MAC column only for C, RT and RE (section 7). No new MAC arm (S5).
24. Field table S4: measured on the first 200 episodes of E, all 50 steps, for C, RT, RT5, RE and RE5 traffic. A rejected RT5 or RE5 cell has no success value, because the design has no MAC-RT5 or MAC-RE5 rollout.
25. First flag: steps 1 to 30; the threshold is the 99th percentile of the 1,000 benign test-episode scores of the same monitor.
26. V2 compares per-episode success and length exactly, in every team of both conditions.
27. V3 also checks RT5 and RE5.
28. NaN teams: no evaluation; they count as failed in V1 and are absent from all analysis sets.
29. The kNN scores use float64 and chunks of 2,048 rows. J3 uses the default numpy (Accelerate) threads; training and rollouts use 1 torch thread.
30. The P and X monitors use the same 5,000 subsampled fit steps.
31. Tag timing with the Python `hmac` module on W1 team 1.
32. The analysis set for the F2 and X rules is the V1 analysis set (all of its teams have monitor scores).
33. The gate (`mrp_gate.sh`) checks: no other heavy lab job (a name pattern and any Python process above 50% CPU); AC power, battery > 50%, charger >= 60 W; swap used < 2,000 MB; free disk >= 10 GB; FREEZE.json hashes; unit tests (S1).
34. File roles: `jobs.py` holds the role of the design's `train.py` (training of one team, the checkpoint and the gate results) and the job control.
35. G-a and G-b compare integer counts of the gate set: 100 x n_C >= 80 x n and 100 x (n_C - n_Z) >= 20 x n, with n = 2,000. Reason: float32 rates can fail an exact boundary (for example a gap of 400 of 2,000). `gate.json` stores the counts.
36. Before the freeze, a separate reviewer agent read the code against this file. It found three defects, and all three were fixed before the freeze: the interval level of P monitors on RN in `stats_check.py` (95%, as in `analysis.py`); the float32 gate rates (item 35); a NaN team that skipped the S2 and time-cap checks in `jobs.py`.

## 15. Files

| File | Content |
|---|---|
| `beacon.py` | Environment, fixed sets, rollouts, training environment |
| `team.py` | Networks, GAE, loss, trainer |
| `channel.py` | Arms, MAC, field sets |
| `evaluate.py` | Gate set, task rollouts, monitor rollouts, checks V2 to V4, field table, tag timing |
| `monitors.py` | Chunked kNN, PCA, context features, block scaling, scores for one team |
| `jobs.py` | J0, team jobs, scoring job, plan rule |
| `analysis.py` | Reading rule and descriptive numbers; writes `results.json` |
| `stats_check.py` | Independent statistics and agreement gate; writes `stats_check.json` |
| `test_marl.py` | 39 synthetic unit tests (no optimizer step on a team, no result file) |
| `freeze.py` | Writes and checks `FREEZE.json` |
| `mrp_gate.sh`, `mrp_watchdog.sh`, `mrp_launch.sh`, `mrp_chain.sh`, `mrp_wait.sh` | Job control |

## 16. Procedure

1. Run the unit tests. Write `FREEZE.json` (`freeze.py write`).
2. Wait for test B v3 (`mrp_wait.sh`).
3. Check the machine (gate). Run J0 through `mrp_launch.sh`.
4. Read `runs/j0/decision.json`. If a stop applies, write REPORT.md and DONE with the reason, and wait for [researcher].
5. Else run `mrp_chain.sh` (J1, J2, J3 by the plan rule).
6. If every job exits 0 and the agreement gate gives AGREE, write REPORT.md with the reading-rule result and every number tied to a file. Write the empty file DONE.
7. If a job stops, write REPORT.md with the reason and DONE with one line, and wait for [researcher].
