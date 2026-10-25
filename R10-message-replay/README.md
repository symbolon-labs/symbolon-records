# R.10 Replay in learned message channels: records

These are the records behind the entry [R.10 Replay in learned message channels](https://symbolonlabs.com/research/message-replay/) on the Symbolon site. All paths below are relative to the root of this repository.

## What the record shows

Two CPU studies ask what a replayed valid message does to a learned message channel, and which defense can detect it. Both studies are defensive. The attacker only records valid messages and sends them again. It has no key and makes no new message.

**Replay test (preregistered).** Forty teams of two agents train on Beacon, a small two-dimensional catch task. The main analysis uses the 20 teams that train with blackout episodes. The other 20 teams train without them. A speaker sees a beacon that moves and sends a learned message of 64 numbers (norm 8) at each step. A listener with no memory reads the message and moves to catch the beacon.

- **Stale replay.** The step-0 message at every later step cuts the team success from 0.987 to 0.615. The loss is 0.372 (98.75% interval 0.360 to 0.384). Verdict: HARM.
- **Foreign replay.** The message stream of another episode cuts the success to 0.245. This is below silence (0.305). The loss is 0.742 (0.733 to 0.751). Verdict: HARM.
- **Wire monitors.** A kNN monitor and a PCA monitor, fit on benign messages, cannot flag either replay. Their AUROC is 0.087 and 0.161 on the stale replay, and 0.502 and 0.508 on the foreign replay. The TPR at 1% FPR is at most 0.012. Verdict: CANNOT FLAG.
- **Context monitors.** Monitors that also read the previous message, the position and the local view of the listener cannot flag either replay (secondary rule).
- **Full MAC.** A message authentication code (MAC) over the message bytes, a session nonce and a step counter rejects both replays. A rejected message becomes the zero vector.
  - On the foreign replay, the MAC gives back 0.060 of success (0.051 to 0.069), through the trained fallback. Verdict: MAC HELPS.
  - On the stale replay, the MAC costs 0.308 of success (0.291 to 0.324), because a stale message still carries useful content. Verdict: MAC HURTS.
- **Silent failures.** With no defense, 38.5% (stale) and 75.5% (foreign) of the episodes fail with no flag. Under the MAC, each failed episode has a logged rejection, so no failure is silent.
- **Trained fallback.** Teams that train with blackout episodes keep 0.137 more success after a rejection (0.129 to 0.145) than teams that train without them. Verdict: FALLBACK TRAINING HELPS.
- **Independent statistics check.** A second script recomputes the intervals and verdicts from the saved per-episode files. It agrees, and the largest difference is 2.6e-12.

**Influence analysis (exploratory).** A post hoc analysis of saved runs asks if an influence score can separate a replayed valid message from a benign message. The score measures how far a message moves the answer distribution of the receiver from a reference. The channel is a two-bit toy channel with a 0.36B receiver.

- **Outcome: E-NULL.** The AUROC is 0.524 (95% CI 0.493 to 0.553, 256 rows in each class). The CI is inside the null band from 0.40 to 0.60.
- **Cause.** The cause is structural. The receiver reads the same query on every row. Thus a replayed message gives exactly the output of the benign message of another row (maximum difference 0.0 in all three sets).
- **The score is not blind.** Content-free random messages have much less influence than benign messages (AUROC 0.058, CI 0.045 to 0.070). Thus the score measures how far a message steers the receiver, not if the content fits the input.

Limits that the records state:

- Toy scale: one environment, one algorithm (A2C), three small networks, one speaker and one listener. The teams are very similar, so the narrow intervals describe this toy only (`records/research/marl_replay_20261023/REPORT.md:283`).
- The listener has no memory. A recurrent listener can resist a replay or carry its effect longer (`REPORT.md:284` in the same folder).
- No adaptive attacker was tested (`REPORT.md:286`).
- The MAC rejections, and the equality of the MAC on the foreign replay with silence, hold by construction. The run measured the success after the fallback and the cost of the tag (`REPORT.md:288`).
- The replay test is not a result on large language models (`REPORT.md:292`).
- There is one run and no outside audit (`REPORT.md:5`).
- The three data sets of the influence analysis share the label order and the random references. Thus the pooled CI is too narrow, and the single-set CIs are the conservative values (`records/research/influence_monitor_20261023/REPORT.md:127-128`, `:146`).
- The toy receiver of the influence analysis has no own view of the input. Thus that analysis cannot show if an influence score works on a receiver whose own context disagrees with a replay (`REPORT.md:144` in the same folder).

## Site values and their sources

This table follows the evidence map of the site (`design/ENTRY_R10_EVIDENCE.md` in the site repository). `<R>` is `records/research`, `<M>` is `<R>/marl_replay_20261023` and `<I>` is `<R>/influence_monitor_20261023`.

The site shows the first 12 hex digits of the SHA-256 of the **original** result file. For each file, `MANIFEST.json` gives `sha256_original` and `sha256_copy`. The result files of this entry are not scrubbed, so the two hashes are equal.

Neither folder has a seal file. For these folders, `sha256_original` is the hash of the study file on 2026-10-25. Each folder has a `FREEZE.json` with the hashes of its frozen files, written before the run.

### Gates

| Gate on the site | Site hash | Result file (key) | Copy scrubbed | Rule source |
|---|---|---|---|---|
| Validity gates (passed) | `b6dfc57421a0` | `<M>/results.json` (`validity`, `gate_set_W1`, `teams`) | no | `<M>/PREREGISTRATION.md:164-175` (G-a, G-b, V1 to V5) |
| Agreement gate (passed) | `f1781fbc5f79` | `<M>/stats_check.json` (`agreement`, `max_abs_diff`, `problems`) | no | `<M>/PREREGISTRATION.md:230`, `:238`; the rule is in `<M>/stats_check.py:236-257` (tolerance 1e-9, line 22) |
| Q1 Replay harms the team (passed: HARM) | `b6dfc57421a0` | `<M>/results.json` (`F1."Q1-RT"`, `F1."Q1-RE"`) | no | `<M>/PREREGISTRATION.md:189-199` |
| Q2 Point monitors flag the replay (failed: CANNOT FLAG) | `b6dfc57421a0` | `<M>/results.json` (`F2`) | no | `<M>/PREREGISTRATION.md:209-215` |
| Context monitors flag the replay (failed: CANNOT FLAG) | `b6dfc57421a0` | `<M>/results.json` (`monitors."X-kNN\|RT"`, `"X-kNN\|RE"`, `"X-PCA\|RT"`, `"X-PCA\|RE"`) | no | `<M>/PREREGISTRATION.md:217-219` |
| Q3 MAC, foreign replay (passed: MAC HELPS) | `b6dfc57421a0` | `<M>/results.json` (`F1."Q3-RE"`) | no | `<M>/PREREGISTRATION.md:201-207` |
| Q3 MAC, stale replay (failed: MAC HURTS) | `b6dfc57421a0` | `<M>/results.json` (`F1."Q3-RT"`) | no | `<M>/PREREGISTRATION.md:201-207` |
| S3 Trained fallback (passed) | `b6dfc57421a0` | `<M>/results.json` (`S3`) | no | `<M>/PREREGISTRATION.md:220` |
| Influence score separates the replay (failed: E-NULL) | `5ba8495e9b68` | `<I>/results.json` (`outcome`, `comparisons.primary_S_rand_replay_vs_benign`) | no | `<I>/PREREGISTRATION.md:111-123` |

A gate on the site is passed when the rule supports the claim in its name, and failed when the rule gives the opposite verdict.

### Controls

| Arm on the site | Site value | Source |
|---|---|---|
| Silence | Success 0.305 (correct 0.987) | `<M>/results.json`, `arms_W1_analysis_set.Z.success.mean` = 0.304775 and `.C.success.mean` = 0.986525; `<M>/REPORT.md:135`, `:139` |
| Random message | Success 0.176; monitors flag it (AUROC 1.000) | `arms_W1_analysis_set.RN.success.mean` = 0.1759; `validity.V5` (mean AUROC 1.0 for both point monitors); `<M>/REPORT.md:71`, `:138`, `:190-191` |
| Benign traffic through the MAC | 0 rejections; success equal to correct | `mac.W1."MAC-C"` (0.0 messages and episodes rejected); `validity.V2`; `<M>/REPORT.md:68`, `:164` |
| No fallback training | Success after a rejection 0.168 (trained 0.305) | `arms_W0_all_ok."MAC-RE".success.mean` = 0.16805; `arms_W1_analysis_set."MAC-RE".success.mean` = 0.304775; `<M>/REPORT.md:142`, `:273` |
| Content-free message (influence) | AUROC 0.058 against benign | `<I>/results.json`, `comparisons.sec5_S_block_content_free_vs_benign` (`auroc` 0.0576, `lo` 0.0449, `hi` 0.0703); `<I>/REPORT.md:102` |

### Other values

| Value on the site | Source |
|---|---|
| Beacon task: unit square, a beacon that moves, catch at distance 0.06, at most 50 steps | `<M>/PREREGISTRATION.md:27-39` |
| Speaker, listener with no memory, centralized critic; A2C | `<M>/PREREGISTRATION.md:45-52` |
| Message of 64 numbers, norm 8 | `<M>/PREREGISTRATION.md:38` |
| 20 teams for each training rule (W1 with blackout episodes, W0 without) | `<M>/PREREGISTRATION.md:56-59`; `<M>/results.json` `validity.V1.n_pass` = 20 |
| The attacker has no key and makes no new message | `<M>/PREREGISTRATION.md:63` |
| MAC fields: session nonce and step counter | `<M>/PREREGISTRATION.md:84-86` |
| Stale replay success 0.615; foreign replay 0.245; random 0.176 | `<M>/results.json` `arms_W1_analysis_set` (`RT` 0.61485, `RE` 0.244875, `RN` 0.1759); `<M>/REPORT.md:136-138` |
| A five-step lag costs 0.030 | `<M>/results.json` `descriptive_differences_W1."C-RT5".mean` = 0.029575; `<M>/REPORT.md:152` |
| Silent failures 38.5% and 75.5% with no defense; 0 under the MAC | `<M>/results.json` `silent_failure.RT.none.mean` = 0.38515, `silent_failure.RE.none.mean` = 0.755125, `.MAC.mean` = 0.0; `<M>/REPORT.md:234-235`, `:241` |
| Influence score in the replay test: 0.365 and 0.463 (descriptive) | `<M>/results.json` `influence_auroc.RT.mean` = 0.3648, `influence_auroc.RE.mean` = 0.4632; `<M>/REPORT.md:224-225`, `:275` |
| About 37 minutes of CPU jobs | `<M>/REPORT.md:54` |
| Influence analysis: same query on every row; replay output equals the output of the donor row | `<I>/PREREGISTRATION.md:31-38`; `<I>/results.json` `descriptive_checks.<set>.D1_max_abs_replay_minus_donor_benign` = 0.0; `<I>/REPORT.md:12`, `:65` |
| Two-bit toy channel with a 0.36B receiver | `<I>/REPORT.md:143` |
| Pooled CI too narrow; single-set CIs are the conservative values | `<I>/REPORT.md:127-128`, `:146` |
| Label "Preregistered" for the replay test | `<M>/REPORT.md:5`; rule `<M>/PREREGISTRATION.md:238` |
| Label "exploratory" for the influence analysis | `<I>/REPORT.md:5`; `<I>/PREREGISTRATION.md:142-147` |
| Opened 2026-10-23 | `<M>/PREREGISTRATION.md:3` and `<I>/PREREGISTRATION.md:3` (both dated 2026-10-23). `<I>/FREEZE.json` field `utc` is 2026-10-22T23:39:36Z, which is 2026-10-23 in local time. |
| Updated 2026-10-23 | `<M>/REPORT.md:3` (run 23:04 to 23:41 local, 2026-10-23) |

## Preregistration

- `<M>/PREREGISTRATION.md` and `<M>/FREEZE.json`. The freeze time is 2026-10-23T18:11:30Z, before the first training step. Section 10 gives the reading rule, section 12 the status label and section 13 the predictions. Section 14 lists the 36 choices that the source design left open.
- `<I>/PREREGISTRATION.md` and `<I>/FREEZE.json`. The freeze time is 2026-10-22T23:39:36Z, before the analysis read any result array. Section 9 gives the reading rule. Section 12 gives the label EXPLORATORY and the reasons.

Three files of this entry have redactions (four in all):

| File | Line | Replacement |
|---|---|---|
| `<M>/PREREGISTRATION.md` | 11 | `[redacted: private chat]` (a quote of an approval) |
| `<I>/PREREGISTRATION.md` | 10 | `[redacted: private chat]` (a quote of an approval) |
| `<I>/PREREGISTRATION.md` | 152 | `[redacted: unpublished result]` (one example that names another study) |
| `<M>/REPORT.md` | 292 | `[redacted: unpublished result]` (one sentence) |

The SHA-256 of each original preregistration agrees with its `FREEZE.json`. The copies do not agree, because of the redactions and the name replacements (see "Study copy"). `SCRUB_LOG.json` (`redactions`, `files`) gives each position.

`<M>/results.json` has a field `label` with a fixed text that `analysis.py` writes in every run. It is not the label of the result. The label of the result is in `<M>/REPORT.md:5`.

## Raw outputs

- `<M>/results.json`: all verdicts and descriptive numbers of the replay test (output of `analysis.py`).
- `<M>/runs/W1_s01/` to `<M>/runs/W1_s20/` and `<M>/runs/W0_s01/` to `<M>/runs/W0_s20/`: `train.json`, `gate.json` and `eval.json` for each of the 40 teams.
- `<M>/runs/j0/decision.json`: the development phase, which selected the budget D1 (489 updates).
- `<M>/runs/logs/*.log` and `<M>/runs/chain.log`: the job logs with the gate output of each job.
- `<M>/posthoc.json`: a post hoc check of the context monitors (not preregistered; `<M>/REPORT.md:245-258`).
- `<I>/results.json` and `<I>/run.log`: all numbers of the influence analysis.
- `<I>/posthoc.json`: post hoc checks of the shared structure of the three sets (not preregistered; `<I>/REPORT.md:111-132`).

The per-episode arrays and the checkpoints are binary files. See "What is missing and why".

## Audit report

- `<M>/stats_check.py` and `<M>/stats_check.json`: independent statistics from the saved per-episode files. The script shares no code with `analysis.py` (`<M>/stats_check.py:1-9`). It gives AGREE, with a largest difference of 2.6e-12.
- `<M>/test_marl.py`: 39 synthetic unit tests. The gate of each job ran them (`<M>/runs/logs/*.log`, "gate result: PASS").
- `<M>/PREREGISTRATION.md:290` (item 36): before the freeze, a separate reviewer agent read the code against the plan. It found three defects, and all three were fixed before the freeze.
- `<I>/test_influence.py`: 15 synthetic unit tests, run before the freeze.
- `<I>/run_analysis.py` (`check_freeze`): it compares the hashes of the frozen files and of the three input files with `<I>/FREEZE.json` before the run.

These checks are second scripts in the same project. No outside party ran them.

## Code

Replay test:

- `<M>/beacon.py` (environment, fixed sets, rollouts)
- `<M>/team.py` (networks, GAE, loss, trainer)
- `<M>/channel.py` (arms, MAC, tag field sets)
- `<M>/evaluate.py` (gate set, task and monitor rollouts, checks V2 to V4, cost of the tag)
- `<M>/monitors.py` (kNN, PCA and context monitors)
- `<M>/jobs.py` (job control and the plan rule)
- `<M>/analysis.py` (reading rule) and `<M>/stats_check.py` (independent statistics)
- `<M>/freeze.py`, `<M>/test_marl.py`, `<M>/posthoc_checks.py`
- `<M>/mrp_gate.sh`, `mrp_watchdog.sh`, `mrp_launch.sh`, `mrp_chain.sh` and `mrp_wait.sh` (job control)

Influence analysis:

- `<I>/influence.py` (scores, AUROC, bootstrap, reading rule)
- `<I>/run_analysis.py` (frozen analysis)
- `<I>/freeze.py`, `<I>/test_influence.py`, `<I>/posthoc_checks.py`

**Caution:** `<M>/analysis.py` and `<M>/stats_check.py` load a detector file of an earlier study of the lab. They use it for the AUROC and TPR functions and for a spot check of the monitor scores. That file is not in this package. `<M>/FREEZE.json` (`inputs`) gives its SHA-256.

## What is missing and why

`MANIFEST.json` lists each file that is not included, with its SHA-256, its size and the reason. For this entry, 162 files (4,322 MB) are not included:

| Reason | Files | Size | Status |
|---|---|---|---|
| Saved arrays of each team (`runs/W*_s*/*.npz`: outcomes, monitor rollouts, scores, silent-failure rows, influence) | 120 | 4,318 MB | Not included, available on request |
| Checkpoints (`team.pt` of 40 teams; `runs/j0/dev_s9001_D1.pt` and `dev_s9002_D1.pt`) | 42 | 4 MB | Not included, available on request |

Other gaps:

- **Inputs from other folders.** The replay test loads the detector file of an earlier study (see "Code"). The influence analysis reads three saved output files (`outputs.pt`) of earlier runs. These files are not in this package. `FREEZE.json` (`inputs`) in each folder gives their SHA-256.
- **Source design and threat model.** `<M>/PREREGISTRATION.md` copies its source design and fixes each open choice (section 14). The design file and the threat model of the lab are not in this package. `<M>/FREEZE.json` (`inputs`) gives the SHA-256 of the design file.
- **No seal.** Neither folder has a seal file. `FREEZE.json` holds the hashes of the frozen files before each run. The copy of `<M>/jobs.py` has two name replacements, so its SHA-256 does not agree with the freeze. The original agrees.
- **Model weights and data sets.** The replay test uses generated data only. The influence analysis reads saved outputs only, and it loads no model.

## Study copy

These folders are copies of two study folders of the Ember project (snapshot of 2026-10). They are not a runnable environment.

- The scripts refer to the project layout, a virtual environment and folders that are not here.
- The researcher's name became `[researcher]` in 15 places in 5 files (rule R14).
- Two quotes of private chat and two passages about other studies are redacted (see "Preregistration").
- Some scripts and logs name the folders of other studies as start conditions of the job chain. Those studies are not part of this package.
- Most scripts need the binary files that are not included.

Use the code to read the method. Do not expect it to run without changes.
