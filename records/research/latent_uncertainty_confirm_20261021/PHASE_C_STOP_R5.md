# Phase C stop: R5 reached its 120-min limit (2026-10-22, 06:41 local)

Written by the parent session. This is a stop under PREREGISTRATION.md Part 7.3, rule 2 ("a limit reached in ... R5"). R5 does not start again without [researcher]'s word (rule 5, P25). No test label was read; no stage before R8 reads one.

## What ran

| Job | Time | Result |
|---|---|---|
| F | 22:48 | FREEZE.json; 36 tests pass; hashes equal to Part 9 |
| R3a, R3b | 22:49 to 01:05 | 4,000 train items (1.93 s/item); R3a stopped at its planned limit; chunk check 14 of 14 OK |
| R4a, R4b | 01:06 to 04:41 | 6,000 test items (2.09 s/item); R4a stopped at its planned limit; chunk check 13 of 13 OK; R4c not needed |
| R5 | 04:41 to 06:41 | ctext done (8 seed-free arms, 1,551 s). Seed 20261101: probes (05:09), T3 and T4 receiver passes (05:14). The L arm then trained from about 05:14 to 06:41 and did not finish. No `latent_*.npz` file exists. |

## Facts about the stop

- Expected time of the L arm: 2 epochs on 4,000 items. The Phase D rate (R2b, rank-64 map) gives about 11 min. It ran about 87 min without an end, so it was at least 8 times slower.
- Swap: 664 MB before R5. 5,705 MB of 7,168 MB at the kill. 1,777 MB about one minute after the process ended. Thus the R5 process itself held about 4 GB in swap. The power guard made no pause (empty `power_guard.log`). The charger gave 140 W.
- In Phase D, the text passes (R2a) and the latent fits (R2b) ran in separate processes. In R5, the 8 seed-free text arms, the T3 and T4 passes and the latent fit ran in one process. A probable cause (not proven) is memory that the text passes kept (for example the MPS cache) and that the latent fit then pushed into swap.

## What a restart does (needs [researcher]'s word)

The same command (`./run_uconfirm.sh confirm 120 --stages seeds --seeds 20261101`, without ctext) starts a new process. It skips every saved part (probes, T3, T4) and starts the L arm at once. This uses the frozen code; it changes no file.

## Restart approval (2026-10-22 07:12 local)

[researcher]'s words, in the parent session: [redacted: private chat]
