# EXPLORATORY. Amendment 1 to PLAN.md (2026-10-20)

Written after run 1 stopped and before any numeric result of T2 or T3.

## What happened

Run 1 (`t2_t3.py run`, frozen in `FREEZE.json`, SHA-256 7ad637bd5e935150e87073b29a016e5f7da677fe8c082129b027d21ce8d6a3e0) stopped at data loading. The script assumed that all saved `selected_3` rows of one ID are equal. This assumption is false for one row: familiar row 35 (ID L7, condition "changed") has routes [0, 0], but its expected slot is 1. Thus its selected value is the value of a different record. No gate, metric or reading was computed. No `RESULT.json` exists for run 1. The record gives the d0 familiar changed count as 31/32 (`latent_binding_convergence_fixed_20260911/REPORT.md`, d0 familiar o/d/c 32/32/31). This row is a candidate for that one failure; this amendment does not check it. The other 127 rows route to their expected slot.

## Change (schema only)

- The selected value of an ID comes only from rows whose two routes equal the expected slot (`meta['routes'] == [slot, slot]`).
- Gate 4 uses these routed rows only (127 of 128). It reports the excluded row. The tolerance (1e-5 relative) is unchanged.
- The slot (`wire64`) of each ID is unchanged: it is the ID's own record slot in every row.

No threshold, metric, seed, variant or reading rule changes. `t2_t3.py` and `FREEZE.json` stay as they are. The successor is `t2_t3_v2.py`, frozen in `FREEZE_v2.json`, with output `RESULT_v2.json`.
