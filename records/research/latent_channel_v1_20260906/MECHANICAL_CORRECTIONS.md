# Mechanical corrections before the first optimizer step

1. Unit tests caught negation applied after `.tolist()` in a reader entropy
   diagnostic. Parentheses fixed it; no model operation changed. Sender capture
   was running during this diagnostics-only edit; capture code was unchanged.
   The initial failed test and corrected three-test PASS were observed locally.
2. First real-model attempt passed Gate0, then evaluation serialization indexed
   an empty diagnostics list for no-reader conditions. It failed before any
   optimizer step. Changed empty-list fallback; no scores or model operations
   changed. Preserve `run_tiny_attempt00_serialization` and
   `channel_attempt00.py`; do not use this attempt for a research conclusion.
3. During pre-training input audit, IDs were found to contain group and class.
   They were never passed into either model or bridge, but this was unnecessary
   metadata exposure. IDs now use only randomized row index and split. Preserve
   original tiny data as `data/tiny_attempt00_metadata`; regenerate the same
   source/query/label/order data with opaque IDs and independently assert model
   inputs and captures match. No architecture, optimizer, seed, layer, decision
   rule, or training budget has changed. No trained outcomes existed at correction.

First attempt code SHA256:
199300aa1bf7822cee095506ab371b98d6a7b473d3188782959bc8592e3df4d6.
Preregistration remains unchanged:
24e0696c72610edc880275ba27c26f6c5392e48b49e3ff903226d6e38b4497eb.
