# Correction from the mechanics-invalid predecessor; one line

Predecessor: `research/latent_binding_convergence_20260911/` (FAILURE.md, seal
`e79399a83d705c53a3a233aa1d3dabb9dfa258ea60540019ccb5fb4a32420d17`). It exited 1
after 907.187s in the sealed-candidate replay, before any fit, with no candidate
and no scientific result.

Exact failure: `runtime.reference()` compares freshly computed scores against the
sealed 822db91d arrays, which requires the bridge to hold the sealed candidate
state. The predecessor loaded only `c.bridge()` (the original d0 state) and ran
the baseline phase with it, then went straight into the reference phase. The
`receiver_only` arm passed because it bypasses the reader entirely; the next arm
(`correct`) compared d0-state scores against candidate-state scores and failed on
`token_lp`.

This tree changes exactly one thing: before the reference phase it loads the
sealed candidate state into the bridge and asserts the full state matches. No
other line, threshold, gate, budget, data row, seed, learning rate or pin
changed. The PLAN, the 400-step continuation, the gates and the interpretation
limits are identical to the predecessor.

Everything else is inherited unchanged, including the progenitor seal pin, the
sealed 822db91d parent pin, the start-state bitwise check, and the historical
baseline replay. The predecessor tree stays sealed and read-only.
