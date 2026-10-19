# Inherited evidence and implementation boundary

Historical trees are immutable: latent_channel_v1_20260906 (failed unbounded
tiny training), latent_channel_v2_20260906 (same-family held-out PASS),
latent_channel_crossfamily_20260906 (Llama tiny-training NO-GO), and
receiver_qualification_20260906 (qualified SmolLM2 oracle reader).

This new tree copies thin orchestration/audit code from the sealed Llama tree,
while importing pinned V1 native inference/training and V2 bounded operations
and statistics. Necessary changes are receiver revision/width/block count/middle
layer, fresh dataset seeds and prior-nonce exclusions, qualified snapshot path
resolution, future-split guards, provenance component paths, and a required
sender-state rerun before final decision. Training objective/optimizer/thresholds
and receiver prompt are unchanged. No source or checkpoint is edited in place.

The earlier environment disappearance was already repaired and documented in
the old tree. No environment repair is presumed necessary for this run. The
Hub1.30 native-file snapshot resolver fix from receiver qualification is retained
for SmolLM2, so unrelated ONNX variants are neither downloaded nor executed.

Oracle qualification is not qualification of the plain latent query or of the
reader's optimization. Those uncertainties must be resolved by the new tiny
gate, not by positive wording. FINAL remains uncreated until explicit freeze.
