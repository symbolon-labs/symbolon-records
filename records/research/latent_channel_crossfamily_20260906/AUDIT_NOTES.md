# Protocol audit notes

Before model inference, structural unit tests instantiated TRAIN and DEV rows
in memory to check nonce disjointness and counterfactual balance. They did not
persist those splits, capture activations, or evaluate endpoint/bridge outputs.
FINAL was not instantiated. This is a literal deviation from the preregistration
sentence reserving TRAIN/DEV generation until the tiny gate passes; it is not
held-out outcome inspection. Record it rather than claiming no earlier data
construction of any kind. Seeds, task and decision rules were not changed.

The Python environment disappeared before the first launch. See
ENVIRONMENT_RECOVERY.md and environment_restored.json for the restoration and
verified identities. The historical pretrain_tests.xml is retained; restored
runtime tests are separately recorded in restored_tests.xml.
