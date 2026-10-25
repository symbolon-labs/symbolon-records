# Phase D selection (copy into PREREGISTRATION.md; development values, not evidence)

- Sender: Qwen/Qwen3-1.7B (blocks [21, 28]).
- Probe family: mlp.
- Formats: {'T2': 'F1', 'T3': 'F1', 'T4': 'F3', 'T5': 'F3', 'T6': 'list'}.
- L: rank 16, lr 0.01, 2 epochs.
- L-probe: lr 0.01, 3 epochs.
- n_test: 6000 (dev-test SD of binary NLL(T4) - NLL(L): 0.480).
- Gates: G-D1 pass, G-D4 pass, G-D2 FAIL, G-D3 FAIL
- Code sha256: 690305212c48de3fa9aed58a41c7a89b16ab156f2c57f06a37b01eb19df74566.
