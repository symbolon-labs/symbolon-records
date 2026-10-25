# Phase D selection (copy into PREREGISTRATION.md; development values, not evidence)

- Sender: Qwen/Qwen3-1.7B (blocks [21, 28]).
- Probe family: logistic.
- Formats: {'T2': 'F3', 'T3': 'F3', 'T4': 'F3', 'T5': 'F3', 'T6': 'counts'}.
- L: rank 64, lr 0.0003, 2 epochs.
- L-probe: lr 0.001, 3 epochs.
- n_test: 6000 (dev-test SD of binary NLL(T4) - NLL(L): 0.328).
- Gates: G-D1 pass, G-D4 pass, G-D2 pass, G-D3 FAIL
- Code sha256: cba5e677c6e98905632c2d5c5483b9155e84e13a3bb8282ddf11b5056d076d4c.
