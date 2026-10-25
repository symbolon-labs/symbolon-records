# Results: replication of the leakage result (TOPK@64 against BOT@64)

Seeds: 20261121, 20261122, 20261123, 20261124, 20261125, 20261126. Valid: True. Checks: V1 pass, V2 pass, V3 pass, V4 pass, V5 pass, V6 pass.

| Row | Value | 95% CI (two-level) | Seed values | Reading |
|---|---|---|---|---|
| PRIMARY Δleak, Decoder-L (dB) | 0.609 | 0.547 to 0.664 | 0.655, 0.580, 0.480, 0.687, 0.647, 0.607 | R1: L+ replicates |
| Δacc (points; secondary) | 13.85 | 13.15 to 14.57 | 13.75, 13.60, 14.70, 14.29, 13.67, 13.11 | A+ replicates |
| Δleak, Decoder-S, new seeds (secondary) | 0.615 | 0.547 to 0.675 | 0.650, 0.591, 0.473, 0.713, 0.657, 0.605 | R1 (secondary) |
| Δleak, Decoder-S, 3 sealed + 6 new seeds (descriptive) | 0.593 | 0.536 to 0.649 | 0.512, 0.489, 0.647, 0.650, 0.591, 0.473, 0.713, 0.657, 0.605 | R1 (descriptive) |

- Seed SD of the primary Δleak: 0.074 dB. Student t 95% interval over the seed means: 0.532 to 0.687 dB. CI lower bound at or above the margin: True.
- Sealed rule (image bootstrap only) on the new seeds: Decoder-L L+ (CI 0.587 to 0.632), Decoder-S L+ (CI 0.593 to 0.636).
- Floors on test: F0 12.693 dB, FY 13.065 dB.
- Sealed seed-1 Δleak with Decoder-L (reference): 0.504 dB.

| Arm | Test acc | PSNR Decoder-L (dB) | PSNR Decoder-S (dB) |
|---|---|---|---|
| TOPK@64 | 0.8269 | 13.426 | 13.425 |
| BOT@64 | 0.6884 | 14.036 | 14.040 |
| LAB | 0.8269 | 13.089 | 13.089 |

N D_dev accuracy per seed: 0.8872, 0.8882, 0.8866, 0.8848, 0.8812, 0.8846.
