# Licenses

## Code and records in this repository

The code and the records in `records/` are the work of the repository owner. They come from the Ember project, which uses the MIT License. `LICENSE` applies the same license to this repository.

## Models

This repository does not contain model weights. The records name each model and its pinned revision. To get a model, download it from the link below. Each model has its own license.

| Model | Pinned revision | License | Use in these records |
|---|---|---|---|
| [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B/tree/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e) | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | Apache-2.0 | Sender in all R.01 and R.06 runs |
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca) | `c1899de289a04d12100db370d81485cdf75e47ca` | Apache-2.0 | Receiver in channel v1 and Gate 3 (v2); control in the receiver qualification |
| [HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/tree/a10cc1512eabd3dde888204e902eca88bddb4951) | `a10cc1512eabd3dde888204e902eca88bddb4951` | Apache-2.0 | Receiver in Gate 4, binding gate V1, E3 and all R.06 tests |
| [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/tree/9213176726f574b556790deb65791e0c5aa438b6) | `9213176726f574b556790deb65791e0c5aa438b6` | Llama 3.2 Community License (gated access) | Receiver in the failed Llama attempt and in the receiver qualification |

The revisions come from the model cache paths in the records (for example `data/tiny/manifest.json` in each channel folder). E3 loads the same Qwen3-1.7B and SmolLM2-360M-Instruct revisions, and `records/research/latent_anchor_program_20260930/lab.py` checks the parameter hashes of both models.

The licenses and revisions were checked on the Hugging Face hub on 2026-10-19.

## Python packages

The records name the Python packages and versions of each run (for example `environment_restored.json` and the `FREEZE.json` files). This repository does not contain these packages.
