# Candidate launch interruption — no inference outcome

The initial smollm_candidate launch stopped in Hugging Face Hub1.30's offline
snapshot resolver before identities/model loading/scoring. It demanded ONNX
variants and repository extras absent from the deliberately limited download.
The native safetensors and tokenizer files were present. The original script
is preserved byte-for-byte as qualify_initial.py; existing Qwen/Llama receipts
refer to its hash. The smollm_candidate/started.json remains untouched.

The replacement attempt explicitly selects native root-level runtime filenames
in snapshot_download(local_files_only=True), then uses the exact same frozen
native loader and scorer. New outputs go to smollm_candidate_snapshot_path_fix.
No weights, prompt, answer strings, precision, task, criterion, or architecture
were changed. No candidate inference results existed before this fix.
