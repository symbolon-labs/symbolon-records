# Pre-inference environment interruption

The first tiny-job launch exited127: `.venv/bin/python` did not exist.
Inspection found `.venv` empty. No tiny-job directory, data capture, model
inference, or optimizer step had been started. The earlier pretrain_tests.xml
is retained as a historical test receipt, not evidence for the restored runtime.
The cause of the missing environment is unknown.

Restore with the same Python3.14.7 interpreter and pinned recorded versions:
torch2.13.0 (expected build2.13.0+cu130), transformers5.10.2, numpy2.4.6.
Verify the installed Qwen runtime hash against the sealed V2 final freeze;
record all installed packages and rerun pre-training tests to a fresh receipt.
Do not claim byte-identical dependency restoration beyond verified hashes.
No experimental task, architecture, seed, or pass-rule change is authorized
by this environment repair. No cloud resources were used.
