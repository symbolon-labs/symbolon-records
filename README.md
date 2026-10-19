# Symbolon records

This repository holds the text records behind two entries of the Symbolon Labs site ([symbolonlabs.com](https://symbolonlabs.com/research/)):

- [R.01 Latent channel](R01-latent-channel/README.md)
- [R.06 Binding failure mechanism](R06-binding-failure/README.md)

Each entry README tells what the record shows, which file backs each value on the site, and what is missing.

## Layout

| Path | Contents |
|---|---|
| `R01-latent-channel/README.md`, `R06-binding-failure/README.md` | The guide to each entry |
| `records/` | The copied records. The layout is the same as the root of the old Ember project. |
| `MANIFEST.json` | One row for each file: original path, new path, SHA-256 of the original, SHA-256 of the copy, and whether the copy is scrubbed. It also lists each file that is not included, and why. |
| `SCRUB_LOG.json` | Each replacement of a local path, with its position |
| `integrity/recovery-manifest-lines.jsonl` | The lines of the recovery manifest for the folders in scope |
| `integrity/verify.py` | A checker that uses the Python standard library only |
| `LICENSES.md`, `LICENSE` | The licenses, and the pinned model revisions |

## Check the records

Run this command from the repository root:

```bash
python3 integrity/verify.py
```

The script checks that:

1. Each copy has the SHA-256 that `MANIFEST.json` gives.
2. Each copy that is not scrubbed has the SHA-256 of the original.
3. The SHA-256 of each original agrees with the recovery manifest and with the seal of its folder.
4. Each replacement in `SCRUB_LOG.json` is at the recorded position.

Each sealed folder in `records/research/` has its own seal file (`seal.json`, `SEAL.json` or `documentation_seal.json`). The seal lists the SHA-256 of each file in the folder at closure. For a copy that is not scrubbed, you can compare the SHA-256 of the copy with the seal directly.

## What changed in the copies

Two kinds of change were made. No other text changed.

1. **Local paths.** In 94 of 648 text files, the scrub replaced the home folder of the original account:
   - The old project root becomes a path relative to `records/`.
   - Other paths in the home folder (for example the model cache) start with `<local>`.
2. **Quotes of private chat.** In 4 files, a direct quote of a private chat message is replaced by `[redacted: private chat]`. A redacted original cannot be rebuilt from this repository.

No number changed. The build compared every number in each file before and after the change. `SCRUB_LOG.json` lists all 742 path replacements (`files`) and all 4 redactions (`redactions`), with the line, the position and the SHA-256 of the original and of the copy.

## What is not included

- **Binary files.** These are activation captures (`data/*/sender.pt`), checkpoints and saved tensors. They are not included, and they are available on request. `MANIFEST.json` gives the SHA-256 and the size of each one. PyTorch files can run code when you load them, so this repository contains text only.
- **Files larger than 20 MB.** These are not included, and they are available on request.
- **Subfolders outside the file list of each entry.** These are not included, and they are available on request. Each entry README names them.
- **Agent session notes.** These are hand-over notes of the agent sessions that ran the work (`HANDOFF.md`, `SESSION_HANDOFF.md`, `AUTHORIZATION.md`). They are not included.
- **An outside review of the program** (`RESEARCH_REVIEW_20260930.md`). It quotes private chat, so it is not included.
- **Model weights.** These are not included. `LICENSES.md` links each model at its pinned revision.

To ask for a file that is not included, open an issue in this repository. Give its `original_path` from `MANIFEST.json`.

## Historical copy

The records come from a recovered historical snapshot of the Ember project (USB backup of 2026-09-20, recovered on 2026-09-30). The recovery tool hashed each file and verified the copy. The snapshot is a historical copy, not a configured, runnable environment. The code refers to paths, environments and binary files that are not here. Use the code to read the method, not to run it without changes.

## Known issues

- `records/research/latent_anchor_program_20260930/lab.py` and `sweep.py` are later versions than the versions that test E3 froze. The frozen versions are not in the snapshot. See the R.01 README.
- Three result files that the site cites by hash are scrubbed copies. The site hash is the hash of the original. See the entry README files.
- Four redacted files do not agree with the seal or the recovery manifest of their folder. `sha256_original` in `MANIFEST.json` is the sealed hash. See the entry README files.

## Author

Symbolon Labs.
