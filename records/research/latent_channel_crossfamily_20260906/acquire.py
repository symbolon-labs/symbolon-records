"""Authorized, revision-pinned model acquisition; never records auth tokens."""
import hashlib
import json
from pathlib import Path
import time

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parent
REPO = "meta-llama/Llama-3.2-1B-Instruct"
REV = "9213176726f574b556790deb65791e0c5aa438b6"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    assert not (ROOT / "acquisition.json").exists()
    start = time.monotonic()
    old = ROOT.parent / "latent_channel_v2_20260906"
    seal = json.loads((old / "seal.json").read_text())
    assert all(sha(old / name) == value for name, value in seal["files"].items())
    info = HfApi().model_info(REPO, revision=REV, files_metadata=True)
    assert info.sha == REV
    directory = Path(snapshot_download(REPO, revision=REV,
        allow_patterns=["model*.safetensors", "*.json", "*.txt", "*.model", "*.jinja"], max_workers=1))
    files = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
             for p in sorted(directory.iterdir()) if p.is_file()}
    upstream_checks = {}
    for entry in info.siblings:
        if entry.rfilename in files and entry.lfs is not None:
            expected = entry.lfs.sha256
            assert files[entry.rfilename]["sha256"] == expected
            upstream_checks[entry.rfilename] = expected
    result = {"repo": REPO, "revision": REV, "path": str(directory), "files": files,
        "upstream_lfs_hashes_verified": upstream_checks, "prior_v2_sealed_files_verified": len(seal["files"]),
        "preregistration_sha256": sha(ROOT / "PREREGISTRATION.md"), "code_sha256": sha(__file__),
        "elapsed_s": time.monotonic()-start, "external_cloud_spend_usd": 0,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with (ROOT / "acquisition.json").open("x") as stream:
        json.dump(result, stream, indent=2); stream.write("\n")
    print("PASS: authorized receiver snapshot downloaded and upstream hashes verified")


if __name__ == "__main__":
    main()
