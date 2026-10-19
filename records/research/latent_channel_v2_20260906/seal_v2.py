"""Seal completed research artifacts; hashes are integrity, not timestamps."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    destination = ROOT / "seal.json"
    assert not destination.exists(), "Never overwrite an existing seal"
    assert json.loads((ROOT / "confirmation_job/state.json").read_text())["status"] == "complete"
    analysis = json.loads((ROOT / "run_final/analysis.json").read_text())
    assert json.loads((ROOT / "postfreeze_input_audit.json").read_text())["status"] == "PASS"
    reproduction = json.loads((ROOT / "sender_reproduction/verification.json").read_text())
    assert reproduction["actual_sender_hidden_bit_exact"] and reproduction["sender_frozen"]
    assert "Status: final confirmation running" not in (ROOT / "REPORT.md").read_text()
    frozen = json.loads((ROOT / "run_full/FINAL_FREEZE.json").read_text())
    for component in frozen["components"].values():
        assert digest(Path(component["path"])) == component["sha256"]
    assert digest(ROOT / "run_full/best.pt") == frozen["checkpoint_sha256"]
    model_checks = {}
    for role, identity in frozen["models"].items():
        for name, entry in identity["files"].items():
            path = Path(identity["path"]) / name
            assert digest(path) == entry["sha256"] and path.stat().st_size == entry["bytes"]
        model_checks[role] = {"all_files_match": True, "n_files": len(identity["files"])}
    old_root = ROOT.parent / "latent_channel_v1_20260906"
    old = json.loads((old_root / "seal.json").read_text())
    assert all(digest(old_root / name) == value for name, value in old["files"].items())
    paths = sorted(p for p in ROOT.rglob("*") if p.is_file()
                   and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts)
    inventory = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    value = {"files": inventory, "file_count": len(inventory),
        "bytes": sum(p.stat().st_size for p in paths), "models_rehashed": model_checks,
        "v1_sealed_files_verified": len(old["files"]), "verdict": analysis["verdict"],
        "external_spend_usd": 0, "scope": analysis["scope"],
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "caveat": "Local integrity seal, not independent timestamp attestation. Caches excluded."}
    with destination.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    # Separate readback and complete payload pass, not only the write-time values.
    loaded = json.loads(destination.read_text())
    assert all(digest(ROOT / name) == value for name, value in loaded["files"].items())
    print(f"PASS: {len(inventory)} files sealed and independently read back; both endpoints rehashed")


if __name__ == "__main__":
    main()
