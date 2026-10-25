"""Write FREEZE.json: SHA-256 of the frozen files and of the input files, and the UTC time.

It reads no result array. It only hashes the bytes of each file. It refuses to
overwrite an existing FREEZE.json.
"""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HARNESS = ROOT.parent / "latent_sabotage_harness_20261011"
FROZEN_FILES = ["PREREGISTRATION.md", "influence.py", "run_analysis.py", "test_influence.py", "freeze.py"]
INPUTS = ["runs/conf2_wire_b2.00/outputs.pt", "runs/conf_wire_b0.25/outputs.pt",
          "runs/fresh_ood_combo/outputs.pt"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    out = ROOT / "FREEZE.json"
    if out.exists():
        raise SystemExit("REFUSED: FREEZE.json exists")
    freeze = dict(
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        statement=("No result array of the input files was read before this freeze. "
                   "Only keys, shapes and dtypes were read, to write the code."),
        files={f: sha256(ROOT / f) for f in FROZEN_FILES},
        inputs={p: sha256(HARNESS / p) for p in INPUTS})
    out.write_text(json.dumps(freeze, indent=2))
    print(json.dumps(freeze, indent=2))
