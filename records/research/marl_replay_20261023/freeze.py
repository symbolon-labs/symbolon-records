"""FREEZE.json: SHA-256 of the frozen files, of the lab input file and of the design, with
the UTC time and the software versions.

  freeze.py write    refuses to overwrite an existing FREEZE.json
  freeze.py verify   exit 0 only if every frozen file still has its hash

It reads no result file. It is written before the first training step.
"""
import datetime
import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent / "latent_sabotage_harness_20261011" / "detector_suite.py"
DESIGN = ROOT.parents[2] / "reports" / "marl_replay_design_20261023.md"
FROZEN_FILES = ["PREREGISTRATION.md", "beacon.py", "team.py", "channel.py", "evaluate.py", "monitors.py",
                "jobs.py", "analysis.py", "stats_check.py", "test_marl.py", "freeze.py",
                "mrp_gate.sh", "mrp_watchdog.sh", "mrp_launch.sh", "mrp_chain.sh", "mrp_wait.sh"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def current():
    return dict(files={f: sha256(ROOT / f) for f in FROZEN_FILES},
                inputs={"latent_sabotage_harness_20261011/detector_suite.py": sha256(LAB),
                        "reports/marl_replay_design_20261023.md": sha256(DESIGN)})


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    out = ROOT / "FREEZE.json"
    if mode == "write":
        if out.exists():
            raise SystemExit("REFUSED: FREEZE.json exists")
        import numpy
        import torch
        freeze = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                      statement=("Written before the first training step. No team was trained and no "
                                 "result file exists. The unit tests used untrained networks and random arrays."),
                      versions=dict(python=sys.version.split()[0], torch=torch.__version__, numpy=numpy.__version__,
                                    platform=platform.platform()),
                      **current())
        out.write_text(json.dumps(freeze, indent=2))
        print(json.dumps(freeze, indent=2))
        return 0
    if mode == "verify":
        if not out.exists():
            print("FAIL: no FREEZE.json")
            return 1
        ref = json.loads(out.read_text())
        now = current()
        bad = [f for f in ref["files"] if now["files"].get(f) != ref["files"][f]]
        bad += [f for f in ref["inputs"] if now["inputs"].get(f) != ref["inputs"][f]]
        print(f"FAIL: changed {bad}" if bad else f"OK: {len(ref['files'])} files and {len(ref['inputs'])} inputs equal FREEZE.json ({ref['utc']})")
        return 1 if bad else 0
    raise SystemExit("usage: freeze.py write | verify")


if __name__ == "__main__":
    sys.exit(main())
