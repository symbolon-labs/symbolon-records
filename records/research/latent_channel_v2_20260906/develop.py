"""Sequential local development job, with persistent subprocess provenance."""
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    gate = json.loads((ROOT / "run_tiny/semantic_gates.json").read_text())
    assert all(gate[g] for g in ["gate0", "gate1", "gate2"])
    text_gate = json.loads((ROOT / "text_tiny/summary.json").read_text())
    assert text_gate["oracle_text_sanity_pass"]
    dest = ROOT / "development_job"
    assert not dest.exists(), "Inspect any previous job; do not restart blindly"
    dest.mkdir()
    started = time.monotonic()
    state = {"pid": os.getpid(), "status": "running", "stages": [],
             "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "external_spend_usd": 0, "final_test_licensed": False}
    save(dest / "state.json", state)
    commands = [("prepare_train", ["prepare", "--split", "train"]),
                ("prepare_dev", ["prepare", "--split", "dev"]),
                ("train_full", ["train", "--mode", "full"])]
    for name, args in commands:
        command = [sys.executable, str(ROOT / "channel_v2.py"), *args]
        step = {"name": name, "command": command, "status": "running",
                "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        t = time.monotonic()
        with open(dest / f"{name}.log", "x") as log:
            child = subprocess.Popen(command, cwd=ROOT.parents[1], stdout=log, stderr=subprocess.STDOUT)
            step["pid"] = child.pid
            state["stages"].append(step)
            save(dest / "state.json", state)
            print(json.dumps({"event": "stage_started", **step}), flush=True)
            code = child.wait()
        step.update({"exit_code": code, "elapsed_s": time.monotonic()-t,
                     "status": "complete" if code == 0 else "failed",
                     "peak_child_rss_kib_so_far": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss})
        state["elapsed_s"] = time.monotonic()-started
        if code != 0:
            state["status"] = "failed"
            save(dest / "state.json", state)
            print(json.dumps({"event": "stage_failed", **step}), flush=True)
            raise SystemExit(code if code > 0 else 1)
        save(dest / "state.json", state)
        print(json.dumps({"event": "stage_complete", **step}), flush=True)
    state["status"] = "complete"
    save(dest / "state.json", state)
    print(json.dumps({"event": "development_complete", "elapsed_s": time.monotonic()-started,
                      "final_test_run": False}), flush=True)


if __name__ == "__main__":
    main()
