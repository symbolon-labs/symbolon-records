"""Fail-closed sequential verification/final queue behind the live development job."""
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def development_state():
    return json.loads((ROOT / "development_job/state.json").read_text())


def wait_for_development(state, dest):
    while True:
        try:
            previous = development_state()
        except json.JSONDecodeError:
            # The existing coordinator uses a short non-atomic JSON rewrite.
            # A transient partial read is not evidence that its process stopped.
            previous = None
        if previous is not None:
            if previous["status"] == "complete":
                assert all(s["exit_code"] == 0 and s["status"] == "complete" for s in previous["stages"])
                assert json.loads((ROOT / "train_full_completion.json").read_text())["status"] == "completed"
                return
            assert previous["status"] == "running", f"Development is terminal: {previous}"
            parent = previous["pid"]
            state["development_pid"] = parent
        else:
            parent = state["development_pid"]
        proc = Path(f"/proc/{parent}/cmdline")
        try:
            command = proc.read_bytes()
        except FileNotFoundError:
            # Check the terminal state once more to cover normal parent exit.
            current = development_state()
            if current["status"] == "complete":
                continue
            raise RuntimeError("Development parent is absent without a completed state; inspect, do not restart")
        assert b"latent_channel_v2_20260906/develop.py" in command, "PID identity mismatch"
        state["last_verified_live_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save(dest / "state.json", state)
        time.sleep(5)


def preflight():
    decision = json.loads((ROOT / "run_full/decision.json").read_text())
    selected = decision["best_step"]
    m = json.loads((ROOT / f"run_full/eval_{selected:04d}/metrics.json").read_text())["conditions"]
    reproduction = json.loads((ROOT / "dev_reproduction/verification.json").read_text())
    assert reproduction["all_conditions_exact"] and reproduction["endpoint_frozen"]
    assert reproduction["selected_step"] == selected
    checks = {"correct_ge90": m["correct"]["accuracy"] >= .90,
        "each_class_ge85": min(m["correct"]["per_class"]) >= .85,
        "baseline_zero_chance": m["receiver_only"]["accuracy"] == m["zero"]["accuracy"] == .25,
        "randomized_controls_le35": all(m[c]["accuracy"] <= .35 for c in ["shuffled", "random_71001", "random_71002"])}
    save(ROOT / "confirmation_job/dev_preflight.json", {"checks": checks, "selected_step": selected,
         "purpose": "Conservative DEV readiness check; not a final-test verdict"})
    assert all(checks.values()), "Selected DEV checkpoint failed readiness; do not burn final data"


def main():
    dest = ROOT / "confirmation_job"
    assert not dest.exists(), "No blind restart or duplicate model queue"
    dest.mkdir()
    initial = development_state()
    state = {"pid": os.getpid(), "development_pid": initial["pid"],
        "status": "waiting_development", "stages": [], "external_spend_usd": 0,
        "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    save(dest / "state.json", state)
    started = time.monotonic()
    try:
        print(json.dumps({"event": "waiting_for_verified_development", "pid": initial["pid"]}), flush=True)
        wait_for_development(state, dest)
        state["status"] = "running"
        stages = [
            ("dev_reproduce", "dev_reproduce.py", []),
            ("text_dev", "text_baseline.py", ["--split", "dev"]),
            ("freeze", "final_v2.py", ["freeze"]),
            ("prepare_final", "channel_v2.py", ["prepare", "--split", "final"]),
            ("score_final", "final_v2.py", ["score"]),
            ("text_final", "text_baseline.py", ["--split", "final"]),
            ("analyze_final", "final_v2.py", ["analyze"]),
        ]
        for name, script, args in stages:
            if name == "freeze":
                preflight()
            command = [sys.executable, str(ROOT / script), *args]
            record = {"name": name, "command": command, "status": "running",
                "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            t = time.monotonic()
            with open(dest / f"{name}.log", "x") as log:
                child = subprocess.Popen(command, cwd=ROOT.parents[1], stdout=log, stderr=subprocess.STDOUT)
                record["pid"] = child.pid
                state["stages"].append(record)
                save(dest / "state.json", state)
                print(json.dumps({"event": "confirmation_stage_started", **record}), flush=True)
                code = child.wait()
            record.update({"exit_code": code, "elapsed_s": time.monotonic()-t,
                "status": "complete" if code == 0 else "failed",
                "peak_child_rss_kib_so_far": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss})
            state["elapsed_s"] = time.monotonic()-started
            save(dest / "state.json", state)
            print(json.dumps({"event": "confirmation_stage_finished", **record}), flush=True)
            assert code == 0, f"{name} failed; preserve the attempt and inspect its log"
        state["status"] = "complete"
        save(dest / "state.json", state)
        print(json.dumps({"event": "confirmation_complete", "cross_family_not_run": True}), flush=True)
    except BaseException:
        state["status"] = "failed"
        state["error"] = traceback.format_exc()
        state["elapsed_s"] = time.monotonic()-started
        save(dest / "state.json", state)
        raise


if __name__ == "__main__":
    main()
