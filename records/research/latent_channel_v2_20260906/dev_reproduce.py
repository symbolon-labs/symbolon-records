"""Actual full DEV rerun of the independently selected checkpoint; no training."""
import json
import time

import torch

from channel_v2 import ROOT, BASE, BridgeModel, c
from final_v2 import development_evaluations, stock_parity
from analysis_v2 import CORE, select_checkpoint


def main():
    start = time.monotonic()
    c.seed_all(53003)
    assert json.loads((ROOT / "train_full_completion.json").read_text())["status"] == "completed"
    dest = ROOT / "dev_reproduction"
    assert not dest.exists()
    evaluations = development_evaluations()
    step = select_checkpoint(evaluations)
    decision = json.loads((ROOT / "run_full/decision.json").read_text())
    assert step == decision["best_step"]
    checkpoint = ROOT / "run_full/best.pt"
    assert c.sha(checkpoint) == decision["best_checkpoint_sha256"]
    dest.mkdir()
    rows, hidden = c.get_data("dev")
    model, tok = c.load("receiver")
    initial = c.parameter_hash(model)
    assert initial == decision["receiver_final_parameter_sha256"]
    bridge = BridgeModel()
    bridge.load_state_dict(torch.load(checkpoint, weights_only=True))
    bridge.eval()
    wrapper = c.ReceiverWrapper(model, bridge)
    parity = stock_parity(wrapper, bridge, tok)
    _, repeats = c.evaluate(wrapper, bridge, tok, rows, hidden, dest / "evaluation")
    original = torch.load(ROOT / f"run_full/eval_{step:04d}/outputs.pt", weights_only=True)["results"]
    checks = {condition: all(torch.equal(original[condition][k], repeats[condition][k])
                           for k in ["lp", "unconstrained"]) for condition in CORE}
    final_hash = c.parameter_hash(model)
    verification = {"all_conditions_exact": all(checks.values()), "conditions": checks,
        "selected_step": step, "checkpoint_sha256": c.sha(checkpoint),
        "receiver_initial_hash": initial, "receiver_final_hash": final_hash,
        "endpoint_frozen": initial == final_hash, "parity": parity,
        "extension_sha256": c.sha(ROOT / "channel_v2.py"), "base_sha256": c.sha(BASE),
        "code_sha256": c.sha(__file__), "elapsed_s": time.monotonic()-start,
        "external_spend_usd": 0}
    c.write_json(dest / "verification.json", verification)
    assert all(checks.values()) and initial == final_hash
    c.event("dev_reproduction_pass", selected_step=step)


if __name__ == "__main__":
    main()
