"""Freeze, evaluate once, and audit the reserved synthetic test; no model selection."""
import argparse
import json
import time

import numpy as np
import torch

from channel import (ROOT, CONDITIONS, QUERY, BridgeModel, ReceiverWrapper, load,
                     get_data, evaluate, parameter_hash, seed_all, sha, write_json,
                     identities, tokens, event)


def freeze():
    run = ROOT / "run_full"
    assert not (run / "FINAL_FREEZE.json").exists()
    d = json.loads((run / "decision.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    assert d["best_checkpoint_sha256"] == sha(run / "best.pt")
    assert manifest["code_sha256"] == sha(ROOT / "channel.py")
    assert manifest["preregistration_sha256"] == sha(ROOT / "PREREGISTRATION.md")
    # Apply the planned selection rule independently, using only reusable DEV.
    dev = [(json.loads(p.read_text())["conditions"]["correct"]["full_vocab_ce"],
            int(p.parent.name.split("_")[1])) for p in run.glob("eval_*/metrics.json")
           if p.parent.name != "eval_0000"]
    assert min(dev)[1] == d["best_step"]
    write_json(run / "FINAL_FREEZE.json", {
        "best_step": d["best_step"], "checkpoint_sha256": sha(run / "best.pt"),
        "channel_code_sha256": sha(ROOT / "channel.py"),
        "evaluation_code_sha256": sha(__file__),
        "preregistration_sha256": sha(ROOT / "PREREGISTRATION.md"),
        "final_seed": 44001, "final_groups": 256,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "decision_rule": "PREREGISTRATION.md Gate 3, unchanged",
        "final_already_materialized": (ROOT / "data" / "final").exists(),
    })
    assert not (ROOT / "data" / "final").exists()
    event("final_frozen", step=d["best_step"])


def run():
    start = time.monotonic()
    seed_all(53001)
    dest = ROOT / "run_final"
    assert not dest.exists(), "Final is one-shot; preserve any partial attempt"
    f = json.loads((ROOT / "run_full" / "FINAL_FREEZE.json").read_text())
    assert f["checkpoint_sha256"] == sha(ROOT / "run_full" / "best.pt")
    assert f["channel_code_sha256"] == sha(ROOT / "channel.py")
    assert f["evaluation_code_sha256"] == sha(__file__)
    dest.mkdir()
    rows, hidden = get_data("final")
    model, tok = load("receiver")
    initial = parameter_hash(model)
    bridge = BridgeModel()
    bridge.load_state_dict(torch.load(ROOT / "run_full" / "best.pt", weights_only=True))
    bridge.eval()
    wrapper = ReceiverWrapper(model, bridge)
    metrics, original = evaluate(wrapper, bridge, tok, rows, hidden, dest / "evaluation")
    payload = torch.load(dest / "evaluation" / "outputs.pt", weights_only=True)
    repeats = {}
    with torch.no_grad():
        for c in CONDITIONS:
            ids = tokens(tok, [r["source"] + "\n" + QUERY if c == "text" else QUERY for r in rows[:32]])
            memory = payload["memories"].get(c)
            lp, unconstrained = [], []
            for i in range(0, 32, 8):
                logits = wrapper(ids[i:i+8], None if memory is None else memory[i:i+8])
                lp.append(logits.log_softmax(-1)[:, payload["answer_ids"]])
                unconstrained.append(logits.argmax(-1))
            repeats[c] = {"lp": torch.cat(lp), "unconstrained": torch.cat(unconstrained)}
            assert all(torch.equal(original[c][key][:32], repeats[c][key]) for key in repeats[c])
    torch.save(repeats, dest / "rerun.pt")
    final_hash = parameter_hash(model)
    assert initial == final_hash
    assert f["checkpoint_sha256"] == sha(ROOT / "run_full" / "best.pt")
    write_json(dest / "execution.json", {"elapsed_s": time.monotonic()-start,
        "receiver_identity": identities("receiver"), "receiver_initial_hash": initial,
        "receiver_final_hash": final_hash, "rerun_exact": True, "rerun_n": 32,
        "external_cost_usd": 0, "final_burned": True, "freeze": f})
    analyze()


def analyze():
    """Independent accuracy/bootstrap arithmetic, reading immutable output records."""
    dest = ROOT / "run_final"
    rows = json.loads((dest / "evaluation" / "per_example.json").read_text())
    execution = json.loads((dest / "execution.json").read_text())
    groups = sorted({r["group"] for r in rows})
    labels = np.array([r["label"] for r in rows])
    group_indices = [[i for i, r in enumerate(rows) if r["group"] == g] for g in groups]
    assert all(sorted(labels[ix].tolist()) == [0, 1, 2, 3] for ix in group_indices)
    predictions = {c: np.array([int(np.argmax(r["conditions"][c]["answer_log_probabilities"])) for r in rows])
                   for c in CONDITIONS}
    correct = {c: predictions[c] == labels for c in CONDITIONS}
    grouped = {c: np.array([correct[c][ix].mean() for ix in group_indices]) for c in CONDITIONS}
    rng = np.random.default_rng(81001)
    draws = rng.integers(0, len(groups), size=(10_000, len(groups)))
    boots = {c: grouped[c][draws].mean(axis=1) for c in CONDITIONS}
    contrasts = {}
    for control in CONDITIONS:
        if control == "correct":
            continue
        b = boots["correct"] - boots[control]
        contrasts[f"correct-minus-{control}"] = {
            "mean": float(correct["correct"].mean()-correct[control].mean()),
            "ci99": np.quantile(b, [0.005, 0.995]).tolist(),
            "ci95": np.quantile(b, [0.025, 0.975]).tolist()}
    equivalence = {}
    for c in ["shuffled", "random_71001", "random_71002"]:
        equivalence[c] = np.quantile(boots[c]-boots["receiver_only"], [0.025, 0.975]).tolist()
    accuracy = {c: {"mean": float(correct[c].mean()),
        "ci95": np.quantile(boots[c], [0.025, 0.975]).tolist(),
        "per_class": [float(correct[c][labels == y].mean()) for y in range(4)]} for c in CONDITIONS}
    payload = torch.load(dest / "evaluation" / "outputs.pt", weights_only=True)
    zero_exact = torch.equal(payload["results"]["zero"]["lp"], payload["results"]["receiver_only"]["lp"])
    checks = {
        "correct_ge90": accuracy["correct"]["mean"] >= .90,
        "each_class_ge85": min(accuracy["correct"]["per_class"]) >= .85,
        "receiver_chance": accuracy["receiver_only"]["mean"] == .25,
        "controls_le35": all(accuracy[c]["mean"] <= .35 for c in equivalence),
        "zero_exact": zero_exact,
        "paired99_gain_gt50": all(v["ci99"][0] > .50 for c, v in contrasts.items() if c != "correct-minus-text"),
        "controls_equivalent": all(v[0] >= -.10 and v[1] <= .10 for v in equivalence.values()),
        "rerun_exact": execution["rerun_exact"],
        "endpoint_frozen": execution["receiver_initial_hash"] == execution["receiver_final_hash"],
    }
    # Audit norm constancy on the transmitted tensor and the expanded reader memory.
    bridge = BridgeModel()
    bridge.load_state_dict(torch.load(ROOT / "run_full" / "best.pt", weights_only=True))
    norms = {}
    with torch.no_grad():
        for c, z in payload["memories"].items():
            transport = z.norm(dim=-1)
            expanded = bridge.projector(z).norm(dim=-1)
            norms[c] = {"transport_min": transport.min().item(), "transport_max": transport.max().item(),
                "expanded_per_class_mean": [expanded[torch.tensor(labels == y)].mean().item() for y in range(4)]}
    checks["constant_transport_norm"] = abs(norms["correct"]["transport_min"]-8) < 1e-5 and abs(norms["correct"]["transport_max"]-8) < 1e-5
    donor_labels = np.array([r["donor_label"] for r in rows])
    passed = all(checks.values())
    write_json(dest / "analysis.json", {"accuracy": accuracy, "contrasts": contrasts,
        "control_equivalence_ci95": equivalence, "checks": checks, "norms": norms,
        "shuffled_prediction_matches_donor": float((predictions["shuffled"] == donor_labels).mean()),
        "bootstrap": {"unit": "nonce counterfactual group", "draws": 10000, "seed": 81001},
        "verdict": "A — LATENT COMMUNICATION CHANNEL ESTABLISHED" if passed else "F — MECHANICS / TRAINING NO-GO",
        "scope": "Same-family synthetic two-bit forced-choice channel only. No cross-family, natural, Arabic or generation claim."})
    event("final_decision", passed=passed, checks=checks, accuracy=accuracy)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["freeze", "run", "analyze"])
    args = p.parse_args()
    {"freeze": freeze, "run": run, "analyze": analyze}[args.command]()
