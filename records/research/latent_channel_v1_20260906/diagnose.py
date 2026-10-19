"""Post-tiny diagnostic audit. No optimizer, architecture changes or new task data."""
import json
import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from channel import (ROOT, ANSWERS, QUERY, BridgeModel, ReceiverWrapper,
                     get_data, load, seed_all, parameter_hash, sha, write_json,
                     evaluate, identities, tokens, CONDITIONS)


def main():
    seed_all(91001)
    run = ROOT / "run_tiny"
    decision = json.loads((run / "decision.json").read_text())
    dest = ROOT / "tiny_audit"
    assert not dest.exists()
    dest.mkdir()
    start = time.monotonic()
    rows, hidden = get_data("tiny")
    checkpoint = run / "last.pt"
    bridge = BridgeModel()
    bridge.load_state_dict(torch.load(checkpoint, weights_only=True))
    bridge.eval()
    labels = np.array([r["label"] for r in rows])
    groups = [[i for i, r in enumerate(rows) if r["group"] == g] for g in range(8)]
    # In a causal LM, all prefix rows preceding the varying symbol must be equal.
    assert all(torch.equal(hidden[ix[0], :-1], hidden[j, :-1]) for ix in groups for j in ix)
    final_state_differences = [float((hidden[ix[0], -1]-hidden[j, -1]).norm())
                               for ix in groups for j in ix[1:]]
    with torch.no_grad():
        z = bridge.resampler(hidden)
        normalized = F.layer_norm(hidden, (2048,))
        attention = (bridge.resampler.queries @ bridge.resampler.key(normalized).transpose(-2, -1) / 8).softmax(-1)
        per_class_norms = {str(y): z[torch.tensor(labels == y)].norm(dim=-1).tolist() for y in range(4)}
        counterfactual_distances = [float((z[ix[0]]-z[j]).norm()) for ix in groups for j in ix[1:]]
        slot_pair_cosine = F.normalize(z, dim=-1) @ F.normalize(z, dim=-1).transpose(-2, -1)
        output = {
            "checkpoint_sha256": sha(checkpoint),
            "prefix_hidden_counterfactual_parity": True,
            "sender_final_row_counterfactual_distances": final_state_differences,
            "slot_counterfactual_distances": counterfactual_distances,
            "slot_pair_cosine_mean_off_diagonal": slot_pair_cosine[:, ~torch.eye(8, dtype=torch.bool)].mean().item(),
            "resampler_attention_last_source_token": attention[:, :, -1].tolist(),
            "resampler_attention_mean_over_examples_slots": attention.mean(dim=(0, 1)).tolist(),
            "resampler_entropy": (-(attention.clamp_min(1e-30).log()*attention).sum(-1)).mean().item(),
            "resampler_max_entropy": math.log(hidden.shape[1]),
            "transport_norms_per_class": per_class_norms,
            "bridge_parameter_norms": {k: float(p.norm()) for k, p in bridge.named_parameters()},
        }
    model, tok = load("receiver")
    initial = parameter_hash(model)
    wrapper = ReceiverWrapper(model, bridge)
    # Rerun all conditions even after a NO-GO, using the last evaluated checkpoint.
    _, repeat = evaluate(wrapper, bridge, tok, rows, hidden, dest / "rerun")
    original = torch.load(run / f"eval_{decision['step']:04d}" / "outputs.pt", weights_only=True)["results"]
    output["trained_rerun_exact"] = all(torch.equal(original[c][k], repeat[c][k])
                                        for c in CONDITIONS for k in ["lp", "unconstrained"])
    assert output["trained_rerun_exact"]
    # Diagnose gradients without an optimizer; frozen endpoints cannot change.
    selected = torch.arange(8)
    source = hidden[selected].detach().requires_grad_(True)
    logits = wrapper(tokens(tok, [QUERY]*8), bridge.resampler(source))
    answer_ids = torch.tensor([tok.encode(s, add_special_tokens=False)[0] for s in ANSWERS])
    loss = F.cross_entropy(logits, answer_ids[torch.tensor(labels[:8])])
    loss.backward()
    output["trained_gradient_norms"] = {k: float(p.grad.norm()) if p.grad is not None else None
                                         for k, p in bridge.named_parameters()}
    output["source_gradient_norm_by_position"] = source.grad.norm(dim=-1).mean(0).tolist()
    output["receiver_gradients_absent"] = all(p.grad is None for p in model.parameters())
    output["receiver_parameter_hash_unchanged"] = initial == parameter_hash(model)
    assert output["receiver_gradients_absent"] and output["receiver_parameter_hash_unchanged"]
    # Re-derive model/tokenizer identities independently of execution manifests.
    output["independent_model_identities"] = {role: identities(role) for role in ["sender", "receiver"]}
    receiver_manifest = json.loads((run / "manifest.json").read_text())
    sender_manifest = json.loads((ROOT / "data" / "tiny" / "manifest.json").read_text())
    assert output["independent_model_identities"]["receiver"] == receiver_manifest["receiver"]
    assert output["independent_model_identities"]["sender"] == sender_manifest["model"]
    output["independent_hash_verification"] = "PASS"
    # Recalculate per-example correctness and grouped uncertainty directly.
    records = json.loads((run / f"eval_{decision['step']:04d}" / "per_example.json").read_text())
    correctness = {c: np.array([np.argmax(r["conditions"][c]["answer_log_probabilities"]) == r["label"]
                                for r in records], dtype=float) for c in CONDITIONS}
    indices = np.random.default_rng(81001).integers(0, 8, size=(10000, 8))
    bootstrap = {c: np.array([v[ix].mean() for ix in groups])[indices].mean(1)
                 for c, v in correctness.items()}
    output["paired_accuracy_diagnostics_not_scientific_evidence"] = {
        c: {"correct_minus_control": float(correctness["correct"].mean()-correctness[c].mean()),
            "ci99": np.quantile(bootstrap["correct"]-bootstrap[c], [.005, .995]).tolist()}
        for c in CONDITIONS if c != "correct"}
    output["elapsed_s"] = time.monotonic()-start
    output["external_spend_usd"] = 0
    write_json(dest / "diagnostics.json", output)
    print(json.dumps({k: output[k] for k in ["trained_rerun_exact", "resampler_entropy",
               "slot_pair_cosine_mean_off_diagonal", "trained_gradient_norms", "independent_hash_verification"]}), flush=True)


if __name__ == "__main__":
    main()
