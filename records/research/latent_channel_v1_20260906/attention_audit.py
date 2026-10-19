"""Inspect actual checkpoint attention routing; no optimization or ablations."""
import json
import time

import torch
from torch.nn import functional as F

from channel import (ROOT, QUERY, BridgeModel, ReceiverWrapper, get_data, load,
                     parameter_hash, seed_all, tokens, sha, write_json)


def main():
    start = time.monotonic()
    seed_all(91002)
    dest = ROOT / "attention_audit"
    assert not dest.exists()
    dest.mkdir()
    rows, hidden = get_data("tiny")
    model, tok = load("receiver")
    model_hash = parameter_hash(model)
    manifest = json.loads((ROOT / "run_tiny" / "manifest.json").read_text())
    assert model_hash == manifest["receiver_initial_parameter_sha256"]
    bridge = BridgeModel()
    bridge.load_state_dict(torch.load(ROOT / "run_tiny" / "last.pt", weights_only=True))
    bridge.eval()
    wrapper = ReceiverWrapper(model, bridge)
    captured = []
    def inspect(module, args):
        h, memory = args
        q = module.query(F.layer_norm(h, (1024,)))
        a = (q @ module.key(memory).transpose(-2, -1) / 8).softmax(-1)
        captured.append({"attention": a.detach(), "hidden": h.detach(),
                         "context": (module.gate * module.output(a @ module.value(memory))).detach()})
    handle = bridge.reader.register_forward_pre_hook(inspect)
    with torch.no_grad():
        z = bridge.resampler(hidden)
        normalized = F.layer_norm(hidden, (2048,))
        resampler_a = (bridge.resampler.queries @ bridge.resampler.key(normalized).transpose(-2, -1)/8).softmax(-1)
        outputs = []
        for i in range(0, 32, 8):
            outputs.append(wrapper(tokens(tok, [QUERY]*8), z[i:i+8]).log_softmax(-1))
    handle.remove()
    receiver_a = torch.cat([x["attention"] for x in captured])
    context = torch.cat([x["context"] for x in captured])
    raw = torch.load(ROOT / "run_tiny" / "eval_0300" / "outputs.pt", weights_only=True)
    assert torch.equal(torch.cat(outputs)[:, raw["answer_ids"]], raw["results"]["correct"]["lp"])
    groups = [[i for i, r in enumerate(rows) if r["group"] == g] for g in range(8)]
    slot_differences = torch.stack([(z[ix[0]]-z[j]).norm(dim=-1) for ix in groups for j in ix[1:]])
    context_differences = torch.stack([(context[ix[0], -1]-context[j, -1]).norm() for ix in groups for j in ix[1:]])
    source_ids = torch.load(ROOT / "data" / "tiny" / "sender.pt", weights_only=True)["source_ids"]
    torch.save({"resampler_attention": resampler_a, "reader_attention": receiver_a,
                "context": context}, dest / "tensors.pt")
    summary = {
        "reader_slot_attention_last_query_mean": receiver_a[:, -1].mean(0).tolist(),
        "resampler_signal_token_attention_per_slot_mean": resampler_a[:, :, -1].mean(0).tolist(),
        "resampler_constant_marker_attention_per_slot_mean": resampler_a[:, :, 19].mean(0).tolist(),
        "counterfactual_latent_L2_per_slot_mean": slot_differences.mean(0).tolist(),
        "counterfactual_context_L2_mean": context_differences.mean().item(),
        "counterfactual_context_L2_max": context_differences.max().item(),
        "attention_composition_source_mass_last_query": (receiver_a[:, -1].unsqueeze(1) @ resampler_a).mean(0).squeeze(0).tolist(),
        "attention_composition_caveat": "Routing diagnostic only, not exact causal contribution; per-slot normalization changes scales.",
        "example_source_tokens": [tok.decode([int(x)]) for x in source_ids[0]],
        "model_matches_training_parameter_hash": True, "actual_logits_match": True,
        "checkpoint_sha256": sha(ROOT / "run_tiny" / "last.pt"),
        "code_sha256": sha(__file__), "elapsed_s": time.monotonic()-start,
        "external_spend_usd": 0,
    }
    write_json(dest / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
