"""Post-primary actual sender-state reproduction; no new data or selection."""
import json
import time

import torch

from channel_cf import ROOT, c
from final_cf import read_freeze


def main():
    # Sequential execution protects the 16GiB host from two model loads.
    assert (ROOT / "run_final/execution.json").exists()
    assert (ROOT / "text_final/summary.json").exists()
    destination = ROOT / "sender_reproduction"
    assert not destination.exists(), "Preserve any partial attempt"
    destination.mkdir()
    start = time.monotonic()
    c.seed_all(53003)
    frozen = read_freeze(check_models=True)
    rows, saved = c.get_data("final")
    expected = saved[:32].clone()
    del saved
    model, tokenizer = c.load("sender")
    initial_hash = c.parameter_hash(model)
    inputs = c.tokens(tokenizer, [r["source"] for r in rows[:32]])
    captured = []

    def capture(module, args, output):
        captured.append(output.detach().clone())

    hook = model.model.layers[13].register_forward_hook(capture)
    with torch.no_grad():
        for i in range(0, 32, 4):
            model.model(input_ids=inputs[i:i+4], use_cache=False)
    hook.remove()
    actual = torch.cat(captured)
    final_hash = c.parameter_hash(model)
    torch.save({"hidden": actual, "source_ids": inputs}, destination / "rerun.pt")
    result = {"n": 32, "source_layer_zero_based": 13,
        "actual_sender_hidden_bit_exact": torch.equal(actual, expected),
        "max_abs_difference": (actual-expected).abs().max().item(),
        "sender_initial_parameter_hash": initial_hash,
        "sender_final_parameter_hash": final_hash,
        "sender_frozen": initial_hash == final_hash,
        "sender_identity": frozen["models"]["sender"],
        "example_ids": [r["id"] for r in rows[:32]],
        "saved_capture_sha256": c.sha(ROOT / "data/final/sender.pt"),
        "rerun_sha256": c.sha(destination / "rerun.pt"),
        "audit_code_sha256": c.sha(__file__), "elapsed_s": time.monotonic()-start,
        "external_spend_usd": 0}
    c.write_json(destination / "verification.json", result)
    assert result["actual_sender_hidden_bit_exact"] and result["sender_frozen"]
    print("PASS: 32 actual sender captures reproduce bit-exactly")


if __name__ == "__main__":
    main()
