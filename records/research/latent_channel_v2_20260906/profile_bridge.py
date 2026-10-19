"""Post-PASS component timings, not a controlled deployment-speed comparison."""
import json
import statistics
import time

import torch

from channel_v2 import ROOT, BridgeModel, c
from final_v2 import read_freeze


def main():
    assert json.loads((ROOT / "confirmation_job/state.json").read_text())["status"] == "complete"
    assert json.loads((ROOT / "run_final/analysis.json").read_text())["gate3"] == "PASS"
    destination = ROOT / "bridge_profile.json"
    assert not destination.exists()
    start = time.monotonic()
    c.seed_all(53004)
    frozen = read_freeze(check_models=True)
    rows, hidden = c.get_data("final")
    source = hidden[:32].clone()
    del hidden
    model, tok = c.load("receiver")
    bridge = BridgeModel().eval()
    bridge.load_state_dict(torch.load(ROOT / "run_full/best.pt", weights_only=True))
    native = []

    def capture(module, args, output):
        native.append(output.detach().clone())

    hook = model.model.layers[13].register_forward_hook(capture)
    with torch.no_grad():
        model(input_ids=c.tokens(tok, [c.QUERY]*8), use_cache=False, logits_to_keep=1)
    hook.remove()
    h = native[0]
    assert h.shape == (8, 8, 1024)
    chunks = list(source.split(8))
    with torch.no_grad():
        memories = [bridge.resampler(chunk) for chunk in chunks]
        projected = [bridge.projector(z) for z in memories]

        def resampler():
            return [bridge.resampler(chunk) for chunk in chunks]

        def projector():
            return [bridge.projector(z) for z in memories]

        def reader():
            return [bridge.reader(h, z) for z in projected]

        def combined():
            return [bridge.reader(h, bridge.projector(bridge.resampler(chunk))) for chunk in chunks]

        timings = {}
        for name, operation in [("resampler", resampler), ("projector", projector),
                                ("reader", reader), ("combined", combined)]:
            for _ in range(5):
                operation()
            samples = []
            for _ in range(20):
                t = time.perf_counter_ns()
                operation()
                samples.append((time.perf_counter_ns()-t)/1e6/32)
            timings[name] = {"unit": "milliseconds per example, batch8",
                "samples": samples, "median": statistics.median(samples),
                "mean": statistics.mean(samples), "min": min(samples), "max": max(samples)}
    assert c.parameter_hash(model) == frozen["receiver_parameter_sha256"]
    result = {"scope": "Warm component timing including Python hooks' diagnostic work; excludes endpoint inference, serialization, transfer, and model loading.",
        "caveat": "Single thermally variable CPU run, not a deployment latency or text-efficiency advantage.",
        "timings": timings, "warmups": 5, "repetitions": 20, "n_examples": 32,
        "example_ids": [r["id"] for r in rows[:32]], "batch_size": 8, "threads": 4,
        "native_receiver_state": "Actual stock block13 output on the unchanged query, batch8",
        "checkpoint_sha256": frozen["checkpoint_sha256"], "code_sha256": c.sha(__file__),
        "receiver_parameter_hash_unchanged": True, "elapsed_s": time.monotonic()-start,
        "external_spend_usd": 0}
    with destination.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({name: row["median"] for name, row in timings.items()}))


if __name__ == "__main__":
    main()
