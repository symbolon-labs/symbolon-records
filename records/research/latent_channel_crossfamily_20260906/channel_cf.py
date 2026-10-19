"""Endpoint-dimension/site adapter over sealed V1/V2; no new inference engine."""
import argparse
import importlib.util
import json
from pathlib import Path
import random
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parent
V2 = ROOT.parent / "latent_channel_v2_20260906"
V2_SEAL = json.loads((V2 / "seal.json").read_text())


def load_pinned(name, path, expected):
    import hashlib
    with Path(path).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == expected, path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v2 = load_pinned("sealed_crossfamily_v2", V2 / "channel_v2.py",
                 V2_SEAL["files"]["channel_v2.py"])
c = v2.c
BASE = v2.BASE
c.ROOT = ROOT
c.MODELS["receiver"] = ("meta-llama/Llama-3.2-1B-Instruct", "9213176726f574b556790deb65791e0c5aa438b6")
c.SPLITS = {"tiny": (11003, 8), "train": (22003, 256),
            "dev": (33003, 64), "final": (44003, 256)}
RECEIVER_LAYER = 7
RECEIVER_WIDTH = 2048
old_evaluate = c.evaluate
OldReceiverWrapper = c.ReceiverWrapper


def dataset(split):
    if split == "final":
        assert (ROOT / "run_full/FINAL_FREEZE.json").exists(), "Final not frozen"
    prior_paths = [BASE.parent / "data/tiny/examples.json"]
    prior_paths += [V2 / "data" / s / "examples.json" for s in ["tiny", "train", "dev", "final"]]
    used = {r["nonce"] for path in prior_paths for r in json.loads(path.read_text())}
    for name, (seed, count) in c.SPLITS.items():
        rng = random.Random(seed)
        rows = []
        for group in range(count):
            while True:
                nonce = " ".join(str(rng.randrange(10)) for _ in range(8))
                if nonce not in used:
                    used.add(nonce)
                    break
            for label in range(4):
                rows.append({"group": group, "nonce": nonce, "label": label,
                    "source": f"Noise: {nonce}\nSignal: {chr(65+label)}", "query": c.QUERY})
        rng.shuffle(rows)
        for index, row in enumerate(rows):
            row["id"] = f"{name}-{index:06d}"
        if name == split:
            return rows
    raise ValueError(split)


def identities(role):
    p = c.model_path(role)
    return {"repo": c.MODELS[role][0], "revision": c.MODELS[role][1], "path": str(p),
        "files": {x.name: {"sha256": c.sha(x), "bytes": x.stat().st_size}
                  for x in sorted(p.iterdir()) if x.is_file()},
        "dtype": "float32", "quantization": "none"}


class BridgeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resampler = v2.BoundedResampler()
        self.projector = torch.nn.Linear(64, RECEIVER_WIDTH, bias=False)
        self.reader = v2.BoundedReader(receiver_width=RECEIVER_WIDTH)
        assert sum(p.numel() for p in self.parameters()) == 918017


class ReceiverWrapper(OldReceiverWrapper):
    # _hook and __call__ are inherited without changes, including finally-reset.
    def __init__(self, model, bridge):
        assert model.config.model_type == "llama"
        assert model.config.hidden_size == RECEIVER_WIDTH
        assert len(model.model.layers) == 16
        self.model, self.bridge = model, bridge
        self.memory, self.enabled = None, False
        self.hook = model.model.layers[RECEIVER_LAYER].register_forward_hook(self._hook)


def stock_parity(wrapper, bridge, tok):
    ids = c.tokens(tok, [c.QUERY]*8)
    with torch.no_grad():
        wrapper.hook.remove()
        try:
            stock = wrapper.model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[:, -1]
        finally:
            wrapper.hook = wrapper.model.model.layers[RECEIVER_LAYER].register_forward_hook(wrapper._hook)
        disabled = wrapper(ids)
        zero = wrapper(ids, torch.zeros(8, 8, 64))
    assert torch.equal(stock, disabled) and torch.equal(stock, zero)
    return {"stock_disabled_full_vocab_exact": True, "stock_zero_full_vocab_exact": True}


def mechanics(wrapper, bridge, tok, hidden, dest):
    parity = stock_parity(wrapper, bridge, tok)
    ids = c.tokens(tok, [c.QUERY]*8)
    with torch.no_grad():
        stock = wrapper(ids)
        z = bridge.resampler(hidden[:8])
        active = wrapper(ids, z)
        assert torch.equal(active, wrapper(ids, z))
        assert not torch.equal(stock, active)
    bridge.zero_grad(set_to_none=True)
    logits = wrapper(ids, bridge.resampler(hidden[:8]))
    target = tok.encode(" A", add_special_tokens=False)
    assert len(target) == 1
    torch.nn.functional.cross_entropy(logits, torch.tensor(target*8)).backward()
    grads = {name: float(p.grad.norm()) if p.grad is not None else 0.0 for name,p in bridge.named_parameters()}
    assert all(c.math.isfinite(g) and g > 0 for g in grads.values())
    assert all(p.grad is None and not p.requires_grad for p in wrapper.model.parameters())
    bridge.zero_grad(set_to_none=True)
    c.write_json(dest / "mechanics.json", {"status": "PASS", **parity,
        "disabled_bit_exact": True, "zero_bit_exact": True, "rerun_bit_exact": True,
        "gradient_norms": grads, "endpoint_gradients": False,
        "max_logit_change": (active-stock).abs().max().item(), "receiver_layer_zero_based": RECEIVER_LAYER})
    c.event("mechanics_pass", gradients=grads)


def evaluate(wrapper, bridge, tok, rows, hidden, dest, frozen_cache=None):
    # Exact original scoring. Only the inherited fixed-width byte annotation
    # is adapted after export; a failed stage is never interpreted as complete.
    result = old_evaluate(wrapper, bridge, tok, rows, hidden, dest, frozen_cache)
    path = dest / "metrics.json"
    metadata = json.loads(path.read_text())
    metadata.update({"expanded_memory_bytes": 8*bridge.projector.out_features*4,
        "receiver_width": RECEIVER_WIDTH, "receiver_layer_zero_based": RECEIVER_LAYER,
        "export_adapter": "Replace V1's fixed1024-width memory byte annotation with actual projector width."})
    c.write_json(path, metadata)
    return result


c.dataset, c.identities = dataset, identities
c.BridgeModel, c.ReceiverWrapper = BridgeModel, ReceiverWrapper
c.mechanics, c.evaluate = mechanics, evaluate


def provenance():
    return {"extension_sha256": c.sha(__file__), "base_sha256": c.sha(BASE),
        "bounded_operations_sha256": c.sha(V2 / "channel_v2.py"),
        "preregistration_sha256": c.sha(ROOT / "PREREGISTRATION.md"),
        "receiver_layer_zero_based": RECEIVER_LAYER, "receiver_width": RECEIVER_WIDTH,
        "source_layer_zero_based": 13, "split_seeds": c.SPLITS, "command": sys.argv,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "external_spend_usd": 0}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--split", choices=c.SPLITS, required=True)
    t = sub.add_parser("train"); t.add_argument("--mode", choices=["tiny", "full"], required=True)
    args = parser.parse_args()
    assert (ROOT / "acquisition.json").exists()
    later = (args.command == "prepare" and args.split != "tiny") or (args.command == "train" and args.mode == "full")
    if later:
        gates = json.loads((ROOT / "run_tiny/semantic_gates.json").read_text())
        assert all(gates[k] for k in ["gate0", "gate1", "gate2"])
    kind = f"prepare_{args.split}" if args.command == "prepare" else f"train_{args.mode}"
    marker = ROOT / f"{kind}_extension.json"
    assert not marker.exists(), "Preserve previous attempts"
    info = provenance(); c.write_json(marker, info)
    start = time.monotonic()
    try:
        if args.command == "prepare":
            c.seed_all(11000); c.prepare(args)
        else:
            c.train(args)
            if args.mode == "tiny":
                d = json.loads((ROOT / "run_tiny/decision.json").read_text())
                c.write_json(ROOT / "run_tiny/semantic_gates.json", {"gate0": d["gate0"],
                    "gate1": d["gate1"], "gate2": d["gate1"] and d["gate2"],
                    "scientific_cross_family_established": False})
    except BaseException:
        c.write_json(ROOT / f"{kind}_error.json", {"traceback": traceback.format_exc(), "provenance": info})
        raise
    c.write_json(ROOT / f"{kind}_completion.json", {"status": "completed", "provenance": info,
        "elapsed_s": time.monotonic()-start})


if __name__ == "__main__":
    main()
