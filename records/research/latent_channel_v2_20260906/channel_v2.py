"""One explicit bounded-attention repair; reuse sealed v1 orchestration."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "latent_channel_v1_20260906" / "channel.py"
spec = importlib.util.spec_from_file_location("sealed_channel_v1", BASE)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
assert c.sha(BASE) == "e2de36c051ba503b441c2d84b959728cb09f8d064791012b5f673462039f6fbd"
c.ROOT = ROOT
c.SPLITS = {"tiny": (11002, 8), "train": (22002, 256),
            "dev": (33002, 64), "final": (44002, 256)}
base_dataset = c.dataset


def dataset(split):
    rows = base_dataset(split)
    previous = json.loads((BASE.parent / "data/tiny/examples.json").read_text())
    assert not {r["nonce"] for r in rows} & {r["nonce"] for r in previous}
    return rows


def bounded_attention(query, key):
    q = F.normalize(query, dim=-1, eps=1e-8)
    k = F.normalize(key, dim=-1, eps=1e-8)
    return (2.0 * (q @ k.transpose(-2, -1))).softmax(-1)


class BoundedResampler(c.LatentResampler):
    def forward(self, hidden):
        hidden = F.layer_norm(hidden, (hidden.shape[-1],))
        a = bounded_attention(self.queries, self.key(hidden))
        slots = a @ self.value(hidden)
        self.last_attention = a.detach()
        self.last_entropy = -(a.detach() * a.detach().clamp_min(1e-30).log()).sum(-1).mean(-1)
        return slots / slots.square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()


class BoundedReader(c.LatentReader):
    def forward(self, hidden, memory):
        q = self.query(F.layer_norm(hidden, (hidden.shape[-1],)))
        a = bounded_attention(q, self.key(memory))
        delta = self.gate * self.output(a @ self.value(memory))
        with torch.no_grad():
            self.last_attention = a.detach()
            self.diagnostics = {
                "reader_entropy": (-(a.clamp_min(1e-30).log()*a).sum(-1).mean(-1)).tolist(),
                "reader_min_attention": a.amin(dim=(-2, -1)).tolist(),
                "reader_max_attention": a.amax(dim=(-2, -1)).tolist(),
                "delta_norm": delta[:, -1].norm(dim=-1).tolist(),
                "residual_norm": hidden[:, -1].norm(dim=-1).tolist(),
                "delta_residual_ratio": (delta[:, -1].norm(dim=-1)/hidden[:, -1].norm(dim=-1)).tolist(),
                "cosine": F.cosine_similarity(hidden[:, -1], (hidden+delta)[:, -1]).tolist(),
            }
        return hidden + delta


# The inherited BridgeModel constructor resolves these explicit implementations.
c.LatentResampler = BoundedResampler
c.LatentReader = BoundedReader
c.dataset = dataset
BridgeModel = c.BridgeModel


def provenance():
    return {"extension_sha256": c.sha(__file__), "base_sha256": c.sha(BASE),
            "base_path": str(BASE), "preregistration_sha256": c.sha(ROOT / "PREREGISTRATION.md"),
            "attention": "2*cosine(q,k), eps1e-8, both bridge attention operations",
            "split_seeds": c.SPLITS, "command": sys.argv,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "external_spend_usd": 0}


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--split", choices=c.SPLITS, required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--mode", choices=["tiny", "full"], required=True)
    args = p.parse_args()
    kind = f"prepare_{args.split}" if args.command == "prepare" else f"train_{args.mode}"
    marker = ROOT / f"{kind}_extension.json"
    assert not marker.exists(), "No silent restart/overwrite"
    if args.command == "prepare" and args.split != "tiny":
        gate = json.loads((ROOT / "run_tiny/semantic_gates.json").read_text())
        assert gate["gate1"] and gate["gate2"]
    info = provenance()
    c.write_json(marker, info)
    start = time.monotonic()
    try:
        if args.command == "prepare":
            c.seed_all(11000)
            c.prepare(args)
        else:
            c.train(args)
            if args.mode == "tiny":
                d = json.loads((ROOT / "run_tiny/decision.json").read_text())
                c.write_json(ROOT / "run_tiny/semantic_gates.json", {
                    "gate0": d["gate0"], "gate1": d["gate1"],
                    "gate2": d["gate1"] and d["gate2"],
                    "clarification": "Low shuffled accuracy only counts after correct accuracy >=95%.",
                    "scientific_channel_established": False,
                    "architecture_hashes": info})
    except BaseException:
        c.write_json(ROOT / f"{kind}_error.json", {"traceback": traceback.format_exc(),
                     "elapsed_s": time.monotonic()-start, "provenance": info})
        raise
    c.write_json(ROOT / f"{kind}_completion.json", {"elapsed_s": time.monotonic()-start,
                 "provenance": info, "status": "completed"})


if __name__ == "__main__":
    main()
