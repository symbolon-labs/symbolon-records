"""Frozen-endpoint, learned-reader experiment. No Ember inference changes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
MODELS = {
    "sender": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
    "receiver": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
}
SPLITS = {"tiny": (11001, 8), "train": (22001, 256),
          "dev": (33001, 64), "final": (44001, 256)}
QUERY = "Hidden signal (A/B/C/D):"
ANSWERS = [" A", " B", " C", " D"]
CONDITIONS = ["receiver_only", "correct", "shuffled", "random_71001",
              "random_71002", "zero", "text"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")


def event(kind, **data):
    print(json.dumps({"event": kind, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      **data}, ensure_ascii=False, allow_nan=False), flush=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def dataset(split):
    """Never constructs later partitions while generating an earlier partition."""
    used = set()
    for name, (seed, count) in SPLITS.items():
        rng = random.Random(seed)
        rows = []
        for group in range(count):
            while True:
                nonce = " ".join(str(rng.randrange(10)) for _ in range(8))
                if nonce not in used:
                    used.add(nonce)
                    break
            for label in range(4):
                rows.append({"id": f"{name}-{group:05d}-{label}",
                             "group": group, "nonce": nonce, "label": label,
                             "source": f"Noise: {nonce}\nSignal: {chr(65 + label)}",
                             "query": QUERY})
        rng.shuffle(rows)
        if name == split:
            return rows
    raise ValueError(split)


def model_path(role):
    name, rev = MODELS[role]
    return Path(snapshot_download(name, revision=rev, local_files_only=True))


def identities(role):
    p = model_path(role)
    files = {x.name: {"sha256": sha(x), "bytes": x.stat().st_size}
             for x in sorted(p.iterdir()) if x.is_file() and x.suffix in {".json", ".safetensors", ".txt"}}
    return {"repo": MODELS[role][0], "revision": MODELS[role][1],
            "path": str(p), "files": files, "dtype": "float32", "quantization": "none"}


def load(role):
    p = model_path(role)
    tok = AutoTokenizer.from_pretrained(p, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(p, dtype=torch.float32,
                attn_implementation="eager", local_files_only=True).eval()
    model.requires_grad_(False)
    assert all(not p.requires_grad for p in model.parameters())
    return model, tok


def tokens(tok, texts):
    lists = [tok.encode(t, add_special_tokens=False) for t in texts]
    assert len({len(x) for x in lists}) == 1, "Unexpected padding/length differences"
    return torch.tensor(lists, dtype=torch.long)


def parameter_hash(module):
    digest = hashlib.sha256()
    for name, p in module.named_parameters():
        digest.update(name.encode())
        digest.update(p.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class LatentResampler(nn.Module):
    def __init__(self, sender_width=2048, width=64, k=8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(k, width) / math.sqrt(width))
        self.key = nn.Linear(sender_width, width, bias=False)
        self.value = nn.Linear(sender_width, width, bias=False)
        self.last_entropy = None

    def forward(self, hidden):
        hidden = F.layer_norm(hidden, (hidden.shape[-1],))
        attention = (self.queries @ self.key(hidden).transpose(-2, -1) / 8).softmax(-1)
        slots = attention @ self.value(hidden)
        self.last_entropy = -(attention.detach() * attention.detach().clamp_min(1e-30).log()).sum(-1).mean(-1)
        return slots / slots.square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()


class LatentReader(nn.Module):
    def __init__(self, receiver_width=1024, width=64):
        super().__init__()
        self.query = nn.Linear(receiver_width, width, bias=False)
        self.key = nn.Linear(receiver_width, width, bias=False)
        self.value = nn.Linear(receiver_width, width, bias=False)
        self.output = nn.Linear(width, receiver_width, bias=False)
        self.gate = nn.Parameter(torch.tensor(0.01))
        self.diagnostics = {}

    def forward(self, hidden, memory):
        q = self.query(F.layer_norm(hidden, (hidden.shape[-1],)))
        a = (q @ self.key(memory).transpose(-2, -1) / 8).softmax(-1)
        delta = self.gate * self.output(a @ self.value(memory))
        with torch.no_grad():
            self.diagnostics = {
                "reader_entropy": (-(a.clamp_min(1e-30).log() * a).sum(-1).mean(-1)).tolist(),
                "delta_norm": delta[:, -1].norm(dim=-1).tolist(),
                "residual_norm": hidden[:, -1].norm(dim=-1).tolist(),
                "delta_residual_ratio": (delta[:, -1].norm(dim=-1) / hidden[:, -1].norm(dim=-1)).tolist(),
                "cosine": F.cosine_similarity(hidden[:, -1], (hidden + delta)[:, -1]).tolist(),
            }
        return hidden + delta


class BridgeModel(nn.Module):
    """Only this object is given to the optimizer; endpoints are separate."""
    def __init__(self):
        super().__init__()
        self.resampler = LatentResampler()
        self.projector = nn.Linear(64, 1024, bias=False)
        self.reader = LatentReader()


class ReceiverWrapper:
    def __init__(self, model, bridge):
        self.model, self.bridge = model, bridge
        self.memory = None
        self.enabled = False
        self.hook = model.model.layers[13].register_forward_hook(self._hook)

    def _hook(self, module, args, hidden):
        if not self.enabled:
            return hidden
        assert self.memory is not None
        return self.bridge.reader(hidden, self.bridge.projector(self.memory))

    def __call__(self, ids, memory=None):
        self.enabled = memory is not None
        self.memory = memory
        try:
            return self.model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[:, -1]
        finally:
            self.enabled = False
            self.memory = None


def prepare(args):
    dest = ROOT / "data" / args.split
    assert not dest.exists(), f"Refusing overwrite: {dest}"
    if args.split == "final":
        assert (ROOT / "run_full" / "FINAL_FREEZE.json").exists(), "Final not licensed"
    dest.mkdir(parents=True)
    rows = dataset(args.split)
    write_json(dest / "examples.json", rows)
    start = time.monotonic()
    identity = identities("sender")
    model, tok = load("sender")
    source_ids = tokens(tok, [r["source"] for r in rows])
    assert all(sum(r["label"] == y for r in rows) == len(rows) // 4 for y in range(4))
    captured = []
    def hook(module, args, output):
        captured.append(output.detach().clone())
    handle = model.model.layers[13].register_forward_hook(hook)
    with torch.no_grad():
        for i in range(0, len(rows), 4):
            model.model(input_ids=source_ids[i:i+4], use_cache=False)
            event("capture", split=args.split, completed=min(i+4, len(rows)), total=len(rows),
                  elapsed_s=time.monotonic()-start)
    handle.remove()
    hidden = torch.cat(captured)
    assert hidden.shape == (len(rows), source_ids.shape[1], 2048)
    torch.save({"hidden": hidden, "source_ids": source_ids}, dest / "sender.pt")
    write_json(dest / "manifest.json", {"model": identity, "examples_sha256": sha(dest / "examples.json"),
               "capture_sha256": sha(dest / "sender.pt"), "code_sha256": sha(__file__),
               "preregistration_sha256": sha(ROOT / "PREREGISTRATION.md"),
               "layer_zero_based": 13, "shape": list(hidden.shape),
               "elapsed_s": time.monotonic()-start, "external_cost_usd": 0})


def get_data(split):
    p = ROOT / "data" / split
    rows = json.loads((p / "examples.json").read_text())
    manifest = json.loads((p / "manifest.json").read_text())
    assert sha(p / "examples.json") == manifest["examples_sha256"]
    assert sha(p / "sender.pt") == manifest["capture_sha256"]
    data = torch.load(p / "sender.pt", weights_only=True)
    assert rows == dataset(split)
    return rows, data["hidden"]


def derangement(n, seed=61001):
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(n)
        if np.all(p != np.arange(n)):
            return p


def make_memories(bridge, hidden):
    with torch.no_grad():
        z = torch.cat([bridge.resampler(h) for h in hidden.split(32)])
    donors = derangement(len(hidden))
    memories = {"correct": z, "shuffled": z[donors], "zero": torch.zeros_like(z)}
    for seed in [71001, 71002]:
        random_z = torch.randn(z.shape, generator=torch.Generator().manual_seed(seed))
        memories[f"random_{seed}"] = random_z / random_z.norm(dim=-1, keepdim=True) * z.norm(dim=-1, keepdim=True)
    return memories, donors


def evaluate(wrapper, bridge, tok, rows, hidden, dest, frozen_cache=None):
    start = time.monotonic()
    dest.mkdir(parents=True, exist_ok=False)
    answer_ids = [tok.encode(s, add_special_tokens=False) for s in ANSWERS]
    assert all(len(a) == 1 for a in answer_ids)
    answer_ids = torch.tensor([a[0] for a in answer_ids])
    labels = torch.tensor([r["label"] for r in rows])
    query_ids = tokens(tok, [QUERY] * len(rows))
    text_ids = tokens(tok, [r["source"] + "\n" + QUERY for r in rows])
    memories, donors = make_memories(bridge, hidden)
    results, diagnostics, times = {}, {}, {}
    bridge.eval()
    with torch.no_grad():
        for condition in CONDITIONS:
            t = time.monotonic()
            if frozen_cache is not None and condition in {"receiver_only", "text"}:
                results[condition] = frozen_cache[condition]
                times[condition] = {"s": 0, "cached": True}
                continue
            ids = text_ids if condition == "text" else query_ids
            memory = memories.get(condition)
            out, ds = [], []
            for i in range(0, len(rows), 8):
                logits = wrapper(ids[i:i+8], None if memory is None else memory[i:i+8])
                lp = logits.log_softmax(-1)
                out.append({"lp": lp[:, answer_ids], "unconstrained": logits.argmax(-1)})
                if memory is not None:
                    ds.extend([{k: v[j] for k, v in bridge.reader.diagnostics.items()}
                               for j in range(len(logits))])
            results[condition] = {"lp": torch.cat([x["lp"] for x in out]),
                                  "unconstrained": torch.cat([x["unconstrained"] for x in out])}
            diagnostics[condition] = ds
            times[condition] = {"s": time.monotonic()-t, "cached": False}
            event("evaluation_condition", condition=condition, directory=str(dest), elapsed_s=time.monotonic()-t)
    assert torch.equal(results["zero"]["lp"], results["receiver_only"]["lp"]), "Zero parity failed"
    metrics = {}
    for condition, result in results.items():
        pred = result["lp"].argmax(-1)
        metrics[condition] = {"accuracy": (pred == labels).float().mean().item(),
             "per_class": [(pred[labels == y] == y).float().mean().item() for y in range(4)],
             "full_vocab_ce": -result["lp"][torch.arange(len(rows)), labels].mean().item(),
             "unconstrained_accuracy": (result["unconstrained"] == answer_ids[labels]).float().mean().item()}
    assert metrics["receiver_only"]["accuracy"] == 0.25
    payload = {"results": results, "memories": memories, "donors": torch.tensor(donors),
               "labels": labels, "answer_ids": answer_ids}
    torch.save(payload, dest / "outputs.pt")
    records = []
    for i, row in enumerate(rows):
        records.append({"id": row["id"], "group": row["group"], "label": row["label"],
            "donor_id": rows[int(donors[i])]["id"], "donor_label": rows[int(donors[i])]["label"],
            "conditions": {c: {"answer_log_probabilities": results[c]["lp"][i].tolist(),
              "prediction": int(results[c]["lp"][i].argmax()),
              "unconstrained_token": int(results[c]["unconstrained"][i]),
              "diagnostics": diagnostics.get(c, [{}] * len(rows))[i]} for c in CONDITIONS}})
    write_json(dest / "per_example.json", records)
    write_json(dest / "metrics.json", {"conditions": metrics, "gate": bridge.reader.gate.item(),
         "times": times, "total_s": time.monotonic()-start,
         "source_tokens": hidden.shape[1], "receiver_query_tokens": query_ids.shape[1],
         "text_receiver_tokens": text_ids.shape[1], "transport_bytes": 2048,
         "expanded_memory_bytes": 32768, "source_bytes": hidden.shape[1]*2048*4,
         "slot_norms": memories["correct"].norm(dim=-1).tolist(),
         "expanded_norms": bridge.projector(memories["correct"]).detach().norm(dim=-1).tolist(),
         "donor_label_agreement": np.mean([rows[i]["label"] == rows[int(donors[i])]["label"] for i in range(len(rows))]).item(),
         "output_sha256": sha(dest / "outputs.pt")})
    event("evaluation", directory=str(dest), metrics=metrics, gate=bridge.reader.gate.item())
    return metrics, results


def mechanics(wrapper, bridge, tok, hidden, dest):
    ids = tokens(tok, [QUERY] * 8)
    with torch.no_grad():
        # Remove the hook to obtain truly stock inference, then reinstall it.
        wrapper.hook.remove()
        stock = wrapper.model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[:, -1]
        wrapper.hook = wrapper.model.model.layers[13].register_forward_hook(wrapper._hook)
        disabled = wrapper(ids)
        z = bridge.resampler(hidden[:8])
        zero = wrapper(ids, torch.zeros_like(z))
        active = wrapper(ids, z)
        repeated = wrapper(ids, z)
    assert torch.equal(stock, disabled) and torch.equal(stock, zero)
    assert torch.equal(active, repeated)
    assert not torch.equal(stock, active)
    bridge.zero_grad(set_to_none=True)
    logits = wrapper(ids, bridge.resampler(hidden[:8]))
    loss = F.cross_entropy(logits, torch.tensor([tok.encode(" A", add_special_tokens=False)[0]]*8))
    loss.backward()
    grads = {name: float(p.grad.norm()) if p.grad is not None else 0.0 for name, p in bridge.named_parameters()}
    assert all(math.isfinite(v) and v > 0 for v in grads.values()), grads
    assert all(p.grad is None and not p.requires_grad for p in wrapper.model.parameters())
    bridge.zero_grad(set_to_none=True)
    write_json(dest / "mechanics.json", {"status": "PASS", "disabled_bit_exact": True,
              "zero_bit_exact": True, "rerun_bit_exact": True, "gradient_norms": grads,
              "max_logit_change": (active-stock).abs().max().item(),
              "endpoint_gradients": False})
    event("mechanics_pass", gradients=grads)


def train(args):
    tiny = args.mode == "tiny"
    dest = ROOT / f"run_{args.mode}"
    assert not dest.exists(), f"Refusing overwrite: {dest}"
    if not tiny:
        gate = json.loads((ROOT / "run_tiny" / "decision.json").read_text())
        assert gate["gate1"] and gate["gate2"]
    dest.mkdir()
    start = time.monotonic()
    seed = 51001 if tiny else 52001
    seed_all(seed)
    rows, hidden = get_data("tiny" if tiny else "train")
    val_rows, val_hidden = (rows, hidden) if tiny else get_data("dev")
    identity = identities("receiver")
    model, tok = load("receiver")
    initial_hash = parameter_hash(model)
    bridge = BridgeModel()
    count = sum(p.numel() for p in bridge.parameters())
    receiver_count = sum(p.numel() for p in model.parameters())
    assert count / receiver_count < 0.01
    wrapper = ReceiverWrapper(model, bridge)
    write_json(dest / "manifest.json", {"receiver": identity, "seed": seed,
        "trainable_parameters": count, "receiver_parameters": receiver_count,
        "parameter_fraction": count / receiver_count, "receiver_initial_parameter_sha256": initial_hash,
        "preregistration_sha256": sha(ROOT / "PREREGISTRATION.md"), "code_sha256": sha(__file__),
        "python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__,
        "platform": platform.platform(), "cpu": platform.processor(), "cuda": torch.cuda.is_available(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "command": sys.argv, "optimizer": "AdamW", "lr": 0.003, "weight_decay": 0,
        "batch_size": 8 if tiny else 16, "max_steps": 300 if tiny else 600})
    mechanics(wrapper, bridge, tok, hidden, dest)
    metrics, cached = evaluate(wrapper, bridge, tok, val_rows, val_hidden, dest / "eval_0000")
    answer_ids = torch.tensor([tok.encode(s, add_special_tokens=False)[0] for s in ANSWERS])
    labels = torch.tensor([r["label"] for r in rows])
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=0.003, weight_decay=0)
    batch_size, interval, max_steps = (8, 20, 300) if tiny else (16, 50, 600)
    generator = torch.Generator().manual_seed(seed + 1)
    order, cursor = torch.randperm(len(rows), generator=generator), 0
    best_ce, stale, best_step = float("inf"), 0, None
    gate1 = gate2 = False
    with open(dest / "curve.jsonl", "x") as curve:
        for step in range(1, max_steps + 1):
            t = time.monotonic()
            if cursor + batch_size > len(rows):
                order, cursor = torch.randperm(len(rows), generator=generator), 0
            idx = order[cursor:cursor+batch_size]
            cursor += batch_size
            bridge.train()
            optimizer.zero_grad(set_to_none=True)
            logits = wrapper(tokens(tok, [QUERY] * len(idx)), bridge.resampler(hidden[idx]))
            loss = F.cross_entropy(logits, answer_ids[labels[idx]])
            assert torch.isfinite(loss)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
            assert torch.isfinite(grad_norm)
            optimizer.step()
            record = {"step": step, "ce": loss.item(), "gradient_norm": float(grad_norm),
                "batch_accuracy": (logits[:, answer_ids].argmax(-1) == labels[idx]).float().mean().item(),
                "gate": bridge.reader.gate.item(), "step_s": time.monotonic()-t,
                "elapsed_s": time.monotonic()-start}
            curve.write(json.dumps(record, allow_nan=False) + "\n")
            curve.flush()
            event("train", **record)
            if step % interval:
                continue
            assert all(p.grad is None for p in model.parameters())
            metrics, _ = evaluate(wrapper, bridge, tok, val_rows, val_hidden,
                                   dest / f"eval_{step:04d}", cached)
            ce = metrics["correct"]["full_vocab_ce"]
            if ce < best_ce:
                improvement = best_ce-ce
                best_ce, best_step = ce, step
                torch.save(bridge.state_dict(), dest / "best.pt")
                stale = 0 if improvement >= 0.001 else stale + 1
            else:
                stale += 1
            if tiny:
                gate1 = metrics["correct"]["accuracy"] >= 0.95
                gate2 = all(metrics[c]["accuracy"] <= 0.50 for c in ["shuffled", "random_71001", "random_71002"])
                if gate1 and gate2:
                    break
            elif stale >= 3:
                break
    final_hash = parameter_hash(model)
    assert initial_hash == final_hash, "Frozen receiver changed"
    torch.save(bridge.state_dict(), dest / "last.pt")
    # Rerun all tiny-set conditions from actual inference, without frozen-output caching.
    if tiny and gate1 and gate2:
        _, rerun = evaluate(wrapper, bridge, tok, rows, hidden, dest / "deterministic_rerun")
        original = torch.load(dest / f"eval_{step:04d}" / "outputs.pt", weights_only=True)["results"]
        assert all(torch.equal(original[c][key], rerun[c][key]) for c in CONDITIONS for key in ["lp", "unconstrained"])
    write_json(dest / "decision.json", {"gate0": True, "gate1": gate1 if tiny else None,
        "gate2": gate2 if tiny else None, "step": step, "best_step": best_step,
        "best_dev_ce": best_ce, "best_checkpoint_sha256": sha(dest / "best.pt"),
        "last_checkpoint_sha256": sha(dest / "last.pt"), "receiver_final_parameter_sha256": final_hash,
        "elapsed_s": time.monotonic()-start, "external_spend_usd": 0,
        "verdict": "MECHANICAL GATES PASS — NOT SCIENTIFIC EVIDENCE" if tiny and gate1 and gate2
                    else "F — MECHANICS / TRAINING NO-GO" if tiny else "DEV TRAINING COMPLETE — FINAL NOT EVALUATED"})


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--split", choices=SPLITS, required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--mode", choices=["tiny", "full"], required=True)
    args = p.parse_args()
    if args.command == "prepare":
        seed_all(11000)
        prepare(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
