"""Independent input/token audit, after freeze; no inference or selection."""
import hashlib
import json
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    freeze_path = ROOT / "run_full/FINAL_FREEZE.json"
    freeze = json.loads(freeze_path.read_text())
    assert not (ROOT / "postfreeze_input_audit.json").exists()
    for entry in freeze["components"].values():
        assert digest(entry["path"]) == entry["sha256"]
    tokenizers = {
        role: AutoTokenizer.from_pretrained(model["path"], local_files_only=True)
        for role, model in freeze["models"].items()
    }
    query = "Hidden signal (A/B/C/D):"
    answers = [" A", " B", " C", " D"]
    used = set()
    old = json.loads((ROOT.parent / "latent_channel_v1_20260906/data/tiny/examples.json").read_text())
    old_nonces = {r["nonce"] for r in old}
    summaries = {}
    for split, expected_count in [("tiny", 32), ("train", 1024), ("dev", 256), ("final", 1024)]:
        directory = ROOT / "data" / split
        manifest = json.loads((directory / "manifest.json").read_text())
        rows = json.loads((directory / "examples.json").read_text())
        assert len(rows) == expected_count
        assert digest(directory / "examples.json") == manifest["examples_sha256"]
        assert digest(directory / "sender.pt") == manifest["capture_sha256"]
        capture = torch.load(directory / "sender.pt", weights_only=True)
        assert torch.isfinite(capture["hidden"]).all()
        nonces = {r["nonce"] for r in rows}
        assert not nonces & (used | old_nonces)
        used |= nonces
        assert len(nonces) * 4 == len(rows)
        assert len({r["id"] for r in rows}) == len(rows)
        for nonce in nonces:
            assert sorted(r["label"] for r in rows if r["nonce"] == nonce) == [0, 1, 2, 3]
        for i, row in enumerate(rows):
            assert row["query"] == query
            assert row["source"] == f"Noise: {row['nonce']}\nSignal: {'ABCD'[row['label']]}"
            sender_ids = tokenizers["sender"].encode(row["source"], add_special_tokens=False)
            assert len(sender_ids) == 22
            assert capture["source_ids"][i].tolist() == sender_ids
            receiver = tokenizers["receiver"]
            for answer in answers:
                continuation = receiver.encode(answer, add_special_tokens=False)
                assert len(continuation) == 1
                for prompt in [query, row["source"] + "\n" + query]:
                    assert receiver.encode(prompt + answer, add_special_tokens=False) == receiver.encode(prompt, add_special_tokens=False) + continuation
        summaries[split] = {"examples": len(rows), "nonce_groups": len(nonces),
            "source_tokens": 22, "query_tokens": len(tokenizers["receiver"].encode(query)),
            "examples_sha256": manifest["examples_sha256"], "capture_sha256": manifest["capture_sha256"]}
        del capture
    prepare = json.loads((ROOT / "prepare_final_extension.json").read_text())
    assert freeze["utc"] <= prepare["utc"]
    result = {"status": "PASS", "splits": summaries, "frozen_component_hashes_match": True,
        "cross_split_nonce_disjoint": True, "v1_observed_nonces_excluded": True,
        "counterfactual_balance": True, "constant_receiver_query": True,
        "source_ids_independently_retokenized": True, "joint_continuation_boundaries_exact": True,
        "final_freeze_utc": freeze["utc"], "final_prepare_utc": prepare["utc"],
        "chronology_caveat": "Local timestamps/hashes are not third-party timestamp attestations.",
        "freeze_sha256": digest(freeze_path), "audit_code_sha256": digest(__file__),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with (ROOT / "postfreeze_input_audit.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("PASS: independently retokenized every source and audited all four splits")


if __name__ == "__main__":
    main()
