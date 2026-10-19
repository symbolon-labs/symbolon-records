"""One-shot V2 final freeze, inference and audited analysis."""
import argparse
import inspect
import json
import sys
import time
import traceback

import numpy as np
import torch
from transformers.models.qwen3 import modeling_qwen3

from channel_v2 import ROOT, BASE, BridgeModel, c
from analysis_v2 import CORE, select_checkpoint, summarize


def components():
    paths = {name: ROOT / name for name in ["channel_v2.py", "final_v2.py",
        "analysis_v2.py", "text_baseline.py", "dev_reproduce.py", "confirm.py", "PREREGISTRATION.md",
        "TEXT_BASELINE_PREREGISTRATION.md", "FINAL_EXECUTION.md"]}
    paths["inherited_v1"] = BASE
    paths["qwen_runtime"] = type(ROOT)(inspect.getfile(modeling_qwen3))
    return {name: {"path": str(path), "sha256": c.sha(path)} for name,path in paths.items()}


def exclusive_json(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def read_freeze(check_models=False):
    f = json.loads((ROOT / "run_full/FINAL_FREEZE.json").read_text())
    assert f["components"] == components(), "Frozen code/runtime changed"
    assert f["checkpoint_sha256"] == c.sha(ROOT / "run_full/best.pt")
    assert f["versions"] == {"torch": torch.__version__, "numpy": np.__version__,
                             "transformers": c.transformers.__version__, "python": sys.version}
    if check_models:
        assert f["models"] == {role: c.identities(role) for role in ["sender", "receiver"]}
    return f


def development_evaluations():
    """Recompute the actual selection metric, not just trust the metric JSON."""
    rows, _ = c.get_data("dev")
    labels = torch.tensor([r["label"] for r in rows])
    evaluations = []
    for path in (ROOT / "run_full").glob("eval_*/metrics.json"):
        step = int(path.parent.name.split("_")[1])
        metric = json.loads(path.read_text())
        assert metric["output_sha256"] == c.sha(path.parent / "outputs.pt")
        output = torch.load(path.parent / "outputs.pt", weights_only=True)
        assert torch.equal(output["labels"], labels)
        ce = -output["results"]["correct"]["lp"][torch.arange(len(rows)), labels].mean().item()
        assert ce == metric["conditions"]["correct"]["full_vocab_ce"]
        evaluations.append((step, metric))
    return evaluations


def freeze():
    run = ROOT / "run_full"
    assert not (run / "FINAL_FREEZE.json").exists()
    assert all(not (ROOT / p).exists() for p in ["data/final", "run_final", "text_final", "prepare_final_extension.json"])
    gate = json.loads((ROOT / "run_tiny/semantic_gates.json").read_text())
    assert all(gate[k] for k in ["gate0", "gate1", "gate2"])
    assert json.loads((ROOT / "development_job/state.json").read_text())["status"] == "complete"
    assert json.loads((ROOT / "train_full_completion.json").read_text())["status"] == "completed"
    text_dev = json.loads((ROOT / "text_dev/summary.json").read_text())
    d = json.loads((run / "decision.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    extension = json.loads((ROOT / "train_full_extension.json").read_text())
    assert extension["extension_sha256"] == c.sha(ROOT / "channel_v2.py")
    assert extension["base_sha256"] == manifest["code_sha256"] == c.sha(BASE)
    assert extension["preregistration_sha256"] == manifest["preregistration_sha256"] == c.sha(ROOT / "PREREGISTRATION.md")
    assert d["receiver_final_parameter_sha256"] == manifest["receiver_initial_parameter_sha256"]
    assert d["best_checkpoint_sha256"] == c.sha(run / "best.pt")
    evaluations = development_evaluations()
    selected = select_checkpoint(evaluations)
    assert selected == d["best_step"]
    selected_metrics = dict(evaluations)[selected]
    reproduction = json.loads((ROOT / "dev_reproduction/verification.json").read_text())
    assert reproduction["all_conditions_exact"] and reproduction["selected_step"] == selected
    assert reproduction["checkpoint_sha256"] == c.sha(run / "best.pt")
    assert reproduction["extension_sha256"] == c.sha(ROOT / "channel_v2.py")
    models = {role: c.identities(role) for role in ["sender", "receiver"]}
    assert models["receiver"] == manifest["receiver"]
    assert models["sender"] == json.loads((ROOT / "data/train/manifest.json").read_text())["model"]
    assert text_dev["code_sha256"] == c.sha(ROOT / "text_baseline.py")
    assert text_dev["text_preregistration_sha256"] == c.sha(ROOT / "TEXT_BASELINE_PREREGISTRATION.md")
    value = {"components": components(), "checkpoint_sha256": c.sha(run / "best.pt"),
        "selected_step": selected, "selected_dev_metrics": selected_metrics,
        "text_dev_metrics": text_dev["metrics"], "models": models,
        "receiver_parameter_sha256": d["receiver_final_parameter_sha256"],
        "versions": {"torch": torch.__version__, "numpy": np.__version__,
                     "transformers": c.transformers.__version__, "python": sys.version},
        "final_seed": 44002, "groups": 256, "examples": 1024,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "final_previously_materialized": False, "decision_rule": "PREREGISTRATION.md unchanged",
        "text_dev_output_sha256": c.sha(ROOT / "text_dev/per_example.json"),
        "dev_reproduction_sha256": c.sha(ROOT / "dev_reproduction/verification.json"),
        "command": sys.argv, "external_spend_usd": 0}
    exclusive_json(run / "FINAL_FREEZE.json", value)
    c.event("final_frozen", selected_step=selected, final_previously_materialized=False)


def stock_parity(wrapper, bridge, tok):
    ids = c.tokens(tok, [c.QUERY]*8)
    with torch.no_grad():
        wrapper.hook.remove()
        stock = wrapper.model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[:, -1]
        wrapper.hook = wrapper.model.model.layers[13].register_forward_hook(wrapper._hook)
        disabled = wrapper(ids)
        zero = wrapper(ids, torch.zeros(8, 8, 64))
    assert torch.equal(stock, disabled) and torch.equal(stock, zero)
    return {"stock_disabled_full_vocab_exact": True, "stock_zero_full_vocab_exact": True}


def score():
    start = time.monotonic()
    c.seed_all(53001)
    f = read_freeze(check_models=True)
    dest = ROOT / "run_final"
    assert not dest.exists(), "One-shot final: preserve partial attempts"
    rows, hidden = c.get_data("final")
    assert len(rows) == 1024 and hidden.shape == (1024, 22, 2048)
    extension = json.loads((ROOT / "prepare_final_extension.json").read_text())
    assert extension["extension_sha256"] == c.sha(ROOT / "channel_v2.py")
    assert json.loads((ROOT / "data/final/manifest.json").read_text())["model"] == f["models"]["sender"]
    dest.mkdir()
    c.write_json(dest / "started.json", {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "freeze_sha256": c.sha(ROOT / "run_full/FINAL_FREEZE.json"), "final_burned": True})
    model, tok = c.load("receiver")
    initial_hash = c.parameter_hash(model)
    assert initial_hash == f["receiver_parameter_sha256"]
    bridge = BridgeModel()
    bridge.load_state_dict(torch.load(ROOT / "run_full/best.pt", weights_only=True))
    bridge.eval()
    wrapper = c.ReceiverWrapper(model, bridge)
    parity = stock_parity(wrapper, bridge, tok)
    metrics, original = c.evaluate(wrapper, bridge, tok, rows, hidden, dest / "evaluation")
    payload = torch.load(dest / "evaluation/outputs.pt", weights_only=True)
    repeats = {}
    with torch.no_grad():
        for condition in CORE:
            ids = c.tokens(tok, [r["source"]+"\n"+c.QUERY if condition=="text" else c.QUERY for r in rows[:32]])
            memory = payload["memories"].get(condition)
            lps, predictions = [], []
            for i in range(0, 32, 8):
                logits = wrapper(ids[i:i+8], None if memory is None else memory[i:i+8])
                lps.append(logits.log_softmax(-1)[:, payload["answer_ids"]])
                predictions.append(logits.argmax(-1))
            repeats[condition] = {"lp": torch.cat(lps), "unconstrained": torch.cat(predictions)}
            assert all(torch.equal(original[condition][k][:32], repeats[condition][k]) for k in repeats[condition])
    torch.save(repeats, dest / "rerun.pt")
    final_hash = c.parameter_hash(model)
    assert initial_hash == final_hash
    read_freeze()
    c.write_json(dest / "execution.json", {"elapsed_s": time.monotonic()-start,
        "receiver_initial_hash": initial_hash, "receiver_final_hash": final_hash,
        "parity": parity, "rerun_exact": True, "rerun_n": 32,
        "core_output_sha256": c.sha(dest / "evaluation/outputs.pt"),
        "per_example_sha256": c.sha(dest / "evaluation/per_example.json"),
        "rerun_sha256": c.sha(dest / "rerun.pt"), "freeze": f,
        "strong_text_pending": True, "final_burned": True, "external_spend_usd": 0})
    c.event("final_core_complete", stronger_text_still_required=True)


def analyze():
    start = time.monotonic()
    c.seed_all(53002)
    f = read_freeze(check_models=True)
    dest = ROOT / "run_final"
    assert not (dest / "analysis.json").exists(), "Preserve the first final decision"
    execution = json.loads((dest / "execution.json").read_text())
    assert execution["freeze"] == f
    for path, key in [("evaluation/outputs.pt", "core_output_sha256"),
                      ("evaluation/per_example.json", "per_example_sha256"),
                      ("rerun.pt", "rerun_sha256")]:
        assert c.sha(dest / path) == execution[key], path
    examples, hidden = c.get_data("final")
    records = json.loads((dest / "evaluation/per_example.json").read_text())
    assert len(examples) == len(records) == 1024
    assert all((a["id"], a["group"], a["label"]) == (b["id"], b["group"], b["label"])
               for a,b in zip(examples, records))
    payload = torch.load(dest / "evaluation/outputs.pt", weights_only=True)
    labels = torch.tensor([r["label"] for r in records])
    assert torch.equal(labels, payload["labels"])
    repeats = torch.load(dest / "rerun.pt", weights_only=True)
    for condition in CORE:
        tensor = payload["results"][condition]
        assert torch.equal(tensor["lp"], torch.tensor([r["conditions"][condition]["answer_log_probabilities"] for r in records]))
        assert tensor["lp"].argmax(-1).tolist() == [r["conditions"][condition]["prediction"] for r in records]
        assert tensor["unconstrained"].tolist() == [r["conditions"][condition]["unconstrained_token"] for r in records]
        assert all(torch.equal(tensor[key][:32], repeats[condition][key]) for key in ["lp", "unconstrained"])
    bridge = BridgeModel()
    bridge.load_state_dict(torch.load(ROOT / "run_full/best.pt", weights_only=True))
    bridge.eval()
    expected_memories, expected_donors = c.make_memories(bridge, hidden)
    assert torch.equal(payload["donors"], torch.tensor(expected_donors))
    donors = payload["donors"].tolist()
    assert sorted(donors) == list(range(len(records))) and all(i != d for i,d in enumerate(donors))
    for i,d in enumerate(donors):
        assert records[i]["donor_id"] == examples[d]["id"]
        assert records[i]["donor_label"] == examples[d]["label"]
    norms = {}
    with torch.no_grad():
        for condition, memory in expected_memories.items():
            assert torch.equal(memory, payload["memories"][condition]), condition
            norm = memory.norm(dim=-1)
            expanded = bridge.projector(memory).norm(dim=-1)
            norms[condition] = {"min": norm.min().item(), "max": norm.max().item(),
                "mean": norm.mean().item(),
                "per_class_mean": [norm[labels==y].mean().item() for y in range(4)],
                "expanded_per_class_mean": [expanded[labels==y].mean().item() for y in range(4)]}
        correct_norm = expected_memories["correct"].norm(dim=-1)
        for condition in ["random_71001", "random_71002"]:
            assert torch.allclose(correct_norm, expected_memories[condition].norm(dim=-1), atol=1e-5, rtol=0)
    zero_exact = all(torch.equal(payload["results"]["zero"][key], payload["results"]["receiver_only"][key]) for key in ["lp", "unconstrained"])
    assert zero_exact
    text_dir = ROOT / "text_final"
    text_summary = json.loads((text_dir / "summary.json").read_text())
    assert text_summary["code_sha256"] == c.sha(ROOT / "text_baseline.py")
    assert text_summary["text_preregistration_sha256"] == c.sha(ROOT / "TEXT_BASELINE_PREREGISTRATION.md")
    assert text_summary["data_sha256"] == c.sha(ROOT / "data/final/examples.json")
    text = json.loads((text_dir / "per_example.json").read_text())
    text_sender = json.loads((text_dir / "sender.json").read_text())
    text_receiver = json.loads((text_dir / "receiver.json").read_text())
    assert text_sender["model"] == f["models"]["sender"]
    assert text_receiver["model"] == f["models"]["receiver"]
    assert len(text_sender["outputs"]) == len(text_sender["prompts"]) == len(examples)
    assert len(text_receiver["outputs"]) == len(text_receiver["prompts"]) == 4
    for output in [*text_receiver["outputs"], text_receiver["missing_output"]]:
        assert output["prediction"] == int(np.argmax(output["answer_log_probabilities"]))
    assert text_receiver["rerun_exact"] and text_receiver["frozen_parameter_hash"] == f["receiver_parameter_sha256"]
    text_by_id = {r["id"]: r for r in text}
    assert len(text) == len(text_by_id) == len(examples)
    for example, sent in zip(examples, text_sender["outputs"]):
        row = text_by_id[example["id"]]
        assert row["sender"] == sent
        predicted = int(np.argmax(sent["answer_log_probabilities"]))
        assert predicted == sent["prediction"]
        assert row["transmitted_text"] == "ABCD"[predicted]
        assert row["text_sender"] == text_receiver["outputs"][predicted]
        parsed = "ABCD".index(example["source"].rsplit("Signal: ", 1)[1])
        assert row["text_oracle"] == text_receiver["outputs"][parsed]
        assert row["receiver_chat_only"] == text_receiver["missing_output"]
    result = summarize(records, text)
    for condition in ["text_sender", "text_oracle", "receiver_chat_only"]:
        assert result["accuracy"][condition]["mean"] == text_summary["metrics"][condition]["accuracy"]
        assert result["accuracy"][condition]["per_class"] == text_summary["metrics"][condition]["per_class"]
    mechanics = {"zero_exact": zero_exact,
        "stock_disabled_parity": execution["parity"]["stock_disabled_full_vocab_exact"],
        "stock_zero_parity": execution["parity"]["stock_zero_full_vocab_exact"],
        "actual_rerun_exact": execution["rerun_exact"],
        "endpoint_frozen": execution["receiver_initial_hash"] == execution["receiver_final_hash"] == f["receiver_parameter_sha256"],
        "constant_transmitted_norm": norms["correct"]["min"] >= 8-1e-5 and norms["correct"]["max"] <= 8+1e-5,
        "all_latent_tensors_independently_reproduced": True,
        "all_model_tokenizer_code_hashes_match": True,
        "strong_text_complete": True,
        "final_frozen_before_materialization": f["final_previously_materialized"] is False}
    passed = all(result["statistical_checks"].values()) and all(mechanics.values())
    result.update({"mechanical_checks": mechanics, "norms": norms,
        "analysis_elapsed_s": time.monotonic()-start, "external_spend_usd": 0,
        "gate3": "PASS" if passed else "NO-GO", "freeze_sha256": c.sha(ROOT / "run_full/FINAL_FREEZE.json"),
        "text_summary": text_summary,
        "verdict": "A — LATENT COMMUNICATION CHANNEL ESTABLISHED" if passed else "F — MECHANICS / TRAINING NO-GO",
        "scope": "Held-out same-family synthetic two-bit channel; no cross-family, natural, Arabic or free-generation claim."})
    exclusive_json(dest / "analysis.json", result)
    c.event("final_decision", gate3=result["gate3"], checks=result["statistical_checks"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["freeze", "score", "analyze"])
    args = p.parse_args()
    try:
        {"freeze": freeze, "score": score, "analyze": analyze}[args.command]()
    except BaseException:
        if args.command == "score" and (ROOT / "run_final").exists():
            c.write_json(ROOT / "run_final/error.json", {"traceback": traceback.format_exc()})
        raise
