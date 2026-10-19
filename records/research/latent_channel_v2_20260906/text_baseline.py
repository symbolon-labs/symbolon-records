"""Frozen sender forced-choice -> compact textual symbol -> frozen receiver."""
import argparse
import gc
import json
import time

import torch

from channel_v2 import ROOT, c, provenance

SENDER_INSTRUCTION = "Copy the value of the Signal field in the record below. Reply with exactly one letter: A, B, C, or D.\n\n"
RECEIVER_INSTRUCTION = "Copy the Signal value. Reply with exactly one letter: A, B, C, or D.\n\n"
LETTERS = list("ABCD")


def render(tok, body):
    return tok.apply_chat_template([{"role": "user", "content": body}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)


def score(model, tok, prompts, batch=4):
    ids = [tok.encode(p, add_special_tokens=False) for p in prompts]
    answers = [tok.encode(x, add_special_tokens=False) for x in LETTERS]
    assert all(len(a) == 1 for a in answers)
    answer_ids = [a[0] for a in answers]
    for p, tokens in zip(prompts, ids):
        for letter, a in zip(LETTERS, answers):
            assert tok.encode(p+letter, add_special_tokens=False) == tokens+a
    results = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch):
            selected = ids[start:start+batch]
            assert len({len(x) for x in selected}) == 1
            logits = model(input_ids=torch.tensor(selected), use_cache=False, logits_to_keep=1).logits[:, -1]
            lp = logits.log_softmax(-1)[:, answer_ids]
            for i in range(len(selected)):
                results.append({"input_ids": selected[i], "answer_ids": answer_ids,
                    "answer_log_probabilities": lp[i].tolist(), "prediction": int(lp[i].argmax()),
                    "unconstrained_token": int(logits[i].argmax())})
            c.event("text_score", completed=min(start+batch, len(prompts)), total=len(prompts))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["tiny", "dev", "final"], required=True)
    args = p.parse_args()
    # Do not compete with tiny training for memory; the local process must finish.
    assert (ROOT / "run_tiny/decision.json").exists()
    if args.split != "tiny":
        gate = json.loads((ROOT / "run_tiny/semantic_gates.json").read_text())
        assert gate["gate1"] and gate["gate2"]
    if args.split == "final":
        assert (ROOT / "run_full/FINAL_FREEZE.json").exists()
    dest = ROOT / f"text_{args.split}"
    assert not dest.exists()
    dest.mkdir()
    c.seed_all(92001)
    start = time.monotonic()
    rows, hidden = c.get_data(args.split)
    del hidden
    source_identity = c.identities("sender")
    model, tok = c.load("sender")
    sender_prompts = [render(tok, SENDER_INSTRUCTION+r["source"]) for r in rows]
    sender_start = time.monotonic()
    sender = score(model, tok, sender_prompts)
    sender_s = time.monotonic()-sender_start
    c.write_json(dest / "sender.json", {"prompts": sender_prompts, "outputs": sender,
                  "model": source_identity, "inference_s": sender_s})
    del model, tok
    gc.collect()
    receiver_identity = c.identities("receiver")
    model, tok = c.load("receiver")
    initial_hash = c.parameter_hash(model)
    compact_prompts = [render(tok, RECEIVER_INSTRUCTION+"Signal: "+letter) for letter in LETTERS]
    missing_prompt = render(tok, RECEIVER_INSTRUCTION+"The Signal value was withheld.")
    receiver_start = time.monotonic()
    receiver = score(model, tok, compact_prompts)
    missing = score(model, tok, [missing_prompt], batch=1)[0]
    receiver_s = time.monotonic()-receiver_start
    # Actual deterministic reread of every unique receiver control, same batches.
    assert receiver == score(model, tok, compact_prompts)
    assert missing == score(model, tok, [missing_prompt], batch=1)[0]
    assert initial_hash == c.parameter_hash(model)
    records = []
    for r, sent in zip(rows, sender):
        oracle_letter = r["source"].rsplit("Signal: ", 1)[1]
        assert oracle_letter in LETTERS
        oracle_index = LETTERS.index(oracle_letter)
        assert oracle_index == r["label"]
        records.append({"id": r["id"], "group": r["group"], "label": r["label"],
            "sender": sent, "text_sender": receiver[sent["prediction"]],
            "text_oracle": receiver[oracle_index], "receiver_chat_only": missing,
            "transmitted_text": LETTERS[sent["prediction"]],
            "transmitted_utf8_bytes": 1})
    metrics = {}
    for condition in ["sender", "text_sender", "text_oracle", "receiver_chat_only"]:
        metrics[condition] = {"accuracy": sum(r[condition]["prediction"]==r["label"] for r in records)/len(records),
            "per_class": [sum(r[condition]["prediction"]==y for r in records if r["label"]==y)/sum(r["label"]==y for r in records) for y in range(4)]}
    assert metrics["receiver_chat_only"]["accuracy"] == .25
    c.write_json(dest / "per_example.json", records)
    c.write_json(dest / "receiver.json", {"prompts": compact_prompts, "outputs": receiver,
         "missing_prompt": missing_prompt, "missing_output": missing, "model": receiver_identity,
         "unique_prompt_cache": True, "inference_s_excluding_rerun": receiver_s,
         "rerun_exact": True, "frozen_parameter_hash": initial_hash})
    c.write_json(dest / "summary.json", {"metrics": metrics,
         "oracle_text_sanity_pass": metrics["text_oracle"]["accuracy"] >= .90,
         "sender_inference_s": sender_s, "receiver_unique_inference_s": receiver_s,
         "elapsed_s": time.monotonic()-start, "external_spend_usd": 0,
         "code_sha256": c.sha(__file__), "text_preregistration_sha256": c.sha(ROOT / "TEXT_BASELINE_PREREGISTRATION.md"),
         "data_sha256": c.sha(ROOT / "data" / args.split / "examples.json"), "provenance": provenance()})
    c.event("text_result", metrics=metrics)


if __name__ == "__main__":
    main()
