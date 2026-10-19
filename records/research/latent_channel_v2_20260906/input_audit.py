"""Tokenization and information-asymmetry checks; no model inference."""
import json
from channel_v2 import ROOT, c, provenance
from transformers import AutoTokenizer


def main():
    rows, hidden = c.get_data("tiny")
    tok = AutoTokenizer.from_pretrained(c.model_path("receiver"), local_files_only=True)
    query_ids = tok.encode(c.QUERY, add_special_tokens=False)
    answer_ids = [tok.encode(a, add_special_tokens=False) for a in c.ANSWERS]
    assert all(len(a) == 1 for a in answer_ids)
    for row in rows:
        for answer, ids in zip(c.ANSWERS, answer_ids):
            for prompt in [c.QUERY, row["source"] + "\n" + c.QUERY]:
                assert tok.encode(prompt+answer, add_special_tokens=False) == tok.encode(prompt, add_special_tokens=False)+ids
        source_ids = tok.encode(row["source"], add_special_tokens=False)
        assert source_ids[-1] == answer_ids[row["label"]][0]
        assert len(source_ids) == 22
        assert row["query"] == c.QUERY
    old_seal = json.loads((c.BASE.parent / "seal.json").read_text()) if hasattr(c, "BASE") else None
    # V1 is accessed through the explicitly pinned inherited source.
    from channel_v2 import BASE
    old_seal = json.loads((BASE.parent / "seal.json").read_text())
    assert all(c.sha(BASE.parent / p) == h for p,h in old_seal["files"].items())
    c.write_json(ROOT / "input_audit.json", {
        "status": "PASS", "receiver_query_ids": query_ids,
        "answer_ids": answer_ids, "joint_continuation_tokenization_exact": True,
        "source_final_token_is_gold": True, "all_source_lengths": 22,
        "receiver_query_identical": True, "all_v1_sealed_hashes_match": True,
        "v1_sealed_files": len(old_seal["files"]), "provenance": provenance(),
    })
    print("PASS: source labels, exact full-continuation boundaries, input lengths, constant query, V1 seal")


if __name__ == "__main__":
    main()
