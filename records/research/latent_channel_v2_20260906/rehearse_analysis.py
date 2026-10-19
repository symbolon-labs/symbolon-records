"""Exercise final statistics against existing tiny records, never final data."""
import json

from channel_v2 import ROOT, c
from analysis_v2 import summarize


def main():
    decision = json.loads((ROOT / "run_tiny/decision.json").read_text())
    records = json.loads((ROOT / f"run_tiny/eval_{decision['step']:04d}/per_example.json").read_text())
    text = json.loads((ROOT / "text_tiny/per_example.json").read_text())
    result = summarize(records, text)
    assert result["accuracy"]["correct"]["mean"] == 1.0
    assert result["accuracy"]["text_sender"]["mean"] == 1.0
    assert result["accuracy"]["receiver_only"]["mean"] == .25
    assert result["shuffled_prediction_matches_donor"] == 1.0
    result.update({"purpose": "Analysis-harness rehearsal on trained-on tiny data, not a final scientific result",
                   "held_out": False, "final_data_materialized": (ROOT / "data/final").exists(),
                   "code_sha256": c.sha(ROOT / "analysis_v2.py")})
    assert not result["final_data_materialized"]
    c.write_json(ROOT / "tiny_analysis_rehearsal.json", result)
    print(json.dumps({"harness": "PASS", "statistical_checks_on_tiny_not_a_gate": result["statistical_checks"],
                      "shuffled_follows_donor": result["shuffled_prediction_matches_donor"]}))


if __name__ == "__main__":
    main()
