"""Post-outcome audit of core records; not a replacement final decision."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    source = ROOT / "run_final/evaluation/per_example.json"
    records = json.loads(source.read_text())
    destination = ROOT / "independent_core_audit.json"
    assert not destination.exists()
    by_id = {r["id"]: r for r in records}
    assert len(by_id) == len(records) == 1024
    for row in records:
        donor = by_id[row["donor_id"]]
        assert donor["id"] != row["id"]
        assert row["donor_label"] == donor["label"]
        assert row["conditions"]["shuffled"]["answer_log_probabilities"] == donor["conditions"]["correct"]["answer_log_probabilities"]
        assert row["conditions"]["shuffled"]["unconstrained_token"] == donor["conditions"]["correct"]["unconstrained_token"]
    # Aggregate four counterfactuals per nonce before resampling; never treat
    # them as independent bootstrap units.
    groups = sorted({r["group"] for r in records})
    controls = ["receiver_only", "shuffled", "random_71001", "random_71002", "zero"]
    names = ["correct", *controls]
    counts = []
    for group in groups:
        block = [r for r in records if r["group"] == group]
        assert sorted(r["label"] for r in block) == [0, 1, 2, 3]
        counts.append([sum(int(np.argmax(r["conditions"][name]["answer_log_probabilities"])) == r["label"] for r in block) / 4 for name in names])
    group_scores = np.array(counts)
    indices = np.random.default_rng(81001).integers(0, len(groups), size=(10000, len(groups)))
    bootstrap = group_scores[indices].mean(axis=1)
    contrasts = {name: np.quantile(bootstrap[:, 0]-bootstrap[:, i], [.005, .995]).tolist()
                 for i, name in enumerate(names) if i}
    result = {"purpose": "Post-outcome independent core audit, not the final verdict; strong text remains mandatory.",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "audit_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "all_shuffled_log_probabilities_equal_matching_donor_correct_exactly": True,
        "shuffled_donor_following": sum(r["conditions"]["shuffled"]["prediction"] == r["donor_label"] for r in records)/len(records),
        "accuracy": dict(zip(names, group_scores.mean(axis=0).tolist())),
        "correct_minus_control_ci99": contrasts,
        "bootstrap_unit": "256 nonce groups, each containing all four classes", "draws": 10000, "seed": 81001}
    with destination.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
