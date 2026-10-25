"""POST HOC checks for test E. Not in PREREGISTRATION.md; written after results.json.

They explain two patterns in results.json: equal S_block and S_zero results, and
equal S_rand medians in sets A and B. They change no preregistered number or the
reading. Output: posthoc.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))  # -I does not put the script folder on sys.path

import numpy as np  # noqa: E402

import influence as inf  # noqa: E402
import run_analysis as ra  # noqa: E402


def main():
    data = {n: ra.load_set(rel) for n, rel in ra.SETS.items()}
    out = {}
    for n, s in data.items():
        lp = {a: s["rec"][a]["lp"] for a in s["rec"]}
        benign = ra.scores_for(s, "correct")
        per_label = {}
        for lab in range(4):
            v = benign["S_block"][s["labels"] == lab]
            per_label[str(lab)] = dict(min=float(v.min()), max=float(v.max()))
        out[n] = dict(
            zero_equals_block_max_abs=float(np.abs(lp["zero"] - lp["receiver_only"]).max()),
            distinct_benign_lp_rows=int(len(np.unique(lp["correct"], axis=0))),
            distinct_benign_S_block=int(len(np.unique(benign["S_block"]))),
            benign_S_block_range_by_label=per_label,
            median_benign_answer_prob=float(np.median(np.exp(inf.answer_logprobs(lp["correct"]).max(1)))))
    # Shared random references across sets (same generator seeds 70000..70004).
    ra_a, ra_b, ra_c = ({r: data[n]["rec"][r]["lp"] for r in ra.RANDOM_ARMS} for n in "ABC")
    out["random_refs_A_vs_B_max_abs"] = float(max(np.abs(ra_a[r] - ra_b[r]).max() for r in ra.RANDOM_ARMS))
    out["random_refs_C_vs_A_first64_max_abs"] = float(
        max(np.abs(ra_c[r] - ra_a[r][:64]).max() for r in ra.RANDOM_ARMS))
    out["labels_A_equal_B"] = bool(np.array_equal(data["A"]["labels"], data["B"]["labels"]))
    out["labels_C_equal_A_first64"] = bool(np.array_equal(data["C"]["labels"], data["A"]["labels"][:64]))
    # Sensitivity: the primary comparison without set B, which repeats set A here.
    names = ["A", "C"]
    donors = {n: inf.recover_donors(data[n]["z_clean"], data[n]["z_replay"]) for n in names}
    pos = np.concatenate([ra.scores_for(data[n], "donor_swap")["S_rand"] for n in names])
    neg = np.concatenate([ra.scores_for(data[n], "correct")["S_rand"] for n in names])
    strata = np.concatenate([np.full(len(data[n]["labels"]), i) for i, n in enumerate(names)])
    clusters = np.concatenate([i * 10**6 + data[n]["groups"] for i, n in enumerate(names)])
    r = inf.bootstrap_auroc(pos, neg, clusters, strata, n_boot=ra.N_BOOT, seed=ra.SEED)
    r["reading_with_the_same_rule"] = inf.reading(r["lo"], r["hi"])
    out["sensitivity_S_rand_replay_vs_benign_sets_A_C"] = r
    harmful = np.concatenate([data[n]["labels"][donors[n]] != data[n]["labels"] for n in names])
    out["sensitivity_harmful_rows_sets_A_C"] = int(harmful.sum())
    (ROOT / "posthoc.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
