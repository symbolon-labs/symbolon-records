"""Descriptive summaries only; does not change the frozen decision rule."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def describe(values):
    assert values and all(math.isfinite(v) for v in values)
    return {"min": min(values), "median": statistics.median(values),
            "mean": statistics.mean(values), "max": max(values)}


def summarize(records):
    output = {}
    for condition in records[0]["conditions"]:
        rows = [r["conditions"][condition]["diagnostics"] for r in records]
        if not rows[0]:
            assert all(not r for r in rows)
            continue
        output[condition] = {
            key: {"overall": describe([r[key] for r in rows]),
                  "per_class": [describe([d[key] for r, d in zip(records, rows) if r["label"] == y]) for y in range(4)]}
            for key in rows[0]
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "final"], required=True)
    args = parser.parse_args()
    if args.split == "final":
        assert (ROOT / "run_final/analysis.json").exists(), "Wait for the frozen primary analysis"
        source = ROOT / "run_final/evaluation/per_example.json"
    else:
        decision = json.loads((ROOT / "run_full/decision.json").read_text())
        source = ROOT / f"run_full/eval_{decision['best_step']:04d}/per_example.json"
    records = json.loads(source.read_text())
    result = {"purpose": "Post-hoc descriptive reader diagnostics; not a selection or pass criterion.",
              "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "diagnostics": summarize(records)}
    with (ROOT / f"{args.split}_reader_diagnostics.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
