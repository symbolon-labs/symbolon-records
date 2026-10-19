"""Pure, independently testable paired statistics for the registered channel gate."""
import numpy as np

CORE = ["receiver_only", "correct", "shuffled", "random_71001", "random_71002", "zero", "text"]
NON_TEXT_CONTROLS = ["receiver_only", "shuffled", "random_71001", "random_71002", "zero"]
RANDOMIZED = ["shuffled", "random_71001", "random_71002"]


def select_checkpoint(evaluations):
    """Minimum recorded DEV full-vocabulary CE, ties choose earlier trained step."""
    candidates = [(float(m["conditions"]["correct"]["full_vocab_ce"]), int(step))
                  for step, m in evaluations if step > 0]
    assert candidates and all(np.isfinite(loss) for loss, _ in candidates)
    return min(candidates)[1]


def summarize(records, text_records=None, draws=10000, seed=81001):
    assert records and len({r["id"] for r in records}) == len(records)
    labels = np.array([r["label"] for r in records], dtype=np.int64)
    assert np.isin(labels, [0, 1, 2, 3]).all()
    group_ids = sorted({r["group"] for r in records})
    groups = np.array([[i for i,r in enumerate(records) if r["group"]==g] for g in group_ids])
    assert groups.shape == (len(group_ids), 4)
    assert all(sorted(labels[ix].tolist()) == [0, 1, 2, 3] for ix in groups)
    lp = {c: np.array([r["conditions"][c]["answer_log_probabilities"] for r in records], dtype=np.float64) for c in CORE}
    if text_records is not None:
        by_id = {r["id"]: r for r in text_records}
        assert len(by_id) == len(records) == len(text_records)
        assert set(by_id) == {r["id"] for r in records}
        ordered_text = [by_id[r["id"]] for r in records]
        assert all(t["label"] == r["label"] and t["group"] == r["group"] for r,t in zip(records, ordered_text))
        for condition in ["text_sender", "text_oracle", "receiver_chat_only"]:
            lp[condition] = np.array([r[condition]["answer_log_probabilities"] for r in ordered_text], dtype=np.float64)
    for condition, array in lp.items():
        assert array.shape == (len(records), 4) and np.isfinite(array).all(), condition
        assert (array <= 1e-6).all(), "Expected full-vocabulary log probabilities, not raw logits"
    predictions = {c: array.argmax(-1) for c,array in lp.items()}
    correctness = {c: pred == labels for c,pred in predictions.items()}
    margin = {}
    for c, array in lp.items():
        other = array.copy()
        other[np.arange(len(records)), labels] = -np.inf
        margin[c] = array[np.arange(len(records)), labels] - other.max(-1)
    indices = np.random.default_rng(seed).integers(0, len(groups), size=(draws, len(groups)))
    boots = {c: values[groups].mean(-1)[indices].mean(-1) for c,values in correctness.items()}
    accuracy = {}
    for condition, values in correctness.items():
        per_class, class_ci = [], []
        for y in range(4):
            # Exactly one member per class in each resampled nonce group.
            group_class = np.array([values[ix[labels[ix] == y][0]] for ix in groups], dtype=float)
            per_class.append(float(group_class.mean()))
            class_ci.append(np.quantile(group_class[indices].mean(-1), [.025,.975]).tolist())
        accuracy[condition] = {"mean": float(values.mean()),
            "ci95": np.quantile(boots[condition], [.025, .975]).tolist(),
            "per_class": per_class, "per_class_ci95": class_ci,
            "full_vocab_ce": float(-lp[condition][np.arange(len(records)), labels].mean()),
            "margin_mean": float(margin[condition].mean()),
            "margin_median": float(np.median(margin[condition]))}
    contrasts = {}
    for condition in lp:
        if condition == "correct":
            continue
        samples = boots["correct"] - boots[condition]
        margin_delta = margin["correct"]-margin[condition]
        margin_samples = margin_delta[groups].mean(-1)[indices].mean(-1)
        contrasts[condition] = {"accuracy_mean": float(correctness["correct"].mean()-correctness[condition].mean()),
            "accuracy_ci99": np.quantile(samples, [.005,.995]).tolist(),
            "accuracy_ci95": np.quantile(samples, [.025,.975]).tolist(),
            "margin_change_mean": float(margin_delta.mean()),
            "margin_change_median": float(np.median(margin_delta)),
            "margin_change_ci99": np.quantile(margin_samples, [.005,.995]).tolist()}
    equivalence = {c: np.quantile(boots[c]-boots["receiver_only"], [.025,.975]).tolist() for c in RANDOMIZED}
    checks = {
        "correct_ge90": accuracy["correct"]["mean"] >= .90,
        "each_class_ge85": min(accuracy["correct"]["per_class"]) >= .85,
        "receiver_exact_chance": accuracy["receiver_only"]["mean"] == .25,
        "randomized_controls_le35": all(accuracy[c]["mean"] <= .35 for c in RANDOMIZED),
        "paired99_gain_gt50": all(contrasts[c]["accuracy_ci99"][0] > .50 for c in NON_TEXT_CONTROLS),
        "randomized_controls_equivalent": all(bounds[0] >= -.10 and bounds[1] <= .10 for bounds in equivalence.values()),
    }
    donor_following = float(np.mean([p == r["donor_label"] for p,r in zip(predictions["shuffled"], records)]))
    return {"accuracy": accuracy, "correct_minus": contrasts,
            "control_equivalence_ci95": equivalence, "statistical_checks": checks,
            "shuffled_prediction_matches_donor": donor_following,
            "bootstrap": {"unit": "nonce group with four counterfactual labels", "seed": seed, "draws": draws},
            "n_examples": len(records), "n_groups": len(groups)}
