import copy
import unittest
from analysis_v2 import CORE, select_checkpoint, summarize


def fixture(correct_mode="perfect"):
    """Fabricated logits only, unrelated to any experimental/final data seed."""
    rows = []
    for group in range(32):
        for y in range(4):
            conditions = {}
            for condition in CORE:
                pred = y if condition in {"correct", "text"} else 0
                if condition == "correct" and correct_mode == "prior":
                    pred = 0
                if condition == "correct" and correct_mode == "asymmetric" and y == 3:
                    pred = 0
                lp = [-12.0]*4
                lp[pred] = -0.01
                conditions[condition] = {"answer_log_probabilities": lp}
            rows.append({"id": f"unit-{group}-{y}", "group": group,
                         "label": y, "donor_label": 0, "conditions": conditions})
    return rows


class AnalysisContracts(unittest.TestCase):
    def test_ideal_separates_and_equates_controls(self):
        summary = summarize(fixture(), draws=500)
        self.assertTrue(all(summary["statistical_checks"].values()))
        self.assertEqual(summary["correct_minus"]["receiver_only"]["accuracy_ci99"], [.75, .75])
        self.assertEqual(summary["correct_minus"]["text"]["accuracy_ci99"], [0, 0])

    def test_generic_prior_fails(self):
        checks = summarize(fixture("prior"), draws=500)["statistical_checks"]
        self.assertFalse(checks["correct_ge90"])
        self.assertFalse(checks["paired99_gain_gt50"])

    def test_asymmetric_fails(self):
        checks = summarize(fixture("asymmetric"), draws=500)["statistical_checks"]
        self.assertFalse(checks["each_class_ge85"])

    def test_random_reproducing_success_fails(self):
        rows = fixture()
        for row in rows:
            row["conditions"]["random_71001"] = copy.deepcopy(row["conditions"]["correct"])
        checks = summarize(rows, draws=500)["statistical_checks"]
        self.assertFalse(checks["randomized_controls_le35"])
        self.assertFalse(checks["paired99_gain_gt50"])

    def test_corrupted_groups_and_text_joins_fail(self):
        rows = fixture()
        broken = copy.deepcopy(rows)
        broken[0]["label"] = 1
        with self.assertRaises(AssertionError):
            summarize(broken, draws=10)
        with self.assertRaises(AssertionError):
            summarize(rows, text_records=[], draws=10)

    def test_checkpoint_not_selected_by_accuracy_or_latest(self):
        entries = [(0, {"conditions": {"correct": {"full_vocab_ce": 0.0}}}),
                   (50, {"conditions": {"correct": {"full_vocab_ce": 1.0}}}),
                   (100, {"conditions": {"correct": {"full_vocab_ce": 0.5}}}),
                   (150, {"conditions": {"correct": {"full_vocab_ce": 0.5}}})]
        self.assertEqual(select_checkpoint(entries), 100)


if __name__ == "__main__":
    unittest.main()
