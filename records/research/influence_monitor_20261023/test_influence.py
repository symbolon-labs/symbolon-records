"""Synthetic unit tests for test E. They use no saved result array and no model.

Run from ember/:
    .venv/bin/python -I research/influence_monitor_20261023/test_influence.py
"""
import itertools
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))  # -I does not put the script folder on sys.path

import numpy as np  # noqa: E402

import influence as inf  # noqa: E402
import run_analysis as ra  # noqa: E402


def brute_auroc(pos, neg):
    s = 0.0
    for p, n in itertools.product(pos, neg):
        s += 1.0 if p > n else 0.5 if p == n else 0.0
    return s / (len(pos) * len(neg))


def synthetic_set(groups=6, seed=0):
    """A fake set with the structure of run_analysis.load_set and the toy's properties.

    The receiver input is the same on every row, so the replay output at row i
    is the benign output of row d(i), and the block and zero outputs are constant.
    """
    rng = np.random.default_rng(seed)
    n = 4 * groups
    labels = np.concatenate([rng.permutation(4) for _ in range(groups)])
    group_ids = np.repeat(np.arange(groups) + 100, 4)
    z = rng.standard_normal((n, 8, 64)).astype(np.float32)
    d = inf.derangement(n, seed=61001)
    lp_correct = rng.normal(-12.0, 1.0, (n, 4))
    lp_correct[np.arange(n), labels] = rng.normal(-0.01, 0.005, n)
    rec = {"correct": lp_correct, "donor_swap": lp_correct[d],
           "receiver_only": np.tile([-0.1, -3.0, -4.0, -5.0], (n, 1)),
           "zero": np.tile([-0.2, -2.5, -4.5, -5.5], (n, 1)),
           "attack": rng.normal(-2.0, 1.0, (n, 4))}
    for s in range(5):
        rec[f"random_{s}"] = rng.normal(-2.0, 1.0, (n, 4))
    rec = {a: {"lp": v, "pred": v.argmax(1)} for a, v in rec.items()}
    return dict(rec=rec, labels=labels, groups=group_ids, z_clean=z, z_replay=z[d],
                summary={}), d


class TestScores(unittest.TestCase):
    def test_renormalized_rows_sum_to_one_and_ignore_a_row_constant(self):
        rng = np.random.default_rng(1)
        lp = rng.normal(-5, 3, (50, 4))
        q = inf.answer_logprobs(lp)
        np.testing.assert_allclose(np.exp(q).sum(1), 1.0, atol=1e-12)
        np.testing.assert_allclose(inf.answer_logprobs(lp + rng.normal(0, 4, (50, 1))), q, atol=1e-12)

    def test_kl_hand_value_zero_and_nonnegative(self):
        p = np.log(np.array([[0.5, 0.25, 0.125, 0.125]]))
        q = np.log(np.array([[0.25, 0.25, 0.25, 0.25]]))
        want = 0.5 * np.log(2) + 0.25 * 0 + 2 * 0.125 * np.log(0.5)
        self.assertAlmostEqual(float(inf.kl(p, q)[0]), want, places=12)
        self.assertAlmostEqual(float(inf.kl(p, p)[0]), 0.0, places=12)
        rng = np.random.default_rng(2)
        a = inf.answer_logprobs(rng.normal(0, 3, (200, 4)))
        b = inf.answer_logprobs(rng.normal(0, 3, (200, 4)))
        self.assertTrue(np.all(inf.kl(a, b) >= -1e-12))

    def test_influence_averages_over_references(self):
        rng = np.random.default_rng(3)
        m, r1, r2 = (rng.normal(0, 2, (10, 4)) for _ in range(3))
        qm = inf.answer_logprobs(m)
        want = (inf.kl(qm, inf.answer_logprobs(r1)) + inf.kl(qm, inf.answer_logprobs(r2))) / 2
        np.testing.assert_allclose(inf.influence(m, [r1, r2]), want, atol=1e-12)


class TestAuroc(unittest.TestCase):
    def test_known_values(self):
        self.assertEqual(inf.auroc([3, 4, 5], [0, 1, 2]), 1.0)
        self.assertEqual(inf.auroc([0, 1, 2], [3, 4, 5]), 0.0)
        self.assertEqual(inf.auroc([1, 2, 3], [3, 2, 1]), 0.5)
        self.assertEqual(inf.auroc([1, 2], [2, 3]), 0.125)

    def test_matches_brute_force_with_ties(self):
        rng = np.random.default_rng(4)
        for _ in range(20):
            pos = rng.integers(0, 6, rng.integers(3, 15)).astype(float)
            neg = rng.integers(0, 6, rng.integers(3, 15)).astype(float)
            self.assertAlmostEqual(inf.auroc(pos, neg), brute_auroc(pos, neg), places=12)

    def test_tpr_at_fpr(self):
        neg = np.arange(100, dtype=float)
        pos = np.array([98.5, 99.5, 50.0, 200.0])
        r = inf.tpr_at_fpr(pos, neg, 0.01)
        self.assertEqual(r["threshold"], 98.0)
        self.assertLessEqual(r["fpr"], 0.01)
        self.assertEqual(r["tpr"], 0.75)


class TestBootstrapAndReading(unittest.TestCase):
    def _design(self, n_groups=64, n_sets=3):
        clusters = np.repeat(np.arange(n_groups), 4)
        strata = clusters % n_sets
        return clusters, strata

    def test_reading_rule_boundaries(self):
        self.assertEqual(inf.reading(0.40, 0.60), "E-NULL")
        self.assertEqual(inf.reading(0.45, 0.55), "E-NULL")
        self.assertEqual(inf.reading(0.61, 0.80), "E-SEP")
        self.assertEqual(inf.reading(0.20, 0.39), "E-SEP-REVERSE")
        self.assertEqual(inf.reading(0.39, 0.55), "E-INCONCLUSIVE")
        self.assertEqual(inf.reading(0.55, 0.65), "E-INCONCLUSIVE")
        self.assertEqual(inf.reading(0.60, 0.70), "E-INCONCLUSIVE")

    def test_planted_separation_is_found(self):
        rng = np.random.default_rng(5)
        cl, st = self._design()
        neg = rng.standard_normal(len(cl))
        pos = rng.standard_normal(len(cl)) + 1.5
        r = inf.bootstrap_auroc(pos, neg, cl, st, n_boot=2000, seed=1)
        self.assertEqual(inf.reading(r["lo"], r["hi"]), "E-SEP")
        r = inf.bootstrap_auroc(-pos, neg, cl, st, n_boot=2000, seed=1)
        self.assertEqual(inf.reading(r["lo"], r["hi"]), "E-SEP-REVERSE")

    def test_null_gives_e_null_at_the_pooled_size(self):
        rng = np.random.default_rng(6)
        cl, st = self._design()
        neg, pos = rng.standard_normal(len(cl)), rng.standard_normal(len(cl))
        r = inf.bootstrap_auroc(pos, neg, cl, st, n_boot=2000, seed=1)
        self.assertEqual(inf.reading(r["lo"], r["hi"]), "E-NULL")
        self.assertLess(r["hi"] - r["lo"], 0.14)

    def test_bootstrap_coverage_under_the_null(self):
        rng = np.random.default_rng(7)
        cl, st = self._design()
        cover = 0
        for _ in range(100):
            neg, pos = rng.standard_normal(len(cl)), rng.standard_normal(len(cl))
            r = inf.bootstrap_auroc(pos, neg, cl, st, n_boot=400, seed=int(rng.integers(1 << 30)))
            cover += r["lo"] <= 0.5 <= r["hi"]
        self.assertGreaterEqual(cover, 88)

    def test_mask_restricts_the_suspect_class_only(self):
        cl, st = self._design(n_groups=8, n_sets=2)
        pos = np.arange(len(cl), dtype=float)
        neg = np.full(len(cl), 10.5)
        mask = pos > 10
        r = inf.bootstrap_auroc(pos, neg, cl, st, n_boot=50, seed=1, pos_mask=mask)
        self.assertEqual(r["auroc"], 1.0)
        self.assertEqual(r["n_pos"], int(mask.sum()))
        self.assertEqual(r["n_neg"], len(cl))


class TestDonorsAndPipeline(unittest.TestCase):
    def test_recover_donors_and_reject_duplicates(self):
        rng = np.random.default_rng(8)
        z = rng.standard_normal((12, 8, 64)).astype(np.float32)
        d = inf.derangement(12)
        np.testing.assert_array_equal(inf.recover_donors(z, z[d]), d)
        z2 = z.copy()
        z2[3] = z2[5]
        with self.assertRaises(ValueError):
            inf.recover_donors(z2, z2[d])
        with self.assertRaises(ValueError):
            inf.recover_donors(z, z[d] + 1e-3)

    def test_derangement_has_no_fixed_point(self):
        for n in [16, 64, 96]:
            d = inf.derangement(n)
            self.assertEqual(sorted(d.tolist()), list(range(n)))
            self.assertTrue(np.all(d != np.arange(n)))

    def test_structural_null_on_a_toy_like_set(self):
        s, d = synthetic_set()
        desc = ra.descriptive_checks(s, d)
        self.assertEqual(desc["D1_max_abs_replay_minus_donor_benign"], 0.0)
        self.assertTrue(desc["D2_holds"])
        self.assertTrue(desc["D3_donor_equals_seed61001"])
        self.assertEqual(desc["harmful_replay_rows"], int(np.sum(s["labels"][d] != s["labels"])))
        benign, replay = ra.scores_for(s, "correct"), ra.scores_for(s, "donor_swap")
        for score in ["S_block", "S_zero"]:
            np.testing.assert_array_equal(replay[score], benign[score][d])
            self.assertEqual(inf.auroc(replay[score], benign[score]), 0.5)
        self.assertNotIn("S_rand", ra.scores_for(s, "random_0"))

    def test_gate_logic_on_a_toy_like_set(self):
        s, d = synthetic_set()
        rel = "synthetic"
        freeze = {"inputs": {rel + "/outputs.pt": "x"}}
        orig = ra.sha256
        try:
            ra.sha256 = lambda path: "x"
            s["summary"] = {"output_sha256": "x"}
            g, d2 = ra.gates("T", rel, s, freeze)
        finally:
            ra.sha256 = orig
        self.assertTrue(all(v for k, v in g.items() if k.startswith("G")))
        np.testing.assert_array_equal(d2, d)
        s["rec"]["correct"]["pred"] = (s["labels"] + 1) % 4
        try:
            ra.sha256 = lambda path: "x"
            g, _ = ra.gates("T", rel, s, freeze)
        finally:
            ra.sha256 = orig
        self.assertFalse(g["G4_benign_validity"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
