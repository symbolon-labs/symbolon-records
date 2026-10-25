"""Unit tests of sleak.py. Synthetic data only, on the CPU; no CIFAR-10 file is read.

Run: ../../.venv/bin/python -I -B sleak.py test
"""
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import sleak as L  # noqa: E402

S = L.S
torch.set_num_threads(2)


class Base(unittest.TestCase):
    def setUp(self):
        self._dev = S.DEV
        S.DEV = torch.device("cpu")

    def tearDown(self):
        S.DEV = self._dev


class TestPlan(Base):
    def test_seeds(self):
        self.assertGreaterEqual(len(L.NEW_SEEDS), 5)
        self.assertEqual(len(set(L.NEW_SEEDS)), len(L.NEW_SEEDS))
        used = set(S.RUN_SEEDS) | {S.SPLIT_SEED, S.SCRATCH_SEED, S.BOOT_SEED}
        self.assertFalse(used & (set(L.NEW_SEEDS) | {L.BOOT2_SEED}))
        self.assertEqual(L.SEALED_SEEDS, S.RUN_SEEDS)

    def test_waiter_runs_the_seeds_in_order(self):
        txt = (L.ROOT / "wait_and_run.sh").read_text()
        m = re.search(r"^SEEDS=\(([0-9 ]+)\)", txt, re.M)
        self.assertIsNotNone(m)
        self.assertEqual(tuple(int(v) for v in m.group(1).split()), L.NEW_SEEDS)

    def test_copies_and_selection(self):
        for f, want in L.SEALED_SHA.items():
            self.assertEqual(S.sha256_file(L.ROOT / f), want)
        sel = S.read_json(L.ROOT / "selection.json")
        self.assertEqual(L.bot_star(sel), "BOT_c16_s2_q1")
        self.assertEqual(sel["TOPK"]["64"], ["all", 0, 6, 0.0])
        self.assertEqual(S.DATA_DIR, L.SEALED / "data")
        self.assertEqual(Path(S.__file__).resolve().parent, L.ROOT)

    def test_planned_units(self):
        sel = S.read_json(L.ROOT / "selection.json")
        first = L.planned_units(L.NEW_SEEDS[0], sel)
        other = L.planned_units(L.NEW_SEEDS[1], sel)
        self.assertEqual(first[:5], ["N", "head", "topk", "BOT_c16_s2_q1", "msgs"])
        self.assertEqual(first[5:], ["L1_TOPK@64", "L1_BOT@64", "L1_LAB", "S1_TOPK@64", "S1_BOT@64", "S1_LAB",
                                     "L1_BOT@64_ctrl", "L1_TOPK@64_ctrl"])
        self.assertEqual(other, first[:-2])


class TestStats(Base):
    def naive(self, D, seed, draws):
        K, n = D.shape
        rng = np.random.default_rng(seed)
        out = []
        for c in range(0, draws, L.BOOT2_CHUNK):
            m = min(L.BOOT2_CHUNK, draws - c)
            si = rng.integers(0, K, size=(m, K))
            ii = rng.integers(0, n, size=(m, n))
            for b in range(m):
                out.append(D[si[b]][:, ii[b]].mean())
        return np.array(out)

    def test_two_level_equals_naive(self):
        rng = np.random.default_rng(3)
        D = rng.normal(size=(4, 37)) + np.arange(4)[:, None]
        got = L.two_level(D, seed=11, draws=600)
        self.assertEqual(got.shape, (600,))
        np.testing.assert_allclose(got, self.naive(D, 11, 600), rtol=0, atol=1e-12)
        np.testing.assert_allclose(L.two_level(D, seed=11, draws=600), got, rtol=0, atol=0)

    def test_two_level_constant_and_seed_spread(self):
        np.testing.assert_allclose(L.two_level(np.full((3, 20), 0.7), draws=300), 0.7, atol=1e-12)
        D = np.repeat(np.array([[0.0], [1.0], [2.0], [3.0], [4.0]]), 50, axis=1)  # no image noise, seed spread only
        m = L.two_level(D, draws=4000)
        self.assertAlmostEqual(m.mean(), 2.0, delta=0.05)
        self.assertAlmostEqual(m.std(), np.sqrt(2.0 / 5.0), delta=0.03)  # population var 2 over 5 seeds

    def test_t_interval(self):
        lo, hi = L.t_interval([0.512, 0.489, 0.647])
        sd = np.std([0.512, 0.489, 0.647], ddof=1)
        self.assertAlmostEqual(hi - lo, 2 * 4.302653 * sd / np.sqrt(3), places=9)

    def test_reading(self):
        self.assertEqual(L.reading(0.55, (0.40, 0.70)), "R1")
        self.assertEqual(L.reading(0.50, (0.01, 0.70)), "R1")
        self.assertEqual(L.reading(0.4999, (0.30, 0.70)), "R2")
        self.assertEqual(L.reading(0.20, (0.0, 0.40)), "R3")
        self.assertEqual(L.reading(0.60, (-0.01, 1.2)), "R3")
        self.assertEqual(L.reading(-0.20, (-0.40, -0.01)), "R4")
        self.assertEqual(L.reading(-0.20, (-0.40, 0.0)), "R3")

    def test_rows(self):
        rng = np.random.default_rng(5)
        D = 0.6 + 0.1 * rng.normal(size=(5, 200))
        r = L.leak_row(D)
        self.assertAlmostEqual(r["d_leak"], D.mean(), places=12)
        self.assertEqual(r["reading"], "R1")
        self.assertEqual(len(r["d_leak_seeds"]), 5)
        c1 = (rng.random((5, 200)) < 0.8).astype(float)
        c0 = (rng.random((5, 200)) < 0.7).astype(float)
        a = L.acc_row(c1, c0)
        self.assertAlmostEqual(a["d_acc"], 100 * (c1 - c0).mean(), places=10)
        self.assertEqual(a["outcome"], "A+ replicates")


class TestChecks(Base):
    """collect_checks on synthetic checks.jsonl files, done markers and decoder outputs."""

    def build(self, res, sel):
        seeds = L.NEW_SEEDS[:2]
        for s in seeds:
            out = res / f"s{s}"
            for sub in ("done", "dec"):
                (out / sub).mkdir(parents=True, exist_ok=True)
            for u in L.planned_units(s, sel):
                S.write_json(out / "done" / f"{u}.json", {"unit": u})
            for d in L.decs_for(s):
                np.savez(out / "dec" / f"{S.dec_tag(*d)}.npz", recon_test=np.zeros(1))
            recs = [{"check": "V6", "ok": True}, {"check": "V3", "where": "head start", "ok": True}]
            recs += [{"check": "V4", "where": f"msgs {a}", "ok": True} for a in L.MSG_NAMES]
            recs += [{"check": "V1", "where": w, "ok": True} for w in ("S1_LAB", "L1_LAB")]
            recs += [{"check": "V2", "where": S.dec_tag(*d), "ok": True} for d in L.decs_for(s) if d[3]]
            (out / "checks.jsonl").write_text("".join(__import__("json").dumps(r) + "\n" for r in recs))
        return seeds

    def edit(self, res, seed, fn):
        p = res / f"s{seed}" / "checks.jsonl"
        lines = p.read_text().splitlines()
        p.write_text("".join(x + "\n" for x in fn(lines)))

    def test_collect_checks(self):
        sel = S.read_json(L.ROOT / "selection.json")
        saved = S.RES
        with tempfile.TemporaryDirectory() as d:
            try:
                S.RES = res = Path(d)
                seeds = self.build(res, sel)
                recs, ok, failed = L.collect_checks(seeds, sel)
                self.assertTrue(all(ok.values()), ok)
                self.assertEqual(failed, [])
                self.assertEqual(len(recs["V2"]), 2)
                # a failed V3 record that a rerun replaces: the last record counts
                self.edit(res, seeds[1], lambda x: x + ['{"check": "V3", "where": "msgs start", "ok": false}',
                                                         '{"check": "V3", "where": "msgs start", "ok": true}'])
                recs, ok, failed = L.collect_checks(seeds, sel)
                self.assertTrue(ok["V3"])
                self.assertEqual(len(failed), 1)
                # a failed last record
                self.edit(res, seeds[1], lambda x: x + ['{"check": "V3", "where": "msgs start", "ok": false}'])
                self.assertFalse(L.collect_checks(seeds, sel)[1]["V3"])
                self.edit(res, seeds[1], lambda x: x[:-1])
                # an absent V1 record of Decoder-L
                self.edit(res, seeds[1], lambda x: [v for v in x if "L1_LAB" not in v])
                ok = L.collect_checks(seeds, sel)[1]
                self.assertFalse(ok["V1"])
                self.assertTrue(ok["V2"] and ok["V3"] and ok["V5"])
                # an absent V2 record
                self.edit(res, seeds[0], lambda x: [v for v in x if "L1_TOPK@64_ctrl" not in v])
                self.assertFalse(L.collect_checks(seeds, sel)[1]["V2"])
                # an absent V6 record and an absent done marker
                self.edit(res, seeds[0], lambda x: [v for v in x if '"V6"' not in v])
                (res / f"s{seeds[0]}" / "done" / "msgs.json").unlink()
                recs, ok, _ = L.collect_checks(seeds, sel)
                self.assertFalse(ok["V6"])
                self.assertFalse(ok["V5"])
                self.assertEqual(recs["V5"][0]["missing"], ["msgs"])
            finally:
                S.RES = saved


class TestShell(unittest.TestCase):
    def test_wrapper_and_waiter(self):
        """test_run_sleak.zsh: exit codes after a pause, the start wait, the pause, time and battery limits, the lock,
        and the wait of the waiter for DONE and for a process of the folder."""
        import subprocess
        r = subprocess.run(["zsh", str(L.ROOT / "test_run_sleak.zsh")], capture_output=True, text=True, timeout=600)
        print(r.stdout)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(r.stdout.count("\nok   ") + r.stdout.startswith("ok   "), 16, r.stdout)


class TestEndToEnd(Base):
    """Two seed jobs and the analysis on tiny synthetic data in a temporary folder, on the CPU, with 1 epoch for each
    part. The freeze check is replaced; every other path is the real code of sleak.py and sebits.py."""

    def test_seed_jobs_and_analysis(self):
        sl = ("NEW_SEEDS", "SEALED", "check_freeze")
        ss = ("ROOT", "RES", "DATA_DIR", "N_TEST", "MEM_N", "N_EPOCHS", "ARM_EPOCHS", "DEC_EPOCHS", "PROBE_EPOCHS",
              "V6_MIN")
        saved_l = {n: getattr(L, n) for n in sl}
        saved_s = {n: getattr(S, n) for n in ss}
        threads = torch.get_num_threads()
        torch.set_num_threads(4)
        with tempfile.TemporaryDirectory() as d:
            try:
                root = Path(d)
                S.ROOT, S.RES, S.DATA_DIR = root, root / "results", root / "data"
                S.N_TEST, S.MEM_N, S.V6_MIN = 60, 20, 0.0
                S.N_EPOCHS = S.ARM_EPOCHS = S.DEC_EPOCHS = S.PROBE_EPOCHS = 1
                L.NEW_SEEDS = (20261121, 20261122)
                L.SEALED = root / "sealed"
                L.check_freeze = lambda files: {}
                shutil.copy(saved_l["SEALED"] / "selection.json", root / "selection.json")
                rng = np.random.default_rng(16)
                y = np.repeat(np.arange(10), 50).astype(np.uint8)
                rng.shuffle(y)
                x = rng.integers(0, 256, (500, 3, 32, 32), dtype=np.uint8)
                x[:, 0] = (y[:, None, None] * 25).astype(np.uint8)  # a class signal in the R plane
                xte = rng.integers(0, 256, (60, 3, 32, 32), dtype=np.uint8)
                S.DATA_DIR.mkdir()
                np.savez(S.DATA_DIR / "cifar10_train.npz", x=x, y=y)
                np.savez(S.DATA_DIR / "cifar10_test_images.npz", x=xte)
                np.savez(S.DATA_DIR / "cifar10_test_labels.npz", y=rng.integers(0, 10, 60).astype(np.uint8))
                for s in L.SEALED_SEEDS:  # stand-ins for the sealed reconstructions (read only by the analysis)
                    p = L.SEALED / "results" / f"s{s}" / "dec"
                    p.mkdir(parents=True)
                    for tag in ("S1_TOPK@64", "S1_BOT@64", "L1_TOPK@64", "L1_BOT@64"):
                        np.savez(p / f"{tag}.npz", recon_test=rng.integers(0, 256, (60, 3, 32, 32), dtype=np.uint8))
                L.run_seed(20261121)
                L.run_seed(20261121)  # every unit is done: all skip
                L.run_seed(20261122)
                with self.assertRaises(S.CheckFail):
                    L.run_seed(20261111)
                sel = S.read_json(root / "selection.json")
                for s in L.NEW_SEEDS:
                    done = sorted(p.stem for p in (S.RES / f"s{s}" / "done").glob("*.json"))
                    self.assertEqual(done, sorted(L.planned_units(s, sel)))
                    self.assertEqual(sorted(p.stem for p in (S.RES / f"s{s}" / "msgs").glob("*.npz")), sorted(L.MSG_NAMES))
                v1 = [r for r in map(__import__("json").loads, (S.RES / "s20261121" / "checks.jsonl").read_text().splitlines())
                      if r["check"] == "V1"]
                self.assertEqual(sorted(r["where"] for r in v1), ["L1_LAB", "S1_LAB"])
                l1 = [r for r in v1 if r["where"] == "L1_LAB"][0]
                s1 = [r for r in v1 if r["where"] == "S1_LAB"][0]
                self.assertAlmostEqual(l1["closed"], s1["closed"], places=9)  # the same closed form as sebits
                R = L.analysis()
                self.assertEqual(R["checks"]["V5"]["ok"], True)
                self.assertTrue(R["checks"]["V3"]["ok"] and R["checks"]["V4"]["ok"] and R["checks"]["V6"]["ok"])
                self.assertEqual(R["checks"]["V2"]["records"], 2)
                self.assertIn(R["primary"]["reading_final"], ("R1", "R2", "R3", "R4", "void"))
                self.assertEqual(len(R["primary"]["d_leak_seeds"]), 2)
                self.assertEqual(R["primary"]["bits"], {"TOPK@64": 58, "BOT@64": 64})
                self.assertEqual(len(R["secondary"]["pooled_S_sealed_and_new"]["d_leak_seeds"]), 5)
                self.assertIn("sealed_seed1_L", R["secondary"])
                self.assertIn(R["secondary"]["sealed_rule_L"]["L"], ("L+", "L=", "L-", "L?"))
                yte = np.load(S.DATA_DIR / "cifar10_test_labels.npz")["y"]
                c = [(np.load(S.RES / f"s{s}" / "msgs" / "TOPK@64.npz")["pred_test"] == yte).mean() for s in L.NEW_SEEDS]
                self.assertAlmostEqual(R["descriptive"]["TOPK@64"]["acc"], float(np.mean(c)), places=12)
                self.assertTrue((S.RES / "table.md").exists() and (S.RES / "results.json").exists())
                (S.RES / "s20261122" / "dec" / "L1_BOT@64.npz").unlink()
                R = L.analysis()
                self.assertEqual(R["primary"]["reading"], "void")
                self.assertEqual(R["checks"]["V5"]["ok"], False)
            finally:
                for n, v in saved_l.items():
                    setattr(L, n, v)
                for n, v in saved_s.items():
                    setattr(S, n, v)
                torch.set_num_threads(threads)


if __name__ == "__main__":
    unittest.main()
