"""Unit tests of sebits.py. Synthetic data only, on the CPU; no CIFAR-10 file is read.

Run: ../../.venv/bin/python -I -B sebits.py test
"""
import io
import itertools
import math
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import sebits as S  # noqa: E402

torch.set_num_threads(2)


def tiny_data(mode="confirm", n_per_class=50, mem_n=10, seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(10), n_per_class)
    rng.shuffle(y)
    x = rng.integers(0, 256, size=(len(y), 3, 32, 32), dtype=np.uint8)
    xte = rng.integers(0, 256, size=(S.N_TEST, 3, 32, 32), dtype=np.uint8) if mode in ("confirm", "analysis") else None
    yte = rng.integers(0, 10, S.N_TEST) if mode == "analysis" else None
    return S.Data(x, y, xte, yte, mode, mem_n=mem_n)


class Base(unittest.TestCase):
    def setUp(self):
        self._dev = S.DEV
        S.DEV = torch.device("cpu")

    def tearDown(self):
        S.DEV = self._dev


class TestBits(Base):
    def test_idx_bits(self):
        self.assertEqual([S.idx_bits(k) for k in range(1, 6)], [4, 7, 10, 13, 15])

    def test_perm_rank_exhaustive(self):
        for k in range(1, 6):
            lists = torch.tensor(list(itertools.permutations(range(10), k)), dtype=torch.int64)
            r = S.perm_rank(lists)
            self.assertTrue(torch.equal(r, torch.arange(math.perm(10, k))))
            self.assertTrue(torch.equal(S.perm_unrank(r, k), lists))

    def test_bits_table(self):
        for b in S.BUDGETS:
            for (c, s, q) in S.RAW_CANDS[b]:
                self.assertEqual(c * s * s * q, b)
                self.assertEqual(S.RawSpec(c, s, q, np.arange(c), np.zeros((c, (1 << q) - 1)), np.zeros((c, 1 << q))).bits, b)
            for (c, s, q) in S.BOT_CANDS[b]:
                self.assertEqual(S.BotSpec(c, s, q).bits, b)
        want = {16: [16, 16, 16, 13], 64: [58, 61, 63], 256: [148], 1024: [148]}
        for b, cands in S.TOPK_CANDS.items():
            got = [S.TopkSpec(f, k, p, np.zeros((10 if f == "all" else k - 1, (1 << p) - 1)),
                              np.zeros((10 if f == "all" else k - 1, 1 << p))).bits for (f, k, p) in cands]
            self.assertEqual(got, want[b])
            self.assertTrue(all(g <= b for g in got))
        self.assertEqual(S.LabSpec(np.ones((10, 10)) / 10).bits, 4)
        self.assertEqual(S.FloatSpec().bits, 320)
        self.assertEqual(S.BotSpec(16, 1, 4, float_mode=True).bits, 16 * 32)
        self.assertEqual(S.RawSpec(64, 8, 8, np.arange(64), np.zeros((64, 255)), np.zeros((64, 256))).bits, 32768)

    def test_pack_roundtrip_every_code(self):
        g = torch.Generator().manual_seed(1)
        widths_list = []
        for b in S.BUDGETS:
            widths_list += [S.BotSpec(c, s, q).widths for (c, s, q) in S.RAW_CANDS[b] + S.BOT_CANDS[b]]
            for (f, k, p) in S.TOPK_CANDS[b]:
                widths_list.append(S.TopkSpec(f, k, p, np.zeros((10 if f == "all" else k - 1, 1)), np.zeros((10 if f == "all" else k - 1, 2))).widths)
        widths_list += [[4], [32] * 10, [8] * 4096, [32] * 16]
        for widths in widths_list:
            hi = torch.tensor([1 << w for w in widths], dtype=torch.int64)
            f = (torch.rand(7, len(widths), generator=g, dtype=torch.float64) * hi).floor().to(torch.int64)
            f[0] = hi - 1
            f[1] = 0
            bits = S.pack(f, widths)
            self.assertEqual(bits.shape[1], sum(widths))
            self.assertTrue(torch.equal(S.unpack(bits, widths), f))
            self.assertTrue(torch.equal(S.from_packed(S.to_packed(bits), sum(widths)), bits))

    def test_pack_range_and_length_checks(self):
        with self.assertRaises(S.CheckFail):
            S.pack(torch.tensor([[16]]), [4])
        with self.assertRaises(S.CheckFail):
            S.pack(torch.tensor([[-1]]), [4])
        with self.assertRaises(S.CheckFail):
            S.unpack(torch.zeros(1, 5, dtype=torch.uint8), [4])

    def test_pack_msb_first(self):
        self.assertEqual(S.pack(torch.tensor([[5, 1]]), [4, 2]).tolist(), [[0, 1, 0, 1, 0, 1]])


class TestQuant(Base):
    def test_fit_cells_equal_probability(self):
        v = np.random.default_rng(2).normal(size=20000)
        e, lv = S.fit_cells(v, 3)
        cell = np.searchsorted(e, v, side="left")
        cnt = np.bincount(cell, minlength=8)
        self.assertTrue(np.all(np.abs(cnt - 2500) <= 2))
        self.assertTrue(np.all(np.diff(lv) > 0))
        for j in range(8):
            self.assertAlmostEqual(lv[j], v[cell == j].mean(), places=5)

    def test_fit_cells_ties(self):
        rng = np.random.default_rng(3)
        v = np.concatenate([np.zeros(6000), rng.exponential(size=4000)])
        e, lv = S.fit_cells(v, 2)
        self.assertEqual(len(e), 3)
        self.assertTrue(np.all(np.isfinite(lv)))
        c0 = np.searchsorted(e, 0.0, side="left")
        self.assertEqual(lv[c0], 0.0)
        e, lv = S.fit_cells(np.array([]), 3)
        self.assertEqual((e.shape, lv.shape, float(np.abs(lv).sum())), ((7,), (8,), 0.0))

    def test_raw_spec_paths_agree(self):
        g = torch.Generator().manual_seed(4)
        h2 = F.relu(torch.randn(300, 64, 8, 8, generator=g))
        for (c, s, q) in [(16, 1, 4), (64, 2, 1), (16, 4, 4), (2, 1, 8), (64, 8, 8)]:
            chans = torch.randperm(64, generator=g)[:c].numpy()
            pooled = F.adaptive_avg_pool2d(h2[:, torch.from_numpy(chans)], s).numpy()
            cells = [S.fit_cells(pooled[:, j], q) for j in range(c)]
            spec = S.RawSpec(c, s, q, chans, np.stack([a for a, _ in cells]), np.stack([b for _, b in cells]))
            f = spec.encode(h2)
            self.assertEqual(f.shape, (300, c * s * s))
            bits = S.pack(f, spec.widths)
            self.assertEqual(bits.shape[1], spec.bits)
            v = spec.decode(S.unpack(bits, spec.widths))
            self.assertTrue(torch.equal(v, spec.quantize_dev(h2)))
            self.assertEqual(spec.dec_in(v).shape, (300, c) if s == 1 else (300, c, s, s))
            # channel-major field order: field j*s*s + r*s + col is channel j at (r, col)
            j, r, col = c - 1, s - 1, 0
            lv = spec.levels[j][f[:, j * s * s + r * s + col]]
            self.assertTrue(torch.equal(lv, v[:, j, r, col]))

    def test_bot_levels_and_ste(self):
        spec = S.BotSpec(2, 2, 2)
        t = torch.tanh(torch.randn(50, 2, 2, 2, generator=torch.Generator().manual_seed(5))).requires_grad_(True)
        f = spec.encode(t.detach())
        self.assertTrue(set(f.unique().tolist()) <= {0, 1, 2, 3})
        v = spec.decode(S.unpack(S.pack(f, spec.widths), spec.widths))
        self.assertTrue(torch.allclose(v, spec.ste(t).detach(), atol=1e-6))
        self.assertTrue(set(np.round(v.unique().double().numpy(), 5).tolist()) <= {-1.0, round(-1 / 3, 5), round(1 / 3, 5), 1.0})
        spec.ste(t).sum().backward()
        self.assertTrue(torch.allclose(t.grad, torch.ones_like(t)))

    def test_float_codes_exact(self):
        x = torch.randn(20, 16, 1, 1) * 1e3
        spec = S.BotSpec(16, 1, 4, float_mode=True)
        self.assertTrue(torch.equal(spec.decode(S.unpack(S.pack(spec.encode(x), spec.widths), spec.widths)), x))
        z = torch.randn(20, 10)
        fs = S.FloatSpec()
        self.assertTrue(torch.equal(fs.decode(S.unpack(S.pack(fs.encode(z), fs.widths), fs.widths)), z))
        self.assertEqual(fs.dec_in(z).shape, (20, 20))


class TestTopk(Base):
    def setUp(self):
        super().setUp()
        self.z_fit = torch.randn(5000, 10, generator=torch.Generator().manual_seed(6), dtype=torch.float64) * 3
        self.z = torch.randn(400, 10, generator=torch.Generator().manual_seed(7)) * 3

    def roundtrip(self, spec, z):
        f = spec.encode(z)
        bits = S.pack(f, spec.widths)
        self.assertEqual(bits.shape[1], spec.bits)
        return spec.decode(S.unpack(bits, spec.widths))

    def test_every_format_prediction_is_argmax(self):
        am = S.top1(self.z)
        for b, cands in S.TOPK_CANDS.items():
            for (form, k, p) in cands:
                spec = S.TopkSpec.fit(form, k, p, self.z_fit.numpy())
                if form == "top":
                    spec.gfill = 2.0 * spec.last_level()
                d = self.roundtrip(spec, self.z)
                self.assertTrue(torch.equal(spec.pred(d), am))
                self.assertTrue(torch.all(d[0].gather(1, am[:, None]) == 0))
                self.assertEqual(spec.dec_in(d).shape, (400, 20))

    def test_top_format_content(self):
        spec = S.TopkSpec.fit("top", 4, 16, self.z_fit.numpy())
        spec.gfill = 1.5 * spec.last_level()
        z, mask, first = self.roundtrip(spec, self.z)
        srt = torch.sort(self.z, dim=1, descending=True, stable=True)
        self.assertTrue(torch.equal(mask.sum(1), torch.full((400,), 4.0)))
        for j in range(1, 4):
            cls = srt.indices[:, j]
            gap = srt.values[:, 0] - srt.values[:, j]
            self.assertTrue(torch.all(mask.gather(1, cls[:, None]) == 1))
            err = (z.gather(1, cls[:, None])[:, 0] + gap).abs()
            self.assertLess(float(torch.quantile(err, 0.99)), 0.05)
        out = mask == 0
        self.assertTrue(torch.all(z[out] == -spec.gfill))

    def test_all_format_content(self):
        spec = S.TopkSpec.fit("all", 0, 16, self.z_fit.numpy())
        z, mask, first = self.roundtrip(spec, self.z)
        ref = self.z - self.z.max(1, keepdim=True).values
        self.assertLess(float(torch.quantile((z - ref).abs().flatten(), 0.99)), 0.05)
        self.assertTrue(torch.all(mask == 1))

    def test_lab_spec(self):
        pred = np.array([0, 0, 1, 2])
        y = np.array([0, 1, 1, 2])
        lab = S.LabSpec.fit(pred, y)
        self.assertTrue(np.allclose(lab.conf.sum(1).numpy(), 1.0))
        self.assertAlmostEqual(float(lab.conf[0, 0]), 2 / 12, places=6)
        self.assertAlmostEqual(float(lab.conf[3, 5]), 1 / 10, places=6)
        f = lab.encode(torch.tensor([[0.0, 5.0, 5.0, 1, 1, 1, 1, 1, 1, 1]]))
        self.assertEqual(f.tolist(), [[1]])
        d = lab.decode(S.unpack(S.pack(f, lab.widths), lab.widths))
        self.assertEqual(lab.pred(d).tolist(), [1])
        self.assertTrue(torch.allclose(lab.logits(d).exp(), lab.conf[1:2]))
        self.assertEqual(lab.dec_in(d).tolist()[0], [0.0, 1.0] + [0.0] * 8)

    def test_spec_state_roundtrip(self):
        spec = S.TopkSpec.fit("top", 3, 3, self.z_fit.numpy())
        spec.gfill = 2.5
        s2 = S.spec_from_state(spec.state())
        self.assertTrue(torch.equal(s2.encode(self.z), spec.encode(self.z)))
        self.assertEqual(s2.gfill, 2.5)


class TestMetrics(Base):
    def test_psnr(self):
        a = np.zeros((2, 3, 32, 32), np.uint8)
        b = a.copy()
        b[0] = 51  # error 0.2 -> MSE 0.04 -> 13.979 dB
        b[1] = 0
        p, mse = S.psnr_u8(b, a)
        self.assertAlmostEqual(p[0], 10 * math.log10(1 / 0.04), places=6)
        self.assertEqual(mse[1], 0.0)

    def test_ssim(self):
        rng = np.random.default_rng(8)
        x = rng.integers(0, 256, (3, 3, 32, 32), dtype=np.uint8)
        self.assertTrue(np.allclose(S.ssim_u8(x, x), 1.0, atol=1e-12))
        y = np.clip(x.astype(int) + rng.integers(-40, 40, x.shape), 0, 255).astype(np.uint8)
        got = S.ssim_u8(y, x)
        self.assertTrue(np.all(got < 1.0))
        g = np.exp(-((np.arange(11) - 5) ** 2) / (2 * 1.5 ** 2))
        w = np.outer(g, g) / np.outer(g, g).sum()
        X, Y = y[0].astype(float) / 255, x[0].astype(float) / 255
        vals = []
        for c in range(3):
            for i in range(22):
                for j in range(22):
                    a, b = X[c, i:i + 11, j:j + 11], Y[c, i:i + 11, j:j + 11]
                    ma, mb = (w * a).sum(), (w * b).sum()
                    va, vb = (w * a * a).sum() - ma ** 2, (w * b * b).sum() - mb ** 2
                    cab = (w * a * b).sum() - ma * mb
                    vals.append(((2 * ma * mb + 1e-4) * (2 * cab + 9e-4)) / ((ma ** 2 + mb ** 2 + 1e-4) * (va + vb + 9e-4)))
        self.assertAlmostEqual(got[0], float(np.mean(vals)), places=10)

    def test_auroc_brute_force(self):
        rng = np.random.default_rng(9)
        s = np.round(rng.normal(size=300), 1)
        lab = rng.integers(0, 2, 300)
        pos, neg = s[lab == 1], s[lab == 0]
        brute = ((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])).mean()
        self.assertAlmostEqual(S.auroc(s, lab), brute, places=12)

    def test_auroc_boot_equals_resample(self):
        rng = np.random.default_rng(10)
        s = np.round(rng.normal(size=200), 1)
        lab = rng.integers(0, 2, 200)
        got = S.auroc_boot(s, lab, draws=300)
        r2 = np.random.default_rng(S.BOOT_SEED)
        idx = r2.integers(0, 200, size=(S.BOOT_CHUNK, 200))
        for d in (0, 17, 249):
            self.assertAlmostEqual(got[d], S.auroc(s[idx[d]], lab[idx[d]]), places=12)

    def test_boot_means(self):
        d = np.random.default_rng(11).normal(size=500)
        m = S.boot_means(d, draws=600)
        r2 = np.random.default_rng(S.BOOT_SEED)
        ref = np.concatenate([d[r2.integers(0, 500, size=(min(S.BOOT_CHUNK, 600 - c), 500))].mean(1) for c in range(0, 600, S.BOOT_CHUNK)])
        self.assertTrue(np.array_equal(m, ref))
        self.assertTrue(np.array_equal(m, S.boot_means(d, draws=600)))

    def test_fit_T(self):
        g = torch.Generator().manual_seed(12)
        z = torch.randn(20000, 10, generator=g) * 3
        y = torch.multinomial(F.softmax(z / 1.5, 1), 1, generator=g)[:, 0].numpy()
        T, nll = S.fit_T(z, y)
        self.assertLess(abs(T - 1.5), 0.15)
        self.assertIn(T, S.TEMPS)
        self.assertEqual(len(S.TEMPS), 51)
        self.assertEqual((S.TEMPS[0], S.TEMPS[-1]), (0.5, 3.0))

    def test_top1_stable(self):
        self.assertEqual(S.top1(torch.tensor([[1.0, 3.0, 3.0, 0.0]])).tolist(), [1])


class TestRule(Base):
    def make(self, n=4000, seed=13):
        rng = np.random.default_rng(seed)
        c = (rng.random((3, n)) < 0.8).astype(float)
        p = rng.normal(15, 2, (3, n))
        return rng, c, p

    def test_cases(self):
        rng, c, p = self.make()
        self.assertEqual(S.compare(c, c, p, p + 1.0 + rng.normal(0, 0.3, p.shape))["case"], "S")
        self.assertEqual(S.compare(c, c, p, p + rng.normal(0, 0.3, p.shape))["case"], "E")
        self.assertEqual(S.compare(c, c, p, p - 1.0 + rng.normal(0, 0.3, p.shape))["case"], "X")
        c_act = c.copy()
        c_act[:, :800] = 1.0
        c_sym = c.copy()
        c_sym[:, :800] = 0.0
        r = S.compare(c_sym, c_act, p, p + 1.0)
        self.assertEqual((r["A"], r["L"], r["case"]), ("A-", "L+", "T"))
        self.assertAlmostEqual(r["rate_db_per_point"], 1.0 / abs(r["d_acc"]))
        self.assertEqual(S.compare(c_sym, c_act, p, p)["case"], "A")
        self.assertEqual(S.compare(c_sym, c_act, p, p - 1.0)["case"], "A")
        shift = np.array([[1.2], [1.2], [-0.4]])  # mean 0.67 dB, one seed negative -> L?
        r = S.compare(c, c, p, p + shift)
        self.assertEqual((r["A"], r["L"], r["case"]), ("A+", "L?", "U"))
        _, c1, p1 = self.make(n=1000)
        c_act, c_sym = np.ones_like(c1), np.ones_like(c1)
        c_sym[:, :50] = 0.0  # SYM wrong, ACT right on 50 images
        c_act[:, 50:95] = 0.0  # ACT wrong, SYM right on 45 images: mean -0.5 points, wide CI
        r = S.compare(c_sym, c_act, p1, p1 + 1.0)
        self.assertEqual((r["A"], r["case"]), ("A?", "U"))
        self.assertAlmostEqual(r["d_acc"], -0.5)

    def test_ssim_margin(self):
        rng, c, p = self.make()
        s = rng.uniform(0.2, 0.4, p.shape)
        self.assertEqual(S.compare(c, c, s, s + 0.05, leak_margin=0.02)["L"], "L+")
        self.assertEqual(S.compare(c, c, s, s + 0.001, leak_margin=0.02)["L"], "L=")

    def test_nll_beats(self):
        rng = np.random.default_rng(14)
        a = rng.normal(1.0, 0.3, (3, 3000))
        self.assertTrue(S.nll_beats(a + 0.05, a)["topk_beats"])
        self.assertFalse(S.nll_beats(a + 0.01, a)["topk_beats"])

    def test_select_cfg(self):
        rows = [{"name": "a", "dev_acc": 0.800, "dev_nll": 0.60, "order": 0},
                {"name": "b", "dev_acc": 0.7985, "dev_nll": 0.55, "order": 1},
                {"name": "c", "dev_acc": 0.790, "dev_nll": 0.40, "order": 2}]
        self.assertEqual(S.select_cfg(rows), "b")
        rows[1]["dev_acc"] = 0.7975
        self.assertEqual(S.select_cfg(rows), "a")


class TestData(Base):
    def test_make_splits(self):
        y = np.repeat(np.arange(10), 5000)
        np.random.default_rng(15).shuffle(y)
        sp = S.make_splits(y)
        self.assertEqual((len(sp["task"]), len(sp["dev"]), len(sp["aux"])), (35000, 5000, 10000))
        for k, n in (("task", 3500), ("dev", 500), ("aux", 1000)):
            self.assertEqual(np.bincount(y[sp[k]], minlength=10).tolist(), [n] * 10)
        allidx = np.concatenate([sp["task"], sp["dev"], sp["aux"]])
        self.assertEqual(len(np.unique(allidx)), 50000)
        self.assertEqual(len(sp["flip_test"]), 10000)
        self.assertTrue(0.45 < sp["flip_aux"].mean() < 0.55)
        self.assertEqual(len(np.intersect1d(sp["mem_in_train"], sp["mem_in_test"])), 0)
        self.assertEqual((len(sp["mem_in_train"]), len(sp["mem_out_train"]), len(sp["mem_out_test"])), (5000, 5000, 5000))
        self.assertTrue(sp["mem_out_test"].max() < 10000 and sp["mem_out_train"].max() < 10000 and sp["mem_in_test"].max() < 35000)
        sp2 = S.make_splits(y)
        self.assertTrue(all(np.array_equal(sp[k], sp2[k]) for k in sp))

    def test_read_archive_layout(self):
        recs = []
        for b in range(6):
            r = np.zeros((10000, S.REC), np.uint8)
            r[:, 0] = np.arange(10000) % 10
            r[0, 1:1025] = 1
            r[0, 1025:2049] = 2
            r[0, 2049:] = 3
            r[1, 1 + 5 * 32 + 7] = 200  # R plane, row 5, column 7
            r[2, 0] = b
            recs.append(r)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.tar.gz"
            with tarfile.open(p, "w:gz", compresslevel=1) as tf:
                for name, r in zip(S.MEMBERS, recs):
                    ti = tarfile.TarInfo(name)
                    raw = r.tobytes()
                    ti.size = len(raw)
                    tf.addfile(ti, io.BytesIO(raw))
            xtr, ytr, xte, yte = S.read_archive(p)
        self.assertEqual((xtr.shape, xte.shape, ytr.shape, yte.shape), ((50000, 3, 32, 32), (10000, 3, 32, 32), (50000,), (10000,)))
        self.assertEqual([int(xtr[0, c, 0, 0]) for c in range(3)], [1, 2, 3])
        self.assertEqual(int(xtr[1, 0, 5, 7]), 200)
        self.assertEqual(int(xtr[1].sum()), 200)
        self.assertEqual([int(ytr[2 + 10000 * b]) for b in range(5)], [0, 1, 2, 3, 4])
        self.assertEqual(int(yte[2]), 5)

    def test_split_records_rejects_bad_label(self):
        r = np.zeros((1, S.REC), np.uint8)
        r[0, 0] = 10
        with self.assertRaises(ValueError):
            S.split_records(r)

    def test_test_label_guard(self):
        d = tiny_data("confirm")
        with self.assertRaises(S.CheckFail):
            d.labels("test")
        d = tiny_data("train")
        with self.assertRaises(S.CheckFail):
            d.images("test")
        d = tiny_data("analysis")
        self.assertEqual(len(d.labels("test")), S.N_TEST)

    def test_scratch_mode_uses_dev(self):
        d = tiny_data("scratch")
        self.assertTrue(torch.equal(d.images("test"), d.images("dev")))
        self.assertTrue(np.array_equal(d.labels("test"), d.labels("dev")))

    def test_augment_is_a_crop_and_flip(self):
        d = tiny_data("train")
        xb = d.images("task")[:6]
        out = d.augment(xb, S.gen(3))
        g = S.gen(3)
        ox, oy = torch.randint(0, 9, (6,), generator=g), torch.randint(0, 9, (6,), generator=g)
        fl = torch.randint(0, 2, (6,), generator=g).bool()
        xp = F.pad(xb.float() / 255, (4, 4, 4, 4))
        for i in range(6):
            crop = xp[i, :, oy[i]:oy[i] + 32, ox[i]:ox[i] + 32]
            if fl[i]:
                crop = crop.flip(2)
            self.assertTrue(torch.allclose(out[i], (crop - d.mean[0]) / d.std[0], atol=1e-6))

    def test_normalization_on_task(self):
        d = tiny_data("train")
        x = d.images("task").float() / 255
        self.assertTrue(torch.allclose(d.mean.view(3), x.mean((0, 2, 3)), atol=1e-5))


class TestModels(Base):
    def test_param_counts(self):
        net = S.Net()

        def conv(m):
            return sum(p.numel() for n, p in m.named_parameters() if p.dim() == 4)

        self.assertEqual((conv(net.s1), conv(net.s2), conv(net.s3)), (10080, 55296, 221184))
        self.assertEqual(sum(p.numel() for p in net.head.parameters()), 1290)
        self.assertTrue(285000 < sum(p.numel() for p in net.parameters()) < 292000)
        h = S.TopkHead()
        self.assertEqual(sum(p.numel() for p in h.parameters()), 64 * 16 + 32 + 1024 * 256 + 256 + 2570)

    def test_macs(self):
        cm, sm = S.arm_macs("RAW@64", {})
        self.assertTrue(24.4e6 < cm < 24.6e6)
        self.assertTrue(14.1e6 < sm < 14.3e6)
        self.assertEqual(S.arm_macs("TOPK@64", {})[0] - cm, 330240)
        self.assertEqual(S.arm_macs("BOT@64", {"c": 64, "s": 1})[0] - cm, 262144)
        self.assertEqual(S.arm_macs("LAB", {})[1], 0)
        self.assertTrue(38.5e6 < S.arm_macs("TOPK_full", {})[0] < 38.8e6)

    def test_decoder_shapes(self):
        for shape in ((20,), (10,), (16, 2, 2), (16, 4, 4), (64, 8, 8)):
            for w in (1, 2):
                dec = S.Decoder(shape, w)
                out = dec(torch.randn((2,) + shape))
                self.assertEqual(out.shape, (2, 3, 32, 32))
                self.assertTrue(bool((out >= 0).all() and (out <= 1).all()))
        dec = S.Decoder((20,), 1)
        trunk = sum(p.numel() for n, p in dec.named_parameters() if not n.startswith("inp"))
        self.assertTrue(1.75e6 < trunk < 1.78e6)

    def test_raw_and_bot_models(self):
        net = S.Net().eval()
        h2 = F.relu(torch.randn(4, 64, 8, 8))
        for (c, s, q) in ((16, 1, 4), (64, 2, 1), (16, 4, 4)):
            spec = S.RawSpec(c, s, q, np.arange(c)[::-1].copy(), np.zeros((c, (1 << q) - 1)), np.ones((c, 1 << q)))
            m = S.RawModel(net, spec)
            self.assertEqual(m(spec.quantize_dev(h2)).shape, (4, 10))
        native = S.RawSpec(64, 8, 8, np.arange(64), np.zeros((64, 255)), np.zeros((64, 256)))
        self.assertTrue(torch.allclose(S.RawModel(net, native).eval()(h2), net.server(h2)))
        for (c, s, q) in ((16, 1, 4), (4, 2, 4), (16, 8, 1)):
            spec = S.BotSpec(c, s, q)
            m = S.BotModel(net, spec)
            out = m(h2)
            self.assertEqual(out.shape, (4, 10))
            out.sum().backward()
            self.assertGreater(float(m.enc.weight.grad.abs().sum()), 0.0)
            self.assertEqual(m.code(h2).shape, (4, c, s, s))

    def test_client_sha(self):
        net = S.Net()
        a = S.client_sha(net)
        self.assertEqual(a, S.client_sha(net))
        with torch.no_grad():
            net.s3[0].weight.add_(1.0)
        self.assertEqual(a, S.client_sha(net))
        with torch.no_grad():
            net.s2[0].weight.add_(1e-3)
        self.assertNotEqual(a, S.client_sha(net))


class TestPipeline(Base):
    def test_messages_decoder_and_probe(self):
        d = tiny_data("confirm", n_per_class=60, mem_n=10)
        torch.manual_seed(0)
        net = S.Net().eval()
        client = net.client
        task = d.images("task")
        h2 = torch.cat([client(d.norm(x)) for x in task.split(100)]).detach()
        pooled = F.adaptive_avg_pool2d(h2[:, :16], 1).numpy()
        cells = [S.fit_cells(pooled[:, j], 4) for j in range(16)]
        spec = S.RawSpec(16, 1, 4, np.arange(16), np.stack([a for a, _ in cells]), np.stack([b for _, b in cells]))
        arm = S.Arm("RAW@64", spec, lambda h: h, S.RawModel(net, spec).eval())
        with torch.no_grad():
            r = S.arm_messages(arm, client, d, d.images("aux"), bs=37)
        self.assertEqual(r["packed"].shape, (len(d.sp["aux"]), 8))
        self.assertTrue(torch.equal(S.dec_in_from_packed(spec, r["packed"]), r["dec_in"]))
        self.assertTrue(np.array_equal(r["pred"], S.top1(r["logits"]).numpy()))
        with torch.no_grad():
            rf = S.arm_messages(arm, client, d, d.images("aux"), flip=True)
        self.assertFalse(np.array_equal(rf["packed"], r["packed"]))
        Xtr = torch.randn(256, 20)
        Ttr = torch.randint(0, 256, (256, 3, 32, 32), dtype=torch.uint8)
        model, norm, info = S.train_decoder(Xtr, Ttr, Xtr[:40], Ttr[:40], 1, seed=1, epochs=2)
        self.assertEqual(info["best_epoch"], 2)
        rec = S.reconstruct(model, norm(Xtr[:5]))
        self.assertEqual((rec.shape, rec.dtype), ((5, 3, 32, 32), np.uint8))
        X = torch.randn(900, 6)
        y = (X[:, 0] > 0).long().numpy()
        lg, ep = S.train_probe(X[:600], y[:600], X[600:700], y[600:700], X[700:], 2, seed=2, epochs=30)
        self.assertEqual(lg.shape, (200, 2))
        self.assertGreater(float((lg.argmax(1).numpy() == y[700:]).mean()), 0.85)
        lg2, _ = S.train_probe(X[:600], y[:600], None, None, X[700:], 2, seed=2, epochs=3, select=False)
        self.assertEqual(lg2.shape, (200, 2))

    def test_project_and_plan(self):
        T = {"N_seconds": 600, "N_dev_acc": 0.85, "prep_seconds": 60, "topk_seconds": 60, "full_seconds": 30,
             "head": {"t_epochs": [10.0], "t_eval": 5.0},
             "arms": {"a": {"t_epochs": [12.0], "t_eval": 5.0}, "b": {"t_epochs": [15.0], "t_eval": 6.0}},
             "msgs": {"x": 10.0, "y": 30.0},
             "dec": {"S1_x": {"size": "S", "t_prep": 5, "t_train_epoch": [9.0], "t_eval": [1.0], "t_recon": 2.0},
                     "L1_x": {"size": "L", "t_prep": 5, "t_train_epoch": [36.0], "t_eval": [3.0], "t_recon": 6.0}},
             "analysis_smoke_seconds": 30}
        proj, plan, stops = S.project(T)
        self.assertEqual(stops, [])
        self.assertAlmostEqual(proj["unit_arm"], (15 * 15 + 6) / 60)
        self.assertAlmostEqual(proj["unit_decS"], (5 + 20 * 9 + 10 * 1 + 2) / 60)
        for job, p in plan.items():
            flat = [u for part in p["parts"] for u in part]
            self.assertEqual(len(flat), len(set(flat)))
            for pm, part in zip(p["part_minutes"], p["parts"]):
                self.assertTrue(pm <= S.JOB_LIMIT_MIN or len(part) == 1)
        self.assertEqual(len(plan["j2"]["parts"][0]) + sum(len(x) for x in plan["j2"]["parts"][1:]), 19)
        T["N_seconds"], T["N_dev_acc"] = 1300, 0.79
        T["dec"]["S1_x"]["t_train_epoch"] = [20.0]
        _, _, stops = S.project(T)
        self.assertEqual(len(stops), 3)

    def test_resolve_plan_names(self):
        class C:
            sel = {"RAW": {"16": "RAW_c16_s1_q1", "64": "RAW_c16_s1_q4", "256": "RAW_c64_s2_q1", "1024": "RAW_c16_s4_q4"},
                   "BOT": {"16": "BOT_c4_s2_q1", "64": "BOT_c16_s1_q4", "256": "BOT_c16_s2_q4", "1024": "BOT_c16_s8_q1"}}
        got = S.resolve_plan_names(C(), ["N", "RAW_sel0", "RAW_sel3", "BOT_sel1", "msgs"])
        self.assertEqual(got, ["N", "RAW_c16_s1_q1", "RAW_c64_s2_q1", "BOT_c16_s2_q4", "msgs"])  # sorted names
        self.assertEqual(S.parse_cfg("BOT_c16_s8_q1"), (16, 8, 1))

    def test_job_lists(self):
        self.assertEqual(len(S.J2_DECS), 18)
        self.assertEqual(len(S.J3_DECS), 9)
        self.assertEqual(len(S.J45_DECS), 15)
        self.assertEqual(S.RUN_SEEDS, (20261111, 20261112, 20261113))
        self.assertNotEqual(S.sub_seed(1, "a"), S.sub_seed(1, "b"))
        self.assertNotEqual(S.sub_seed(1, "a"), S.sub_seed(2, "a"))


class TestEndToEnd(Base):
    """J0 to J6 on tiny synthetic data in a temporary folder, on the CPU, with 1 epoch for each part.
    The freeze check and the nested test run of J0 are replaced; every other path is the real code."""

    def test_all_jobs(self):
        names = ("ROOT", "RES", "DATA_DIR", "N_TEST", "MEM_N", "N_EPOCHS", "ARM_EPOCHS", "DEC_EPOCHS", "PROBE_EPOCHS",
                 "V6_MIN", "check_freeze", "run_tests")
        saved = {n: getattr(S, n) for n in names}
        threads = torch.get_num_threads()
        torch.set_num_threads(4)

        class FakeResult:
            testsRun, failures, errors = 0, [], []

        with tempfile.TemporaryDirectory() as d:
            try:
                root = Path(d)
                S.ROOT, S.RES, S.DATA_DIR = root, root / "results", root / "data"
                S.N_TEST, S.MEM_N, S.V6_MIN = 60, 20, 0.0
                S.N_EPOCHS = S.ARM_EPOCHS = S.DEC_EPOCHS = S.PROBE_EPOCHS = 1
                S.check_freeze = lambda files: {}
                S.run_tests = lambda: FakeResult()
                rng = np.random.default_rng(16)
                y = np.repeat(np.arange(10), 50).astype(np.uint8)
                rng.shuffle(y)
                x = rng.integers(0, 256, (500, 3, 32, 32), dtype=np.uint8)
                x[:, 0] = (y[:, None, None] * 25).astype(np.uint8)  # a class signal in the R plane
                S.DATA_DIR.mkdir()
                np.savez(S.DATA_DIR / "cifar10_train.npz", x=x, y=y)
                np.savez(S.DATA_DIR / "cifar10_test_images.npz", x=rng.integers(0, 256, (60, 3, 32, 32), dtype=np.uint8))
                np.savez(S.DATA_DIR / "cifar10_test_labels.npz", y=rng.integers(0, 10, 60).astype(np.uint8))
                S.j0()
                self.assertFalse(S.read_json(root / "j0_result.json")["stop"])
                plan = S.read_json(root / "job_plan.json")
                S.run_job("j1", 1)
                for p in range(2, len(plan["j1"]["parts"]) + 1):
                    S.run_job("j1", p)
                S.run_job("j1", 0)  # every unit is done: all skip
                sel = S.read_json(root / "selection.json")
                self.assertEqual(sorted(sel["RAW"]), ["1024", "16", "256", "64"])
                S.run_job("j2", 0)
                S.run_job("j3", 0)
                for p in range(1, len(plan["j4"]["parts"]) + 1):
                    S.run_job("j4", p)
                S.run_job("j5", 0)
                S.j6()
                R = S.read_json(root / "results" / "results.json")
                self.assertIn(R["primary"]["case_final"], ("S", "E", "X", "T", "A", "U", "0"))
                self.assertIn("decoder_dependent", R["primary"])
                self.assertEqual(R["checks"]["V5"]["ok"], True)
                self.assertTrue(R["checks"]["V3"]["ok"] and R["checks"]["V4"]["ok"])
                self.assertEqual(len(R["secondary"]), 14)  # 1 + 2 x 3 + 4 + 2 + 1 (Part 7.7)
                self.assertEqual(sorted(R["attributes"]["arms"]), sorted(S.PROBE_ARMS))
                self.assertEqual(R["descriptive"]["TOPK@1024"]["acc"], R["descriptive"]["TOPK@256"]["acc"])
                self.assertEqual(R["descriptive"]["LAB"]["acc"], R["descriptive"]["TOPK@64"]["acc"])
                self.assertEqual(R["descriptive"]["RAW_native"]["bits"], 32768)
                self.assertTrue((root / "results" / "table.md").exists() and (root / "results" / "acc_psnr.svg").exists())
                dec = sorted(p.stem for p in (S.RES / "s20261111" / "dec").glob("*.npz"))
                self.assertEqual(len(dec), 18 + 9)
                self.assertEqual(len(list((S.RES / "s20261113" / "dec").glob("*.npz"))), 15)
                with self.assertRaises(S.CheckFail):
                    S.run_job("j2", 9)
            finally:
                for n, v in saved.items():
                    setattr(S, n, v)
                torch.set_num_threads(threads)


if __name__ == "__main__":
    unittest.main()
