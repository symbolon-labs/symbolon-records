"""EXPLORATORY. Synthetic tests for t5.py. No receiver, no sender, no sealed model."""
import sys
import unittest
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import torch.nn.functional as F
import t5


def fake_state(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"reader.output.weight": torch.randn(960, 64, generator=g) * 0.01,
            "reader.gate": torch.tensor(0.7), "reader.norm_scale": torch.tensor(1.3),
            "reader.query_center": torch.randn(2, 960, generator=g)}


class FakeForward:
    def __init__(self, vocab=20, seed=1):
        g = torch.Generator().manual_seed(seed)
        self.emb = torch.randn(vocab, 16, generator=g)
        self.proj = torch.randn(960, 16, generator=g) * 0.05
        self.out = torch.randn(16, vocab, generator=g)

    def __call__(self, seq, payload, keep):
        h = self.emb[seq]
        if payload is not None:
            h = h + (payload.mean(1) @ self.proj)[:, None, :]
        return (h @ self.out)[:, -keep:]


class Tests(unittest.TestCase):
    def test_additive_init_recovers_additive_codes(self):
        g = torch.Generator().manual_seed(2)
        a, b, c = torch.randn(16, 64, generator=g, dtype=torch.float64), torch.randn(8, 64, generator=g, dtype=torch.float64), torch.randn(64, generator=g, dtype=torch.float64)
        names = [l + d for l in "ABEG" for d in "0236"][:-1]  # one in-block combination left out
        codes = torch.stack([c + a[ord(n[0]) - 65] + b[int(n[1])] for n in names])
        c0, a0, b0 = t5.additive_init(codes, names)
        for n in names + ["G6"]:
            l, d = t5.parts(n)
            self.assertTrue(torch.allclose(c0 + a0[l] + b0[d], c + a[l] + b[d], atol=1e-9))

    def test_values_are_additive(self):
        g = torch.Generator().manual_seed(3)
        m = t5.Factored(torch.randn(64, dtype=torch.float64), torch.randn(16, 64, dtype=torch.float64), torch.randn(8, 64, dtype=torch.float64),
                        torch.rand(64, dtype=torch.float64) + 0.5, torch.linalg.qr(torch.randn(64, 64, dtype=torch.float64))[0])
        with torch.no_grad():
            for p in m.parameters():
                p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float64))
        L, D = torch.tensor([0, 5, 0, 5]), torch.tensor([1, 1, 7, 7])
        v = m.values(L, D)
        self.assertTrue(torch.allclose(v[0] - v[1], v[2] - v[3], atol=1e-9))

    def test_render_shape_and_zero(self):
        st = fake_state()
        render = t5.make_render(st)
        out = render(torch.zeros(3, 64, dtype=torch.float64))
        self.assertEqual(tuple(out.shape), (3, 2, 960))
        self.assertTrue(torch.equal(out, st["reader.query_center"][None].expand(3, 2, 960)))

    def test_fit_loss_is_mean_token_ce_and_moves_only_theta(self):
        st = fake_state()
        render = t5.make_render(st)
        fwd = FakeForward()
        g = torch.Generator().manual_seed(4)
        n = 19
        q = torch.randint(0, 20, (n, 5), generator=g)
        tgt = torch.randint(0, 20, (n, 2), generator=g)
        L, D = torch.randint(0, 16, (n,), generator=g), torch.randint(0, 8, (n,), generator=g)
        m = t5.Factored(torch.zeros(64), torch.zeros(16, 64), torch.zeros(8, 64), torch.ones(64), torch.eye(64))
        with torch.no_grad():
            logits = fwd(torch.cat([q, tgt[:, :1]], 1), render(m.values(L, D)), 2)
            expected = float(F.cross_entropy(logits.reshape(-1, 20), tgt.reshape(-1)))
        buffers = {k: v.clone() for k, v in m.named_buffers()}
        hist = t5.fit(fwd, render, m, q, tgt, L, D, 2, 0.1, "test")
        self.assertAlmostEqual(hist[0]["loss"], expected, places=6)
        self.assertTrue(all(torch.equal(v, dict(m.named_buffers())[k]) for k, v in buffers.items()))
        self.assertGreater(float(m.theta_a.abs().sum()), 0)
        self.assertLess(hist[1]["loss"], hist[0]["loss"])

    def test_counts_joint_and_sequential(self):
        ids = [chr(65 + i // 8) + str(i % 8) for i in range(128)]
        tokens = torch.tensor([[100 + i // 8, 200 + i % 8] for i in range(128)])
        token_lp = torch.full((2, 128, 2), -10.0)
        token_lp[0, ids.index("C4"), :] = 0.0
        token_lp[0, [ids.index("C" + str(d)) for d in range(8)], 0] = -0.1
        token_lp[0, ids.index("C4"), 0] = -0.1
        token_lp[1, ids.index("G3"), :] = 0.0
        lp = token_lp.sum(-1)
        out = {"token_lp": token_lp, "lp": lp, "prediction": lp.argmax(-1)}
        c = t5.counts(out, ["C4", "G2"], tokens, ids)
        self.assertEqual((c["joint"], c["sequential"], c["rows"]), (1, 1, 2))

    def test_shift_map_is_cyclic_with_no_fixed_point(self):
        names = ["B3", "A6", "A3", "C1"]
        m = t5.shift_map(names)
        self.assertEqual(m, {"A3": "A6", "A6": "B3", "B3": "C1", "C1": "A3"})


if __name__ == "__main__":
    unittest.main()
