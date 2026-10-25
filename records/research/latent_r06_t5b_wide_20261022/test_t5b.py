"""EXPLORATORY. Synthetic tests for t5b.py (PLAN.md section B, 4.4). No receiver, no sealed model."""
import sys
import unittest
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import t5
import t5b


def fake_state(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"reader.output.weight": torch.randn(960, 64, generator=g) * 0.01,
            "reader.gate": torch.tensor(0.7), "reader.norm_scale": torch.tensor(1.3),
            "reader.query_center": torch.randn(2, 960, generator=g)}


def basis(state):
    A = (state["reader.gate"] * state["reader.norm_scale"]).double() * state["reader.output.weight"].double()
    U, _, _ = torch.linalg.svd(A, full_matrices=True)
    return U[:, :64], U[:, 64:].contiguous()


def wide(seed=1):
    st = fake_state()
    u, Q = basis(st)
    g = torch.Generator().manual_seed(seed)
    m = t5b.Wide(torch.randn(64, generator=g, dtype=torch.float64), torch.randn(16, 64, generator=g, dtype=torch.float64),
                 torch.randn(8, 64, generator=g, dtype=torch.float64), torch.rand(64, generator=g, dtype=torch.float64) + 0.5,
                 torch.linalg.qr(torch.randn(64, 64, generator=g, dtype=torch.float64))[0], Q)
    return st, u, Q, m


class FakeForward:
    def __init__(self, vocab=20, seed=1):
        g = torch.Generator().manual_seed(seed)
        self.emb = torch.randn(vocab, 16, generator=g)
        self.proj = torch.randn(960, 16, generator=g) * 0.05
        self.out = torch.randn(16, vocab, generator=g)

    def __call__(self, seq, payload, keep):
        h = self.emb[seq] + (payload.mean(1) @ self.proj)[:, None, :]
        return (h @ self.out)[:, -keep:]


class Tests(unittest.TestCase):
    def test_offset_lies_in_complement(self):
        st, u, Q, m = wide()
        with torch.no_grad():
            for p in [m.psi_c, m.psi_a, m.psi_b]:
                p.copy_(torch.randn(p.shape, dtype=torch.float64))
        _, off = m.values(torch.tensor([0, 3, 15]), torch.tensor([1, 2, 7]))
        self.assertLess(float((u.T @ off.T).abs().max()), 1e-10)

    def test_zero_psi_gives_factored_value_and_zero_offset(self):
        st, u, Q, m = wide()
        fac = t5.Factored(m.c0, m.a0, m.b0, m.s, m.vh)
        L, D = torch.tensor([0, 5, 9]), torch.tensor([1, 4, 6])
        v, off = m.values(L, D)
        self.assertTrue(torch.equal(v, fac.values(L, D)))
        self.assertTrue(torch.equal(off, torch.zeros(3, 960, dtype=torch.float64)))

    def test_values_are_additive(self):
        st, u, Q, m = wide()
        with torch.no_grad():
            for p in m.parameters():
                p.copy_(torch.randn(p.shape, dtype=torch.float64))
        L, D = torch.tensor([0, 5, 0, 5]), torch.tensor([1, 1, 7, 7])
        v, off = m.values(L, D)
        self.assertTrue(torch.allclose(v[0] - v[1], v[2] - v[3], atol=1e-9))
        self.assertTrue(torch.allclose(off[0] - off[1], off[2] - off[3], atol=1e-9))

    def test_render_wide_with_zero_offset_equals_render(self):
        st, u, Q, m = wide()
        L, D = torch.tensor([0, 5, 9]), torch.tensor([1, 4, 6])
        a = t5b.make_render_wide(st)(m.values(L, D))
        b = t5.make_render(st)(t5.Factored(m.c0, m.a0, m.b0, m.s, m.vh).values(L, D))
        self.assertTrue(torch.equal(a, b))

    def test_fit_moves_psi_and_lr0_moves_nothing(self):
        st, u, Q, m = wide()
        render = t5b.make_render_wide(st)
        g = torch.Generator().manual_seed(4)
        q = torch.randint(0, 20, (11, 5), generator=g)
        tgt = torch.randint(0, 20, (11, 2), generator=g)
        L, D = torch.randint(0, 16, (11,), generator=g), torch.randint(0, 8, (11,), generator=g)
        t5.fit(FakeForward(), render, m, q, tgt, L, D, 1, 0.0, "lr0")
        self.assertEqual(float(m.psi_a.detach().abs().max()), 0.0)
        t5.fit(FakeForward(), render, m, q, tgt, L, D, 2, 0.1, "move")
        self.assertGreater(float(m.psi_a.detach().abs().max()), 0.0)
        self.assertGreater(float(m.theta_a.detach().abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
