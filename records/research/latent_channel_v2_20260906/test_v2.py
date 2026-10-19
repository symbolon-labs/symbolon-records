import math
import unittest
import torch
from channel_v2 import BridgeModel, bounded_attention, dataset, c


class Contracts(unittest.TestCase):
    def test_bounded_attention_does_not_close(self):
        torch.manual_seed(1)
        q = (torch.randn(3, 8, 64)*1e12).requires_grad_()
        k = (torch.randn(3, 22, 64)*1e12).requires_grad_()
        a = bounded_attention(q, k)
        self.assertGreaterEqual(a.min().item(), math.exp(-4)/22 - 1e-7)
        self.assertTrue(torch.allclose(a.sum(-1), torch.ones(3, 8)))
        a[..., -1].sum().backward()
        self.assertGreater(q.grad.norm().item(), 0)
        self.assertGreater(k.grad.norm().item(), 0)

    def test_zero_gradient_and_parameter_contract(self):
        torch.manual_seed(2)
        b = BridgeModel()
        h = torch.randn(4, 22, 2048)
        native = torch.randn(4, 8, 1024)
        z = b.resampler(h)
        self.assertTrue(torch.allclose(z.norm(dim=-1), torch.full((4, 8), 8.0)))
        self.assertTrue(torch.equal(native, b.reader(native, b.projector(torch.zeros_like(z)))))
        b.reader(native, b.projector(z)).square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in b.parameters()))
        self.assertEqual(sum(p.numel() for p in b.parameters()), 590337)

    def test_data_no_leakage(self):
        seen = set()
        for split in ["tiny", "train", "dev"]:
            rows = dataset(split)
            nonces = {r["nonce"] for r in rows}
            self.assertFalse(seen & nonces)
            seen |= nonces
            self.assertEqual({r["query"] for r in rows}, {c.QUERY})
            for nonce in nonces:
                self.assertEqual(sorted(r["label"] for r in rows if r["nonce"]==nonce), [0,1,2,3])


if __name__ == "__main__":
    unittest.main()
