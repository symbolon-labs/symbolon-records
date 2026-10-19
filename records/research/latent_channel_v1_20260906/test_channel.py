import unittest
import torch
from channel import BridgeModel, dataset, derangement, QUERY


class Contracts(unittest.TestCase):
    def test_counterfactuals_and_split_disjointness(self):
        used = set()
        for split in ["tiny", "train", "dev"]:  # Do not materialize reserved final.
            rows = dataset(split)
            nonces = {r["nonce"] for r in rows}
            self.assertFalse(used & nonces)
            used |= nonces
            for nonce in nonces:
                self.assertEqual(sorted(r["label"] for r in rows if r["nonce"] == nonce), list(range(4)))
            self.assertEqual({r["query"] for r in rows}, {QUERY})

    def test_zero_and_scale_and_gradients(self):
        torch.manual_seed(5)
        b = BridgeModel()
        h = torch.randn(4, 23, 2048)
        z = b.resampler(h)
        self.assertEqual(z.shape, (4, 8, 64))
        self.assertTrue(torch.allclose(z.norm(dim=-1), torch.full((4, 8), 8.0)))
        receiver = torch.randn(4, 8, 1024)
        self.assertTrue(torch.equal(receiver, b.reader(receiver, b.projector(torch.zeros_like(z)))))
        b.reader(receiver, b.projector(z)).square().mean().backward()
        self.assertTrue(all(p.grad is not None and p.grad.norm() > 0 for p in b.parameters()))
        self.assertLess(sum(p.numel() for p in b.parameters()), 1_000_000)

    def test_shuffle(self):
        d = derangement(32)
        self.assertEqual(sorted(d.tolist()), list(range(32)))
        self.assertTrue(all(i != v for i, v in enumerate(d)))


if __name__ == "__main__":
    unittest.main()
