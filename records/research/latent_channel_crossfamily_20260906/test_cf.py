import json
import unittest
import torch
from transformers import AutoTokenizer

import channel_cf as cf

torch.set_num_threads(4)


class Contracts(unittest.TestCase):
    def test_bridge_unchanged_bottleneck_and_zero(self):
        torch.manual_seed(1)
        b = cf.BridgeModel()
        self.assertEqual(sum(p.numel() for p in b.parameters()), 918017)
        h = torch.randn(4, 22, 2048)
        z = b.resampler(h)
        self.assertEqual(tuple(z.shape), (4, 8, 64))
        self.assertTrue(torch.allclose(z.norm(dim=-1), torch.full((4,8),8.0)))
        native = torch.randn(4,8,2048)
        self.assertTrue(torch.equal(native, b.reader(native,b.projector(torch.zeros_like(z)))))
        b.reader(native,b.projector(z)).square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in b.parameters()))

    def test_old_and_new_splits_disjoint(self):
        used = set()
        for s in ['tiny','train','dev','final']:
            used.update(r['nonce'] for r in json.loads((cf.V2/'data'/s/'examples.json').read_text()))
        used.update(r['nonce'] for r in json.loads((cf.BASE.parent/'data/tiny/examples.json').read_text()))
        for split in ['tiny','train','dev']:
            rows = cf.dataset(split); nonces = {r['nonce'] for r in rows}
            self.assertFalse(used & nonces); used |= nonces
            self.assertEqual({r['query'] for r in rows}, {cf.c.QUERY})
            self.assertEqual(len({r['id'] for r in rows}),len(rows))
            for nonce in nonces:
                self.assertEqual(sorted(r['label'] for r in rows if r['nonce']==nonce),[0,1,2,3])
        self.assertFalse((cf.ROOT/'data/final').exists())
        if not (cf.ROOT/'run_full/FINAL_FREEZE.json').exists():
            with self.assertRaises(AssertionError): cf.dataset('final')

    def test_tokenization_and_text_date(self):
        from text_cf import render, text
        rows = cf.dataset('tiny')
        for role in ['sender','receiver']:
            tok = AutoTokenizer.from_pretrained(cf.c.model_path(role),local_files_only=True)
            lengths = set()
            for row in rows:
                if role=='receiver':
                    for prompt in [cf.c.QUERY,row['source']+'\n'+cf.c.QUERY]:
                        for answer in cf.c.ANSWERS:
                            tail=tok.encode(answer,add_special_tokens=False)
                            self.assertEqual(len(tail),1)
                            self.assertEqual(tok.encode(prompt+answer,add_special_tokens=False),tok.encode(prompt,add_special_tokens=False)+tail)
                lengths.add(len(tok.encode(row['source'],add_special_tokens=False)))
            self.assertEqual(len(lengths),1)
            prompt=render(tok,text.RECEIVER_INSTRUCTION+'Signal: A')
            for answer in text.LETTERS:
                self.assertEqual(tok.encode(prompt+answer,add_special_tokens=False),tok.encode(prompt,add_special_tokens=False)+tok.encode(answer,add_special_tokens=False))
            if role=='receiver': self.assertIn('06 Sep 2026',prompt)


if __name__=='__main__': unittest.main()
