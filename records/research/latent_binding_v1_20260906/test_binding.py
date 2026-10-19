import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import AutoTokenizer
import binding as b

torch.set_num_threads(4)


class BindingTests(unittest.TestCase):
    def test_architecture_is_exact_parent(self):
        self.assertIs(b.BridgeModel,b.cf.BridgeModel)
        torch.manual_seed(1); bridge=b.BridgeModel()
        self.assertEqual(sum(p.numel() for p in bridge.parameters()),569857)
        z=bridge.resampler(torch.randn(8,19,2048))
        self.assertEqual(z.shape,(8,8,64))
        self.assertTrue(torch.allclose(z.norm(dim=-1),torch.full((8,8),8.)))
        h=torch.randn(8,5,960)
        self.assertTrue(torch.equal(h,bridge.reader(h,bridge.projector(torch.zeros_like(z)))))

    def test_pools_and_tiny_only(self):
        pools=b.pools(); self.assertFalse(set(pools['known']) & set(pools['novel']))
        self.assertEqual(set(pools['known'])|set(pools['novel']),set(b.IDS))
        for pool in pools.values():
            for letter in 'ABCDEFGHIJKLMNOP': self.assertEqual(sum(x[0]==letter for x in pool),4)
            for digit in '01234567': self.assertEqual(sum(x[1]==digit for x in pool),8)
        data=b.dataset('tiny')
        self.assertEqual(len(data['records']),8); self.assertEqual(len(data['rows']),32)
        for rec in data['records']:
            self.assertEqual(set(rec['bindings'].values()),set(rec['wrong_bindings'].values()))
            self.assertTrue(all(rec['bindings'][c]!=rec['wrong_bindings'][c] for c in b.COLORS))
        with tempfile.TemporaryDirectory() as tmp,patch.object(b,'ROOT',Path(tmp)):
            for split in ['train','dev','final_known','final_novel']:
                with self.assertRaises((AssertionError,FileNotFoundError)): b.dataset(split)

    def test_tokenization(self):
        data=b.dataset('tiny')
        for role in ['sender','receiver']:
            tok=AutoTokenizer.from_pretrained(b.c.model_path(role),local_files_only=True)
            bank=b.AnswerBank(tok)
            q=bank.prompts([r['query'] for r in data['rows']]); self.assertEqual(q.shape,(32,5))
            sources=[r[k] for r in data['records'] for k in ['source','wrong_source']]
            self.assertEqual({len(tok.encode(s,add_special_tokens=False)) for s in sources},{19})
            chatbank=b.AnswerBank(tok,leading=False)
            text=tok.apply_chat_template([{'role':'user','content':'Copy the answer ID. Reply with exactly the ID and nothing else.\n\nAnswer ID: A0'}],
                tokenize=False,add_generation_prompt=True,enable_thinking=False,date_string='06 Sep 2026')
            chatbank.prompts([text])

    def test_full_id_scoring_with_native_shaped_fake(self):
        class Bank:
            tokens=torch.tensor([[i//8,i%8+16] for i in range(128)])
            prefixes=list(range(16))
            indices={p:torch.arange(p*8,(p+1)*8) for p in range(16)}
        def forward(ids,memory=None,keep=1):
            # Causal deterministic position-specific logits with prefix dependence.
            out=torch.zeros(len(ids),ids.shape[1],32)
            for pos in range(ids.shape[1]):
                out[:,pos]=torch.arange(32)[None,:]*.01*(1+ids[:,:pos+1].sum(1)[:,None])
            return out[:,-keep:]
        ids=torch.tensor([[1,2,3],[2,3,4]])
        fast=b.score(forward,ids,Bank())
        slow=b.direct_score(forward,ids,Bank())
        self.assertTrue(torch.equal(fast['lp'],slow))
        self.assertTrue(torch.equal(fast['lp'],fast['token_lp'].sum(-1)))


if __name__=='__main__': unittest.main()
