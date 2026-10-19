"""Synthetic boundary and implementation checks, no fresh bank or endpoint."""
import ast
import copy
import inspect
import unittest
from pathlib import Path
import numpy as np
import torch
import core as c
import audit as a
import stats
import scalar
class Tests(unittest.TestCase):
    def test_query_join_reorders_and_rejects_absent(self):
        old=torch.tensor([[1,2],[3,4],[1,2]])
        self.assertEqual(c.query_join(old[[1,0]],old).tolist(),[1,0])
        with self.assertRaises(AssertionError):c.query_join(torch.tensor([[8,9]]),old)
    def test_exact_platform_allowance(self):
        import provenance as v
        expected={'parent':{'platform':v.OLD,'framework':'fixed'},'other':42}
        actual={'parent':{'platform':v.NEW,'framework':'fixed'},'other':42}
        v.compare(actual,expected,['parent','platform'])
        for bad in [{**actual,'other':43},{'parent':{'platform':v.NEW,'framework':'changed'},'other':42},expected]:
            with self.assertRaises(AssertionError):v.compare(bad,expected,['parent','platform'])
    def test_import_identity(self):
        import runtime as rt
        self.assertEqual(Path(rt.__file__).resolve(),c.ROOT/'runtime.py');self.assertIs(rt.c,c)
        self.assertIs(c.evaluation.c.b,c.b);self.assertIs(c.evaluation.c.guard,c.s.guard);self.assertIs(c.b,c.n.r.b)
        self.assertIs(c.bridge,c.p.bridge);self.assertIs(c.fixed,c.p.fixed)
    def test_historical_last_token_shape(self):
        ref=torch.arange(21,dtype=torch.float32).reshape(3,7);raw=ref[:,None,:].clone()
        self.assertTrue(torch.equal(c.last_logits(raw,ref),ref))
        for bad in [raw[:,0],raw.expand(3,2,7),raw[:2]]:
            with self.assertRaises(AssertionError):c.last_logits(bad,ref)
    def test_gate_boundaries(self):
        counts=dict(receiver_only=4,zero=4,correct=31,shuffled=8,wrong_binding=8,random_71001=8,random_71002=8,canonical_only=4)
        paired={part:{name:{metric:dict(ci95=[.1,.2]) for metric in ['accuracy','margin']} for name in names}
            for part,names in [('original',['zero','shuffled','wrong_binding','random_71001','random_71002']),('changed',['zero','correct','random_71001','random_71002'])]}
        def compare(values,donor,changed):
            metrics={k:dict(accuracy=v/32,donor_following=None) for k,v in values.items()};metrics['shuffled']['donor_following']=donor/32;metrics['wrong_binding']['donor_following']=changed/32
            got=c.gate(metrics,paired);ind=a.decision(values,donor,changed,paired)
            self.assertEqual(got['native_view_pass'],ind['native_view_pass']);self.assertEqual(got['verdict'],ind['verdict'])
            self.assertFalse(got['text_measured']);self.assertFalse(got['novel_binding_established']);return got['native_view_pass']
        self.assertTrue(compare(counts,31,28));self.assertFalse(compare(counts,30,28));self.assertFalse(compare(counts,31,27))
        for name,value in [('receiver_only',5),('zero',3),('correct',30),('shuffled',9),('wrong_binding',9),('random_71001',9),('random_71002',9),('canonical_only',5)]:self.assertFalse(compare({**counts,name:value},31,28))
        paired['changed']['random_71002']['margin']['ci95'][0]=0;self.assertFalse(compare(counts,31,28))
    def test_dev_bootstrap_independent_and_exact_code(self):
        rows=[dict(group=i//4) for i in range(32)];x=np.arange(32,dtype=float)%7-3
        self.assertEqual(stats.boot(x,rows),scalar.interval(x,rows));self.assertEqual(stats.boot(x,rows)['seed'],95002)
        def function(path,name):return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
        for name in ['boot','vectors','summary']:
            old=function(c.ROOT.parent/'latent_binding_query_only_value_patch_20260908/core.py',name)
            for node in ast.walk(old):
                if isinstance(node,ast.Constant) and node.value==113001:node.value=95002
            self.assertEqual(ast.dump(old),ast.dump(function(c.ROOT/'stats.py',name)))
    def test_burn_before_forward_and_no_training(self):
        tree=ast.parse((c.ROOT/'runtime.py').read_text());text=(c.ROOT/'runtime.py').read_text()
        self.assertLess(text.index("b.write(ROOT/'NOVEL_LOCK.json'"),text.index("model,tok=b.c.load('receiver')"))
        for node in ast.walk(tree):
            if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Attribute):continue
            self.assertNotIn(node.func.attr,{'backward','step','optim','train','fit','generate'})
            if isinstance(node.func.value,ast.Name) and node.func.value.id=='b' and node.func.attr=='event':
                inspect.signature(c.b.event).bind(*[None for _ in node.args],**{k.arg:None for k in node.keywords if k.arg is not None})
if __name__=='__main__':unittest.main()
