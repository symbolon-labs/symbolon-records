import ast
import copy
import inspect
import unittest
import torch
import core as c
import audit as a
class Tests(unittest.TestCase):
    def test_requests_and_prompt_exclusion(self):
        rows=c.requests(c.b.read(c.PARENT/'run/data.json'));self.assertEqual(len(rows),64)
        for row in rows:
            other={**row,'answer':'SECRET_LABEL','example_id':'SECRET_ID','group':999,'row':999}
            self.assertEqual(c.sender_body(row),c.sender_body(other));self.assertNotIn('SECRET',c.sender_body(other))
            self.assertEqual(c.b.parse_source(row['source'])[row['color']],row['answer'])
    def test_actual_message_union_not_oracle_replacement(self):
        rows=c.requests(c.b.read(c.PARENT/'run/data.json'));pred=[0]*64
        self.assertIn('A0',c.message_ids(rows,pred));self.assertEqual(c.message_ids(rows,pred),sorted({'A0'}|{r['answer'] for r in rows}))
        with self.assertRaises(AssertionError):c.message_ids(rows,[-1]*64)
    def test_receiver_has_only_message(self):
        body=c.receiver_body('D4');self.assertTrue(body.endswith('Answer ID: D4'))
        self.assertNotIn('Which ID',body);self.assertNotIn(' = ',body)
    def test_full_two_token_scalar_score(self):
        z=torch.zeros(2,128,2);z[:,0]=torch.tensor([10.,-20.]);z[:,1]=torch.tensor([2.,2.]);lp=z.sum(-1);out=dict(token_lp=z,lp=lp,prediction=lp.argmax(-1))
        self.assertEqual(a.scores(out)[1].tolist(),[1,1])
        with self.assertRaises(AssertionError):a.scores({**out,'prediction':z[:,:,0].argmax(-1)})
    def test_event_signatures_and_no_training(self):
        tree=ast.parse((c.ROOT/'runtime.py').read_text())
        for n in ast.walk(tree):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):
                self.assertNotIn(n.func.attr,{'backward','step','generate','train'})
                if isinstance(n.func.value,ast.Name) and n.func.value.id=='b' and n.func.attr=='event':inspect.signature(c.b.event).bind(*[None for _ in n.args],**{k.arg:None for k in n.keywords if k.arg})
    def test_existing_runtime_identity(self):
        self.assertIs(c.r.b,c.b);self.assertEqual(c.CAPS,{'sender':None,'receiver':None})
        self.assertEqual(list(inspect.signature(c.b.native_forward).parameters),['model','ids','memory','keep'])
    def test_inference_functions_unchanged(self):
        old=ast.parse((c.PREVIOUS/'runtime.py').read_text());new=ast.parse((c.ROOT/'runtime.py').read_text())
        for name in ['parity','body','run','successful','prepare_receiver']:
            left=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
            right=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(left),ast.dump(right),name)
    def test_no_wall_time_kill(self):
        self.assertTrue(all(v is None for v in c.CAPS.values()))
        source=(c.ROOT/'job.py').read_text()
        self.assertIn('assert cap is None',source);self.assertIn('proc.wait(timeout=cap)',source)
if __name__=='__main__':unittest.main()
