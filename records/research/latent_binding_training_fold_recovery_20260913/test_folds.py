import unittest
import torch
import common as h
torch.set_num_threads(4)
from seam import replace
class Tests(unittest.TestCase):
 def test_replay_rejects_optimizer_mutation(self):
  a={'state':{0:{'exp_avg':torch.zeros(2),'step':torch.tensor(50.)}},'history':[1.0]}
  import copy
  z=copy.deepcopy(a);h.exact_tree(a,z);z['state'][0]['exp_avg'][0]=1
  with self.assertRaises(AssertionError):h.exact_tree(a,z)
 def test_matched_parity_qualification(self):
  d=h.load(h.DIAGNOSTIC/'run/batch8.pt');single=h.load(h.DIAGNOSTIC/'run/batch1.pt')
  self.assertLessEqual(float((d['fast']['lp']-d['direct']).abs().max()),1e-4)
  self.assertGreater(float((d['fast']['lp'][:1]-single['direct']).abs().max()),1e-4)
  self.assertTrue(torch.equal(d['fast']['prediction'],d['direct'].argmax(-1)))
 def test_partial_batch_gradient(self):
  torch.manual_seed(13);a=torch.randn(13,2,7,dtype=torch.float64,requires_grad=True);t=torch.randint(0,7,(13,2))
  whole=h.batch_loss(a,t,13);g=torch.autograd.grad(whole,a)[0]
  batched=sum(h.batch_loss(a[i:i+8],t[i:i+8],13) for i in range(0,13,8));g2=torch.autograd.grad(batched,a)[0]
  self.assertTrue(torch.allclose(g,g2,atol=1e-15,rtol=1e-15));self.assertAlmostEqual(float(whole.detach()),float(batched.detach()),places=14)
 def test_fold_exclusion_and_coordinates(self):
  state=h.load(h.CANDIDATE);x=h.load(h.ROOT/'features.pt')['projected'];data=h.read(h.ROOT/'inputs.json');folds=h.read(h.ROOT/'folds.json')['folds'];saved=h.load(h.ROOT/'coordinates.pt')
  self.assertEqual(sorted(i for f in folds for i in f['validation_rows']),list(range(224)))
  for f,co in zip(folds,saved):
   fi=[data['ids'].index(n) for n in f['fit_ids']];vi=[data['ids'].index(n) for n in f['validation_ids']]
   z=x.clone();z[vi]=1e6;other=h.coordinates(state,z[fi]);again=h.coordinates(state,x[fi])
   for k in ['input_basis','output_basis','base']:self.assertTrue(torch.equal(other[k],again[k]))
   self.assertEqual(co['rank'],len(fi));self.assertTrue(torch.equal(h.weight(torch.zeros(64,len(fi),dtype=torch.float64),co),state[h.NAME]))
   self.assertAlmostEqual(float((x[fi].double()@co['input_basis']).square().sum(-1).mean()),1,places=10)
   self.assertFalse(set(f['fit_ids'])&set(f['validation_ids']))
 def test_selection_tie_and_both_metrics(self):
  totals={0:dict(joint=10,sequential=10),50:dict(joint=11,sequential=12),100:dict(joint=20,sequential=12),200:dict(joint=20,sequential=11)}
  self.assertEqual(h.select(totals)['selected_step'],50);self.assertTrue(h.select(totals)['qualifies'])
  totals[50]['joint']=9;self.assertFalse(h.select(totals)['qualifies'])
 def test_export_and_support(self):
  x=torch.randn(3,6,960);z=torch.randn(3,2,960,requires_grad=True);y=replace(x,z);self.assertTrue(torch.equal(y[:,:3],x[:,:3]));self.assertTrue(torch.equal(y[:,5:],x[:,5:]));y.sum().backward();self.assertTrue(torch.equal(z.grad,torch.ones_like(z)))
  co=h.load(h.ROOT/'coordinates.pt')[0];t=torch.zeros(64,co['rank'],dtype=torch.float64,requires_grad=True);w=h.weight(t,co);w.square().sum().backward();self.assertTrue(torch.isfinite(t.grad).all());self.assertGreater(float(t.grad.norm()),0);self.assertEqual(w.dtype,torch.float32)
 def test_non_target_mutation_rejected(self):
  state=h.load(h.CANDIDATE);new={k:v.clone() for k,v in state.items()};new[h.NAME][0,0]+=1;h.fixed(new,state);new['reader.gate'].add_(1)
  with self.assertRaises(AssertionError):h.fixed(new,state)
if __name__=='__main__':unittest.main()
