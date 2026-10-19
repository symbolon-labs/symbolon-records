import unittest,numpy as np,torch
from analyze import scores
class Mechanics(unittest.TestCase):
 def test_full_vocabulary_gold_scores(self):
  a=torch.tensor([[[1.,2.,3.],[4.,1.,0.]],[[0.,0.,0.],[-1.,2.,4.]]]);t=torch.tensor([[2,0],[1,1]]);g,p,m,e=scores(a,t)
  np.testing.assert_array_equal(p,[[2,0],[0,2]]);np.testing.assert_array_equal(m,[[1,3],[0,-2]]);self.assertLessEqual(e,1e-8)
 def test_seam_only_two_sites(self):
  import importlib.util
  from pathlib import Path
  spec=importlib.util.spec_from_file_location('test_existing_seam',Path(__file__).resolve().parent.parent/'latent_binding_training_fold_recovery_20260913/seam.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
  x=torch.randn(3,6,960);p=torch.randn(3,2,960);z=s.replace(x,p);self.assertTrue(torch.equal(z[:,:3],x[:,:3]));self.assertTrue(torch.equal(z[:,5:],x[:,5:]));self.assertTrue(torch.equal(z[:,3:5],p))
if __name__=='__main__':unittest.main()
