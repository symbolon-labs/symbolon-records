import unittest
import numpy as np
import runtime as r
class Qualification(unittest.TestCase):
 def setUp(self):self.rng=np.random.default_rng(1409)
 def test_dual_primal_and_torch(self):
  for n,d in [(11,5),(7,19)]:
   x=self.rng.normal(size=(n,d));y=self.rng.normal(size=(n,24));z=self.rng.normal(size=(3,d))
   score,p=r.ridge(x,y,z);xc=x-x.mean(0)
   w=np.linalg.solve(xc.T@xc+float(p['lam'])*np.eye(d),xc.T@(y-y.mean(0)))
   np.testing.assert_allclose(score,(z-x.mean(0))@w+y.mean(0),atol=1e-10,rtol=1e-10)
   ts,_=r.ridge(x,y,z,'torch');np.testing.assert_allclose(score,ts,atol=1e-10,rtol=1e-10)
 def test_heldout_excluded_and_fit_centered(self):
  x=self.rng.normal(size=(9,6));y=self.rng.normal(size=(9,24));fi=[0,2,4,6,8];va=[1,3,5,7]
  a,p=r.ridge(x[fi],y[fi],x[va]);x[va]+=100;y[va]+=1000
  b,q=r.ridge(x[fi],y[fi],x[va])
  for k in p:np.testing.assert_array_equal(p[k],q[k])
  np.testing.assert_array_equal(p['xmean'],x[fi].mean(0));self.assertFalse(np.array_equal(a,b))
 def test_shift_inside_fit_order(self):
  ids=['A0','C2','P7'];y=r.targets(ids);shift=np.roll(y,1,axis=0)
  np.testing.assert_array_equal(shift,r.targets(['P7','A0','C2']))
  self.assertEqual(r.prediction(shift),['P7','A0','C2'])
 def test_metrics_and_margins(self):
  z=r.targets(['A0','B1']);rr=r.records(['A0','C1'],z)
  self.assertEqual(r.counts(rr),dict(n=2,letter=1,digit=2,joint=1))
  self.assertEqual(rr[1]['gold_margins'],[-1.,1.])
 def test_exact_argmax_ties(self):
  z=np.zeros((1,24));self.assertEqual(r.prediction(z),['A0'])
  z[0,1]=1e-12;self.assertEqual(r.prediction(z),['B0'])
if __name__=='__main__':unittest.main()
