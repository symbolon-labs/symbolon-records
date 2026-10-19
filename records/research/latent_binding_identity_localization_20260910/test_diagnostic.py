import unittest
import torch
from diagnostic import metric,reference_indices

class Tests(unittest.TestCase):
    def test_affine_nearest_reference(self):
        x=torch.tensor([[1.,2.],[3.,-1.],[0.,0.]],dtype=torch.float64)
        c=torch.tensor([[1.,1.],[-1.,0.]],dtype=torch.float64)
        self.assertTrue(torch.equal(metric(x,c).argmin(-1),(2*x@c.T-(c*c).sum(-1)).argmax(-1)))
    def test_exact_collision(self):
        x=torch.tensor([[1.,2.],[1.,2.],[2.,3.]])
        self.assertEqual(float(metric(x,x)[0,1]),0)
        self.assertGreater(float(metric(x,x)[0,2]),0)
    def test_reference_identity_not_row_order(self):
        self.assertEqual(reference_indices(['b','a','b','c','a']),[1,0,3])
    def test_shifted_codebook_control(self):
        c=torch.eye(8);pred=metric(c,c).argmin(-1)
        self.assertEqual(int((pred==torch.arange(8)).sum()),8)
        self.assertEqual(int(((pred+1)%8==torch.arange(8)).sum()),0)
if __name__=='__main__':unittest.main()
