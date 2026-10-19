import unittest
import torch
from seam import replace,Boundary
class Tests(unittest.TestCase):
    def test_support_both_lengths(self):
        for length in [5,6]:
            h=torch.arange(2*length*960).reshape(2,length,960).float();p=torch.ones(2,2,960)
            x=replace(h,p);self.assertTrue(torch.equal(x[:,:3],h[:,:3]));self.assertTrue(torch.equal(x[:,5:],h[:,5:]));self.assertTrue(torch.equal(x[:,3:5],p));self.assertTrue(torch.equal(replace(h,h[:,3:5]),h))
    def test_wrong_shape_rejected(self):
        with self.assertRaises(AssertionError):replace(torch.zeros(1,52,960),torch.zeros(1,2,960))
    def test_reset_on_exception(self):
        wrapper=object.__new__(Boundary);wrapper.payload=None
        def fail(**kwargs):raise ValueError('synthetic')
        wrapper.model=fail
        with self.assertRaises(ValueError):wrapper.forward(torch.ones(1,5,dtype=torch.long),torch.zeros(1,2,960))
        self.assertIsNone(wrapper.payload)
if __name__=='__main__':unittest.main()

class GradientTests(unittest.TestCase):
    def test_only_replaced_sites_send_gradient_to_payload(self):
        hidden=torch.zeros(2,6,960,requires_grad=True)
        payload=torch.zeros(2,2,960,requires_grad=True)
        replace(hidden,payload).sum().backward()
        self.assertTrue(torch.equal(payload.grad,torch.ones_like(payload)))
        self.assertEqual(hidden.grad[:,3:5].abs().sum().item(),0)
        self.assertTrue(torch.equal(hidden.grad[:,:3],torch.ones_like(hidden.grad[:,:3])))
        self.assertTrue(torch.equal(hidden.grad[:,5:],torch.ones_like(hidden.grad[:,5:])))
