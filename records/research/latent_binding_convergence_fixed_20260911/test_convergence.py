import unittest
import torch
import common as h
from seam import replace


def sealed():
    fit=torch.load(h.ALIGN/'run/fit.pt',weights_only=True)
    return fit['theta'],load_coord(),torch.load(h.ALIGN/'run/candidate.pt',weights_only=True),fit['optimizer']


def load_coord():return torch.load(h.ALIGN/'coordinates.pt',weights_only=True)


class Tests(unittest.TestCase):
    def test_start_reproduces_sealed_candidate(self):
        theta,coord,state,_=sealed()
        self.assertEqual(tuple(theta.shape),(64,56))
        self.assertTrue(torch.equal(h.weight(theta,coord),state[h.NAME]))
        self.assertEqual(len(state),11)

    def test_start_step_is_the_frozen_budget(self):
        _,_,_,optimizer=sealed()
        moments=optimizer['state'][0]
        self.assertEqual(int(moments['step']),h.STEPS)
        self.assertEqual(tuple(moments['exp_avg'].shape),(64,56))
        self.assertEqual(tuple(moments['exp_avg_sq'].shape),(64,56))
        self.assertTrue(bool(torch.isfinite(moments['exp_avg']).all()))
        self.assertTrue(bool(torch.isfinite(moments['exp_avg_sq']).all()))
        self.assertTrue(bool((moments['exp_avg_sq']>0).all()))

    def test_optimizer_state_roundtrip(self):
        theta,_,_,optimizer=sealed()
        parameter=torch.nn.Parameter(theta.clone());opt=torch.optim.Adam([parameter],lr=h.LR)
        opt.load_state_dict(optimizer)
        self.assertEqual(int(opt.state[parameter]['step']),h.STEPS)
        self.assertEqual(opt.param_groups[0]['lr'],h.LR)
        self.assertEqual(h.EXTRA+h.STEPS,h.TOTAL)

    def test_budget_arithmetic(self):
        self.assertEqual(h.STEPS,200)
        self.assertEqual(h.EXTRA,400)
        self.assertEqual(h.TOTAL,600)
        self.assertEqual(h.CAP,14400)

    def test_replace_only_query_sites(self):
        x=torch.randn(3,6,960);z=torch.randn(3,2,960,requires_grad=True);out=replace(x,z)
        self.assertTrue(torch.equal(out[:,:3],x[:,:3]));self.assertTrue(torch.equal(out[:,5:],x[:,5:]))
        out.sum().backward();self.assertTrue(torch.equal(z.grad,torch.ones_like(z)))

    def test_only_one_entry_may_change(self):
        _,_,state,_=sealed();new={k:v.clone() for k,v in state.items()};new[h.NAME][0,0]+=1;h.fixed(new,state)
        new['reader.gate'].add_(1)
        with self.assertRaises(AssertionError):h.fixed(new,state)

    def test_parent_inputs_are_the_continuation_data(self):
        x=h.read(h.ALIGN/'inputs.json')
        self.assertEqual(len(x['training_ids']),56);self.assertEqual(len(x['trainrows']),224)
        self.assertEqual(sorted(x['banks']),['familiar','novel'])
        self.assertEqual(len(x['banks']['novel']['targets']['original']),32)
        self.assertTrue(all(row['feature']==i//4 for i,row in enumerate(x['trainrows'])))

    def test_scalar_counts_match_library(self):
        import analysis
        x=h.read(h.ALIGN/'inputs.json');info=x['banks']['familiar']
        for arm in ['correct','wrong_binding']:
            out=torch.load(h.reference_scores('familiar')/f'{arm}.pt',weights_only=True)['first']
            for condition in ['original','changed']:
                self.assertEqual(analysis.scalar_counts(out,info['targets'][condition],x['candidate_tokens']),
                    h.counts(out,info['targets'][condition],torch.tensor(x['candidate_tokens'])))


if __name__=='__main__':
    unittest.main()
