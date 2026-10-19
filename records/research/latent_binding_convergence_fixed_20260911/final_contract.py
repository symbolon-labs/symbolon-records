"""Closing component audit on saved states; no receiver model, fit or new scores."""
import argparse
import torch
import common as h
from common import ROOT,load,read,write


def audit():
    h.check();status=read(ROOT/'run/status.json');assert status['completed']
    state=load(h.ALIGN/'run/candidate.pt');candidate=load(ROOT/'run/candidate.pt');h.fixed(candidate,state)
    bridge=h.c.bridge();bridge.load_state_dict(candidate);bridge.requires_grad_(False);h.c.s.guard(bridge)
    results={}
    for pool in ['familiar']+(['novel'] if status['novel_evaluated'] else []):
        results[pool]={}
        for arm in ['correct','wrong_binding']:
            five=load(ROOT/'run'/f'trained_{pool}'/f'prefill_{arm}.pt')['first']['capture']
            six=load(h.native(pool)/'run'/f'six_{arm}.pt')['first']['capture'];outs=[];sixouts=[]
            with torch.no_grad():
                for start in range(0,32,8):
                    out=bridge.reader(five['hidden'][start:start+8],five['memory'][start:start+8]);outs.append(out.clone())
                    z=bridge.reader(six['hidden'][start:start+8],six['memory'][start:start+8]);sixouts.append(z.clone());cap=bridge.reader.capture
                    assert torch.equal(z[:,5:],six['hidden'][start:start+8,5:]) and not cap['used_delta'][:,5:].any()
                    assert torch.equal(cap['hard'],six['hard'][start:start+8]) and torch.equal(cap['scores'],six['scores'][start:start+8])
            assert torch.equal(torch.cat(outs),five['output']);assert torch.equal(torch.cat(sixouts)[:,5:],six['hidden'][:,5:])
            results[pool][arm]=dict(exported_reader_five_state_exact=True,saved_six_prefix_exact=True,routing_exact=True)
        for label in ['baseline','trained']:
            d=load(ROOT/'run'/f'{label}_{pool}'/'direct.pt');err=float((d['fast']['lp']-d['direct']).abs().max())
            assert err==d['error'] and err<=1e-4 and torch.equal(d['fast']['prediction'],d['direct'].argmax(-1))
        for arm in ['correct','wrong_binding']:
            ref=load(h.reference_scores(pool)/f'{arm}.pt')['first'];got=load(ROOT/'run'/f'reference_{pool}'/f'{arm}.pt')['first']
            for key in ['token_lp','lp','prediction']:assert torch.equal(ref[key],got[key]),key
    assert all(torch.equal(v,candidate[k]) for k,v in bridge.state_dict().items());h.check()
    return dict(status='PASS',receiver_loaded=False,new_native_scores=False,results=results,
        scope='Component replay on saved actual states; unchanged receiver before the sole block15 hook makes those input states applicable to the exported candidate. QueryOnlyReader concatenates hidden[:,5:] unchanged for every input.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--save',action='store_true');args=parser.parse_args();result=audit()
    if args.save:write(ROOT/'FINAL_CONTRACT.json',result)
    else:assert result==read(ROOT/'FINAL_CONTRACT.json')
    print(result['status'])
