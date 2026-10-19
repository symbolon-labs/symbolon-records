"""Post-training diagnostic only: no optimizer, selection, new task or new data."""
import json
import time

import torch
import channel_cf as cf

c,ROOT=cf.c,cf.ROOT


def main():
    started=time.monotonic(); c.seed_all(53003)
    decision=json.loads((ROOT/'run_tiny/decision.json').read_text())
    assert not decision['gate1'], 'This is the negative-result diagnostic path'
    assert json.loads((ROOT/'tiny_job/state.json').read_text())['status']=='complete'
    dest=ROOT/'tiny_terminal_audit'; assert not dest.exists(); dest.mkdir()
    rows,hidden=c.get_data('tiny')
    model,tok=c.load('receiver'); initial=c.parameter_hash(model)
    assert initial==decision['receiver_final_parameter_sha256']
    bridge=cf.BridgeModel()
    assert c.sha(ROOT/'run_tiny/last.pt')==decision['last_checkpoint_sha256']
    bridge.load_state_dict(torch.load(ROOT/'run_tiny/last.pt',weights_only=True)); bridge.eval()
    wrapper=c.ReceiverWrapper(model,bridge)
    parity=cf.stock_parity(wrapper,bridge,tok)
    _,repeated=c.evaluate(wrapper,bridge,tok,rows,hidden,dest/'evaluation')
    prior=torch.load(ROOT/f"run_tiny/eval_{decision['step']:04d}/outputs.pt",weights_only=True)
    exact={cond:all(torch.equal(prior['results'][cond][key],repeated[cond][key])
                    for key in ['lp','unconstrained']) for cond in c.CONDITIONS}
    donors=prior['donors']
    donor_exact=torch.equal(repeated['shuffled']['lp'],repeated['correct']['lp'][donors])
    final=c.parameter_hash(model)
    c.write_json(dest/'verification.json',dict(all_conditions_exact=all(exact.values()),conditions=exact,
        donor_probability_equivariance=donor_exact,parity=parity,receiver_initial_hash=initial,
        receiver_final_hash=final,endpoint_frozen=initial==final,step=decision['step'],
        elapsed_s=time.monotonic()-started,external_spend_usd=0,
        code_sha256=c.sha(__file__),provenance=cf.provenance(),
        caveat='Terminal training-set rerun, not held-out evidence or a second optimization seed.'))
    assert all(exact.values()) and donor_exact and initial==final
    print('PASS: negative terminal result reproduces exactly; no training performed',flush=True)


if __name__=='__main__': main()
