"""Two-site oracle substitution after the existing receiver layer15."""
import torch

def replace(hidden,payload):
    assert hidden.ndim==3 and hidden.shape[1] in [5,6] and hidden.shape[2]==960
    assert payload.shape==(len(hidden),2,960) and payload.dtype==hidden.dtype
    return torch.cat([hidden[:,:3],payload,hidden[:,5:]],dim=1)

class Boundary:
    def __init__(self,model):
        assert len(model.model.layers)==32 and model.config.hidden_size==960
        self.model=model;self.payload=None;self.capture=None;self.observe=False
        self.hook=model.model.layers[15].register_forward_hook(self._hook)
    def _hook(self,module,args,hidden):
        out=hidden if self.payload is None else replace(hidden,self.payload)
        if self.payload is not None:
            assert torch.equal(out[:,:3],hidden[:,:3]) and torch.equal(out[:,5:],hidden[:,5:])
            assert torch.equal(out[:,3:5],self.payload)
        if self.observe:self.capture=dict(before=hidden.detach().clone(),after=out.detach().clone())
        return out
    def forward(self,ids,memory=None,keep=1):
        assert self.payload is None
        self.payload=memory
        try:return self.model(input_ids=ids,use_cache=False,logits_to_keep=keep).logits
        finally:self.payload=None
    def close(self):self.hook.remove()
