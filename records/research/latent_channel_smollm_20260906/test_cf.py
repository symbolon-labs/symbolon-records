import json
from unittest.mock import patch
import pytest
import torch
from transformers import AutoTokenizer
import channel_cf as cf

torch.set_num_threads(4)


def test_bottleneck_zero_and_gradients():
    torch.manual_seed(1); b=cf.BridgeModel()
    assert sum(p.numel() for p in b.parameters())==569857
    h=torch.randn(4,22,2048); z=b.resampler(h); native=torch.randn(4,12,960)
    assert z.shape==(4,8,64)
    assert torch.allclose(z.norm(dim=-1),torch.full((4,8),8.))
    assert torch.equal(native,b.reader(native,b.projector(torch.zeros_like(z))))
    b.reader(native,b.projector(z)).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in b.parameters())


def test_only_tiny_data_licensed(tmp_path):
    rows=cf.dataset('tiny')
    prior=[cf.BASE.parent/'data/tiny/examples.json',cf.ROOT.parent/'latent_channel_crossfamily_20260906/data/tiny/examples.json']
    prior += [cf.V2/'data'/s/'examples.json' for s in ['tiny','train','dev','final']]
    used={r['nonce'] for p in prior for r in json.loads(p.read_text())}
    assert not used & {r['nonce'] for r in rows}
    assert len(rows)==32 and len({r['id'] for r in rows})==32
    assert {r['query'] for r in rows}=={cf.c.QUERY}
    for nonce in {r['nonce'] for r in rows}:
        assert sorted(r['label'] for r in rows if r['nonce']==nonce)==[0,1,2,3]
    # Guard calls cannot instantiate later rows, even if real tiny gates pass later.
    with patch.object(cf,'ROOT',tmp_path):
        for split in ['train','dev','final']:
            with pytest.raises(FileNotFoundError): cf.dataset(split)


def test_native_token_boundaries():
    from text_cf import render,text
    rows=cf.dataset('tiny')
    for role in ['sender','receiver']:
        tok=AutoTokenizer.from_pretrained(cf.c.model_path(role),local_files_only=True)
        assert len({len(tok.encode(r['source'],add_special_tokens=False)) for r in rows})==1
        if role=='receiver':
            assert len(tok.encode(cf.c.QUERY,add_special_tokens=False))==12
            for prompt in [cf.c.QUERY,*[r['source']+'\n'+cf.c.QUERY for r in rows]]:
                for a in cf.c.ANSWERS:
                    tail=tok.encode(a,add_special_tokens=False); assert len(tail)==1
                    assert tok.encode(prompt+a,add_special_tokens=False)==tok.encode(prompt,add_special_tokens=False)+tail
        for letter in 'ABCD':
            prompt=render(tok,text.RECEIVER_INSTRUCTION+'Signal: '+letter)
            for answer in 'ABCD':
                tail=tok.encode(answer,add_special_tokens=False); assert len(tail)==1
                assert tok.encode(prompt+answer,add_special_tokens=False)==tok.encode(prompt,add_special_tokens=False)+tail
