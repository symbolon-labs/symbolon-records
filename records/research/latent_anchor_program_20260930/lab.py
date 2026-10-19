"""Scratch-mode library for the latent anchor program (E0-E2).

Ported from the sealed binding V1 harness:
  latent_channel_v1_20260906/channel.py      (resampler/reader constructors, wrapper, tokens, derangement)
  latent_channel_v2_20260906/channel_v2.py   (bounded 2*cosine attention forwards)
  latent_channel_smollm_20260906/channel_cf.py (960-wide BridgeModel, receiver layer 15)
  latent_binding_v1_20260906/binding.py      (AnswerBank, score, direct_score, train_logits)
  latent_binding_v1_20260906/run.py          (capture, memories/controls, text prompts)
  latent_binding_training_fold_recovery_20260913/common.py (constrained sequential counts)

Changes from the sealed code, all deliberate:
  * device-safe (MPS) tensors; endpoints load on CPU, are hashed, then move to the device
  * memory shape is configurable (k slots x width); the reader's internal width stays 64
    as in the sealed bridge; the 569,857 assert holds only for 8x64
  * optional sender-side symbol channel (E2-source): for each source line k, the learned
    vector S[letter_k] + S[digit_k] is added to the layer-normed sender states at both of
    that line's ID token positions (DESIGN.md section 4). Symbols come only from the source text.
  * score_cached: KV-cache scorer used by the sweep (validated in E0_data/VALIDATION.json)
  * score_fast: one forward per batch over all 16 first-token prefixes (keep=2),
    validated against the sealed score()/direct_score() within 1e-4
  * reader diagnostics are not computed (they force host syncs on MPS)
  * no bit-exact assertions; parity is reported as max-abs differences
Nothing in research/ is imported or written.
"""
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PROG = Path(__file__).resolve().parent
R = PROG.parent
COLORS = ['red', 'green', 'blue', 'yellow']
IDS = [a + str(d) for a in 'ABCDEFGHIJKLMNOP' for d in range(8)]
BURNED = ['A0', 'A2', 'C4', 'I2', 'K0', 'K3', 'M0', 'M3']
SENDER_HASH = 'deb5e69d2fcd24d139ce2927f8375c2c445867b196bc179df4688d98990e0ec4'
RECEIVER_HASH = '6a3e1782ebad58d1e9323d5eeaff191d112848b5f0058394a998d1ac54c492b3'
SENDER_LAYER, RECEIVER_LAYER = 13, 15          # zero-based
SENDER_WIDTH, RECEIVER_WIDTH = 2048, 960
SOURCE_TOKENS = 19
ID_POSITIONS = [(5 * k, 5 * k + 1) for k in range(4)]   # (letter, digit) token positions per source line
SENDER_INSTRUCTION = 'Read the bindings and answer the question. Reply with exactly one ID and nothing else.\n\n'
RECEIVER_INSTRUCTION = 'Copy the answer ID. Reply with exactly the ID and nothing else.\n\n'
SPLITS_DIR = PROG / 'E0_data/splits'
CAPTURE_DIR = PROG / 'E0_data/captures'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def device():
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def sync(dev):
    if dev.type == 'mps':
        torch.mps.synchronize()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def event(kind, **data):
    print(json.dumps(dict(event=kind, utc=time.strftime('%H:%M:%S', time.gmtime()), **data)), flush=True)


# ----------------------------------------------------------------------------- endpoints

def model_paths():
    return json.loads((PROG / 'E0_data/model_paths.json').read_text())


def parameter_hash(module):
    digest = hashlib.sha256()
    for name, p in module.named_parameters():
        digest.update(name.encode())
        digest.update(p.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def load(role, dev=None, check_hash=True):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = model_paths()[role]
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, attn_implementation='eager',
                                                 local_files_only=True).eval()
    model.requires_grad_(False)
    digest = parameter_hash(model) if check_hash else None
    if check_hash:
        assert digest == {'sender': SENDER_HASH, 'receiver': RECEIVER_HASH}[role], f'{role} parameters differ'
    if dev is not None:
        model.to(dev)
    return model, tok, digest


def tokens(tok, texts):
    lists = [tok.encode(t, add_special_tokens=False) for t in texts]
    assert len({len(x) for x in lists}) == 1, 'Unexpected padding/length differences'
    return torch.tensor(lists, dtype=torch.long)


def render(tok, body):
    return tok.apply_chat_template([dict(role='user', content=body)], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False, date_string='06 Sep 2026')


class AnswerBank:
    def __init__(self, tok, leading=True):
        self.strings = [(' ' if leading else '') + x for x in IDS]
        self.tokens = torch.tensor([tok.encode(x, add_special_tokens=False) for x in self.strings])
        assert self.tokens.shape == (128, 2)
        assert len(set(self.tokens[:, 0].tolist())) == 16
        self.prefixes = sorted(set(self.tokens[:, 0].tolist()))
        self.indices = {p: torch.where(self.tokens[:, 0] == p)[0] for p in self.prefixes}
        self.prefix_index = torch.tensor([self.prefixes.index(int(t)) for t in self.tokens[:, 0]])
        self.tokenizer = tok

    def prompts(self, texts):
        token_lists = [self.tokenizer.encode(s, add_special_tokens=False) for s in texts]
        assert len(set(map(len, token_lists))) == 1, 'Padding not permitted'
        for text, ids in zip(texts[:8], token_lists[:8]):
            for label, tail in zip(self.strings, self.tokens.tolist()):
                assert self.tokenizer.encode(text + label, add_special_tokens=False) == ids + tail
        return torch.tensor(token_lists)


# ----------------------------------------------------------------------------- bridge

def bounded_attention(query, key, scale=2.0):
    """softmax(scale * cos(q, k)); the sealed bridge uses a fixed scale of 2."""
    q = F.normalize(query, dim=-1, eps=1e-8)
    k = F.normalize(key, dim=-1, eps=1e-8)
    return (scale * (q @ k.transpose(-2, -1))).softmax(-1)


class Temperature(nn.Module):
    """Attention scale exp(log_scale). Created after all sealed parameters and draws no RNG,
    so a bridge with temperature=None reproduces the sealed initialisation exactly."""
    def __init__(self, init, learnable):
        super().__init__()
        t = torch.tensor(math.log(init))
        if learnable:
            self.log_scale = nn.Parameter(t)
        else:
            self.register_buffer('log_scale', t)

    def forward(self):
        return self.log_scale.exp()


def entropy(a):
    return -(a.clamp_min(1e-30).log() * a).sum(-1).mean().item()


class Resampler(nn.Module):
    def __init__(self, sender_width=SENDER_WIDTH, width=64, k=8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(k, width) / math.sqrt(width))
        self.key = nn.Linear(sender_width, width, bias=False)
        self.value = nn.Linear(sender_width, width, bias=False)
        self.temperature = None
        self.track = False
        self.last_entropy = None

    def scale(self):
        return 2.0 if self.temperature is None else self.temperature()

    def forward(self, hidden, symbols=None):
        hidden = F.layer_norm(hidden, (hidden.shape[-1],))
        if symbols is not None:
            hidden = hidden + symbols
        a = bounded_attention(self.queries, self.key(hidden), self.scale())
        if self.track:
            self.last_entropy = entropy(a.detach())
        slots = a @ self.value(hidden)
        return slots / slots.square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()


class Reader(nn.Module):
    def __init__(self, receiver_width=RECEIVER_WIDTH, width=64):
        super().__init__()
        self.query = nn.Linear(receiver_width, width, bias=False)
        self.key = nn.Linear(receiver_width, width, bias=False)
        self.value = nn.Linear(receiver_width, width, bias=False)
        self.output = nn.Linear(width, receiver_width, bias=False)
        self.gate = nn.Parameter(torch.tensor(0.01))
        self.temperature = None
        self.track = False
        self.last_entropy = None

    def scale(self):
        return 2.0 if self.temperature is None else self.temperature()

    def forward(self, hidden, memory):
        q = self.query(F.layer_norm(hidden, (hidden.shape[-1],)))
        a = bounded_attention(q, self.key(memory), self.scale())
        if self.track:
            self.last_entropy = entropy(a.detach()[:, -1])
        return hidden + self.gate * self.output(a @ self.value(memory))


class Symbols(nn.Module):
    """E2-source: 16 letter + 8 digit vectors in layer-normed sender space, zero-initialised.
    Line k's vector S[letter_k] + S[digit_k] is added at both of its ID token positions."""
    def __init__(self, width=SENDER_WIDTH):
        super().__init__()
        self.table = nn.Parameter(torch.zeros(24, width))

    def forward(self, letter, digit, n_tokens=SOURCE_TOKENS):
        # letter, digit: LongTensor [B, 4] in source-line order
        B = letter.shape[0]
        out = torch.zeros(B, n_tokens, self.table.shape[1], device=self.table.device, dtype=self.table.dtype)
        for k, (pl, pd) in enumerate(ID_POSITIONS):
            v = self.table[letter[:, k]] + self.table[16 + digit[:, k]]
            out[:, pl] = v
            out[:, pd] = v
        return out


class Oracle(nn.Module):
    """Receiver-side ceiling test: one memory slot per source line, built from learned
    color + letter + digit embeddings (gold symbolic table, no sender). RMS-normalised like
    the resampler's slots."""
    def __init__(self, width=64):
        super().__init__()
        self.color = nn.Embedding(4, width)
        self.letter = nn.Embedding(16, width)
        self.digit = nn.Embedding(8, width)

    def forward(self, letter, digit, color):
        slots = self.color(color) + self.letter(letter) + self.digit(digit)      # [B, 4, w]
        return slots / slots.square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()


class LineSlots(nn.Module):
    """Structured sender read: slot k comes from source line k's own sender states.
    slot_k = W_id * mean(LN h[5k], LN h[5k+1])  +  W_col * LN h[5k+3]   (ID tokens, color token).
    Same per-line structure as the oracle, but every value comes from the frozen sender."""
    def __init__(self, sender_width=SENDER_WIDTH, width=64, factorized=False, symbols=False,
                 use_id=True, id_dropout=0.0, eq_input=False, aux_probes=False):
        super().__init__()
        self.eq_input = eq_input          # identity input from the ' =' token after each ID
        # E5: linear probes on the continuous slot (training-time pressure only; not in the memory path)
        self.probes = nn.ModuleList([nn.Linear(width, 16), nn.Linear(width, 8), nn.Linear(width, 4)]) if aux_probes else None
        self.last_slots = None
        self.use_id = use_id              # False: identity only from source-text symbols
        self.id_dropout = id_dropout      # training-time per-slot dropout of the continuous identity term
        self.id_map = nn.Linear(sender_width, width, bias=False)
        self.color_map = nn.Linear(sender_width, width, bias=False)
        self.digit_map = nn.Linear(sender_width, width, bias=False) if factorized else None
        # E2 on the structured read: letter/digit vectors from the sender's own source text
        self.sym_letter = nn.Embedding(16, width) if symbols else None
        self.sym_digit = nn.Embedding(8, width) if symbols else None
        if symbols:
            nn.init.zeros_(self.sym_letter.weight)
            nn.init.zeros_(self.sym_digit.weight)

    def forward(self, hidden, letter=None, digit=None, use_hidden=True, use_symbols=True):
        h = F.layer_norm(hidden, (hidden.shape[-1],))
        if self.eq_input:
            idv = self.id_map(torch.stack([h[:, a + 2] for a, _ in ID_POSITIONS], 1))
        elif self.digit_map is None:
            ids = torch.stack([(h[:, a] + h[:, b]) / 2 for a, b in ID_POSITIONS], 1)    # [B, 4, 2048]
            idv = self.id_map(ids)
        else:
            idv = self.id_map(torch.stack([h[:, a] for a, _ in ID_POSITIONS], 1)) + \
                  self.digit_map(torch.stack([h[:, b] for _, b in ID_POSITIONS], 1))
        if not self.use_id:
            idv = torch.zeros_like(idv)
        elif self.training and self.id_dropout > 0:
            keep = (torch.rand(idv.shape[:2], device=idv.device) >= self.id_dropout).to(idv.dtype)
            idv = idv * keep.unsqueeze(-1)
        col = torch.stack([h[:, a + 3] for a, _ in ID_POSITIONS], 1)
        slots = idv + self.color_map(col)
        if not use_hidden:
            slots = torch.zeros_like(slots)
        if self.sym_letter is not None and use_symbols:
            slots = slots + self.sym_letter(letter) + self.sym_digit(digit)
        slots = slots / slots.square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
        self.last_slots = slots
        return slots

    def probe_loss(self, letter, digit, color):
        s = self.last_slots
        pl, pd, pc = (p(s) for p in self.probes)
        return (F.cross_entropy(pl.reshape(-1, 16), letter.reshape(-1)) + F.cross_entropy(pd.reshape(-1, 8), digit.reshape(-1))
                + F.cross_entropy(pc.reshape(-1, 4), color.reshape(-1)))

    def probe_accuracy(self, hidden, letter, digit, color):
        with torch.no_grad():
            s = self.forward(hidden)
            return [(p(s).argmax(-1) == y).float().mean().item() for p, y in zip(self.probes, (letter, digit, color))]


class Factored(nn.Module):
    """Sender-derived factored memory. Linear heads on frozen sender states give soft letter and
    digit distributions (from the ' =' token after each ID) and a soft color distribution (from the
    color token). Slot k = p_letter E_letter + p_digit E_digit + p_color E_color, RMS-normalised.
    Every input is a sender hidden state; the source text is used only for optional auxiliary
    supervision of the heads (the sender has its own input)."""
    def __init__(self, sender_width=SENDER_WIDTH, width=64):
        super().__init__()
        self.letter_head = nn.Linear(sender_width, 16)
        self.digit_head = nn.Linear(sender_width, 8)
        self.color_head = nn.Linear(sender_width, 4)
        self.letter = nn.Embedding(16, width)
        self.digit = nn.Embedding(8, width)
        self.color = nn.Embedding(4, width)
        self.last_logits = None

    def forward(self, hidden):
        h = F.layer_norm(hidden, (hidden.shape[-1],))
        eq = torch.stack([h[:, a + 2] for a, _ in ID_POSITIONS], 1)        # ' =' token of each line
        col = torch.stack([h[:, a + 3] for a, _ in ID_POSITIONS], 1)       # color token of each line
        ll, dl, cl = self.letter_head(eq), self.digit_head(eq), self.color_head(col)
        self.last_logits = (ll, dl, cl)
        if getattr(self, 'hard', False):   # eval-time option: one-hot argmax of each head
            pl, pd, pc = (F.one_hot(x.argmax(-1), x.shape[-1]).to(x.dtype) for x in (ll, dl, cl))
        else:
            pl, pd, pc = ll.softmax(-1), dl.softmax(-1), cl.softmax(-1)
        slots = pl @ self.letter.weight + pd @ self.digit.weight + pc @ self.color.weight
        return slots / slots.square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()

    def aux_loss(self, letter, digit, color):
        ll, dl, cl = self.last_logits
        return (F.cross_entropy(ll.reshape(-1, 16), letter.reshape(-1)) + F.cross_entropy(dl.reshape(-1, 8), digit.reshape(-1))
                + F.cross_entropy(cl.reshape(-1, 4), color.reshape(-1)))


class Bridge(nn.Module):
    """Construction order matches channel_cf.BridgeModel, so 8x64 reproduces the sealed init."""
    def __init__(self, k=8, width=64, symbols=False, reader_width=64, temp_init=None, learn_temp=False,
                 oracle=False, gate_init=None, lineslots=False, factorized=False, line_symbols=False,
                 line_use_id=True, id_dropout=0.0, factored=False, eq_input=False, aux_probes=False):
        super().__init__()
        self.resampler = Resampler(width=width, k=k)
        self.projector = nn.Linear(width, RECEIVER_WIDTH, bias=False)
        self.reader = Reader(receiver_width=RECEIVER_WIDTH, width=reader_width)
        if (k, width) == (8, 64):
            assert sum(p.numel() for p in self.parameters()) == 569857
        self.symbols = Symbols() if symbols else None
        self.oracle = Oracle(width) if oracle else None
        self.factored = Factored(width=width) if factored else None
        self.lineslots = LineSlots(width=width, factorized=factorized, symbols=line_symbols,
                                   use_id=line_use_id, id_dropout=id_dropout, eq_input=eq_input,
                                   aux_probes=aux_probes) if lineslots else None
        if gate_init is not None:
            with torch.no_grad():
                self.reader.gate.fill_(gate_init)
        if temp_init is not None:
            self.resampler.temperature = Temperature(temp_init, learn_temp)
            self.reader.temperature = Temperature(temp_init, learn_temp)

    def encode(self, hidden, letter=None, digit=None, use_hidden=True, use_symbols=True, color=None):
        """hidden, letter and digit may come from different tables (hybrid control arm)."""
        if self.oracle is not None:
            return self.oracle(letter, digit, color)
        if self.factored is not None:
            return self.factored(hidden if use_hidden else torch.zeros_like(hidden))
        if self.lineslots is not None:
            return self.lineslots(hidden, letter, digit, use_hidden=use_hidden, use_symbols=use_symbols)
        sym = None
        if self.symbols is not None and use_symbols:
            sym = self.symbols(letter, digit, hidden.shape[1])
        if not use_hidden:
            hidden = torch.zeros_like(hidden)
        return self.resampler(hidden, sym)


class Receiver:
    def __init__(self, model, bridge):
        assert model.config.model_type == 'llama' and model.config.hidden_size == RECEIVER_WIDTH
        assert len(model.model.layers) == 32
        self.model, self.bridge = model, bridge
        self.memory, self.enabled = None, False
        self.hook = model.model.layers[RECEIVER_LAYER].register_forward_hook(self._hook)

    def _hook(self, module, args, hidden):
        if not self.enabled:
            return hidden
        assert isinstance(hidden, torch.Tensor), 'decoder layer must return a tensor (transformers 5.x)'
        return self.bridge.reader(hidden, self.bridge.projector(self.memory))

    def forward(self, ids, memory=None, keep=1):
        self.enabled = memory is not None
        self.memory = memory
        try:
            return self.model(input_ids=ids, use_cache=False, logits_to_keep=keep).logits
        finally:
            self.enabled, self.memory = False, None

    def stock(self, ids, keep=1):
        self.hook.remove()
        try:
            return self.model(input_ids=ids, use_cache=False, logits_to_keep=keep).logits
        finally:
            self.hook = self.model.model.layers[RECEIVER_LAYER].register_forward_hook(self._hook)


def native_forward(model):
    return lambda ids, memory=None, keep=1: model(input_ids=ids, use_cache=False, logits_to_keep=keep).logits


def on_device(forward, dev):
    """Wrap a forward so CPU ids/memory go to dev and float32 logits come back to CPU."""
    def wrapped(ids, memory=None, keep=1):
        return forward(ids.to(dev), None if memory is None else memory.to(dev), keep).float().cpu()
    return wrapped


# ----------------------------------------------------------------------------- scoring

def score(forward, ids, bank, memory=None, batch=8):
    """Sealed reference scorer: 1 + 16 no-cache forwards per batch (forward must return CPU logits)."""
    all_parts = []
    with torch.no_grad():
        for start in range(0, len(ids), batch):
            q = ids[start:start + batch]
            z = None if memory is None else memory[start:start + batch]
            first = forward(q, z, 1)[:, -1].log_softmax(-1)
            parts = torch.empty(len(q), 128, 2)
            parts[:, :, 0] = first[:, bank.tokens[:, 0]]
            for prefix in bank.prefixes:
                seq = torch.cat([q, torch.full((len(q), 1), prefix, dtype=torch.long)], dim=1)
                second = forward(seq, z, 1)[:, -1].log_softmax(-1)
                ix = bank.indices[prefix]
                parts[:, ix, 1] = second[:, bank.tokens[ix, 1]]
            all_parts.append(parts)
    parts = torch.cat(all_parts)
    return dict(token_lp=parts, lp=parts.sum(-1), prediction=parts.sum(-1).argmax(-1))


def direct_score(forward, ids, bank, memory=None):
    """Sealed independent slow teacher-forced loop over all 128 candidates."""
    outputs = []
    with torch.no_grad():
        for tail in bank.tokens:
            seq = torch.cat([ids, tail[:1].expand(len(ids), 1)], dim=1)
            lp = forward(seq, memory, 2).log_softmax(-1)
            outputs.append(lp[:, 0, tail[0]] + lp[:, 1, tail[1]])
    return torch.stack(outputs, dim=1)


def score_fast(forward_dev, ids, bank, memory=None, batch=32, dev=None):
    """One forward per batch: every query extended by all 16 first tokens, keep=2.

    forward_dev takes device tensors and returns device logits. Position -2 gives the
    first-token distribution (identical across prefixes in a causal model), position -1
    gives the second-token distribution conditional on each prefix.
    """
    dev = dev or device()
    P = len(bank.prefixes)
    prefixes = torch.tensor(bank.prefixes, device=dev)
    first_tok = bank.tokens[:, 0].to(dev)
    second_tok = bank.tokens[:, 1].to(dev)
    pidx = bank.prefix_index.to(dev)
    out = []
    with torch.no_grad():
        for start in range(0, len(ids), batch):
            q = ids[start:start + batch].to(dev)
            n = len(q)
            seq = torch.cat([q.repeat_interleave(P, 0), prefixes.repeat(n).unsqueeze(1)], 1)
            z = None
            if memory is not None:
                z = memory[start:start + batch].to(dev).repeat_interleave(P, 0)
            lp = forward_dev(seq, z, 2).float().log_softmax(-1).view(n, P, 2, -1)
            parts = torch.empty(n, 128, 2, device=dev)
            parts[:, :, 0] = lp[:, 0, 0, :][:, first_tok]
            second = lp[:, :, 1, :]                                   # [n, P, V]
            parts[:, :, 1] = second[:, pidx, second_tok]
            out.append(parts.cpu())
    parts = torch.cat(out)
    return dict(token_lp=parts, lp=parts.sum(-1), prediction=parts.sum(-1).argmax(-1))


def score_cached(receiver, ids, bank, memory=None, batch=64, dev=None):
    """KV-cache scorer: one cached forward over the query, then one token for each of 16 prefixes.

    Equivalent to score()/score_fast() because the reader acts position-wise and the query
    positions are identical across prefixes. Validated against direct_score in E0_data/validate.py.
    """
    dev = dev or device()
    P = len(bank.prefixes)
    prefixes = torch.tensor(bank.prefixes, device=dev)
    first_tok, second_tok = bank.tokens[:, 0].to(dev), bank.tokens[:, 1].to(dev)
    pidx = bank.prefix_index.to(dev)
    model = receiver.model
    out = []
    with torch.no_grad():
        for start in range(0, len(ids), batch):
            q = ids[start:start + batch].to(dev)
            n = len(q)
            z = None if memory is None else memory[start:start + batch].to(dev)
            receiver.enabled, receiver.memory = z is not None, z
            try:
                o1 = model(input_ids=q, use_cache=True, logits_to_keep=1)
                first = o1.logits[:, -1].float().log_softmax(-1)
                cache = o1.past_key_values
                cache.batch_repeat_interleave(P)
                receiver.memory = None if z is None else z.repeat_interleave(P, 0)
                o2 = model(input_ids=prefixes.repeat(n).unsqueeze(1), past_key_values=cache,
                           use_cache=True, logits_to_keep=1)
                second = o2.logits[:, -1].float().log_softmax(-1).view(n, P, -1)
            finally:
                receiver.enabled, receiver.memory = False, None
            parts = torch.empty(n, 128, 2, device=dev)
            parts[:, :, 0] = first[:, first_tok]
            parts[:, :, 1] = second[:, pidx, second_tok]
            out.append(parts.cpu())
    parts = torch.cat(out)
    return dict(token_lp=parts, lp=parts.sum(-1), prediction=parts.sum(-1).argmax(-1))


def counts(out, targets, bank_tokens):
    """Joint and candidate-constrained sequential correct counts (sealed common.py:79-83)."""
    groups = [torch.where(bank_tokens[:, 0] == p)[0] for p in dict.fromkeys(bank_tokens[:, 0].tolist())]
    reps = torch.tensor([int(g[0]) for g in groups])
    pred = []
    for part in out['token_lp']:
        g = groups[int(part[reps, 0].argmax())]
        pred.append(int(g[part[g, 1].argmax()]))
    targets = torch.as_tensor(targets)
    return dict(joint=int((out['prediction'] == targets).sum()),
                sequential=int((torch.tensor(pred) == targets).sum()), rows=len(targets),
                sequential_prediction=torch.tensor(pred))


def train_logits(receiver, bank_tokens_dev, ids, labels, memory):
    tails = bank_tokens_dev[labels]
    seq = torch.cat([ids, tails[:, :1]], dim=1)
    logits = receiver.forward(seq, memory, 2)
    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tails.reshape(-1), reduction='none')
    return losses.reshape(len(ids), 2).sum(-1).mean()


# ----------------------------------------------------------------------------- data

def derangement(n, seed=61001):
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(n)
        if np.all(p != np.arange(n)):
            return p


def load_split(name):
    data = json.loads((SPLITS_DIR / f'{name}.json').read_text())
    payload = torch.load(CAPTURE_DIR / f'{name}.pt', weights_only=True)
    n = len(data['records'])
    assert payload['hidden'].shape == (n, SOURCE_TOKENS, SENDER_WIDTH)
    letter = torch.tensor([[ord(line.split(' = ')[0][0]) - 65 for line in r['source'].split('\n')]
                           for r in data['records']])
    digit = torch.tensor([[int(line.split(' = ')[0][1]) for line in r['source'].split('\n')]
                          for r in data['records']])
    wletter = torch.tensor([[ord(line.split(' = ')[0][0]) - 65 for line in r['wrong_source'].split('\n')]
                            for r in data['records']])
    wdigit = torch.tensor([[int(line.split(' = ')[0][1]) for line in r['wrong_source'].split('\n')]
                           for r in data['records']])
    assert torch.equal(letter, wletter) and torch.equal(digit, wdigit), 'wrong-binding twin must keep ID order'
    color = torch.tensor([[COLORS.index(line.split(' = ')[1]) for line in r['source'].split('\n')] for r in data['records']])
    wcolor = torch.tensor([[COLORS.index(line.split(' = ')[1]) for line in r['wrong_source'].split('\n')] for r in data['records']])
    rows = data['rows']
    groups = torch.tensor([r['group'] for r in rows])
    labels = torch.tensor([IDS.index(r['answer']) for r in rows])
    wrong_labels = torch.tensor([IDS.index(r['wrong_answer']) for r in rows])
    novel = torch.tensor([bool(r.get('novel_answer', False)) for r in rows])
    novel_wrong = torch.tensor([bool(r.get('novel_wrong_answer', False)) for r in rows])
    return dict(name=name, data=data, hidden=payload['hidden'], wrong_hidden=payload['wrong_hidden'],
                letter=letter, digit=digit, color=color, wcolor=wcolor, groups=groups, labels=labels, wrong_labels=wrong_labels,
                novel=novel, novel_wrong=novel_wrong, colors=[r['color'] for r in rows])


def memories(bridge, split, dev, arms):
    """Memories for each control arm, one row per query. Random controls drawn on CPU (sealed streams)."""
    g = split['groups']
    n = len(split['data']['records'])
    H = split['hidden'].to(dev)
    W = split['wrong_hidden'].to(dev)
    L, D = split['letter'].to(dev), split['digit'].to(dev)
    C, WC = split['color'].to(dev), split['wcolor'].to(dev)
    has_sym = bridge.symbols is not None or (bridge.lineslots is not None and bridge.lineslots.sym_letter is not None)

    def enc(Hx, Cx, **kw):
        return torch.cat([bridge.encode(h, l, d, color=c, **kw) for h, l, d, c in
                          zip(Hx.split(64), L.split(64), D.split(64), Cx.split(64))]).cpu()
    with torch.no_grad():
        z = enc(H, C)
        values = {}
        if 'correct' in arms:
            values['correct'] = z[g]
        donors = derangement(n)
        if 'shuffled' in arms:
            values['shuffled'] = z[torch.tensor(donors)[g]]
        if 'wrong_binding' in arms:
            w = enc(W, WC)
            values['wrong_binding'] = w[g]
        if 'zero' in arms:
            values['zero'] = torch.zeros_like(z)[g]
        for seed in [71001, 71002]:
            if f'random_{seed}' in arms:
                rz = torch.randn(z.shape, generator=torch.Generator().manual_seed(seed))
                rz = rz / rz.norm(dim=-1, keepdim=True) * z.norm(dim=-1, keepdim=True)
                values[f'random_{seed}'] = rz[g]
        if 'symbols_only' in arms and has_sym:
            s = enc(H, C, use_hidden=False)
            values['symbols_only'] = s[g]
        if 'hybrid' in arms and has_sym:
            # own table's symbols, donor table's continuous states
            Hd = H[torch.tensor(donors, device=H.device)]
            hy = enc(Hd, C)
            values['hybrid'] = hy[g]
        if 'continuous_only' in arms and has_sym:
            c = enc(H, C, use_symbols=False)
            values['continuous_only'] = c[g]
    donor_rows = {(r['group'], r['color']): i for i, r in enumerate(split['data']['rows'])}
    donor_index = torch.tensor([donor_rows[(int(donors[r['group']]), r['color'])] for r in split['data']['rows']])
    # positional target for the hybrid arm: own table's ID on the source line where the donor
    # table lists the queried color
    recs = split['data']['records']
    positional = []
    for r in split['data']['rows']:
        donor_lines = [l.split(' = ') for l in recs[int(donors[r['group']])]['source'].split('\n')]
        own_lines = [l.split(' = ') for l in recs[r['group']]['source'].split('\n')]
        k = [c for _, c in donor_lines].index(r['color'])
        positional.append(IDS.index(own_lines[k][0]))
    return values, dict(donor_index=donor_index, positional=torch.tensor(positional))


# ----------------------------------------------------------------------------- logging

CSV_FIELDS = ['experiment', 'variant', 'run', 'seed', 'lr', 'slots', 'slot_dim', 'params', 'step', 'kind',
              'train_ce', 'train_subset_joint', 'dev_known_joint', 'dev_known_seq',
              'oob_pure_joint', 'oob_pure_seq', 'inblock_pure_joint', 'inblock_pure_seq',
              'oob_mixed_novel_joint', 'oob_mixed_all_joint', 'inblock_mixed_novel_joint',
              'oob_pure_donor_following', 'oob_pure_changed_following', 'oob_pure_in_table',
              'oob_pure_shuffled', 'oob_pure_random_71001', 'oob_pure_random_71002', 'oob_pure_zero',
              'oob_pure_receiver_only', 'oob_pure_symbols_only', 'oob_pure_continuous_only',
              'oob_pure_hybrid_own', 'oob_pure_hybrid_donor', 'oob_pure_hybrid_positional',
              'dev_known_donor_following', 'dev_known_changed_following',
              'gate', 'symbol_norm', 'temperature', 'resampler_entropy', 'reader_entropy',
              'wall_s', 'device', 'lab_sha256', 'notes']


def append_csv(path, row):
    path = Path(path)
    if path.exists():
        with path.open(newline='') as f:
            reader = csv.DictReader(f)
            header, old_rows = reader.fieldnames, list(reader)
        if header != CSV_FIELDS:      # columns were added: rewrite with the current header
            with path.open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
                w.writeheader()
                for r in old_rows:
                    w.writerow({k: r.get(k, '') for k in CSV_FIELDS})
    new = not path.exists()
    with path.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, '') for k in CSV_FIELDS})
