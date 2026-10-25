"""Channel arms and the MAC (design sections 4 and 5.1). Standard library hmac and hashlib.

The attacker sits on the transport. It records messages and their tags and resends
them. It has no key and computes no new message. The receiver checks the tag over
the message bytes, its own session nonce and its own step counter. A rejected
message becomes the zero vector (the fixed fallback).
"""
import hashlib
import hmac
import struct

import numpy as np
import torch

import beacon as B

PREFIX = b"beacon-v1"
SENDER_ID = b"S0"
FIELD_SETS = ("bytes", "bytes+nonce", "bytes+counter", "full")


def team_key(seed):
    """K = SHA-256("marl-replay|key|" + team seed). A real system uses a random secret key."""
    return hashlib.sha256(f"marl-replay|key|{int(seed)}".encode()).digest()


def session_nonce(set_seed, episode):
    """16-byte session nonce N_e from SHA-256 of the set seed and the episode index."""
    return hashlib.sha256(f"marl-replay|nonce|{int(set_seed)}|{int(episode)}".encode()).digest()[:16]


def msg_bytes(m):
    """256 bytes: the 64 floats as float32, little-endian."""
    return np.ascontiguousarray(m, dtype="<f4").tobytes()


def tag(key, nonce, t, mb, fields="full"):
    """tau = HMAC-SHA256(K, "beacon-v1" || sender_id || N_e || uint32(t) || bytes(m)).
    The other field sets (design section 5.1 field table) drop N_e and/or t."""
    parts = [PREFIX, SENDER_ID]
    if fields in ("bytes+nonce", "full"):
        parts.append(nonce)
    if fields in ("bytes+counter", "full"):
        parts.append(struct.pack("<I", int(t)))
    parts.append(mb)
    return hmac.digest(key, b"".join(parts), "sha256")


class Receiver:
    """Receiver rule: compute the expected tag from its own N_e, its own t and the
    received bytes; deliver the message only if the tags are equal (constant-time
    compare); accept at most one message for each pair (N_e, t)."""

    def __init__(self, key, fields="full"):
        self.key, self.fields = key, fields
        self.seen = set()

    def accept(self, nonce, t, mb, received_tag):
        ok = hmac.compare_digest(tag(self.key, nonce, t, mb, self.fields), received_tag)
        if ok and (nonce, t) in self.seen:
            return False
        if ok:
            self.seen.add((nonce, t))
        return ok


def sign_stream(key, set_seed, msgs, fields="full", episodes=None):
    """Tags of a recorded stream msgs (n, T, 64) under its own session nonces and steps.
    episodes: the episode index of each row in its own set (default 0..n-1)."""
    a = msgs.numpy() if isinstance(msgs, torch.Tensor) else msgs
    n, T = a.shape[:2]
    ep = range(n) if episodes is None else [int(e) for e in episodes]
    return [[tag(key, session_nonce(set_seed, e), t, msg_bytes(a[i, t]), fields) for t in range(T)]
            for i, e in enumerate(ep)]


def through_mac(key, set_seed, wire_msgs, wire_tags, fields="full"):
    """Pass a wire stream through the receiver of each episode of the set.

    wire_msgs: (n, T, 64) float32 tensor; wire_tags: n lists of T tags.
    Returns the delivered stream (rejected -> zero vector) and the accept mask (n, T).
    """
    a = wire_msgs.numpy()
    n, T = a.shape[:2]
    acc = np.zeros((n, T), dtype=bool)
    for e in range(n):
        rx = Receiver(key, fields)
        nonce = session_nonce(set_seed, e)
        for t in range(T):
            acc[e, t] = rx.accept(nonce, t, msg_bytes(a[e, t]), wire_tags[e][t])
    acc_t = torch.from_numpy(acc)
    delivered = torch.where(acc_t.unsqueeze(-1), wire_msgs, torch.zeros_like(wire_msgs))
    return delivered, acc_t


def random_sphere(g, shape):
    """Vectors uniform on the sphere of norm 8: 8 g / |g| for a standard normal g."""
    x = torch.randn(*shape, B.MSG_DIM, generator=g)
    return B.MSG_NORM * x / x.norm(dim=-1, keepdim=True)


def replay_t(m):
    """RT: m_0 at every step (the attacker resends the step-0 message)."""
    return m[:, :1].expand_as(m).clone()


def replay_e(donor_m, donor_idx, T):
    """RE: the step-t message of donor stream d(e), in lockstep from t = 0."""
    return donor_m[donor_idx, :T].clone()


def delay(m, lag=5):
    """RT5: m_t for t < 5; m_{t-5} for t >= 5."""
    out = m.clone()
    out[:, lag:] = m[:, :-lag]
    return out


def splice(m, other, start=5):
    """RE5: m_t for t < 5; the donor message m^d_t for t >= 5."""
    out = m.clone()
    out[:, start:] = other[:, start:]
    return out


def wire_tags_rt(tags):
    """The attacker resends (m_0, tau_0) at every step."""
    return [[row[0]] * len(row) for row in tags]


def wire_tags_re(donor_tags, donor_idx, T):
    return [donor_tags[int(d)][:T] for d in donor_idx]


def wire_tags_delay(tags, lag=5):
    return [row[:lag] + row[:-lag] for row in tags]


def wire_tags_splice(tags, other_tags, start=5):
    return [row[:start] + o[start:] for row, o in zip(tags, other_tags)]
