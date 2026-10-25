#!/usr/bin/env python3
"""Split learning at equal bits: activations against a symbolic message (CONFIRMATORY code).

The rule, the arms and every fixed choice are in PREREGISTRATION.md. FREEZE.json holds the sha256 of
this file. Design: reports/split_equal_bits_design_20261023.md. Run each job through run_sebits.sh.
Run Python with -I -B.

  sebits.py test                 unit tests of test_sebits.py (synthetic data; no CIFAR-10 file is read)
  sebits.py freeze [label]       run the tests and write FREEZE.json (a label only for an approved re-freeze)
  sebits.py data <archive>       check the archive, read it with tarfile and numpy, write data/*.npz
  sebits.py j0                   smoke in scratch mode (seed 20261100; the test split is not read)
  sebits.py j1 ... j5 [--part i] confirmatory jobs (part i of job_plan.json, or all units)
  sebits.py j6                   analysis; the only job that reads the test labels
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import sys
import tarfile
import time
import traceback
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RES = ROOT / "results"
FROZEN = ("PREREGISTRATION.md", "sebits.py", "test_sebits.py", "run_sebits.sh")

ARCHIVE_URL = "https://cave.cs.toronto.edu/kriz/cifar-10-binary.tar.gz"
ARCHIVE_MD5 = "c32a1d4ab5d03f1284b67883e8d87530"  # dataset page, read 2026-10-23
MEMBERS = [f"cifar-10-batches-bin/data_batch_{i}.bin" for i in range(1, 6)] + ["cifar-10-batches-bin/test_batch.bin"]
REC = 3073  # 1 label byte and 3,072 pixel bytes (R, G, B planes, row-major)
N_TEST = 10000

SPLIT_SEED = 20261110
RUN_SEEDS = (20261111, 20261112, 20261113)
SCRATCH_SEED = 20261100
BOOT_SEED = 20261024
N_BOOT = 10000
BOOT_CHUNK = 250
MEM_N = 5000

BUDGETS = (16, 64, 256, 1024)
BSTAR = 64
RAW_CANDS = {16: [(16, 1, 1), (4, 1, 4), (2, 1, 8)], 64: [(64, 1, 1), (16, 1, 4), (8, 1, 8)],
             256: [(64, 1, 4), (64, 2, 1), (32, 1, 8)], 1024: [(64, 2, 4), (64, 4, 1), (16, 4, 4)]}
BOT_CANDS = {16: [(16, 1, 1), (4, 1, 4), (4, 2, 1)], 64: [(16, 1, 4), (4, 2, 4), (16, 2, 1)],
             256: [(64, 1, 4), (16, 2, 4), (16, 4, 1)], 1024: [(16, 4, 4), (64, 2, 4), (16, 8, 1)]}
# TOPK formats: ("top", k, p) = F-top(k, p); ("all", 0, p) = F-all(p)
TOPK_CANDS = {16: [("top", 2, 9), ("top", 3, 3), ("top", 4, 1), ("all", 0, 1)],
              64: [("all", 0, 6), ("top", 4, 16), ("top", 5, 12)],
              256: [("all", 0, 16)], 1024: [("all", 0, 16)]}
GFILL_MULTS = (1.0, 1.5, 2.0, 3.0)
TEMPS = [round(0.5 + 0.05 * i, 2) for i in range(51)]
SEL_TIE_POINTS = 0.2  # RAW and BOT selection: within 0.2 points of the best D_dev accuracy, take the lower D_dev NLL

N_EPOCHS, ARM_EPOCHS, DEC_EPOCHS, PROBE_EPOCHS = 30, 15, 20, 30
BATCH = 128
JOB_LIMIT_MIN = 110
V6_MIN = 0.80

BUDGET_ARMS = [f"RAW@{b}" for b in BUDGETS] + [f"BOT@{b}" for b in BUDGETS]
TOPK_ARMS = ["TOPK@16", "TOPK@64", "TOPK@256"]  # TOPK@1024 is TOPK@256: F-all(16) at both budgets
DEC_ARMS = BUDGET_ARMS + TOPK_ARMS + ["LAB", "LAB_full", "TOPK_full", "TOPK_float", "RAW_native"]
J2_DECS = [(a, "S", 1, False) for a in DEC_ARMS] + [("BOT@64", "S", 1, True), ("TOPK@64", "S", 1, True)]
J3_DECS = ([(a, "L", 1, False) for a in ("TOPK@64", "BOT@64", "RAW@64", "LAB", "RAW_native")]
           + [("TOPK@64", "S", 2, False), ("TOPK@64", "S", 3, False), ("BOT@64", "S", 2, False), ("BOT@64", "S", 3, False)])
J45_DECS = [(a, "S", 1, False) for a in DEC_ARMS if a != "TOPK_float"]
MSG_ARMS = DEC_ARMS + ["BOT_float"]
PROBE_ARMS = ["RAW@64", "BOT@64", "TOPK@64", "LAB", "LAB_full", "TOPK_full"]
ALL_SETS = ("aux", "auxF", "dev", "devF", "test", "testF", "mem")

DEV = torch.device("cpu")


class StopRule(Exception):
    """A preregistered stop rule fired (design section 10.3)."""


class CheckFail(Exception):
    """A validity check (V3, V4 or V5) or a freeze check failed."""


# ------------------------------------------------------------------------------------------------ utilities

def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(*a):
    print(f"[{now()}]", *a, flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sub_seed(seed, tag):
    """Each run seed sets every initialization and data order through a tag (design 9.1)."""
    return int.from_bytes(hashlib.sha256(f"{seed}/{tag}".encode()).digest()[:4], "big")


def gen(seed):
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.tolist()
    raise TypeError(type(o))


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_js))
    tmp.replace(p)


def read_json(p):
    return json.loads(Path(p).read_text())


def append_jsonl(p, obj):
    with open(p, "a") as f:
        f.write(json.dumps(obj, default=_js) + "\n")


def sync():
    if DEV.type == "mps":
        torch.mps.synchronize()


def cosine(base, step, total):
    return base * 0.5 * (1.0 + math.cos(math.pi * step / total))


def finite_or_fail(t, what):
    if not torch.isfinite(t).all():
        raise CheckFail(f"V5: loss is not finite in {what}")


# ------------------------------------------------------------------------------------------------ data

def split_records(rec):
    rec = np.asarray(rec, dtype=np.uint8).reshape(-1, REC)
    y = rec[:, 0].copy()
    if y.max() > 9:
        raise ValueError("a label byte is above 9")
    return rec[:, 1:].reshape(-1, 3, 32, 32).copy(), y


def read_archive(path):
    """Read the six batch files from the tar.gz archive. No file is extracted to disk."""
    raw = {}
    with tarfile.open(path, "r:gz") as tf:
        for name in MEMBERS:
            m = tf.getmember(name)
            if not m.isfile() or m.size != 10000 * REC:
                raise ValueError(f"{name}: not a file of {10000 * REC} bytes")
            raw[name] = np.frombuffer(tf.extractfile(m).read(), dtype=np.uint8)
    xtr, ytr = split_records(np.concatenate([raw[n] for n in MEMBERS[:5]]))
    xte, yte = split_records(raw[MEMBERS[5]])
    return xtr, ytr, xte, yte


def make_splits(y, n_test=None, mem_n=None):
    """Stratified splits with the split seed: 70% D_task, 10% D_dev, 20% D_aux of each class (3,500, 500, 1,000)."""
    n_test = N_TEST if n_test is None else n_test
    mem_n = MEM_N if mem_n is None else mem_n
    y = np.asarray(y)
    rng = np.random.default_rng(SPLIT_SEED)
    parts = {"task": [], "dev": [], "aux": []}
    for c in range(10):
        idx = rng.permutation(np.flatnonzero(y == c))
        a, b = len(idx) * 7 // 10, len(idx) * 8 // 10
        parts["task"].append(idx[:a])
        parts["dev"].append(idx[a:b])
        parts["aux"].append(idx[b:])
    sp = {k: np.sort(np.concatenate(v)) for k, v in parts.items()}
    sp["flip_aux"] = rng.integers(0, 2, len(sp["aux"])).astype(np.uint8)
    sp["flip_dev"] = rng.integers(0, 2, len(sp["dev"])).astype(np.uint8)
    sp["flip_test"] = rng.integers(0, 2, n_test).astype(np.uint8)
    pt = rng.permutation(len(sp["task"]))[:2 * mem_n]  # positions in D_task
    sp["mem_in_train"], sp["mem_in_test"] = np.sort(pt[:mem_n]), np.sort(pt[mem_n:])
    sp["mem_out_train"] = np.sort(rng.permutation(len(sp["aux"]))[:mem_n])  # positions in D_aux
    sp["mem_out_test"] = np.sort(rng.permutation(n_test)[:mem_n])  # positions in the test split
    return sp


class Data:
    """Images stay on the device as uint8. mode: scratch (J0: D_dev is in the test role), train (J1), confirm, analysis."""

    def __init__(self, xtr, ytr, xte, yte, mode, mem_n=None):
        self.mode = mode
        mem_n = MEM_N if mem_n is None else mem_n
        self.sp = sp = make_splits(ytr, N_TEST, mem_n)
        self.ytr_np = np.asarray(ytr).astype(np.int64)
        if mode == "scratch":
            xte = xtr[sp["dev"]]
            sp["flip_test"] = sp["flip_dev"].copy()
            sp["mem_out_test"] = np.arange(min(mem_n, len(sp["dev"])))
            yte = self.ytr_np[sp["dev"]]
        self.yte_np = None if yte is None else np.asarray(yte).astype(np.int64)
        mean, std = [], []
        for c in range(3):
            v = xtr[sp["task"], c].astype(np.float64) / 255.0
            mean.append(v.mean())
            std.append(v.std())
        self.mean = torch.tensor(mean, dtype=torch.float32, device=DEV).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32, device=DEV).view(1, 3, 1, 1)
        self.xtr = torch.from_numpy(np.ascontiguousarray(xtr)).to(DEV)
        self.ytr = torch.from_numpy(self.ytr_np).to(DEV)
        self.xte = None if xte is None else torch.from_numpy(np.ascontiguousarray(xte)).to(DEV)
        self._idx = {k: torch.from_numpy(sp[k]).to(DEV) for k in ("task", "dev", "aux")}
        self.has_test = xte is not None

    def idx(self, name):
        return self._idx[name]

    def images(self, name):
        sp = self.sp
        if name in ("task", "dev", "aux"):
            return self.xtr[self._idx[name]]
        if name == "test":
            if self.xte is None:
                raise CheckFail("this job does not read the test images")
            return self.xte
        if name == "mem":
            pos = np.concatenate([sp["mem_in_train"], sp["mem_in_test"]])
            return self.xtr[torch.from_numpy(sp["task"][pos]).to(DEV)]
        raise KeyError(name)

    def labels(self, name):
        if name in ("task", "dev", "aux"):
            return self.ytr_np[self.sp[name]]
        if name == "test":
            if self.yte_np is None:
                raise CheckFail("test labels are read only by J6")
            return self.yte_np
        raise KeyError(name)

    def norm(self, x):
        return (x.float() / 255.0 - self.mean) / self.std

    def augment(self, xb, g):
        """Random crop after a zero pad of 4 pixels, and a random horizontal flip (design 3.2)."""
        B = xb.shape[0]
        xp = F.pad(xb.float() / 255.0, (4, 4, 4, 4))
        ox = torch.randint(0, 9, (B,), generator=g)
        oy = torch.randint(0, 9, (B,), generator=g)
        fl = torch.randint(0, 2, (B,), generator=g).bool()
        ar = torch.arange(32)
        rows = oy[:, None] + ar
        cols = ox[:, None] + ar
        cols = torch.where(fl[:, None], cols.flip(1), cols)
        rows, cols = rows.to(xb.device), cols.to(xb.device)
        bi = torch.arange(B, device=xb.device)[:, None, None, None]
        ci = torch.arange(3, device=xb.device)[None, :, None, None]
        x = xp[bi, ci, rows[:, None, :, None], cols[:, None, None, :]]
        return (x - self.mean) / self.std


def load_data(mode):
    """J0 and J1 open the train file only. J2 to J5 also open the test images. Only J6 opens the test labels."""
    tr = np.load(DATA_DIR / "cifar10_train.npz", allow_pickle=False)
    xtr, ytr = tr["x"], tr["y"]
    xte = yte = None
    if mode in ("confirm", "analysis"):
        xte = np.load(DATA_DIR / "cifar10_test_images.npz", allow_pickle=False)["x"]
    if mode == "analysis":
        yte = np.load(DATA_DIR / "cifar10_test_labels.npz", allow_pickle=False)["y"]
    log(f"data: mode {mode}; files opened: train" + (", test images" if xte is not None else "")
        + (", test labels" if yte is not None else ""))
    return Data(xtr, ytr, xte, yte, mode)


# ------------------------------------------------------------------------------------------------ bits

def idx_bits(k):
    """ceil(log2(10!/(10-k)!)) bits for an ordered list of k different classes (design 5.1, rule 6)."""
    return (math.perm(10, k) - 1).bit_length()


def perm_rank(cls):
    """Lexicographic rank of each ordered list of k different classes among the 10!/(10-k)! lists."""
    cls = cls.to(torch.int64).cpu()
    B, k = cls.shape
    used = torch.zeros(B, 10, dtype=torch.bool)
    ar = torch.arange(10)
    r = torch.zeros(B, dtype=torch.int64)
    rows = torch.arange(B)
    for i in range(k):
        c = cls[:, i]
        d = c - (used & (ar[None, :] < c[:, None])).sum(1)
        r = r * (10 - i) + d
        used[rows, c] = True
    return r


def perm_unrank(r, k):
    r = r.to(torch.int64).cpu().clone()
    B = r.shape[0]
    d = torch.zeros(B, k, dtype=torch.int64)
    for i in range(k - 1, -1, -1):
        d[:, i] = r % (10 - i)
        r = r // (10 - i)
    used = torch.zeros(B, 10, dtype=torch.bool)
    cls = torch.zeros(B, k, dtype=torch.int64)
    rows = torch.arange(B)
    for i in range(k):
        cum = (~used).to(torch.int64).cumsum(1) - 1
        sel = ((cum == d[:, i:i + 1]) & ~used).to(torch.int64).argmax(1)
        cls[:, i] = sel
        used[rows, sel] = True
    return cls


def _runs(widths):
    out, i = [], 0
    while i < len(widths):
        j = i
        while j < len(widths) and widths[j] == widths[i]:
            j += 1
        out.append((i, j, widths[i]))
        i = j
    return out


def pack(fields, widths):
    """Fixed-length code: each field as `width` bits, most significant bit first, in field order."""
    fields = fields.to(torch.int64).cpu()
    if fields.shape[1] != len(widths):
        raise CheckFail("V4: field count differs from the declared code")
    cols = []
    for i, j, w in _runs(widths):
        f = fields[:, i:j]
        if bool((f < 0).any()) or bool((f >= (1 << w)).any()):
            raise CheckFail("V4: a field is outside its range")
        sh = torch.arange(w - 1, -1, -1, dtype=torch.int64)
        cols.append(((f[:, :, None] >> sh) & 1).reshape(f.shape[0], -1).to(torch.uint8))
    return torch.cat(cols, 1)


def unpack(bits, widths):
    if bits.shape[1] != sum(widths):
        raise CheckFail("V4: the bit string has the wrong length")
    out, pos = [], 0
    for i, j, w in _runs(widths):
        n = j - i
        b = bits[:, pos:pos + n * w].reshape(bits.shape[0], n, w).to(torch.int64)
        out.append((b << torch.arange(w - 1, -1, -1, dtype=torch.int64)).sum(2))
        pos += n * w
    return torch.cat(out, 1)


def to_packed(bits):
    return np.packbits(bits.numpy(), axis=1)


def from_packed(packed, nbits):
    return torch.from_numpy(np.unpackbits(packed, axis=1, count=nbits))


def fit_cells(values, q):
    """2^q cells of equal probability (quantile edges); each cell decodes to the mean of its values (rule 7).
    Cell j holds the values v with edge[j-1] < v <= edge[j], so a value equal to an edge goes to the lower cell
    (exact zeros of a ReLU then decode to 0). An empty cell decodes to the midpoint of its edges."""
    v = np.sort(np.asarray(values, dtype=np.float64).ravel())
    m = 1 << q
    if v.size == 0:  # a slot with no D_task value (not possible with a trained head): all edges and levels 0
        return np.zeros(m - 1, np.float32), np.zeros(m, np.float32)
    edges = np.quantile(v, np.arange(1, m) / m)
    cell = np.searchsorted(edges, v, side="left")
    cnt = np.bincount(cell, minlength=m)
    tot = np.bincount(cell, weights=v, minlength=m)
    lo = np.concatenate([[v[0]], edges])
    hi = np.concatenate([edges, [v[-1]]])
    levels = np.where(cnt > 0, tot / np.maximum(cnt, 1), (lo + hi) / 2)
    return edges.astype(np.float32), levels.astype(np.float32)


def f32_to_fields(x):
    return x.detach().float().cpu().contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF


def fields_to_f32(f):
    f = f.to(torch.int64)
    f = torch.where(f >= 2 ** 31, f - 2 ** 32, f)
    return f.to(torch.int32).view(torch.float32)


# ------------------------------------------------------------------------------------------------ message specs
# A spec is the free part that client and server agree on before deployment (rule 5): it has the code
# widths, the quantizer cells and the decode. encode() returns int64 fields on the CPU; decode() reads the
# fields that unpack() gives and works on the CPU.

class RawSpec:
    kind = "raw"

    def __init__(self, c, s, q, chans, edges, levels):
        self.c, self.s, self.q = int(c), int(s), int(q)
        self.chans = torch.as_tensor(chans, dtype=torch.int64).cpu()
        self.edges = torch.as_tensor(edges, dtype=torch.float32).cpu()
        self.levels = torch.as_tensor(levels, dtype=torch.float32).cpu()
        self.widths = [self.q] * (self.c * self.s * self.s)
        self.bits = sum(self.widths)

    def values(self, h2):
        return F.adaptive_avg_pool2d(h2[:, self.chans.to(h2.device)], self.s)

    def _cells(self, v):
        B = v.shape[0]
        flat = v.permute(1, 0, 2, 3).reshape(self.c, -1).contiguous()
        cell = torch.searchsorted(self.edges.to(v.device), flat, right=False)
        return cell.reshape(self.c, B, self.s, self.s).permute(1, 0, 2, 3)

    def encode(self, h2):
        return self._cells(self.values(h2)).reshape(h2.shape[0], -1).to(torch.int64).cpu()

    def _levels(self, cell, lv):
        B = cell.shape[0]
        flat = cell.permute(1, 0, 2, 3).reshape(self.c, -1)
        return torch.gather(lv, 1, flat).reshape(self.c, B, self.s, self.s).permute(1, 0, 2, 3).contiguous()

    def decode(self, fields):
        return self._levels(fields.reshape(fields.shape[0], self.c, self.s, self.s), self.levels)

    def quantize_dev(self, h2):
        """Training path on the device; equal to decode(encode(h2))."""
        return self._levels(self._cells(self.values(h2)), self.levels.to(h2.device))

    def dec_in(self, v):
        return v.reshape(v.shape[0], -1) if self.s == 1 else v

    def state(self):
        return {"kind": "raw", "c": self.c, "s": self.s, "q": self.q, "chans": self.chans,
                "edges": self.edges, "levels": self.levels}


class BotSpec:
    kind = "bot"

    def __init__(self, c, s, q, float_mode=False):
        self.c, self.s, self.q, self.float_mode = int(c), int(s), int(q), bool(float_mode)
        w = 32 if self.float_mode else self.q
        self.widths = [w] * (self.c * self.s * self.s)
        self.bits = sum(self.widths)

    def encode(self, t):
        B = t.shape[0]
        if self.float_mode:
            return f32_to_fields(t.reshape(B, -1))
        L = (1 << self.q) - 1
        return torch.round((t + 1) / 2 * L).clamp(0, L).reshape(B, -1).to(torch.int64).cpu()

    def decode(self, fields):
        B = fields.shape[0]
        if self.float_mode:
            v = fields_to_f32(fields)
        else:
            v = -1.0 + 2.0 * fields.to(torch.float32) / ((1 << self.q) - 1)
        return v.reshape(B, self.c, self.s, self.s)

    def ste(self, t):
        """Uniform levels on [-1, 1] with a straight-through gradient (design 4.1)."""
        if self.float_mode:
            return t
        L = (1 << self.q) - 1
        tq = -1.0 + 2.0 * torch.round((t + 1) / 2 * L) / L
        return t + (tq - t).detach()

    def dec_in(self, v):
        return v.reshape(v.shape[0], -1) if self.s == 1 else v

    def state(self):
        return {"kind": "bot", "c": self.c, "s": self.s, "q": self.q, "float_mode": self.float_mode}


class TopkSpec:
    kind = "topk"

    def __init__(self, form, k, p, edges, levels, gfill=0.0):
        self.form, self.k, self.p = form, int(k), int(p)
        self.edges = torch.as_tensor(edges, dtype=torch.float32).cpu()
        self.levels = torch.as_tensor(levels, dtype=torch.float32).cpu()
        self.gfill = float(gfill)
        self.widths = [idx_bits(self.k)] + [self.p] * (self.k - 1) if form == "top" else [4] + [self.p] * 9
        self.bits = sum(self.widths)

    @staticmethod
    def fit(form, k, p, z):
        """One set of gap cells for each slot (each rank in F-top, each class in F-all), fit on D_task logits."""
        z = np.asarray(z, dtype=np.float64)
        order = np.argsort(-z, axis=1, kind="stable")
        zs = np.take_along_axis(z, order, 1)
        if form == "top":
            cells = [fit_cells(zs[:, 0] - zs[:, j], p) for j in range(1, k)]
        else:
            a = order[:, 0]
            top = z[np.arange(len(z)), a]
            cells = [fit_cells((top - z[:, c])[a != c], p) for c in range(10)]
        return TopkSpec(form, k, p, np.stack([e for e, _ in cells]), np.stack([lv for _, lv in cells]))

    def last_level(self):
        """The largest decoded level of the last sent gap (g_fill base, design 5.3)."""
        return float(self.levels[self.k - 2].max())

    def encode(self, z):
        z = z.detach().float().cpu()
        B = z.shape[0]
        srt = torch.sort(z, dim=1, descending=True, stable=True)
        if self.form == "top":
            cls = srt.indices[:, :self.k]
            gaps = (srt.values[:, :1] - srt.values[:, 1:self.k]).T.contiguous()
            cell = torch.searchsorted(self.edges, gaps, right=False).T
            return torch.cat([perm_rank(cls)[:, None], cell.to(torch.int64)], 1)
        a = srt.indices[:, 0]
        gaps = (z.gather(1, a[:, None]) - z).T.contiguous()
        cell = torch.searchsorted(self.edges, gaps, right=False).T
        keep = torch.ones(B, 10, dtype=torch.bool)
        keep[torch.arange(B), a] = False
        return torch.cat([a[:, None], cell[keep].reshape(B, 9).to(torch.int64)], 1)

    def decode(self, fields):
        B = fields.shape[0]
        if self.form == "top":
            cls = perm_unrank(fields[:, 0], self.k)
            z = torch.full((B, 10), -self.gfill)
            z.scatter_(1, cls[:, :1], 0.0)
            if self.k > 1:
                gaps = torch.stack([self.levels[j][fields[:, 1 + j]] for j in range(self.k - 1)], 1)
                z.scatter_(1, cls[:, 1:], -gaps)
            mask = torch.zeros(B, 10).scatter_(1, cls, 1.0)
            return z, mask, cls[:, 0].clone()
        a = fields[:, 0]
        ar = torch.arange(10).expand(B, 10)
        others = ar[ar != a[:, None]].reshape(B, 9)
        gaps = self.levels[others, fields[:, 1:]]
        z = torch.zeros(B, 10).scatter_(1, others, -gaps)
        return z, torch.ones(B, 10), a.clone()

    def pred(self, d):
        """The server's top-1 prediction is the first class in the message (design 5.3)."""
        return d[2].cpu()

    def dec_in(self, d):
        return torch.cat([d[0], d[1]], 1)

    def state(self):
        return {"kind": "topk", "form": self.form, "k": self.k, "p": self.p, "edges": self.edges,
                "levels": self.levels, "gfill": self.gfill}


class LabSpec:
    kind = "lab"

    def __init__(self, conf):
        self.conf = torch.as_tensor(conf, dtype=torch.float32).cpu()
        self.widths = [4]
        self.bits = 4

    @staticmethod
    def fit(pred_dev, y_dev):
        """The D_dev confusion row of each sent label, with add-one smoothing (design 4.1)."""
        cnt = np.zeros((10, 10))
        np.add.at(cnt, (np.asarray(pred_dev), np.asarray(y_dev)), 1)
        return LabSpec((cnt + 1) / (cnt.sum(1, keepdims=True) + 10))

    def encode(self, z):
        return torch.sort(z.detach().float().cpu(), dim=1, descending=True, stable=True).indices[:, :1].to(torch.int64)

    def decode(self, fields):
        return fields[:, 0].clone()

    def logits(self, lab):
        return torch.log(self.conf.to(lab.device)[lab])

    def pred(self, lab):
        """The label is the prediction (design 4.1)."""
        return lab.cpu()

    def dec_in(self, lab):
        return F.one_hot(lab, 10).float()

    def state(self):
        return {"kind": "lab", "conf": self.conf}


class FloatSpec:
    """TOPK-float: the 10 logits as float32 (320 bits)."""
    kind = "float"

    def __init__(self):
        self.widths = [32] * 10
        self.bits = 320

    def encode(self, z):
        return f32_to_fields(z)

    def decode(self, fields):
        return fields_to_f32(fields)

    def dec_in(self, z):
        return torch.cat([z, torch.ones_like(z)], 1)

    def state(self):
        return {"kind": "float"}


def spec_from_state(st):
    k = st["kind"]
    if k == "raw":
        return RawSpec(st["c"], st["s"], st["q"], st["chans"], st["edges"], st["levels"])
    if k == "bot":
        return BotSpec(st["c"], st["s"], st["q"], st["float_mode"])
    if k == "topk":
        return TopkSpec(st["form"], st["k"], st["p"], st["edges"], st["levels"], st["gfill"])
    if k == "lab":
        return LabSpec(st["conf"])
    if k == "float":
        return FloatSpec()
    raise ValueError(k)


def topk_name(form, k, p):
    return f"F-all({p})" if form == "all" else f"F-top({k},{p})"


# ------------------------------------------------------------------------------------------------ models

def conv_bn(ci, co):
    return [nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True)]


class Net(nn.Module):
    """Network N: three stages of two 3x3 conv layers (BN, ReLU) and a 2x2 max pool, then GAP and a linear head."""

    def __init__(self):
        super().__init__()
        self.s1 = nn.Sequential(*conv_bn(3, 32), *conv_bn(32, 32), nn.MaxPool2d(2))
        self.s2 = nn.Sequential(*conv_bn(32, 64), *conv_bn(64, 64), nn.MaxPool2d(2))
        self.s3 = nn.Sequential(*conv_bn(64, 128), *conv_bn(128, 128), nn.MaxPool2d(2))
        self.head = nn.Linear(128, 10)

    def client(self, x):
        return self.s2(self.s1(x))

    def server(self, h):
        return self.head(self.s3(h).mean((2, 3)))

    def forward(self, x):
        return self.server(self.client(x))


class ServerPart(nn.Module):
    """Stage 3 and head, initialized from N."""

    def __init__(self, net):
        super().__init__()
        self.s3 = copy.deepcopy(net.s3)
        self.head = copy.deepcopy(net.head)

    def forward(self, h):
        return self.head(self.s3(h).mean((2, 3)))


class RawModel(nn.Module):
    """Server of RAW: nearest upsample to 8x8, zeros for the dropped channels, then stage 3 and head."""

    def __init__(self, net, spec):
        super().__init__()
        self.server = ServerPart(net)
        self.register_buffer("chans", spec.chans.clone())

    def forward(self, v):
        if v.shape[-1] != 8:
            v = F.interpolate(v, size=(8, 8), mode="nearest")
        full = v.new_zeros(v.shape[0], 64, 8, 8).index_copy(1, self.chans, v)
        return self.server(full)


class BotModel(nn.Module):
    """BOT: 1x1 conv 64->c', average pool to s'xs', tanh, quantizer (client); 1x1 conv c'->64, BN, ReLU,
    nearest upsample to 8x8, stage 3 and head (server)."""

    def __init__(self, net, spec):
        super().__init__()
        self.spec = spec
        self.enc = nn.Conv2d(64, spec.c, 1)
        self.adapt = nn.Sequential(nn.Conv2d(spec.c, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.server = ServerPart(net)

    def code(self, h2):
        return torch.tanh(F.adaptive_avg_pool2d(self.enc(h2), self.spec.s))

    def server_logits(self, v):
        a = self.adapt(v)
        if a.shape[-1] != 8:
            a = F.interpolate(a, size=(8, 8), mode="nearest")
        return self.server(a)

    def forward(self, h2):
        return self.server_logits(self.spec.ste(self.code(h2)))


class TopkHead(nn.Module):
    """TOPK client head: 1x1 conv 64->16, BN, ReLU, flatten (1,024), linear 1,024->256, ReLU, linear 256->10."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(64, 16, 1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, h2):
        h = F.relu(self.bn(self.conv(h2))).flatten(1)
        return self.fc2(F.relu(self.fc1(h)))


class Decoder(nn.Module):
    """Decoder-S (width 1) or Decoder-L (width 2): four stages from 4x4 to 32x32 (design 6.2)."""

    def __init__(self, in_shape, width=1):
        super().__init__()
        ch = [256 * width, 128 * width, 64 * width, 32 * width]
        self.ch0 = ch[0]
        self.vector = len(in_shape) == 1
        self.up_in = 1
        if self.vector:
            self.inp = nn.Linear(in_shape[0], ch[0] * 16)
            start = 0
        else:
            c, s = in_shape[0], in_shape[1]
            if s in (2, 4):
                self.up_in, start = 4 // s, 0
            elif s == 8:
                start = 1
            else:
                raise ValueError(f"map size {s}")
            self.inp = nn.Conv2d(c, ch[0], 1)
        stages = []
        for i in range(start, 4):
            cin = ch[0] if i == 0 else ch[i - 1]
            stages.append(nn.Sequential(*conv_bn(cin, ch[i]), *conv_bn(ch[i], ch[i])))
        self.stages = nn.ModuleList(stages)
        self.out = nn.Conv2d(ch[3], 3, 3, padding=1)

    def forward(self, m):
        if self.vector:
            h = self.inp(m).view(m.shape[0], self.ch0, 4, 4)
        else:
            if self.up_in > 1:
                m = F.interpolate(m, scale_factor=self.up_in, mode="nearest")
            h = self.inp(m)
        for j, st in enumerate(self.stages):
            if j > 0:
                h = F.interpolate(h, scale_factor=2, mode="nearest")
            h = st(h)
        return torch.sigmoid(self.out(h))


def client_sha(net):
    """V3: sha256 of the parameters and buffers of stages 1 and 2."""
    h = hashlib.sha256()
    for pre, mod in (("s1", net.s1), ("s2", net.s2)):
        for k, v in sorted(mod.state_dict().items()):
            h.update(f"{pre}.{k}".encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def conv_macs(ci, co, k, hw):
    return ci * co * k * k * hw


def arm_macs(name, cfg):
    """Client and server MACs for each image (design 4.1; bias, BN, ReLU and pool omitted)."""
    st12 = conv_macs(3, 32, 3, 1024) + conv_macs(32, 32, 3, 1024) + conv_macs(32, 64, 3, 256) + conv_macs(64, 64, 3, 256)
    st3 = conv_macs(64, 128, 3, 64) + conv_macs(128, 128, 3, 64) + 128 * 10
    head = 64 * 16 * 64 + 1024 * 256 + 256 * 10
    fam = name.split("@")[0]
    if fam == "RAW" or name == "RAW_native":
        return st12, st3
    if fam == "BOT" or name == "BOT_float":
        c, s = cfg["c"], cfg["s"]
        return st12 + 64 * c * 64, c * 64 * s * s + st3
    if fam in ("TOPK", "LAB") or name == "TOPK_float":
        return st12 + head, 0
    if name in ("TOPK_full", "LAB_full"):
        return st12 + st3, 0
    raise ValueError(name)


# ------------------------------------------------------------------------------------------------ arms

class Arm:
    """spec plus the client part before encode (pre) and the server part after decode (post)."""

    def __init__(self, name, spec, pre, post, T=1.0, cfg=None):
        self.name, self.spec, self.pre, self.post, self.T = name, spec, pre, post, float(T)
        self.cfg = cfg or {}


def to_dev(d):
    if isinstance(d, tuple):
        return tuple(t.to(DEV) for t in d)
    return d.to(DEV)


@torch.no_grad()
def arm_messages(arm, client, data, x_u8, flip=False, bs=500, want_dec=True):
    """Client -> fields -> pack -> unpack -> decode -> server, in batches. V4 is checked on every batch."""
    packed, decs, logits, preds = [], [], [], []
    for i in range(0, len(x_u8), bs):
        x = data.norm(x_u8[i:i + bs])
        if flip:
            x = x.flip(3)
        f = arm.spec.encode(arm.pre(client(x)))
        bits = pack(f, arm.spec.widths)
        if bits.shape[1] != arm.spec.bits:
            raise CheckFail(f"V4: {arm.name} wrote {bits.shape[1]} bits, declared {arm.spec.bits}")
        f2 = unpack(bits, arm.spec.widths)
        if not torch.equal(f, f2):
            raise CheckFail(f"V4: unpack(pack(m)) != m for {arm.name}")
        d = arm.spec.decode(f2)
        logits.append(arm.post(to_dev(d)).float().cpu())
        preds.append(arm.spec.pred(d) if hasattr(arm.spec, "pred") else top1(logits[-1]))
        if want_dec:
            decs.append(arm.spec.dec_in(d).float())
        packed.append(to_packed(bits))
    return {"packed": np.concatenate(packed), "logits": torch.cat(logits), "pred": torch.cat(preds).numpy(),
            "dec_in": torch.cat(decs) if want_dec else None}


def dec_in_from_packed(spec, packed):
    out = []
    step = max(64, (1 << 22) // spec.bits)
    for i in range(0, len(packed), step):
        bits = from_packed(packed[i:i + step], spec.bits)
        if bits.shape[1] != spec.bits:
            raise CheckFail("V4: a saved message has the wrong length")
        f = unpack(bits, spec.widths)
        if not torch.equal(pack(f, spec.widths), bits):
            raise CheckFail("V4: pack(unpack(b)) != b for a saved message")
        out.append(spec.dec_in(spec.decode(f)).float())
    return torch.cat(out)


def fit_T(logits, y):
    """One softmax temperature from the D_dev NLL, grid 0.5 to 3.0 in steps of 0.05 (design 4.2)."""
    lg = logits.double().cpu()
    yy = torch.as_tensor(np.asarray(y), dtype=torch.int64)
    best = None
    for T in TEMPS:
        nll = F.cross_entropy(lg / T, yy).item()
        if best is None or nll < best[1]:
            best = (T, nll)
    return best


def acc_nll(logits, y, T):
    lg = logits.double().cpu()
    yy = torch.as_tensor(np.asarray(y), dtype=torch.int64)
    acc = (lg.argmax(1) == yy).double().mean().item()
    return acc, F.cross_entropy(lg / T, yy).item()


def top1(logits):
    """Prediction from the first class of a sorted message (stable sort; ties go to the lower class index)."""
    return torch.sort(logits.float().cpu(), dim=1, descending=True, stable=True).indices[:, 0]


# ------------------------------------------------------------------------------------------------ training

def train_net(data, seed, epochs=None):
    """SGD with Nesterov momentum 0.9, weight decay 5e-4, batch 128; linear warm-up to 0.05 over 2 epochs, then cosine."""
    epochs = epochs or N_EPOCHS
    torch.manual_seed(sub_seed(seed, "N/init"))
    net = Net().to(DEV)
    g = gen(sub_seed(seed, "N/data"))
    opt = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9, nesterov=True, weight_decay=5e-4)
    task = data.idx("task")
    n = len(task)
    spe = n // BATCH
    total, warm = epochs * spe, min(2 * spe, epochs * spe)
    step = 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, generator=g).to(DEV)
        tot = torch.zeros((), device=DEV)
        for b in range(spe):
            lr = 0.05 * (step + 1) / warm if step < warm else cosine(0.05, step - warm, max(total - warm, 1))
            for grp in opt.param_groups:
                grp["lr"] = lr
            bi = task[perm[b * BATCH:(b + 1) * BATCH]]
            loss = F.cross_entropy(net(data.augment(data.xtr[bi], g)), data.ytr[bi])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.detach()
            step += 1
        finite_or_fail(tot, f"N epoch {ep + 1}")
        log(f"N epoch {ep + 1}/{epochs}: mean loss {tot.item() / spe:.4f}")
    net.eval()
    return net


def train_part(module, new_params, old_params, fwd, data, client, seed, tag, epochs=None):
    """AdamW, weight decay 1e-4, batch 128, cosine decay; 1e-3 for new layers, 3e-4 for layers from N (design 4.2)."""
    epochs = epochs or ARM_EPOCHS
    g = gen(sub_seed(seed, f"{tag}/data"))
    groups = []
    if new_params:
        groups.append({"params": new_params, "lr": 1e-3, "base": 1e-3})
    if old_params:
        groups.append({"params": old_params, "lr": 3e-4, "base": 3e-4})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    task = data.idx("task")
    n = len(task)
    spe = n // BATCH
    total = epochs * spe
    step = 0
    t_epochs = []
    for ep in range(epochs):
        t0 = time.time()
        module.train()
        perm = torch.randperm(n, generator=g).to(DEV)
        tot = torch.zeros((), device=DEV)
        for b in range(spe):
            for grp in opt.param_groups:
                grp["lr"] = cosine(grp["base"], step, total)
            bi = task[perm[b * BATCH:(b + 1) * BATCH]]
            x = data.augment(data.xtr[bi], g)
            with torch.no_grad():
                h2 = client(x)
            loss = F.cross_entropy(fwd(h2), data.ytr[bi])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.detach()
            step += 1
        finite_or_fail(tot, f"{tag} epoch {ep + 1}")
        sync()
        t_epochs.append(time.time() - t0)
        log(f"{tag} epoch {ep + 1}/{epochs}: mean loss {tot.item() / spe:.4f} ({t_epochs[-1]:.1f} s)")
    module.eval()
    return t_epochs


def train_decoder(Xtr, Ttr, Xsel, Tsel, width, seed, epochs=None):
    """Adam 1e-3 with cosine decay, batch 128, MSE on pixels in [0, 1]; keep the epoch with the lowest D_dev
    MSE, checked every 2 epochs (design 6.2). Inputs are standardized per element on the training inputs."""
    epochs = epochs or DEC_EPOCHS
    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, unbiased=False, keepdim=True)
    scale = torch.where(sd > 1e-6, 1.0 / sd.clamp_min(1e-6), torch.zeros_like(sd))

    def norm(X):
        return ((X - mu) * scale).to(DEV)

    torch.manual_seed(sub_seed(seed, "init"))
    model = Decoder(tuple(Xtr.shape[1:]), width).to(DEV)
    g = gen(sub_seed(seed, "data"))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xd, Td, Xs = norm(Xtr), Ttr.to(DEV), norm(Xsel)
    n = len(Xd)
    spe = n // BATCH
    total = epochs * spe
    step = 0
    best = (float("inf"), None, 0)
    hist, t_train, t_eval = [], [], []
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n, generator=g).to(DEV)
        tot = torch.zeros((), device=DEV)
        for b in range(spe):
            for grp in opt.param_groups:
                grp["lr"] = cosine(1e-3, step, total)
            bi = perm[b * BATCH:(b + 1) * BATCH]
            loss = F.mse_loss(model(Xd[bi]), Td[bi].float() / 255.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.detach()
            step += 1
        finite_or_fail(tot, f"decoder epoch {ep + 1}")
        sync()
        t_train.append(time.time() - t0)
        if (ep + 1) % 2 == 0 or (epochs < 2 and ep == epochs - 1):
            t0 = time.time()
            mse = dec_mse(model, Xs, Tsel)
            t_eval.append(time.time() - t0)
            hist.append({"epoch": ep + 1, "train_mse": tot.item() / spe, "dev_mse": mse})
            if mse < best[0]:
                best = (mse, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep + 1)
            log(f"  decoder epoch {ep + 1}: train MSE {tot.item() / spe:.5f}, D_dev MSE {mse:.5f}")
    model.load_state_dict(best[1])
    model.eval()
    return model, norm, {"best_epoch": best[2], "best_dev_mse": best[0], "hist": hist,
                         "t_train_epoch": t_train, "t_eval": t_eval}


@torch.no_grad()
def dec_mse(model, Xs, T_u8, bs=500):
    model.eval()
    tot = 0.0
    for i in range(0, len(Xs), bs):
        out = model(Xs[i:i + bs])
        tot += F.mse_loss(out, T_u8[i:i + bs].to(DEV).float() / 255.0, reduction="sum").item()
    return tot / (len(Xs) * 3 * 32 * 32)


@torch.no_grad()
def reconstruct(model, Xn, bs=500):
    model.eval()
    out = []
    for i in range(0, len(Xn), bs):
        out.append((model(Xn[i:i + bs]) * 255.0).round().clamp(0, 255).to(torch.uint8).cpu())
    return torch.cat(out).numpy()


# ------------------------------------------------------------------------------------------------ metrics

def psnr_u8(rec, tgt):
    """PSNR = 10 log10(1 / MSE) for each image, MSE over the 3x32x32 values in [0, 1]. Returns (psnr, mse)."""
    d = rec.astype(np.float64) / 255.0 - tgt.astype(np.float64) / 255.0
    mse = (d * d).reshape(len(d), -1).mean(1)
    return 10.0 * np.log10(1.0 / np.maximum(mse, 1e-10)), mse


def ssim_u8(rec, tgt, bs=1000):
    """SSIM: 11x11 Gaussian window, sigma 1.5, K1 0.01, K2 0.03, data range 1, each RGB channel over the
    22x22 valid positions, then the mean (design 6.2)."""
    g = torch.exp(-((torch.arange(11, dtype=torch.float64) - 5) ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :]).expand(3, 1, 11, 11).contiguous()
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    out = []
    for i in range(0, len(rec), bs):
        x = torch.from_numpy(np.asarray(rec[i:i + bs])).double() / 255.0
        y = torch.from_numpy(np.asarray(tgt[i:i + bs])).double() / 255.0

        def cv(t):
            return F.conv2d(t, w, groups=3)

        mx, my = cv(x), cv(y)
        sxx, syy, sxy = cv(x * x) - mx * mx, cv(y * y) - my * my, cv(x * y) - mx * my
        m = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx * mx + my * my + C1) * (sxx + syy + C2))
        out.append(m.mean((1, 2, 3)).numpy())
    return np.concatenate(out)


def boot_means(d, draws=N_BOOT, seed=BOOT_SEED):
    """Paired bootstrap over images: the same index draws for every call with the same n."""
    d = np.asarray(d, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n, out = len(d), []
    for c in range(0, draws, BOOT_CHUNK):
        idx = rng.integers(0, n, size=(min(BOOT_CHUNK, draws - c), n))
        out.append(d[idx].mean(1))
    return np.concatenate(out)


def ci95(m):
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def auroc(score, label):
    """AUROC by ranks, with the mean rank for ties."""
    score = np.asarray(score, dtype=np.float64)
    label = np.asarray(label).astype(bool)
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    avg = np.cumsum(cnt) - (cnt - 1) / 2.0
    r = avg[inv]
    n1 = label.sum()
    n0 = len(label) - n1
    return float((r[label].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def auroc_boot(score, label, draws=N_BOOT, seed=BOOT_SEED):
    """AUROC on each bootstrap draw (the same draws as boot_means for the same n); ties count one half."""
    score = np.asarray(score, dtype=np.float64)
    label = np.asarray(label).astype(bool)
    _, inv = np.unique(score, return_inverse=True)
    G = int(inv.max()) + 1
    rng = np.random.default_rng(seed)
    n, out = len(score), []
    for c in range(0, draws, BOOT_CHUNK):
        D = min(BOOT_CHUNK, draws - c)
        idx = rng.integers(0, n, size=(D, n))
        gi = inv[idx] + (np.arange(D)[:, None] * G)
        lab = label[idx]
        wp = np.bincount(gi[lab], minlength=D * G).reshape(D, G).astype(np.float64)
        wn = np.bincount(gi[~lab], minlength=D * G).reshape(D, G).astype(np.float64)
        below = np.cumsum(wn, 1) - wn
        out.append((wp * (below + 0.5 * wn)).sum(1) / (wp.sum(1) * wn.sum(1)))
    return np.concatenate(out)


def train_probe(Xtr, ytr, Xsel, ysel, Xte, ncls, seed, epochs=None, select=True, bs=BATCH):
    """MLP input -> 256 -> 256 -> classes (ReLU), Adam 1e-3, 30 epochs, epoch with the best D_dev accuracy."""
    epochs = epochs or PROBE_EPOCHS
    Xtr, Xte = Xtr.float().cpu(), Xte.float().cpu()
    mu, sd = Xtr.mean(0), Xtr.std(0, unbiased=False)
    sc = torch.where(sd > 1e-6, 1.0 / sd.clamp_min(1e-6), torch.zeros_like(sd))

    def f(X):
        return (X.float().cpu() - mu) * sc

    torch.manual_seed(sub_seed(seed, "init"))
    m = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, ncls))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    g = gen(sub_seed(seed, "data"))
    Xn, yt = f(Xtr), torch.as_tensor(np.asarray(ytr), dtype=torch.int64)
    Xs = f(Xsel) if select else None
    ys = torch.as_tensor(np.asarray(ysel), dtype=torch.int64) if select else None
    best = (-1.0, None, epochs)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(Xn), generator=g)
        for b in range(0, len(Xn), bs):
            bi = perm[b:b + bs]
            loss = F.cross_entropy(m(Xn[bi]), yt[bi])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if select:
            m.eval()
            with torch.no_grad():
                acc = (m(Xs).argmax(1) == ys).double().mean().item()
            if acc > best[0]:
                best = (acc, {k: v.clone() for k, v in m.state_dict().items()}, ep + 1)
    if select:
        m.load_state_dict(best[1])
    m.eval()
    with torch.no_grad():
        return m(f(Xte)), best[2]


# ------------------------------------------------------------------------------------------------ reading rule

CASES = {("A+", "L+"): "S", ("A+", "L="): "E", ("A+", "L-"): "X", ("A+", "L?"): "U",
         ("A-", "L+"): "T", ("A-", "L="): "A", ("A-", "L-"): "A", ("A-", "L?"): "U"}
CASE_TEXT = {"S": "symbolic dominates", "E": "equal", "X": "activation leaks less, accuracy kept",
             "T": "trade-off", "A": "activation dominates", "U": "unresolved", "0": "invalid"}


def compare(c_sym, c_act, p_sym, p_act, acc_margin=1.0, leak_margin=0.5):
    """Design 8.1 and 8.4. c_*: [seeds, images] correctness; p_*: [seeds, images] leakage score (PSNR or SSIM).
    Delta acc favors SYM when positive; Delta leak is positive when ACT leaks more."""
    c_sym, c_act = np.asarray(c_sym, np.float64), np.asarray(c_act, np.float64)
    p_sym, p_act = np.asarray(p_sym, np.float64), np.asarray(p_act, np.float64)
    d_acc = 100.0 * (c_sym.mean(0) - c_act.mean(0))
    d_leak = p_act.mean(0) - p_sym.mean(0)
    acc, leak = float(d_acc.mean()), float(d_leak.mean())
    acc_ci, leak_ci = ci95(boot_means(d_acc)), ci95(boot_means(d_leak))
    acc_s = (100.0 * (c_sym - c_act).mean(1)).tolist()
    leak_s = (p_act - p_sym).mean(1).tolist()
    if acc_ci[0] > -acc_margin:
        A = "A+"
    elif acc <= -acc_margin and acc_ci[1] < 0 and all(v < 0 for v in acc_s):
        A = "A-"
    else:
        A = "A?"
    lp = leak >= leak_margin and leak_ci[0] > 0 and all(v > 0 for v in leak_s)
    le = leak_ci[0] > -leak_margin and leak_ci[1] < leak_margin
    lm = leak <= -leak_margin and leak_ci[1] < 0 and all(v < 0 for v in leak_s)
    hits = [n for n, h in (("L+", lp), ("L=", le), ("L-", lm)) if h]
    L = hits[0] if len(hits) == 1 else "L?"
    case = CASES.get((A, L), "U")
    out = {"d_acc": acc, "d_acc_ci": acc_ci, "d_acc_seeds": acc_s, "d_leak": leak, "d_leak_ci": leak_ci,
           "d_leak_seeds": leak_s, "A": A, "L": L, "case": case, "L_hits": hits}
    if case == "T":
        out["rate_db_per_point"] = leak / abs(acc)
    return out


def nll_beats(n_lab, n_topk):
    """R8 rule for NLL: TOPK beats LAB if NLL(LAB) - NLL(TOPK) >= 0.02 nats, CI low > 0, and > 0 in each seed."""
    d = np.asarray(n_lab, np.float64) - np.asarray(n_topk, np.float64)
    m, c = float(d.mean(0).mean()), ci95(boot_means(d.mean(0)))
    s = d.mean(1).tolist()
    return {"d_nll": m, "d_nll_ci": c, "d_nll_seeds": s, "topk_beats": bool(m >= 0.02 and c[0] > 0 and all(v > 0 for v in s))}


# ------------------------------------------------------------------------------------------------ job context

class Ctx:
    def __init__(self, job, seed, out, data):
        self.job, self.seed, self.out, self.data = job, seed, Path(out), data
        for sub in ("arms", "msgs", "dec", "done"):
            (self.out / sub).mkdir(parents=True, exist_ok=True)
        self._net = self._head = None
        self.sel = read_json(ROOT / "selection.json") if (ROOT / "selection.json").exists() else None

    @property
    def net(self):
        if self._net is None:
            net = Net().to(DEV)
            net.load_state_dict(torch.load(self.out / "N.pt", weights_only=True, map_location=DEV))
            self._net = net.eval()
        return self._net

    @property
    def head(self):
        if self._head is None:
            h = TopkHead().to(DEV)
            h.load_state_dict(torch.load(self.out / "arms" / "HEAD.pt", weights_only=True, map_location=DEV))
            self._head = h.eval()
        return self._head

    def client(self, x):
        return self.net.client(x)

    def check_client(self, where):
        want = read_json(self.out / "N.json")["client_sha256"]
        got = client_sha(self.net)
        ok = got == want
        append_jsonl(self.out / "checks.jsonl", {"check": "V3", "where": where, "ok": ok, "time": now()})
        if not ok:
            raise CheckFail(f"V3: the frozen client changed ({where})")

    def save_arm(self, name, spec, model_kind, model_state, T, info):
        torch.save({"spec": spec.state(), "model_kind": model_kind, "model": model_state, "T": float(T)},
                   self.out / "arms" / f"{name}.pt")
        write_json(self.out / "arms" / f"{name}.json", dict(info, name=name, T=float(T), bits=spec.bits))

    def resolve(self, name):
        """RAW@B and BOT@B go to the selected candidate; TOPK@1024 is TOPK@256."""
        if name == "TOPK@1024":
            return "TOPK@256"
        fam, _, b = name.partition("@")
        if fam in ("RAW", "BOT") and b:
            return self.sel[fam][b]
        return name

    def arm(self, name):
        real = self.resolve(name)
        st = torch.load(self.out / "arms" / f"{real}.pt", weights_only=True, map_location=DEV)
        spec = spec_from_state({k: v.cpu() if torch.is_tensor(v) else v for k, v in st["spec"].items()})
        kind = st["model_kind"]
        if kind == "raw":
            m = RawModel(self.net, spec).to(DEV)
            m.load_state_dict(st["model"])
            m.eval()
            return Arm(name, spec, lambda h: h, m, st["T"], {"c": spec.c, "s": spec.s, "q": spec.q})
        if kind == "bot":
            m = BotModel(self.net, spec).to(DEV)
            m.load_state_dict(st["model"])
            m.eval()
            return Arm(name, spec, m.code, m.server_logits, st["T"], {"c": spec.c, "s": spec.s, "q": spec.q})
        if kind == "net_server":
            pre = (lambda h: h) if spec.kind == "raw" else self.net.server
            post = self.net.server if spec.kind == "raw" else _post(spec)
            return Arm(name, spec, pre, post, st["T"])
        if kind == "head":
            return Arm(name, spec, self.head, _post(spec), st["T"])
        raise ValueError(kind)


def _post(spec):
    if spec.kind == "topk":
        return lambda d: d[0]
    if spec.kind == "lab":
        return spec.logits
    if spec.kind == "float":
        return lambda z: z
    raise ValueError(spec.kind)


# ------------------------------------------------------------------------------------------------ units

def unit_N(ctx, epochs=None):
    epochs = epochs or N_EPOCHS
    t0 = time.time()
    net = train_net(ctx.data, ctx.seed, epochs)
    sync()
    t_train = time.time() - t0
    ctx._net = net
    with torch.no_grad():
        lg = torch.cat([net(ctx.data.norm(x)).float().cpu() for x in ctx.data.images("dev").split(1000)])
    acc = (top1(lg).numpy() == ctx.data.labels("dev")).mean()
    torch.save(net.state_dict(), ctx.out / "N.pt")
    info = {"dev_acc": float(acc), "train_seconds": t_train, "epochs": epochs, "client_sha256": client_sha(net)}
    write_json(ctx.out / "N.json", info)
    log(f"N: D_dev accuracy {acc:.4f}, {t_train / 60:.1f} min")
    append_jsonl(ctx.out / "checks.jsonl", {"check": "V6", "dev_acc": float(acc), "ok": bool(acc >= V6_MIN), "time": now()})
    return info


def check_v6(ctx):
    acc = read_json(ctx.out / "N.json")["dev_acc"]
    if acc < V6_MIN:
        raise StopRule(f"V6: the D_dev accuracy of N is {acc:.4f}, below {V6_MIN:.2f}")


@torch.no_grad()
def client_states(ctx, name):
    return torch.cat([ctx.client(ctx.data.norm(x)).float().cpu() for x in ctx.data.images(name).split(1000)])


def unit_prep(ctx, raw_cfgs):
    """RAW channel ranking (variance of the global average on D_task, no labels) and the quantile cells of
    each RAW configuration and of RAW-native, all fit on D_task without augmentation."""
    ctx.check_client("prep start")
    h2 = client_states(ctx, "task")
    gap = h2.mean((2, 3)).double().numpy()
    var = gap.var(0)
    order = np.argsort(-var, kind="stable")
    write_json(ctx.out / "raw_rank.json", {"order": order, "variance": var})
    specs = {}
    for (c, s, q) in raw_cfgs:
        chans = order[:c]
        pooled = F.adaptive_avg_pool2d(h2[:, torch.from_numpy(chans)], s).numpy()
        cells = [fit_cells(pooled[:, j], q) for j in range(c)]
        specs[f"RAW_c{c}_s{s}_q{q}"] = RawSpec(c, s, q, chans, np.stack([e for e, _ in cells]),
                                                 np.stack([lv for _, lv in cells]))
    hn = h2.numpy()
    cells = [fit_cells(hn[:, j], 8) for j in range(64)]
    specs["RAW_native"] = RawSpec(64, 8, 8, np.arange(64), np.stack([e for e, _ in cells]), np.stack([lv for _, lv in cells]))
    for k, sp in specs.items():
        torch.save(sp.state(), ctx.out / "arms" / f"{k}.spec.pt")
    ctx.check_client("prep end")


def eval_dev(ctx, arm):
    r = arm_messages(arm, ctx.client, ctx.data, ctx.data.images("dev"), want_dec=False)
    y = ctx.data.labels("dev")
    T = fit_T(r["logits"], y)[0]
    _, nll = acc_nll(r["logits"], y, T)
    return T, float((r["pred"] == y).mean()), nll


def unit_head(ctx, epochs=None):
    ctx.check_client("head start")
    torch.manual_seed(sub_seed(ctx.seed, "HEAD/init"))
    head = TopkHead().to(DEV)
    t_ep = train_part(head, list(head.parameters()), [], head, ctx.data, ctx.client, ctx.seed, "HEAD", epochs)
    torch.save(head.state_dict(), ctx.out / "arms" / "HEAD.pt")
    ctx._head = head.eval()
    t0 = time.time()
    spec = FloatSpec()
    T, acc, nll = eval_dev(ctx, Arm("TOPK_float", spec, head, _post(spec)))
    ctx.save_arm("TOPK_float", spec, "head", None, T, {"dev_acc": acc, "dev_nll": nll, "t_epochs": t_ep})
    ctx.check_client("head end")
    return {"t_epochs": t_ep, "t_eval": time.time() - t0}


@torch.no_grad()
def logits_on(ctx, fn, name):
    return torch.cat([fn(ctx.client(ctx.data.norm(x))).float().cpu() for x in ctx.data.images(name).split(1000)])


def fit_topk_arm(ctx, form, k, p, z_task, pre, mults=None):
    """Fit the cells on D_task; select g_fill (F-top) and the temperature by the D_dev NLL."""
    spec = TopkSpec.fit(form, k, p, z_task.numpy())
    y = ctx.data.labels("dev")
    best = None
    for mult in (mults or (GFILL_MULTS if form == "top" else (0.0,))):
        spec.gfill = mult * spec.last_level() if form == "top" else 0.0
        r = arm_messages(Arm("topk", spec, pre, _post(spec)), ctx.client, ctx.data, ctx.data.images("dev"), want_dec=False)
        T, nll = fit_T(r["logits"], y)
        if best is None or nll < best["nll"]:
            best = {"mult": mult, "gfill": spec.gfill, "T": T, "nll": nll, "acc": float((r["pred"] == y).mean())}
    spec.gfill = best["gfill"]
    return spec, best


def unit_topk(ctx, fixed=None):
    """TOPK formats on the head logits and LAB. fixed = the J1 selection (seeds 2 and 3)."""
    ctx.check_client("topk start")
    z_task = logits_on(ctx, ctx.head, "task")
    table, chosen = {}, {}
    for b in (16, 64, 256):
        cands = TOPK_CANDS[b] if fixed is None else [tuple(fixed[str(b)][:3])]
        res = []
        for (form, k, p) in cands:
            mults = None if fixed is None else (fixed[str(b)][3],)
            spec, best = fit_topk_arm(ctx, form, k, p, z_task, ctx.head, mults)
            res.append((best["nll"], len(res), spec, best, (form, k, p)))
            table[f"{b}:{topk_name(form, k, p)}"] = dict(best, bits=spec.bits)
        nll, _, spec, best, fkp = min(res, key=lambda r: (r[0], r[1]))
        chosen[str(b)] = [fkp[0], fkp[1], fkp[2], best["mult"]]
        ctx.save_arm(f"TOPK@{b}", spec, "head", None, best["T"],
                     {"dev_acc": best["acc"], "dev_nll": best["nll"], "format": topk_name(*fkp), "gfill_mult": best["mult"]})
    chosen["1024"] = chosen["256"]
    y = ctx.data.labels("dev")
    zd = logits_on(ctx, ctx.head, "dev")
    lab = LabSpec.fit(top1(zd).numpy(), y)
    T, acc, nll = eval_dev(ctx, Arm("LAB", lab, ctx.head, _post(lab)))
    ctx.save_arm("LAB", lab, "head", None, T, {"dev_acc": acc, "dev_nll": nll})
    write_json(ctx.out / "topk_formats.json", {"candidates": table, "chosen": chosen})
    ctx.check_client("topk end")
    return chosen


def unit_full(ctx, fixed=None):
    """Reference rows from N: TOPK-full at 64 bits, LAB-full, and RAW-native with the server of N."""
    ctx.check_client("full start")
    z_task = logits_on(ctx, ctx.net.server, "task")
    cands = TOPK_CANDS[64] if fixed is None else [tuple(fixed[:3])]
    res = []
    for (form, k, p) in cands:
        mults = None if fixed is None else (fixed[3],)
        spec, best = fit_topk_arm(ctx, form, k, p, z_task, ctx.net.server, mults)
        res.append((best["nll"], len(res), spec, best, (form, k, p)))
    nll, _, spec, best, fkp = min(res, key=lambda r: (r[0], r[1]))
    ctx.save_arm("TOPK_full", spec, "net_server", None, best["T"],
                 {"dev_acc": best["acc"], "dev_nll": best["nll"], "format": topk_name(*fkp), "gfill_mult": best["mult"],
                  "candidates": {topk_name(*r[4]): r[3] for r in res}})
    y = ctx.data.labels("dev")
    zd = logits_on(ctx, ctx.net.server, "dev")
    lab = LabSpec.fit(top1(zd).numpy(), y)
    T, acc, nll = eval_dev(ctx, Arm("LAB_full", lab, ctx.net.server, _post(lab)))
    ctx.save_arm("LAB_full", lab, "net_server", None, T, {"dev_acc": acc, "dev_nll": nll})
    rn = spec_from_state(torch.load(ctx.out / "arms" / "RAW_native.spec.pt", weights_only=True, map_location="cpu"))
    T, acc, nll = eval_dev(ctx, Arm("RAW_native", rn, lambda h: h, ctx.net.server))
    ctx.save_arm("RAW_native", rn, "net_server", None, T, {"dev_acc": acc, "dev_nll": nll})
    ctx.check_client("full end")
    return [fkp[0], fkp[1], fkp[2], best["mult"]]


def unit_raw(ctx, name, epochs=None):
    ctx.check_client(f"{name} start")
    spec = spec_from_state(torch.load(ctx.out / "arms" / f"{name}.spec.pt", weights_only=True, map_location="cpu"))
    torch.manual_seed(sub_seed(ctx.seed, f"{name}/init"))
    m = RawModel(ctx.net, spec).to(DEV)
    t_ep = train_part(m, [], list(m.parameters()), lambda h2: m(spec.quantize_dev(h2)),
                      ctx.data, ctx.client, ctx.seed, name, epochs)
    t0 = time.time()
    T, acc, nll = eval_dev(ctx, Arm(name, spec, lambda h: h, m))
    ctx.save_arm(name, spec, "raw", m.state_dict(), T, {"dev_acc": acc, "dev_nll": nll, "t_epochs": t_ep})
    ctx.check_client(f"{name} end")
    log(f"{name}: D_dev acc {acc:.4f}, NLL {nll:.4f}, T {T}")
    return {"t_epochs": t_ep, "t_eval": time.time() - t0}


def unit_bot(ctx, name, c, s, q, float_mode=False, epochs=None):
    ctx.check_client(f"{name} start")
    spec = BotSpec(c, s, q, float_mode)
    torch.manual_seed(sub_seed(ctx.seed, f"{name}/init"))
    m = BotModel(ctx.net, spec).to(DEV)
    new = list(m.enc.parameters()) + list(m.adapt.parameters())
    t_ep = train_part(m, new, list(m.server.parameters()), m, ctx.data, ctx.client, ctx.seed, name, epochs)
    t0 = time.time()
    T, acc, nll = eval_dev(ctx, Arm(name, spec, m.code, m.server_logits))
    ctx.save_arm(name, spec, "bot", m.state_dict(), T, {"dev_acc": acc, "dev_nll": nll, "t_epochs": t_ep})
    ctx.check_client(f"{name} end")
    log(f"{name}: D_dev acc {acc:.4f}, NLL {nll:.4f}, T {T}")
    return {"t_epochs": t_ep, "t_eval": time.time() - t0}


def raw_name(c, s, q):
    return f"RAW_c{c}_s{s}_q{q}"


def parse_cfg(name):
    """RAW_c16_s1_q4 or BOT_c16_s1_q4 -> (16, 1, 4)."""
    return tuple(int(v[1:]) for v in name.split("_")[1:])


def bot_name(c, s, q):
    return f"BOT_c{c}_s{s}_q{q}"


def select_cfg(rows):
    """Highest D_dev accuracy; among candidates within 0.2 points of it, the lowest D_dev NLL (design 7)."""
    best = max(r["dev_acc"] for r in rows)
    near = [r for r in rows if r["dev_acc"] >= best - SEL_TIE_POINTS / 100.0 - 1e-12]
    return min(near, key=lambda r: (r["dev_nll"], r["order"]))["name"]


def unit_select(ctx, topk_chosen, full_chosen):
    sel = {"RAW": {}, "BOT": {}, "TOPK": topk_chosen, "TOPK_full": full_chosen, "time": now(), "seed": ctx.seed}
    for fam, cands, nm in (("RAW", RAW_CANDS, raw_name), ("BOT", BOT_CANDS, bot_name)):
        for b in BUDGETS:
            rows = []
            for i, cfg in enumerate(cands[b]):
                r = read_json(ctx.out / "arms" / f"{nm(*cfg)}.json")
                rows.append({"name": nm(*cfg), "dev_acc": r["dev_acc"], "dev_nll": r["dev_nll"], "order": i})
            sel[fam][str(b)] = select_cfg(rows)
            sel[f"{fam}_rows_{b}"] = rows
    write_json(ROOT / "selection.json", sel)
    ctx.sel = sel
    log("selection: " + json.dumps({k: sel[k] for k in ("RAW", "BOT", "TOPK", "TOPK_full")}))
    return sel


def message_sets(ctx, name):
    if name == "BOT_float":
        return ("dev", "test")
    if name == "RAW_native":
        return ("aux", "auxF", "dev", "test")
    return ALL_SETS


def unit_msgs(ctx, names):
    """Messages of each arm on every set, through pack and unpack; logits on D_dev and test (no metric)."""
    ctx.check_client("msgs start")
    times = {}
    for name in names:
        t0 = time.time()
        arm = ctx.arm(name)
        out = {"T": np.float64(arm.T), "nbits": np.int64(arm.spec.bits)}
        for s in message_sets(ctx, name):
            base, flip = (s[:-1], True) if s.endswith("F") else (s, False)
            r = arm_messages(arm, ctx.client, ctx.data, ctx.data.images(base), flip=flip, want_dec=False)
            out[f"bits_{s}"] = r["packed"]
            if s in ("dev", "test"):
                out[f"logits_{s}"] = r["logits"].numpy()
                out[f"pred_{s}"] = r["pred"]
        np.savez(ctx.out / "msgs" / f"{name}.npz", **out)
        append_jsonl(ctx.out / "checks.jsonl", {"check": "V4", "where": f"msgs {name}", "ok": True, "bits": arm.spec.bits,
                                                "time": now()})
        times[name] = time.time() - t0
        log(f"messages {name}: {arm.spec.bits} bits ({times[name]:.1f} s)")
    ctx.check_client("msgs end")
    return times


def dec_tag(arm, size, rep, shuffle):
    return f"{size}{rep}_{arm}" + ("_ctrl" if shuffle else "")


def unit_decoder(ctx, arm_name, size, rep, shuffle, epochs=None):
    """Train one decoder on D_aux and its flips (20,000 pairs); save the test reconstructions as uint8."""
    tag = dec_tag(arm_name, size, rep, shuffle)
    ctx.check_client(f"{tag} start")
    t0 = time.time()
    real = ctx.resolve(arm_name)
    spec = spec_from_state(torch.load(ctx.out / "arms" / f"{real}.pt", weights_only=True, map_location="cpu")["spec"])
    z = np.load(ctx.out / "msgs" / f"{arm_name}.npz", allow_pickle=False)
    X = {s: dec_in_from_packed(spec, z[f"bits_{s}"]) for s in ("aux", "auxF", "dev", "test")}
    aux = ctx.data.images("aux")
    Xtr = torch.cat([X["aux"], X["auxF"]])
    Ttr = torch.cat([aux, aux.flip(3)])
    dev_u8 = ctx.data.images("dev")
    Xsel = X["dev"]
    seed = sub_seed(ctx.seed, f"dec/{tag}")
    if shuffle:
        Xtr = Xtr[torch.randperm(len(Xtr), generator=gen(sub_seed(seed, "pairs")))]
        Xsel = Xsel[torch.randperm(len(Xsel), generator=gen(sub_seed(seed, "devpairs")))]
    t_prep = time.time() - t0
    model, norm, info = train_decoder(Xtr, Ttr, Xsel, dev_u8, 1 if size == "S" else 2, seed, epochs)
    t1 = time.time()
    rec_dev = reconstruct(model, norm(X["dev"]))
    rec_test = reconstruct(model, norm(X["test"]))
    t_rec = time.time() - t1
    dev_np = dev_u8.cpu().numpy()
    psnr_dev, _ = psnr_u8(rec_dev, dev_np)
    np.savez(ctx.out / "dec" / f"{tag}.npz", recon_test=rec_test, psnr_dev=psnr_dev)
    Ttr_np = Ttr.cpu().numpy().astype(np.float64)
    res = dict(info, arm=arm_name, real=real, size=size, rep=rep, shuffle=shuffle, dev_psnr=float(psnr_dev.mean()),
               t_prep=t_prep, t_recon=t_rec, seconds=time.time() - t0)
    f0 = np.round(Ttr_np.mean(0)).astype(np.uint8)
    res["dev_psnr_F0"] = float(psnr_u8(np.broadcast_to(f0, dev_np.shape), dev_np)[0].mean())
    if shuffle:
        res["V2_ok"] = bool(res["dev_psnr"] <= res["dev_psnr_F0"] + 0.2)
        append_jsonl(ctx.out / "checks.jsonl", {"check": "V2", "where": tag, "ok": res["V2_ok"], "dev_psnr": res["dev_psnr"],
                                                "F0": res["dev_psnr_F0"], "time": now()})
    if arm_name == "LAB" and size == "S" and rep == 1 and not shuffle:
        lab_tr = torch.cat([X["aux"], X["auxF"]]).argmax(1).numpy()
        lab_dev = X["dev"].argmax(1).numpy()
        means = np.stack([Ttr_np[lab_tr == c].mean(0) if (lab_tr == c).any() else Ttr_np.mean(0) for c in range(10)])
        closed = np.round(means).astype(np.uint8)[lab_dev]
        res["dev_psnr_closed"] = float(psnr_u8(closed, dev_np)[0].mean())
        res["V1_ok"] = bool(res["dev_psnr"] >= res["dev_psnr_closed"] - 0.2)
        append_jsonl(ctx.out / "checks.jsonl", {"check": "V1", "where": tag, "ok": res["V1_ok"], "dev_psnr": res["dev_psnr"],
                                                "closed": res["dev_psnr_closed"], "time": now()})
    write_json(ctx.out / "dec" / f"{tag}.json", res)
    ctx.check_client(f"{tag} end")
    log(f"decoder {tag}: D_dev PSNR {res['dev_psnr']:.3f} dB (F0 {res['dev_psnr_F0']:.3f}), best epoch {info['best_epoch']}, "
        f"{res['seconds'] / 60:.1f} min")
    return res


# ------------------------------------------------------------------------------------------------ jobs

def seed_dir(seed):
    return RES / f"s{seed}"


def check_freeze(files_used):
    fz = read_json(ROOT / "FREEZE.json")
    for f in FROZEN:
        if sha256_file(ROOT / f) != fz["sha256"][f]:
            raise CheckFail(f"freeze: {f} differs from FREEZE.json")
    if files_used:
        d = fz.get("data")
        if not d:
            raise CheckFail("freeze: FREEZE.json has no data record")
        for f in files_used:
            if sha256_file(DATA_DIR / f) != d["npz_sha256"][f]:
                raise CheckFail(f"freeze: data/{f} differs from FREEZE.json")
    return fz


def j1_units(ctx):
    raw_cfgs = [cfg for b in BUDGETS for cfg in RAW_CANDS[b]]
    bot_cfgs = [cfg for b in BUDGETS for cfg in BOT_CANDS[b]]
    st = {}

    def n_unit():
        unit_N(ctx)
        check_v6(ctx)

    def topk():
        st["topk"] = unit_topk(ctx)

    def full():
        st["full"] = unit_full(ctx)

    def select():
        tk = read_json(ctx.out / "topk_formats.json")["chosen"]
        fu = read_json(ctx.out / "arms" / "TOPK_full.json")
        fk = [c for c in TOPK_CANDS[64] if topk_name(*c) == fu["format"]][0]
        unit_select(ctx, tk, [fk[0], fk[1], fk[2], fu["gfill_mult"]])

    def botf():
        unit_bot(ctx, "BOT_float", *parse_cfg(ctx.sel["BOT"][str(BSTAR)]), float_mode=True)

    units = [("N", n_unit), ("prep", lambda: unit_prep(ctx, raw_cfgs)), ("head", lambda: unit_head(ctx)),
             ("topk", topk), ("full", full)]
    units += [(raw_name(*cfg), (lambda cfg=cfg: unit_raw(ctx, raw_name(*cfg)))) for cfg in raw_cfgs]
    units += [(bot_name(*cfg), (lambda cfg=cfg: unit_bot(ctx, bot_name(*cfg), *cfg))) for cfg in bot_cfgs]
    units += [("select", select), ("BOT_float", botf)]
    return units


def seed23_units(ctx):
    sel = ctx.sel
    raw_sel = sorted({sel["RAW"][str(b)] for b in BUDGETS})
    bot_sel = sorted({sel["BOT"][str(b)] for b in BUDGETS})
    raw_cfgs = [parse_cfg(n) for n in raw_sel]

    def n_unit():
        unit_N(ctx)
        check_v6(ctx)

    units = [("N", n_unit), ("prep", lambda: unit_prep(ctx, raw_cfgs)), ("head", lambda: unit_head(ctx)),
             ("topk", lambda: unit_topk(ctx, fixed=sel["TOPK"])), ("full", lambda: unit_full(ctx, fixed=sel["TOPK_full"]))]
    units += [(n, (lambda n=n: unit_raw(ctx, n))) for n in raw_sel]
    units += [(n, (lambda n=n: unit_bot(ctx, n, *parse_cfg(n)))) for n in bot_sel]
    names = [a for a in MSG_ARMS if a != "BOT_float"]
    units += [("msgs", lambda: unit_msgs(ctx, names))]
    units += [(dec_tag(*d), (lambda d=d: unit_decoder(ctx, *d))) for d in J45_DECS]
    return units


def job_units(job, ctx):
    if job == "j1":
        return j1_units(ctx)
    if job == "j2":
        return [("msgs", lambda: unit_msgs(ctx, MSG_ARMS))] + [(dec_tag(*d), (lambda d=d: unit_decoder(ctx, *d))) for d in J2_DECS]
    if job == "j3":
        return [(dec_tag(*d), (lambda d=d: unit_decoder(ctx, *d))) for d in J3_DECS]
    if job in ("j4", "j5"):
        return seed23_units(ctx)
    raise ValueError(job)


def run_units(ctx, units, part):
    names = [u for u, _ in units]
    if part:
        plan_sha = sha256_file(ROOT / "job_plan.json")
        if plan_sha != read_json(ROOT / "j0_result.json")["job_plan_sha256"]:
            raise CheckFail("job_plan.json differs from the file that J0 wrote")
        plan = read_json(ROOT / "job_plan.json")[ctx.job]["parts"]
        if not 1 <= part <= len(plan):
            raise CheckFail(f"job_plan.json has {len(plan)} parts for {ctx.job}, not part {part}")

        def resolved(p):
            return resolve_plan_names(ctx, p) if ctx.job in ("j4", "j5") else p

        keep = set(resolved(plan[part - 1]))
        if not keep <= set(names):
            raise CheckFail("job_plan.json names a unit that this job does not have")
        earlier = [u for p in plan[:part - 1] for u in resolved(p)]
        missing = [u for u in earlier if not (ctx.out / "done" / f"{u}.json").exists()]
        if missing:
            raise CheckFail(f"part {part} needs the earlier units {missing}")
    else:
        keep = set(names)
    code = sha256_file(Path(__file__).resolve())
    for name, fn in units:
        if name not in keep:
            continue
        marker = ctx.out / "done" / f"{name}.json"
        if marker.exists():
            log(f"unit {name}: done before; skip")
            continue
        log(f"unit {name}: start")
        t0 = time.time()
        fn()
        sync()
        dt = time.time() - t0
        write_json(marker, {"unit": name, "seconds": dt, "time": now(), "sebits_sha256": code})
        append_jsonl(ctx.out / "times.jsonl", {"job": ctx.job, "unit": name, "seconds": dt, "time": now()})
        log(f"unit {name}: end ({dt / 60:.1f} min)")


JOB_SEED = {"j1": 0, "j2": 0, "j3": 0, "j4": 1, "j5": 2}


def run_job(job, part):
    check_freeze(["cifar10_train.npz"] if job == "j1" else ["cifar10_train.npz", "cifar10_test_images.npz"])
    seed = RUN_SEEDS[JOB_SEED[job]]
    data = load_data("train" if job == "j1" else "confirm")
    ctx = Ctx(job, seed, seed_dir(seed), data)
    if job != "j1" and ctx.sel is None:
        raise CheckFail("selection.json does not exist (J1 writes it)")
    if job in ("j2", "j3") or (ctx.out / "N.json").exists():
        check_v6(ctx)
    run_units(ctx, job_units(job, ctx), part)


# ------------------------------------------------------------------------------------------------ J0 smoke

def j0():
    """Scratch mode (design 10.2): unit tests, one full N with seed 20261100, one epoch of each arm type and
    of each decoder size; D_dev is in the test role. Writes times.json, job_plan.json and j0_result.json."""
    check_freeze(["cifar10_train.npz"])
    T = {"time": now()}
    t0 = time.time()
    r = run_tests()
    T["tests"] = {"run": r.testsRun, "failures": len(r.failures), "errors": len(r.errors), "seconds": time.time() - t0}
    if r.failures or r.errors:
        raise CheckFail("unit tests failed")
    data = load_data("scratch")
    ctx = Ctx("j0", SCRATCH_SEED, RES / "j0", data)
    info = unit_N(ctx)
    T["N_seconds"], T["N_dev_acc"] = info["train_seconds"], info["dev_acc"]
    raw_cfgs = [cfg for b in BUDGETS for cfg in RAW_CANDS[b]]
    t0 = time.time()
    unit_prep(ctx, raw_cfgs)
    T["prep_seconds"] = time.time() - t0
    T["head"] = unit_head(ctx, epochs=1)
    t0 = time.time()
    unit_topk(ctx)
    T["topk_seconds"] = time.time() - t0
    t0 = time.time()
    unit_full(ctx)
    T["full_seconds"] = time.time() - t0
    T["arms"] = {}
    for cfg in ((16, 1, 4), (64, 2, 1), (16, 4, 4)):
        T["arms"][raw_name(*cfg)] = unit_raw(ctx, raw_name(*cfg), epochs=1)
    for cfg in ((16, 1, 4), (16, 8, 1)):
        T["arms"][bot_name(*cfg)] = unit_bot(ctx, bot_name(*cfg), *cfg, epochs=1)
    T["arms"]["BOT_float"] = unit_bot(ctx, "BOT_float", 16, 1, 4, float_mode=True, epochs=1)
    ctx.sel = {"RAW": {"64": raw_name(16, 1, 4)}, "BOT": {"64": bot_name(16, 1, 4)}}
    names = ["TOPK@64", "LAB", "RAW@64", "BOT@64", raw_name(64, 2, 1), raw_name(16, 4, 4), bot_name(16, 8, 1),
             "RAW_native", "BOT_float"]
    T["msgs"] = unit_msgs(ctx, names)
    T["dec"] = {}
    for d in (("TOPK@64", "S", 1, False), (raw_name(64, 2, 1), "S", 1, False), (raw_name(16, 4, 4), "S", 1, False),
              ("RAW_native", "S", 1, False), (bot_name(16, 8, 1), "S", 1, False), ("TOPK@64", "S", 1, True),
              ("BOT@64", "L", 1, False)):
        T["dec"][dec_tag(*d)] = unit_decoder(ctx, *d, epochs=1)
    t0 = time.time()
    smoke = j6_seed_metrics(ctx, probe_epochs=1, probe_arms=["BOT@64"])
    T["analysis_smoke_seconds"] = time.time() - t0
    T["analysis_smoke"] = smoke
    write_json(RES / "j0" / "j0_times_raw.json", T)
    proj, plan, stops = project(T)
    write_json(ROOT / "times.json", {"measured": T, "projected_minutes": proj, "time": now()})
    write_json(ROOT / "job_plan.json", plan)
    result = {"stop_rules": stops, "stop": bool(stops), "N_dev_acc": T["N_dev_acc"], "N_minutes": T["N_seconds"] / 60,
              "job_plan_sha256": sha256_file(ROOT / "job_plan.json"),
              "decS_minutes": proj["unit_decS"], "projected_job_minutes": {k: proj[k] for k in proj if k.startswith("J")}}
    write_json(ROOT / "j0_result.json", result)
    log("J0 result: " + json.dumps(result))
    if stops:
        raise StopRule("; ".join(stops))


def project(T):
    """Unit times from J0 -> job times (minutes) and the parts of each job (design 10.2: <= 110 min each)."""
    m = 60.0
    arm_ep = max(float(np.mean(a["t_epochs"])) for a in T["arms"].values())
    arm_ev = max(a["t_eval"] for a in T["arms"].values())
    arm = (ARM_EPOCHS * arm_ep + arm_ev) / m
    head = (ARM_EPOCHS * float(np.mean(T["head"]["t_epochs"])) + T["head"]["t_eval"]) / m

    def dec(kind):
        rows = [d for k, d in T["dec"].items() if d["size"] == kind]
        return max((d["t_prep"] + DEC_EPOCHS * float(np.mean(d["t_train_epoch"])) + (DEC_EPOCHS // 2) * float(np.mean(d["t_eval"]))
                    + d["t_recon"]) / m for d in rows)

    decS, decL = dec("S"), dec("L")
    msg = max(T["msgs"].values()) / m
    N = T["N_seconds"] / m
    proj = {"unit_N": N, "unit_arm": arm, "unit_head": head, "unit_decS": decS, "unit_decL": decL, "unit_msg_arm": msg,
            "unit_prep": T["prep_seconds"] / m, "unit_topk": T["topk_seconds"] / m, "unit_full": T["full_seconds"] / m}
    unit_min = {"N": N, "prep": proj["unit_prep"], "head": head, "topk": proj["unit_topk"], "full": proj["unit_full"],
                "select": 0.0, "BOT_float": arm}
    plan = {}
    names = {"j1": ["N", "prep", "head", "topk", "full"] + [raw_name(*c) for b in BUDGETS for c in RAW_CANDS[b]]
             + [bot_name(*c) for b in BUDGETS for c in BOT_CANDS[b]] + ["select", "BOT_float"],
             "j2": ["msgs"] + [dec_tag(*d) for d in J2_DECS],
             "j3": [dec_tag(*d) for d in J3_DECS]}
    names["j4"] = names["j5"] = (["N", "prep", "head", "topk", "full"] + [f"RAW_sel{i}" for i in range(4)]
                                 + [f"BOT_sel{i}" for i in range(4)] + ["msgs"] + [dec_tag(*d) for d in J45_DECS])

    def est(job, u):
        if u in unit_min:
            return unit_min[u]
        if u.startswith(("RAW_", "BOT_")):
            return arm
        if u == "msgs":
            return msg * (len(MSG_ARMS) if job == "j2" else len(MSG_ARMS) - 1)
        if u.startswith("S"):
            return decS
        if u.startswith("L"):
            return decL
        raise ValueError(u)

    for job, us in names.items():
        mins = [est(job, u) for u in us]
        parts, cur, acc = [], [], 0.0
        for u, t in zip(us, mins):
            if cur and acc + t > JOB_LIMIT_MIN:
                parts.append(cur)
                cur, acc = [], 0.0
            cur.append(u)
            acc += t
        parts.append(cur)
        proj[f"J{job[1]}"] = sum(mins)
        plan[job] = {"minutes": sum(mins), "parts": parts, "part_minutes": [sum(est(job, u) for u in p) for p in parts]}
    proj["J6"] = T["analysis_smoke_seconds"] / m * 3 * PROBE_EPOCHS  # rough: 3 seeds, 30 probe epochs
    stops = []
    if N > 20:
        stops.append(f"N takes {N:.1f} min (> 20 min)")
    if decS > 6:
        stops.append(f"one Decoder-S takes {decS:.1f} min (> 6 min)")
    if T["N_dev_acc"] < V6_MIN:
        stops.append(f"N D_dev accuracy {T['N_dev_acc']:.4f} (< {V6_MIN:.2f})")
    return proj, plan, stops


def resolve_plan_names(ctx, plan_part):
    """job_plan.json uses RAW_sel<i> and BOT_sel<i> for seeds 2 and 3; map them to the selected names."""
    raw_sel = sorted({ctx.sel["RAW"][str(b)] for b in BUDGETS})
    bot_sel = sorted({ctx.sel["BOT"][str(b)] for b in BUDGETS})
    out = []
    for u in plan_part:
        if u.startswith("RAW_sel"):
            i = int(u[7:])
            if i < len(raw_sel):
                out.append(raw_sel[i])
        elif u.startswith("BOT_sel"):
            i = int(u[7:])
            if i < len(bot_sel):
                out.append(bot_sel[i])
        else:
            out.append(u)
    return out


# ------------------------------------------------------------------------------------------------ J6 analysis

def flip_bit_set(z, spec, base, flips):
    a = dec_in_from_packed(spec, z[f"bits_{base}"])
    b = dec_in_from_packed(spec, z[f"bits_{base}F"])
    fl = torch.from_numpy(flips.astype(bool))
    shape = (-1,) + (1,) * (a.dim() - 1)
    return torch.where(fl.view(shape), b, a)


def luma(x_u8):
    """Mean luma of each image: 0.299 R + 0.587 G + 0.114 B, in [0, 1] (design 6.4)."""
    x = x_u8.astype(np.float64) / 255.0
    return (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).reshape(len(x), -1).mean(1)


def j6_seed_metrics(ctx, probe_epochs=None, probe_arms=None):
    """Per-image test metrics of one seed: correctness, NLL, PSNR, SSIM, MSE; probe outputs."""
    probe_arms = PROBE_ARMS if probe_arms is None else probe_arms
    data = ctx.data
    y_test = data.labels("test")
    test_u8 = data.images("test").cpu().numpy()
    out = {"arms": {}, "probes": {}}
    for f in sorted((ctx.out / "msgs").glob("*.npz")):
        name = f.stem
        z = np.load(f, allow_pickle=False)
        lg = torch.from_numpy(z["logits_test"]).double()
        T = float(z["T"])
        yy = torch.from_numpy(y_test)
        out["arms"][name] = {"correct": (z["pred_test"] == y_test).astype(np.float64),
                             "nll": F.cross_entropy(lg / T, yy, reduction="none").numpy(), "bits": int(z["nbits"])}
    for p in sorted((ctx.out / "dec").glob("*.npz")):
        tag = p.stem
        rec = np.load(p, allow_pickle=False)["recon_test"]
        ps, mse = psnr_u8(rec, test_u8)
        out.setdefault("dec", {})[tag] = {"psnr": ps, "mse": mse, "ssim": ssim_u8(rec, test_u8)}
    lum_aux = luma(data.images("aux").cpu().numpy())
    edges = np.quantile(lum_aux, [1 / 3, 2 / 3])
    b_aux = np.searchsorted(edges, lum_aux, side="right")
    b_dev = np.searchsorted(edges, luma(data.images("dev").cpu().numpy()), side="right")
    b_test = np.searchsorted(edges, luma(test_u8), side="right")
    sp = data.sp
    y_aux, y_dev, y_task = data.labels("aux"), data.labels("dev"), data.labels("task")
    oh = lambda y: F.one_hot(torch.as_tensor(np.asarray(y), dtype=torch.int64), 10).float()
    ps = sub_seed(ctx.seed, "probe")
    lab_logits, _ = train_probe(oh(y_aux), b_aux, oh(y_dev), b_dev, oh(y_test), 3, sub_seed(ps, "bright/label"), probe_epochs)
    out["probes"]["label_only"] = {"bright_correct": (lab_logits.argmax(1).numpy() == b_test).astype(np.float64)}
    for name in probe_arms:
        real = ctx.resolve(name)
        spec = spec_from_state(torch.load(ctx.out / "arms" / f"{real}.pt", weights_only=True, map_location="cpu")["spec"])
        z = np.load(ctx.out / "msgs" / f"{name}.npz", allow_pickle=False)
        Xa, Xd, Xt = (dec_in_from_packed(spec, z[f"bits_{s}"]).flatten(1) for s in ("aux", "dev", "test"))
        Xm = dec_in_from_packed(spec, z["bits_mem"]).flatten(1)
        fa, fd, ft = (flip_bit_set(z, spec, s, sp[f"flip_{s}"]).flatten(1) for s in ("aux", "dev", "test"))
        lg, ep_f = train_probe(fa, sp["flip_aux"], fd, sp["flip_dev"], ft, 2, sub_seed(ps, f"flip/{name}"), probe_epochs)
        flip_c = (lg.argmax(1).numpy() == sp["flip_test"]).astype(np.float64)
        lg, ep_b = train_probe(Xa, b_aux, Xd, b_dev, Xt, 3, sub_seed(ps, f"bright/{name}"), probe_epochs)
        bright_c = (lg.argmax(1).numpy() == b_test).astype(np.float64)
        n = len(sp["mem_in_train"])
        y_mem = y_task[np.concatenate([sp["mem_in_train"], sp["mem_in_test"]])]
        Xtr = torch.cat([torch.cat([Xm[:n], oh(y_mem[:n])], 1),
                         torch.cat([Xa[sp["mem_out_train"]], oh(y_aux[sp["mem_out_train"]])], 1)])
        ytr = np.concatenate([np.ones(n), np.zeros(len(sp["mem_out_train"]))]).astype(np.int64)
        Xte = torch.cat([torch.cat([Xm[n:], oh(y_mem[n:])], 1),
                         torch.cat([Xt[sp["mem_out_test"]], oh(y_test[sp["mem_out_test"]])], 1)])
        yte = np.concatenate([np.ones(len(sp["mem_in_test"])), np.zeros(len(sp["mem_out_test"]))]).astype(np.int64)
        lg, _ = train_probe(Xtr, ytr, None, None, Xte, 2, sub_seed(ps, f"member/{name}"), probe_epochs, select=False)
        score = (lg[:, 1] - lg[:, 0]).numpy()
        out["probes"][name] = {"flip_correct": flip_c, "bright_correct": bright_c, "member_score": score,
                               "member_label": yte, "flip_epoch": ep_f, "bright_epoch": ep_b}
    if ctx.job == "j0":
        summary = {"arms": {k: {"acc": float(v["correct"].mean()), "nll": float(v["nll"].mean())} for k, v in out["arms"].items()},
                   "dec": {k: {"psnr": float(v["psnr"].mean()), "ssim": float(v["ssim"].mean())} for k, v in out.get("dec", {}).items()},
                   "probes": {k: {kk: float(np.mean(vv)) for kk, vv in v.items() if kk.endswith("correct")} for k, v in out["probes"].items()}}
        for k, v in out["probes"].items():
            if "member_score" in v:
                summary["probes"][k]["member_auroc"] = auroc(v["member_score"], v["member_label"])
        return summary
    return out


def j6():
    check_freeze(["cifar10_train.npz", "cifar10_test_images.npz", "cifar10_test_labels.npz"])
    t_start = time.time()
    data = load_data("analysis")
    sel = read_json(ROOT / "selection.json")
    test_u8 = data.images("test").cpu().numpy()
    y_test = data.labels("test")
    aux = data.images("aux").cpu().numpy()
    tr = np.concatenate([aux, aux[:, :, :, ::-1]]).astype(np.float64)
    y_aux2 = np.concatenate([data.labels("aux")] * 2)
    f0 = np.round(tr.mean(0)).astype(np.uint8)
    fy = np.stack([np.round(tr[y_aux2 == c].mean(0)).astype(np.uint8) for c in range(10)])
    floors = {}
    for nm, rec in (("F0", np.broadcast_to(f0, test_u8.shape)), ("FY", fy[y_test])):
        p, mse = psnr_u8(rec, test_u8)
        floors[nm] = {"psnr": float(p.mean()), "mse": float(mse.mean()), "ssim": float(ssim_u8(np.ascontiguousarray(rec), test_u8).mean())}
    mse_f0 = floors["F0"]["mse"]
    seeds = {}
    checks = {"V1": [], "V2": [], "V3": [], "V4": [], "V5": [], "V6": []}
    for si, seed in enumerate(RUN_SEEDS):
        ctx = Ctx("j6", seed, seed_dir(seed), data)
        ctx.sel = sel
        for unit in ("N", "msgs"):
            if not (ctx.out / "done" / f"{unit}.json").exists():
                raise CheckFail(f"V5: seed {seed} has no finished unit {unit}")
        log(f"J6: seed {seed}")
        seeds[seed] = j6_seed_metrics(ctx)
        for line in (ctx.out / "checks.jsonl").read_text().splitlines():
            r = json.loads(line)
            checks[r["check"]].append(dict(r, seed=seed))
        decs_needed = [dec_tag(*d) for d in (J2_DECS if si == 0 else J45_DECS)] + ([dec_tag(*d) for d in J3_DECS] if si == 0 else [])
        missing = [t for t in decs_needed if t not in seeds[seed].get("dec", {})]
        checks["V5"].append({"seed": seed, "ok": not missing, "missing": missing})
    S = RUN_SEEDS

    def corr(a):
        return np.stack([seeds[s]["arms"][a]["correct"] for s in S])

    def nll(a):
        return np.stack([seeds[s]["arms"][a]["nll"] for s in S])

    def dmet(a, key="psnr", size="S", rep=1, only=None):
        ss = only or S
        return np.stack([seeds[s]["dec"][dec_tag(a, size, rep, False)][key] for s in ss])

    def arm_of(name):
        return "TOPK@256" if name == "TOPK@1024" else name

    v_ok = {k: all(r.get("ok", False) for r in v) and len(v) > 0 for k, v in checks.items()}
    valid = all(v_ok.values())
    R = {"time": now(), "floors": floors, "checks": {k: {"ok": v_ok[k], "records": len(checks[k])} for k in checks},
         "check_failures": {k: [r for r in v if not r.get("ok", False)] for k, v in checks.items()}}
    if not v_ok["V5"]:
        R["primary"] = {"case_final": "0", "reason": "V5: a planned decoder output is absent; no comparison is computed"}
        RES.mkdir(exist_ok=True)
        write_json(RES / "results.json", R)
        (RES / "table.md").write_text("# Results table\n\nCase 0 (invalid): V5 failed. Absent decoder outputs:\n\n"
                                      + "\n".join(f"- seed {r['seed']}: {r['missing']}" for r in checks["V5"] if r["missing"]) + "\n")
        log("J6: case 0 (V5 failed)")
        return
    prim = compare(corr("TOPK@64"), corr("BOT@64"), dmet("TOPK@64"), dmet("BOT@64"))
    R["primary"] = dict(prim, sym="TOPK@64", act="BOT@64", case_final=prim["case"] if valid else "0")
    d_l = compare(corr("TOPK@64")[:1], corr("BOT@64")[:1], dmet("TOPK@64", size="L", only=[S[0]]),
                  dmet("BOT@64", size="L", only=[S[0]]))
    d_s1 = compare(corr("TOPK@64")[:1], corr("BOT@64")[:1], dmet("TOPK@64", only=[S[0]]), dmet("BOT@64", only=[S[0]]))
    dep = (np.sign(d_l["d_leak"]) != np.sign(d_s1["d_leak"])) or abs(d_l["d_leak"] - d_s1["d_leak"]) > 1.0
    R["decoder_check"] = {"d_leak_L_seed1": d_l["d_leak"], "d_leak_S_seed1": d_s1["d_leak"], "decoder_dependent": bool(dep)}
    R["primary"]["decoder_dependent"] = bool(dep)
    R["primary"]["case_label"] = R["primary"]["case_final"] + (" (decoder-dependent)" if dep else "")
    reps = {}
    for a in ("TOPK@64", "BOT@64"):
        vals = [float(seeds[S[0]]["dec"][dec_tag(a, "S", r, False)]["psnr"].mean()) for r in (1, 2, 3)]
        reps[a] = {"psnr_reps": vals, "sd": float(np.std(vals, ddof=1))}
    R["decoder_seed_sd"] = reps
    sec = {}
    sec["TOPK_vs_RAW@64"] = compare(corr("TOPK@64"), corr("RAW@64"), dmet("TOPK@64"), dmet("RAW@64"))
    for b in (16, 256, 1024):
        t = arm_of(f"TOPK@{b}")
        sec[f"TOPK_vs_BOT@{b}"] = compare(corr(t), corr(f"BOT@{b}"), dmet(t), dmet(f"BOT@{b}"))
        sec[f"TOPK_vs_RAW@{b}"] = compare(corr(t), corr(f"RAW@{b}"), dmet(t), dmet(f"RAW@{b}"))
    for b in BUDGETS:
        t = arm_of(f"TOPK@{b}")
        row = compare(corr("LAB"), corr(t), dmet("LAB"), dmet(t))
        row["nll"] = nll_beats(nll("LAB"), nll(t))
        sec[f"LAB_vs_TOPK@{b}"] = row
    sec["LAB_full_vs_BOT@64"] = compare(corr("LAB_full"), corr("BOT@64"), dmet("LAB_full"), dmet("BOT@64"))
    sec["TOPK_full_vs_BOT@64"] = compare(corr("TOPK_full"), corr("BOT@64"), dmet("TOPK_full"), dmet("BOT@64"))
    sec["SSIM_TOPK_vs_BOT@64"] = compare(corr("TOPK@64"), corr("BOT@64"), dmet("TOPK@64", "ssim"), dmet("BOT@64", "ssim"),
                                         leak_margin=0.02)
    R["secondary"] = sec
    attr = {}
    lab_b = np.stack([seeds[s]["probes"]["label_only"]["bright_correct"] for s in S])
    for a in PROBE_ARMS:
        fl = np.stack([seeds[s]["probes"][a]["flip_correct"] for s in S])
        br = np.stack([seeds[s]["probes"][a]["bright_correct"] for s in S])
        au = [auroc(seeds[s]["probes"][a]["member_score"], seeds[s]["probes"][a]["member_label"]) for s in S]
        au_b = np.mean([auroc_boot(seeds[s]["probes"][a]["member_score"], seeds[s]["probes"][a]["member_label"]) for s in S], 0)
        attr[a] = {"flip_acc": float(fl.mean()), "flip_acc_ci": ci95(boot_means(fl.mean(0))),
                   "bright_acc": float(br.mean()), "bright_excess": float((br - lab_b).mean()),
                   "bright_excess_ci": ci95(boot_means((br - lab_b).mean(0))),
                   "member_auroc": float(np.mean(au)), "member_auroc_seeds": au, "member_auroc_ci": ci95(au_b)}
    R["attributes"] = {"label_only_bright_acc": float(lab_b.mean()), "arms": attr}
    desc = {}
    rows_all = [a for a in DEC_ARMS] + ["TOPK@1024", "BOT_float"]
    for a in rows_all:
        aa = arm_of(a)
        ss = [s for s in S if aa in seeds[s]["arms"]]
        if not ss:
            continue
        d = {"bits": seeds[ss[0]]["arms"][aa]["bits"], "acc": float(np.mean([seeds[s]["arms"][aa]["correct"].mean() for s in ss])),
             "acc_seeds": [float(seeds[s]["arms"][aa]["correct"].mean()) for s in ss],
             "nll": float(np.mean([seeds[s]["arms"][aa]["nll"].mean() for s in ss])), "seeds": len(ss)}
        dk = dec_tag(aa, "S", 1, False)
        sd = [s for s in S if dk in seeds[s].get("dec", {})]
        if sd:
            ps = np.stack([seeds[s]["dec"][dk]["psnr"] for s in sd])
            d.update({"psnr": float(ps.mean()), "psnr_seeds": ps.mean(1).tolist(),
                      "ssim": float(np.mean([seeds[s]["dec"][dk]["ssim"].mean() for s in sd])),
                      "r2_pix": float(1 - np.mean([seeds[s]["dec"][dk]["mse"].mean() for s in sd]) / mse_f0),
                      "excess_over_FY": float(ps.mean() - floors["FY"]["psnr"]), "dec_seeds": len(sd)})
        dl = dec_tag(aa, "L", 1, False)
        if dl in seeds[S[0]].get("dec", {}):
            d["psnr_L_seed1"] = float(seeds[S[0]]["dec"][dl]["psnr"].mean())
        cfg = {}
        real = sel["RAW"][a.split("@")[1]] if a.startswith("RAW@") else sel["BOT"][a.split("@")[1]] if a.startswith("BOT@") else None
        if real:
            d["config"] = real
            cfg = {"c": int(real.split("_")[1][1:]), "s": int(real.split("_")[2][1:])}
        if a == "BOT_float":
            r = sel["BOT"][str(BSTAR)]
            cfg = {"c": int(r.split("_")[1][1:]), "s": int(r.split("_")[2][1:])}
        d["client_macs"], d["server_macs"] = arm_macs(a, cfg)
        desc[a] = d
    for b in BUDGETS:
        t = arm_of(f"TOPK@{b}")
        desc[f"TOPK@{b}"]["c_quant"] = float(nll(t).mean() - nll("TOPK_float").mean())
    R["descriptive"] = desc
    R["selection"] = {k: sel[k] for k in ("RAW", "BOT", "TOPK", "TOPK_full")}
    RES.mkdir(exist_ok=True)
    write_json(RES / "results.json", R)
    (RES / "table.md").write_text(table_md(R))
    (RES / "acc_psnr.svg").write_text(svg_plot(desc, floors))
    log(f"J6: primary case {R['primary']['case_label']} ({CASE_TEXT[R['primary']['case_final']]}); "
        f"{(time.time() - t_start) / 60:.1f} min")


def fmt(v, n=3):
    return "n/a" if v is None else f"{v:.{n}f}"


def table_md(R):
    L = ["# Results table (written by sebits.py j6)", ""]
    p = R["primary"]
    L += ["## Primary pair: TOPK@64 against BOT@64 (cut C2)", "",
          "| Quantity | Value | 95% CI | Seeds |", "|---|---|---|---|",
          f"| Delta acc (points; + favors TOPK) | {fmt(p['d_acc'])} | {fmt(p['d_acc_ci'][0])} to {fmt(p['d_acc_ci'][1])} | "
          + ", ".join(fmt(v) for v in p["d_acc_seeds"]) + " |",
          f"| Delta leak (dB; + = BOT leaks more) | {fmt(p['d_leak'])} | {fmt(p['d_leak_ci'][0])} to {fmt(p['d_leak_ci'][1])} | "
          + ", ".join(fmt(v) for v in p["d_leak_seeds"]) + " |", "",
          f"Accuracy outcome {p['A']}; leakage outcome {p['L']}; case {p['case_label']} ({CASE_TEXT[p['case_final']]}).", "",
          f"Decoder check: Delta leak with Decoder-L (seed 1) {fmt(R['decoder_check']['d_leak_L_seed1'])} dB; with Decoder-S "
          f"(seed 1) {fmt(R['decoder_check']['d_leak_S_seed1'])} dB; decoder-dependent: {R['decoder_check']['decoder_dependent']}.", "",
          "## Validity checks", "", "| Check | Pass | Records |", "|---|---|---|"]
    L += [f"| {k} | {v['ok']} | {v['records']} |" for k, v in R["checks"].items()]
    L += ["", "## Secondary rows (no case claim)", "", "| Row | Delta acc | CI | Delta leak | CI | A | L | Case |",
          "|---|---|---|---|---|---|---|---|"]
    for k, v in R["secondary"].items():
        L.append(f"| {k} | {fmt(v['d_acc'])} | {fmt(v['d_acc_ci'][0])} to {fmt(v['d_acc_ci'][1])} | {fmt(v['d_leak'])} | "
                 f"{fmt(v['d_leak_ci'][0])} to {fmt(v['d_leak_ci'][1])} | {v['A']} | {v['L']} | {v['case']} |")
    L += ["", "NLL rows (LAB against TOPK; + = TOPK has the lower NLL):", "", "| Budget | Delta NLL | CI | TOPK beats LAB |",
          "|---|---|---|---|"]
    for b in BUDGETS:
        n = R["secondary"][f"LAB_vs_TOPK@{b}"]["nll"]
        L.append(f"| {b} | {fmt(n['d_nll'], 4)} | {fmt(n['d_nll_ci'][0], 4)} to {fmt(n['d_nll_ci'][1], 4)} | {n['topk_beats']} |")
    L += ["", "## Arms (descriptive; means over seeds)", "",
          "| Arm | Bits | Acc | NLL | PSNR | SSIM | R2_pix | Excess over FY | PSNR Decoder-L | Client MACs | Server MACs |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, d in R["descriptive"].items():
        L.append(f"| {k} | {d['bits']} | {fmt(d['acc'], 4)} | {fmt(d['nll'], 4)} | {fmt(d.get('psnr'))} | {fmt(d.get('ssim'))} | "
                 f"{fmt(d.get('r2_pix'))} | {fmt(d.get('excess_over_FY'))} | {fmt(d.get('psnr_L_seed1'))} | {d['client_macs']:,} | "
                 f"{d['server_macs']:,} |")
    L += ["", f"Floors on test: F0 PSNR {fmt(R['floors']['F0']['psnr'])} dB; FY PSNR {fmt(R['floors']['FY']['psnr'])} dB.", "",
          "## Attribute and membership rows at B* (descriptive)", "",
          f"Label-only brightness probe accuracy: {fmt(R['attributes']['label_only_bright_acc'], 4)}.", "",
          "| Arm | Flip acc | CI | Brightness excess | CI | Membership AUROC | CI |", "|---|---|---|---|---|---|---|"]
    for k, a in R["attributes"]["arms"].items():
        L.append(f"| {k} | {fmt(a['flip_acc'], 4)} | {fmt(a['flip_acc_ci'][0], 4)} to {fmt(a['flip_acc_ci'][1], 4)} | "
                 f"{fmt(a['bright_excess'], 4)} | {fmt(a['bright_excess_ci'][0], 4)} to {fmt(a['bright_excess_ci'][1], 4)} | "
                 f"{fmt(a['member_auroc'], 4)} | {fmt(a['member_auroc_ci'][0], 4)} to {fmt(a['member_auroc_ci'][1], 4)} |")
    return "\n".join(L) + "\n"


def svg_plot(desc, floors):
    """Accuracy against PSNR (Decoder-S) for all arms and budgets; plain SVG (no plotting package)."""
    pts = [(k, d["psnr"], d["acc"]) for k, d in desc.items() if "psnr" in d and k != "TOPK@1024"]
    if not pts:
        return "<svg xmlns='http://www.w3.org/2000/svg'/>"
    xs = [p[1] for p in pts] + [floors["F0"]["psnr"], floors["FY"]["psnr"]]
    ys = [p[2] for p in pts]
    x0, x1 = math.floor(min(xs)) - 0.5, math.ceil(max(xs)) + 0.5
    y0, y1 = max(0.0, math.floor(min(ys) * 20) / 20 - 0.05), min(1.0, math.ceil(max(ys) * 20) / 20 + 0.02)
    W, H, ml, mb = 760, 520, 70, 60

    def X(v):
        return ml + (v - x0) / (x1 - x0) * (W - ml - 20)

    def Y(v):
        return H - mb - (v - y0) / (y1 - y0) * (H - mb - 20)

    col = {"RAW": "#1f77b4", "BOT": "#d62728", "TOPK": "#2ca02c", "LAB": "#9467bd"}
    s = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' font-family='sans-serif' font-size='11'>",
         f"<rect width='{W}' height='{H}' fill='white'/>",
         f"<line x1='{ml}' y1='{H - mb}' x2='{W - 20}' y2='{H - mb}' stroke='black'/>",
         f"<line x1='{ml}' y1='20' x2='{ml}' y2='{H - mb}' stroke='black'/>",
         f"<text x='{W / 2}' y='{H - 15}' text-anchor='middle'>PSNR of Decoder-S on test (dB; higher = more leakage)</text>",
         f"<text x='18' y='{H / 2}' transform='rotate(-90 18 {H / 2})' text-anchor='middle'>Test accuracy</text>"]
    for v in np.arange(math.ceil(x0), x1, 1.0):
        s.append(f"<text x='{X(v):.1f}' y='{H - mb + 16}' text-anchor='middle'>{v:.0f}</text>")
    for v in np.arange(y0, y1 + 1e-9, 0.05):
        s.append(f"<text x='{ml - 6}' y='{Y(v) + 4:.1f}' text-anchor='end'>{v:.2f}</text>")
    for nm in ("F0", "FY"):
        xv = X(floors[nm]["psnr"])
        s.append(f"<line x1='{xv:.1f}' y1='20' x2='{xv:.1f}' y2='{H - mb}' stroke='#999' stroke-dasharray='4 3'/>")
        s.append(f"<text x='{xv + 3:.1f}' y='30'>{nm}</text>")
    for k, px, py in pts:
        fam = k.split("@")[0].split("_")[0]
        c = col.get(fam, "#555")
        s.append(f"<circle cx='{X(px):.1f}' cy='{Y(py):.1f}' r='4' fill='{c}'/>")
        s.append(f"<text x='{X(px) + 6:.1f}' y='{Y(py) - 4:.1f}' fill='{c}'>{k}</text>")
    s.append("</svg>")
    return "\n".join(s) + "\n"


# ------------------------------------------------------------------------------------------------ commands

def run_tests():
    spec = importlib.util.spec_from_file_location("test_sebits", ROOT / "test_sebits.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["test_sebits"] = mod
    spec.loader.exec_module(mod)
    suite = unittest.defaultTestLoader.loadTestsFromModule(mod)
    return unittest.TextTestRunner(verbosity=1, stream=sys.stdout).run(suite)


def cmd_freeze(amend=None):
    """One freeze. A re-freeze needs a dated amendment in PREREGISTRATION.md and [researcher]'s word: `freeze <label>` keeps
    the old file as FREEZE_<label>.json and copies its data record."""
    old = None
    if (ROOT / "FREEZE.json").exists():
        if not amend:
            raise CheckFail("FREEZE.json exists; a re-freeze needs an amendment label (sebits.py freeze <label>)")
        old = read_json(ROOT / "FREEZE.json")
        if (ROOT / f"FREEZE_{amend}.json").exists():
            raise CheckFail(f"FREEZE_{amend}.json exists")
    r = run_tests()
    if r.failures or r.errors:
        raise CheckFail("unit tests failed; no freeze")
    if old is not None:
        (ROOT / "FREEZE.json").replace(ROOT / f"FREEZE_{amend}.json")
    write_json(ROOT / "FREEZE.json", {"time": now(), "sha256": {f: sha256_file(ROOT / f) for f in FROZEN},
                                     "tests": {"run": r.testsRun, "failures": 0, "errors": 0},
                                     "data": old.get("data") if old else None,
                                     "amends": f"FREEZE_{amend}.json" if old else None,
                                     "rule": "Every job checks these hashes and stops on a difference. The data step "
                                             "fills 'data' one time (archive and npz sha256)."})
    log("FREEZE.json written")


def cmd_data(archive):
    fz = check_freeze([])
    if fz.get("data"):
        raise CheckFail("FREEZE.json has a data record already")
    archive = Path(archive)
    h, md5 = hashlib.sha256(), hashlib.md5()
    with open(archive, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            md5.update(chunk)
    rec = {"url": ARCHIVE_URL, "archive": archive.name, "bytes": archive.stat().st_size, "sha256": h.hexdigest(),
           "md5": md5.hexdigest(), "md5_page": ARCHIVE_MD5, "time": now()}
    if rec["md5"] != ARCHIVE_MD5:
        raise CheckFail(f"data: md5 {rec['md5']} differs from the dataset page ({ARCHIVE_MD5})")
    xtr, ytr, xte, yte = read_archive(archive)
    counts = np.bincount(ytr, minlength=10).tolist(), np.bincount(yte, minlength=10).tolist()
    if counts[0] != [5000] * 10 or counts[1] != [1000] * 10:
        raise CheckFail(f"data: class counts {counts}")
    DATA_DIR.mkdir(exist_ok=True)
    np.savez(DATA_DIR / "cifar10_train.npz", x=xtr, y=ytr)
    np.savez(DATA_DIR / "cifar10_test_images.npz", x=xte)
    np.savez(DATA_DIR / "cifar10_test_labels.npz", y=yte)
    rec["npz_sha256"] = {f: sha256_file(DATA_DIR / f) for f in
                         ("cifar10_train.npz", "cifar10_test_images.npz", "cifar10_test_labels.npz")}
    rec["class_counts"] = {"train": counts[0], "test": counts[1]}
    fz["data"] = rec
    write_json(ROOT / "FREEZE.json", fz)
    log("data record: " + json.dumps(rec))


def main(argv=None):
    global DEV
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["test", "freeze", "data", "j0", "j1", "j2", "j3", "j4", "j5", "j6"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--part", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "test":
        r = run_tests()
        return 0 if not (r.failures or r.errors) else 1
    try:
        if a.cmd == "freeze":
            cmd_freeze(a.arg)
        elif a.cmd == "data":
            cmd_data(a.arg)
        else:
            if a.cmd != "j6":
                if not torch.backends.mps.is_available():
                    raise CheckFail("MPS is not available")
                DEV = torch.device("mps")
            log(f"job {a.cmd} start (device {DEV}, part {a.part or 'all'})")
            if a.cmd == "j0":
                j0()
            elif a.cmd == "j6":
                j6()
            else:
                run_job(a.cmd, a.part)
            log(f"job {a.cmd} end")
        return 0
    except StopRule as e:
        log(f"STOP RULE: {e}")
        write_json(RES / f"STOP_{a.cmd}.json", {"time": now(), "stop_rule": str(e)})
        return 3
    except CheckFail as e:
        log(f"CHECK FAIL: {e}")
        write_json(RES / f"CHECKFAIL_{a.cmd}.json", {"time": now(), "check": str(e)})
        return 4


if __name__ == "__main__":
    sys.exit(main())
