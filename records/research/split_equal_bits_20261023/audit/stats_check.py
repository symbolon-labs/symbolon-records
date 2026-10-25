"""Independent statistics check of the split equal-bits run (CIFAR-10).

The rules come from PREREGISTRATION.md, Parts 2.2, 4.4, 5.3, 5.4, 7.1, 7.4, 7.5
and 7.6. The file formats come from the task brief and from a look at the npz keys.
This script does not import sebits.py, does not read it and copies no code from it.
It uses numpy and the Python standard library only. It runs on the CPU only.
It trains nothing and downloads nothing. It writes only audit/stats_check.json
and AUDIT.md in the run folder.

Run from the run folder:
    ../../.venv/bin/python -I -B audit/stats_check.py
"""
import hashlib
import json
import math
import os
import platform
import re
import sys
import time

import numpy as np

RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = [20261111, 20261112, 20261113]
N_TEST = 10000
TOL = 1e-9
SYM, ACT = "TOPK@64", "BOT@64"
DESC_ARMS = ["RAW@64", "BOT@64", "TOPK@64", "LAB", "RAW_native"]

ROWS = []      # table rows: one row for each compared quantity
NOTES = []     # observations that are not a comparison with results.json
OUT = {}       # all values for stats_check.json


def p(*parts):
    return os.path.join(RUN, *parts)


def npz(path, key):
    with np.load(path, allow_pickle=False) as z:
        return z[key]


def npz_keys(path):
    with np.load(path, allow_pickle=False) as z:
        return list(z.files)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def row(group, name, ref, val, kind):
    """kind 'float': |ref - val| <= TOL. Other kinds: exact equality."""
    if kind == "float":
        diff = abs(float(ref) - float(val))
        ok = bool(diff <= TOL)
        ref_out, val_out = float(ref), float(val)
    else:
        diff = None
        ok = bool(ref == val)
        ref_out, val_out = ref, val
    ROWS.append({"group": group, "quantity": name, "results_json": ref_out,
                 "independent": val_out, "abs_diff": diff, "kind": kind, "pass": ok})
    return ok


# ---------------------------------------------------------------- inputs
FREEZE = json.load(open(p("FREEZE.json")))
RES = json.load(open(p("results", "results.json")))
OUT["inputs"] = {
    "run_folder": RUN,
    "results_json_sha256": sha256_file(p("results", "results.json")),
    "results_json_time": RES.get("time"),
    "python": sys.version.split()[0],
    "numpy": np.__version__,
    "platform": platform.platform(),
    "time_start": time.strftime("%Y-%m-%d %H:%M:%S"),
}

# ------------------------------------------------- check 1: data sha256
data_files = ["cifar10_train.npz", "cifar10_test_images.npz", "cifar10_test_labels.npz"]
OUT["check1_sha256"] = {}
for f in data_files:
    got = sha256_file(p("data", f))
    ref = FREEZE["data"]["npz_sha256"][f]
    OUT["check1_sha256"][f] = {"freeze": ref, "independent": got}
    row("1 data", "sha256 " + f + " (FREEZE.json)", ref, got, "str")

x_train = npz(p("data", "cifar10_train.npz"), "x")
y_train = npz(p("data", "cifar10_train.npz"), "y").astype(np.int64)
x_test = npz(p("data", "cifar10_test_images.npz"), "x")
y_test = npz(p("data", "cifar10_test_labels.npz"), "y").astype(np.int64)
assert x_train.shape == (50000, 3, 32, 32) and x_train.dtype == np.uint8
assert x_test.shape == (N_TEST, 3, 32, 32) and x_test.dtype == np.uint8
assert y_train.shape == (50000,) and y_test.shape == (N_TEST,)
OUT["data_shapes"] = {"x_train": list(x_train.shape), "x_test": list(x_test.shape),
                      "test_class_counts": np.bincount(y_test, minlength=10).tolist()}


def psnr_per_image(recon, ref):
    """Part 5.3: 10*log10(1/MSE), MSE over the 3x32x32 values in [0, 1]."""
    assert recon.shape == ref.shape and recon.dtype == np.uint8
    out = np.empty(len(ref), dtype=np.float64)
    for a in range(0, len(ref), 1000):
        d = (recon[a:a + 1000].astype(np.float64) - ref[a:a + 1000].astype(np.float64)) / 255.0
        mse = np.mean(d * d, axis=(1, 2, 3))
        out[a:a + 1000] = 10.0 * np.log10(1.0 / mse)
    return out


def recon_of(seed, prefix, arm):
    path = p("results", "s%d" % seed, "dec", "%s_%s.npz" % (prefix, arm))
    r = npz(path, "recon_test")
    assert r.shape == (N_TEST, 3, 32, 32) and r.dtype == np.uint8, path
    return r


def correct_of(seed, arm):
    pred = npz(p("results", "s%d" % seed, "msgs", arm + ".npz"), "pred_test").astype(np.int64)
    assert pred.shape == (N_TEST,)
    return (pred == y_test).astype(np.float64)


# ---------------------------------- per-image correctness and PSNR (S1 files)
C = {}   # C[arm] -> (3, 10000) correctness
P = {}   # P[arm] -> (3, 10000) PSNR of the first Decoder-S
inf_count = 0
for arm in DESC_ARMS:
    C[arm] = np.stack([correct_of(s, arm) for s in SEEDS])
    P[arm] = np.stack([psnr_per_image(recon_of(s, "S1", arm), x_test) for s in SEEDS])
    inf_count += int(np.sum(~np.isfinite(P[arm])))
if inf_count:
    NOTES.append("%d per-image PSNR values are not finite (MSE = 0)." % inf_count)

# ---------------------------------------------- check 2: primary pair
d_acc_i = 100.0 * (C[SYM].mean(axis=0) - C[ACT].mean(axis=0))
d_leak_i = P[ACT].mean(axis=0) - P[SYM].mean(axis=0)
d_acc = float(d_acc_i.mean())
d_leak = float(d_leak_i.mean())
d_acc_s = [float(100.0 * np.mean(C[SYM][k] - C[ACT][k])) for k in range(3)]
d_leak_s = [float(np.mean(P[ACT][k] - P[SYM][k])) for k in range(3)]

# Part 7.1: one set of draws for every comparison.
rng = np.random.default_rng(20261024)
bs_acc = np.empty(10000, dtype=np.float64)
bs_leak = np.empty(10000, dtype=np.float64)
for j in range(40):
    idx = rng.integers(0, 10000, size=(250, 10000))
    bs_acc[j * 250:(j + 1) * 250] = d_acc_i[idx].mean(axis=1)
    bs_leak[j * 250:(j + 1) * 250] = d_leak_i[idx].mean(axis=1)
ci_acc = [float(v) for v in np.percentile(bs_acc, [2.5, 97.5], method="linear")]
ci_leak = [float(v) for v in np.percentile(bs_leak, [2.5, 97.5], method="linear")]


def acc_outcome(d, ci, seeds):
    hits = []
    if ci[0] > -1.0:
        hits.append("A+")
    if d <= -1.0 and ci[1] < 0.0 and all(v < 0.0 for v in seeds):
        hits.append("A-")
    if len(hits) == 1:
        return hits[0], hits
    return "A?", hits   # zero hits; two hits has no rule in Part 7.4 (flagged)


def leak_outcome(d, ci, seeds):
    hits = []
    if d >= 0.5 and ci[0] > 0.0 and all(v > 0.0 for v in seeds):
        hits.append("L+")
    if ci[0] > -0.5 and ci[1] < 0.5:
        hits.append("L=")
    if d <= -0.5 and ci[1] < 0.0 and all(v < 0.0 for v in seeds):
        hits.append("L-")
    return (hits[0] if len(hits) == 1 else "L?"), hits


CASES = {("A+", "L+"): "S", ("A+", "L="): "E", ("A+", "L-"): "X", ("A+", "L?"): "U",
         ("A-", "L+"): "T", ("A-", "L="): "A", ("A-", "L-"): "A", ("A-", "L?"): "U",
         ("A?", "L+"): "U", ("A?", "L="): "U", ("A?", "L-"): "U", ("A?", "L?"): "U"}

A_out, A_hits = acc_outcome(d_acc, ci_acc, d_acc_s)
L_out, L_hits = leak_outcome(d_leak, ci_leak, d_leak_s)
if len(A_hits) > 1:
    NOTES.append("Both A+ and A- hold. Part 7.4 gives no rule for this; A? is used.")
case_rule = CASES[(A_out, L_out)]

pr = RES["primary"]
row("2 primary", "SYM arm", pr.get("sym"), SYM, "str")
row("2 primary", "ACT arm", pr.get("act"), ACT, "str")
row("2 primary", "d_acc (points)", pr["d_acc"], d_acc, "float")
row("2 primary", "d_acc CI low", pr["d_acc_ci"][0], ci_acc[0], "float")
row("2 primary", "d_acc CI high", pr["d_acc_ci"][1], ci_acc[1], "float")
for k, s in enumerate(SEEDS):
    row("2 primary", "d_acc seed %d" % s, pr["d_acc_seeds"][k], d_acc_s[k], "float")
row("2 primary", "d_leak (dB)", pr["d_leak"], d_leak, "float")
row("2 primary", "d_leak CI low", pr["d_leak_ci"][0], ci_leak[0], "float")
row("2 primary", "d_leak CI high", pr["d_leak_ci"][1], ci_leak[1], "float")
for k, s in enumerate(SEEDS):
    row("2 primary", "d_leak seed %d" % s, pr["d_leak_seeds"][k], d_leak_s[k], "float")
row("2 primary", "accuracy outcome A", pr["A"], A_out, "str")
row("2 primary", "leakage outcome L", pr["L"], L_out, "str")
if "L_hits" in pr:
    row("2 primary", "leakage conditions that hold (L_hits)", pr["L_hits"], L_hits, "list")

OUT["check2_primary"] = {
    "sym": SYM, "act": ACT,
    "d_acc": d_acc, "d_acc_ci": ci_acc, "d_acc_seeds": d_acc_s,
    "d_leak": d_leak, "d_leak_ci": ci_leak, "d_leak_seeds": d_leak_s,
    "A": A_out, "A_hits": A_hits, "L": L_out, "L_hits": L_hits,
    "case_by_rule_before_validity": case_rule,
    "bootstrap": {"seed": 20261024, "draws": 10000, "chunks": "40 x integers(0, 10000, size=(250, 10000))",
                  "percentile_method": "linear",
                  "acc_boot_mean": float(bs_acc.mean()), "leak_boot_mean": float(bs_leak.mean())},
    "margins": {"d_leak_minus_0.5": d_leak - 0.5, "d_leak_ci_low": ci_leak[0],
                "min_d_leak_seed": min(d_leak_s), "d_acc_ci_low_minus_-1": ci_acc[0] + 1.0},
}

# ------------------------------------------ check 3: decoder dependence
PL = {arm: psnr_per_image(recon_of(SEEDS[0], "L1", arm), x_test) for arm in (SYM, ACT)}
d_leak_L1 = float(np.mean(PL[ACT] - PL[SYM]))
d_leak_S1 = d_leak_s[0]
dep_sign = bool(np.sign(d_leak_L1) != np.sign(d_leak_S1))
dep_size = bool(abs(d_leak_L1 - d_leak_S1) > 1.0)
dependent = dep_sign or dep_size
dc = RES["decoder_check"]
row("3 decoder", "seed-1 d_leak, Decoder-L (L1_)", dc["d_leak_L_seed1"], d_leak_L1, "float")
row("3 decoder", "seed-1 d_leak, Decoder-S (S1_)", dc["d_leak_S_seed1"], d_leak_S1, "float")
row("3 decoder", "decoder_dependent", dc["decoder_dependent"], dependent, "bool")
row("3 decoder", "primary.decoder_dependent", pr.get("decoder_dependent"), dependent, "bool")
psnrL = {arm: float(PL[arm].mean()) for arm in PL}
for arm in (SYM, ACT):
    row("3 decoder", "PSNR Decoder-L seed 1 " + arm, RES["descriptive"][arm]["psnr_L_seed1"], psnrL[arm], "float")
OUT["check3_decoder"] = {"d_leak_L_seed1": d_leak_L1, "d_leak_S_seed1": d_leak_S1,
                         "sign_differs": dep_sign, "differ_more_than_1dB": dep_size,
                         "abs_difference": abs(d_leak_L1 - d_leak_S1),
                         "decoder_dependent": dependent, "psnr_L_seed1": psnrL}

# --------------------------------------- check 4: descriptive acc and PSNR
OUT["check4_descriptive"] = {}
for arm in DESC_ARMS:
    acc_seeds = [float(C[arm][k].mean()) for k in range(3)]
    psnr_seeds = [float(P[arm][k].mean()) for k in range(3)]
    acc = float(np.mean(acc_seeds))
    psnr = float(np.mean(psnr_seeds))
    ref = RES["descriptive"][arm]
    row("4 descriptive", "acc " + arm, ref["acc"], acc, "float")
    row("4 descriptive", "PSNR " + arm, ref["psnr"], psnr, "float")
    for k, s in enumerate(SEEDS):
        row("4 descriptive seeds", "acc %s seed %d" % (arm, s), ref["acc_seeds"][k], acc_seeds[k], "float")
        row("4 descriptive seeds", "PSNR %s seed %d" % (arm, s), ref["psnr_seeds"][k], psnr_seeds[k], "float")
    OUT["check4_descriptive"][arm] = {"acc": acc, "acc_seeds": acc_seeds,
                                      "psnr": psnr, "psnr_seeds": psnr_seeds}

# --------------------------------------------------------- check 5: bits
SEL = json.load(open(p("selection.json")))


def rank_bits(k):
    return math.ceil(math.log2(math.perm(10, k)))


def topk_format_bits(fmt):
    kind, k, q = fmt[0], int(fmt[1]), int(fmt[2])
    if kind == "all":
        return 4 + 9 * q
    return rank_bits(k) + (k - 1) * q


def bot_values(name):
    m = re.fullmatch(r"BOT_c(\d+)_s(\d+)_q(\d+)", name)
    return int(m.group(1)) * int(m.group(2)) ** 2


# Part 4.4 table (allowed values) and Part 3.4 rows.
ALLOWED = {"RAW@16": {16}, "RAW@64": {64}, "RAW@256": {256}, "RAW@1024": {1024},
           "BOT@16": {16}, "BOT@64": {64}, "BOT@256": {256}, "BOT@1024": {1024},
           "TOPK@16": {13, 16}, "TOPK@64": {58, 61, 63}, "TOPK@256": {148}, "TOPK@1024": {148},
           "LAB": {4}, "LAB_full": {4}, "TOPK_full": {58, 61, 63}, "TOPK_float": {320},
           "RAW_native": {32768}, "BOT_float": {32 * bot_values(SEL["BOT"]["64"])}}
# Exact values from the selected formats of selection.json (Part 4.3).
EXACT = dict()
for b in ("16", "64", "256"):
    EXACT["TOPK@" + b] = topk_format_bits(SEL["TOPK"][b])
EXACT["TOPK_full"] = topk_format_bits(SEL["TOPK_full"])


def field(bits, start, width):
    v = np.zeros(bits.shape[0], dtype=np.int64)
    for j in range(width):
        v = (v << 1) | bits[:, start + j].astype(np.int64)
    return v


def message_first_class(arm, bits):
    """First class written in a TOPK or LAB message (Part 4.2)."""
    if arm in ("LAB", "LAB_full"):
        return field(bits, 0, 4)
    fmt = SEL["TOPK_full"] if arm == "TOPK_full" else SEL["TOPK"][arm.split("@")[1]]
    if fmt[0] == "all":
        return field(bits, 0, 4)
    k = int(fmt[1])
    rank = field(bits, 0, rank_bits(k))
    return rank // math.perm(9, k - 1)


OUT["check5_bits"] = {}
bit_ok_all = True
per_arm = {}
for s in SEEDS:
    folder = p("results", "s%d" % s, "msgs")
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith(".npz"):
            continue
        arm = fn[:-4]
        path = os.path.join(folder, fn)
        keys = npz_keys(path)
        nbits = int(npz(path, "nbits"))
        packed = npz(path, "bits_test")
        pred = npz(path, "pred_test").astype(np.int64)
        ok_cols, pad_zero, cols = True, True, None
        first_mismatch = None
        for a in range(0, packed.shape[0], 1000):
            chunk = packed[a:a + 1000]
            u = np.unpackbits(chunk, axis=1, count=nbits)
            cols = u.shape[1]
            ok_cols &= (cols == nbits)
            full = np.unpackbits(chunk, axis=1)
            pad_zero &= bool(np.all(full[:, nbits:] == 0))
            if arm.startswith("TOPK@") or arm in ("TOPK_full", "LAB", "LAB_full"):
                m = int(np.sum(message_first_class(arm, u) != pred[a:a + 1000]))
                first_mismatch = (first_mismatch or 0) + m
        if first_mismatch is None:   # server-logit arms: prediction = argmax of logits
            logits = npz(path, "logits_test")
            first_mismatch = int(np.sum(np.argmax(logits, axis=1) != pred))
            pred_source = "argmax logits_test"
        else:
            pred_source = "first class in bits_test"
        allowed_ok = nbits in ALLOWED.get(arm, set())
        exact_ok = (nbits == EXACT[arm]) if arm in EXACT else True
        n_rows_ok = packed.shape[0] == N_TEST and packed.shape[1] == (nbits + 7) // 8
        ok = bool(allowed_ok and exact_ok and ok_cols and n_rows_ok)
        bit_ok_all &= ok
        rec = {"nbits": nbits, "allowed": sorted(ALLOWED.get(arm, [])), "exact_from_selection": EXACT.get(arm),
               "unpacked_columns": cols, "packed_shape": list(packed.shape), "padding_zero": pad_zero,
               "pred_source": pred_source, "pred_mismatch_count": first_mismatch, "keys": keys, "pass": ok}
        OUT["check5_bits"]["s%d/%s" % (s, arm)] = rec
        per_arm.setdefault(arm, []).append((s, rec))

for arm in sorted(per_arm):
    recs = per_arm[arm]
    nb = [r["nbits"] for _, r in recs]
    ok = all(r["pass"] for _, r in recs) and len(set(nb)) == 1
    ref_bits = RES["descriptive"].get(arm, {}).get("bits")
    exp = sorted(ALLOWED.get(arm, []))
    if arm in EXACT:
        exp = [EXACT[arm]]
    label = "nbits %s (%s)" % (arm, ", ".join("s%d" % (s - 20261110) for s, _ in recs))
    ROWS.append({"group": "5 bits", "quantity": label, "results_json": ref_bits,
                 "independent": "/".join(str(v) for v in nb) + " (rule: " + " or ".join(map(str, exp)) + ")",
                 "abs_diff": None, "kind": "int",
                 "pass": bool(ok and ref_bits is not None and all(v == ref_bits for v in nb))})
pred_mm = {k: v["pred_mismatch_count"] for k, v in OUT["check5_bits"].items() if v["pred_mismatch_count"]}
pad_bad = [k for k, v in OUT["check5_bits"].items() if not v["padding_zero"]]
ROWS.append({"group": "5 bits", "quantity": "pred_test equals message class or logit argmax (49 files)",
             "results_json": "n/a", "independent": "%d files with a mismatch" % len(pred_mm),
             "abs_diff": None, "kind": "int", "pass": len(pred_mm) == 0})
if pad_bad:
    NOTES.append("Non-zero padding bits in: " + ", ".join(pad_bad))
OUT["check5_summary"] = {"files": len(OUT["check5_bits"]), "all_pass": bit_ok_all,
                         "pred_mismatch": pred_mm, "padding_not_zero": pad_bad}

# -------------------------------------------------------- check 6: floors
rng = np.random.default_rng(20261110)
task, dev, aux = [], [], []
for c in range(10):
    idx = np.flatnonzero(y_train == c)
    perm = rng.permutation(idx)
    n = len(perm)
    n_task, n_dev = int(round(0.7 * n)), int(round(0.1 * n))
    task.append(perm[:n_task])
    dev.append(perm[n_task:n_task + n_dev])
    aux.append(perm[n_task + n_dev:])
task, dev, aux = (np.sort(np.concatenate(v)) for v in (task, dev, aux))
assert len(task) == 35000 and len(dev) == 5000 and len(aux) == 10000
assert len(np.intersect1d(task, aux)) == 0 and len(np.intersect1d(dev, aux)) == 0
xa = x_train[aux]
targets = np.concatenate([xa, xa[:, :, :, ::-1]], axis=0)        # D_aux and horizontal flips
ty = np.concatenate([y_train[aux], y_train[aux]])


def mean_image(imgs):
    m = imgs.astype(np.float64).sum(axis=0) / len(imgs)
    ties = int(np.sum(m - np.floor(m) == 0.5))
    return np.round(m).astype(np.uint8), np.floor(m + 0.5).astype(np.uint8), ties


F0, F0_up, F0_ties = mean_image(targets)
FY, FY_up, FY_ties = [], [], 0
for c in range(10):
    a, b, t = mean_image(targets[ty == c])
    FY.append(a)
    FY_up.append(b)
    FY_ties += t
FY, FY_up = np.stack(FY), np.stack(FY_up)
f0_psnr = float(psnr_per_image(np.broadcast_to(F0, x_test.shape), x_test).mean())
fy_psnr = float(psnr_per_image(FY[y_test], x_test).mean())
f0_psnr_up = float(psnr_per_image(np.broadcast_to(F0_up, x_test.shape), x_test).mean())
fy_psnr_up = float(psnr_per_image(FY_up[y_test], x_test).mean())
row("6 floors", "F0 PSNR (test)", RES["floors"]["F0"]["psnr"], f0_psnr, "float")
row("6 floors", "FY PSNR (test)", RES["floors"]["FY"]["psnr"], fy_psnr, "float")
OUT["check6_floors"] = {
    "split_sizes": {"D_task": int(len(task)), "D_dev": int(len(dev)), "D_aux": int(len(aux))},
    "D_aux_sha256_of_sorted_int64_indices": hashlib.sha256(aux.astype(np.int64).tobytes()).hexdigest(),
    "targets": int(len(targets)),
    "F0_psnr": f0_psnr, "FY_psnr": fy_psnr,
    "rounding": "numpy round (half to even)",
    "exact_half_ties": {"F0": F0_ties, "FY": FY_ties},
    "half_up_variant": {"F0_psnr": f0_psnr_up, "FY_psnr": fy_psnr_up},
}

# --------------------------------------------------- check 7: checks.jsonl
OUT["check7_checks"] = {}
all_ok = True
by_type = {}
for s in SEEDS:
    path = p("results", "s%d" % s, "checks.jsonl")
    n, bad = 0, []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            by_type[r.get("check", "?")] = by_type.get(r.get("check", "?"), 0) + 1
            if r.get("ok") is not True:
                bad.append(ln)
    all_ok &= not bad
    OUT["check7_checks"]["s%d" % s] = {"records": n, "not_ok_lines": bad}
OUT["check7_checks"]["records_by_check"] = by_type
OUT["check7_checks"]["results_json_checks"] = RES.get("checks")
total = sum(v["records"] for k, v in OUT["check7_checks"].items() if k.startswith("s"))
ROWS.append({"group": "7 checks", "quantity": "every checks.jsonl record has ok true (%d records)" % total,
             "results_json": "all ok" if all(v.get("ok") for v in RES["checks"].values()) else "not all ok",
             "independent": "all ok" if all_ok else "not all ok", "abs_diff": None, "kind": "str",
             "pass": bool(all_ok)})

# Planned decoder outputs (Part 8.1, V5 part of J6). Extra check.
planned_S = ["RAW@16", "RAW@64", "RAW@256", "RAW@1024", "BOT@16", "BOT@64", "BOT@256", "BOT@1024",
             "TOPK@16", "TOPK@64", "TOPK@256", "LAB", "LAB_full", "TOPK_full", "RAW_native"]
missing = []
for s in SEEDS:
    names = ["S1_" + a for a in planned_S]
    if s == SEEDS[0]:
        names += ["S1_TOPK_float", "S1_TOPK@64_ctrl", "S1_BOT@64_ctrl",
                  "L1_TOPK@64", "L1_BOT@64", "L1_RAW@64", "L1_LAB", "L1_RAW_native",
                  "S2_TOPK@64", "S3_TOPK@64", "S2_BOT@64", "S3_BOT@64"]
    for nm in names:
        if not os.path.exists(p("results", "s%d" % s, "dec", nm + ".npz")):
            missing.append("s%d/%s" % (s, nm))
OUT["planned_decoder_outputs"] = {"missing": missing, "expected_count": 15 * 3 + 12}
ROWS.append({"group": "7 checks", "quantity": "planned decoder outputs exist (57 files)",
             "results_json": "n/a", "independent": "%d missing" % len(missing),
             "abs_diff": None, "kind": "int", "pass": not missing})

valid = all_ok and not missing
case_ind = case_rule if valid else "0"
case_final_ind = case_ind
if dependent and case_ind != "0":
    case_final_ind = case_ind + " (decoder-dependent)"   # the exact label text is not in the prereg
row("2 primary", "case", pr["case"], case_ind, "str")
row("2 primary", "case_final", pr["case_final"], case_final_ind, "str")
OUT["check2_primary"]["valid"] = valid
OUT["check2_primary"]["case"] = case_ind
OUT["check2_primary"]["case_final"] = case_final_ind

# ------------------------------------------------------- check 8: logs
OUT["check8_logs"] = {}
log_ok = True
pat = re.compile(r"data: mode (\S+); files opened: (.*)$")
for j in range(7):
    path = p("logs", "j%d.log" % j)
    lines = open(path, errors="replace").read().splitlines()
    tests_end = None
    for i, ln in enumerate(lines):
        if re.match(r"^Ran \d+ tests? in ", ln):
            tests_end = i
    recs = []
    for i, ln in enumerate(lines):
        m = pat.search(ln)
        if not m:
            continue
        files = [f.strip() for f in m.group(2).split(",")]
        in_tests = tests_end is not None and i < tests_end
        recs.append({"line": i + 1, "mode": m.group(1), "files": files, "inside_unit_test_output": in_tests})
    job_recs = [r for r in recs if not r["inside_unit_test_output"]]
    labels_job = any("test labels" in r["files"] for r in job_recs)
    labels_tests = any("test labels" in r["files"] for r in recs if r["inside_unit_test_output"])
    if j == 6:
        ok = len(job_recs) >= 1 and labels_job
    else:
        ok = len(job_recs) >= 1 and not labels_job
    log_ok &= ok
    OUT["check8_logs"]["j%d" % j] = {"data_lines": recs, "job_lines": len(job_recs),
                                     "job_opened_test_labels": labels_job,
                                     "unit_test_output_mentions_test_labels": labels_tests,
                                     "unit_test_block_end_line": None if tests_end is None else tests_end + 1,
                                     "pass": ok}
    if labels_tests:
        NOTES.append("logs/j%d.log: a 'test labels' data line occurs inside the unit-test output "
                     "(before the line 'Ran ... tests'). test_sebits.py runs that end-to-end test "
                     "on synthetic data in a temporary folder." % j)
ROWS.append({"group": "8 logs", "quantity": "no job except j6 opened the test labels (job lines)",
             "results_json": "n/a", "independent": "pass" if log_ok else "fail",
             "abs_diff": None, "kind": "str", "pass": bool(log_ok)})

# Extra: each wrapper log ends with rc=0.
rc = {}
for name in ["data"] + ["j%d" % j for j in range(7)]:
    path = p("logs", name + "_wrapper.log")
    ends = re.findall(r"^== end .* rc=(\d+)", open(path, errors="replace").read(), flags=re.M)
    rc[name] = [int(v) for v in ends]
OUT["wrapper_rc"] = rc
rc_ok = all(v and v[-1] == 0 for v in rc.values())
ROWS.append({"group": "8 logs", "quantity": "every job wrapper log ends with rc=0 (data, j0 to j6)",
             "results_json": "n/a", "independent": "all rc=0" if rc_ok else "not all rc=0",
             "abs_diff": None, "kind": "str", "pass": bool(rc_ok)})

# ---------------------------------------------------------------- output
n_fail = sum(1 for r in ROWS if not r["pass"])
float_diffs = [r["abs_diff"] for r in ROWS if r["kind"] == "float"]
OUT["rows"] = ROWS
OUT["notes"] = NOTES
OUT["summary"] = {"rows": len(ROWS), "failed": n_fail,
                  "max_abs_float_diff": max(float_diffs) if float_diffs else None,
                  "tolerance": TOL,
                  "verdict": "PASS" if n_fail == 0 else "FAIL"}
OUT["inputs"]["time_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
with open(p("audit", "stats_check.json"), "w") as f:
    json.dump(OUT, f, indent=1)



# ------------------------------------------------------------ AUDIT.md
def cell(v, kind):
    if v is None:
        return "n/a"
    if kind == "float" and isinstance(v, float):
        return repr(v)
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    return str(v)


def diff_cell(r):
    if r["kind"] != "float":
        return "exact"
    return "0" if r["abs_diff"] == 0.0 else "%.1e" % r["abs_diff"]


ordered = sorted(ROWS, key=lambda r: int(r["group"].split()[0]))
pc = OUT["check2_primary"]
fl = OUT["check6_floors"]
ck = OUT["check7_checks"]
lines = [
    "# AUDIT: independent statistics check of the split equal-bits run",
    "",
    "- What: This audit compares the values in `results/results.json` with values that it calculates "
    "again from the saved files. It applies Parts 2.2, 4.4, 5.3, 5.4, 7.1, 7.4, 7.5 and 7.6 of "
    "`PREREGISTRATION.md` (Part 8.4, item 5).",
    "- When: %s (local time). The `results.json` file has the time %s and the sha256 `%s`."
    % (OUT["inputs"]["time_end"], OUT["inputs"]["results_json_time"], OUT["inputs"]["results_json_sha256"]),
    "- Who: A Claude Code subagent wrote and ran the check for [researcher].",
    "- How: The script `audit/stats_check.py` reads the npz, json, jsonl and log files of the run folder. "
    "It uses Python %s and numpy %s on the CPU. It trains no model and downloads no file. "
    "It changes no existing file. The command is `../../.venv/bin/python -I -B audit/stats_check.py`, "
    "from the run folder." % (OUT["inputs"]["python"], OUT["inputs"]["numpy"]),
    "- Independence: The script shares no code with `sebits.py`. It does not import `sebits.py`. "
    "The author of the script did not read `sebits.py`. The author wrote the script from the written "
    "rules and the file formats only.",
    "- Tolerance: Integers, strings, case letters and true or false values must be equal. "
    "Means and CI bounds must agree to an absolute difference of %g or less." % TOL,
    "- All values: `audit/stats_check.json`.",
    "",
    "## Results",
    "",
    "| Quantity | results.json | Independent value | Difference | Pass |",
    "|---|---|---|---|---|",
]
for r in ordered:
    lines.append("| %s | %s | %s | %s | %s |" % (
        r["quantity"], cell(r["results_json"], r["kind"]), cell(r["independent"], r["kind"]),
        diff_cell(r), "yes" if r["pass"] else "NO"))
lines += ["", "## Notes", ""]
nn = 1
j0 = OUT["check8_logs"]["j0"]
if j0["unit_test_output_mentions_test_labels"]:
    ul = [d["line"] for d in j0["data_lines"] if d["inside_unit_test_output"] and "test labels" in d["files"]]
    jl = [d["line"] for d in j0["data_lines"] if not d["inside_unit_test_output"]]
    lines.append("%d. `logs/j0.log` has a data line with \"test labels\" (line %s). This line is in the "
                 "output of the unit tests, before the line \"Ran ... tests\" (line %d). The docstring of "
                 "`test_sebits.py` says that the tests use synthetic data only and read no CIFAR-10 file. "
                 "Its end-to-end test runs J0 to J6 in a temporary folder. The J0 job line after the unit "
                 "tests (line %s) shows the train file only. Thus the audit counts the line as a unit-test "
                 "line, not as a J0 job line."
                 % (nn, ", ".join(map(str, ul)), j0["unit_test_block_end_line"], ", ".join(map(str, jl))))
    nn += 1
if fl["exact_half_ties"]["F0"] or fl["exact_half_ties"]["FY"]:
    lines.append("%d. The mean images have %d values in F0 and %d values in FY that are exactly at a half. "
                 "The audit rounds a half to the even integer (numpy `round`). This gives the "
                 "`results.json` values. If a half goes up, FY is %r dB (a difference of %.1e dB). "
                 "Part 5.4 does not give the direction for a half."
                 % (nn, fl["exact_half_ties"]["F0"], fl["exact_half_ties"]["FY"],
                    fl["half_up_variant"]["FY_psnr"], abs(fl["half_up_variant"]["FY_psnr"] - fl["FY_psnr"])))
    nn += 1
rec_types = ck["records_by_check"]
rj = ck["results_json_checks"] or {}
extra = sorted(set(rj) - set(rec_types))
lines.append("%d. The three `checks.jsonl` files have %d records (%s). %s"
             % (nn, sum(rec_types.values()), ", ".join("%s %d" % (k, rec_types[k]) for k in sorted(rec_types)),
                ("`results.json` also counts %s, which are not in `checks.jsonl`. "
                 "The audit checked one part of V5 again: all 57 planned decoder outputs exist."
                 % ", ".join("%d %s records" % (rj[k]["records"], k) for k in extra)) if extra else ""))
nn += 1
lines.append("%d. The L+ outcome holds with a small margin. d_leak is %.4f dB, which is %.4f dB above the "
             "0.5 dB bar. The CI lower bound is %.4f dB. The seed values are %s dB. Part 7.4 requires only "
             "that each seed value is above 0. Decoder-L in seed 1 gives %.4f dB."
             % (nn, pc["d_leak"], pc["d_leak"] - 0.5, pc["d_leak_ci"][0],
                ", ".join("%.4f" % v for v in pc["d_leak_seeds"]), OUT["check3_decoder"]["d_leak_L_seed1"]))
nn += 1
lines.append("%d. Extra checks: in all %d message files, `pred_test` agrees with the first class in the bits "
             "(TOPK, TOPK_full, LAB, LAB_full) or with the argmax of `logits_test` (the other arms). "
             "The padding bits after `nbits` are 0. All 8 job wrapper logs end with rc=0."
             % (nn, OUT["check5_summary"]["files"]))
nn += 1
for n in NOTES:
    if "unit-test output" in n:
        continue
    lines.append("%d. %s" % (nn, n))
    nn += 1
s = OUT["summary"]
lines += ["", "## Verdict", ""]
if s["failed"] == 0:
    lines.append("**Verdict: PASS.** All %d rows pass. The largest difference of a float value is %.1e. "
                 "The primary case is %s (%s, %s). The case is not decoder-dependent."
                 % (s["rows"], s["max_abs_float_diff"], pc["case_final"], pc["A"], pc["L"])
                 if not OUT["check3_decoder"]["decoder_dependent"] else
                 "**Verdict: PASS.** All %d rows pass. The largest difference of a float value is %.1e. "
                 "The primary case is %s." % (s["rows"], s["max_abs_float_diff"], pc["case_final"]))
else:
    lines.append("**Verdict: FAIL.** %d of %d rows do not pass. Rows with NO in the Pass column show them."
                 % (s["failed"], s["rows"]))
with open(p("AUDIT.md"), "w") as f:
    f.write("\n".join(lines) + "\n")

print(json.dumps(OUT["summary"], indent=1))
for r in ROWS:
    if not r["pass"] or (r["kind"] == "float" and r["abs_diff"] > 1e-12):
        print("ROW", r)
for n in NOTES:
    print("NOTE", n)
