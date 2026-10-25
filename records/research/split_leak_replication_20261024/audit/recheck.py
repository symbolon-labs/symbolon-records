#!/usr/bin/env python3
"""Independent recomputation of results/results.json and results/table.md.

Sources: the text of PREREGISTRATION.md (this folder, Parts 5 to 9) and of the sealed
PREREGISTRATION.md (Parts 2.2, 5.3, 5.4, 7.1, 7.4, 7.5), FREEZE.json, and the saved data
files (npz, json, jsonl). This script does not open, import or copy sleak.py, sebits.py,
test_sleak.py or any audit/ file of the sealed study. It only hashes the frozen files
byte by byte, as FREEZE.json requires. Standard library and numpy only. CPU only.

Run: <venv python> -I -B audit/recheck.py
Output: audit/recheck.json and audit/table_rows.md.
"""
import hashlib
import json
import math
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
F = os.path.dirname(HERE)
SS = os.path.join(os.path.dirname(F), "split_equal_bits_20261023")
RES = os.path.join(F, "results")
DATA = os.path.join(SS, "data")

NEW = [20261121, 20261122, 20261123, 20261124, 20261125, 20261126]
SEALED = [20261111, 20261112, 20261113]
ARMS = ["TOPK@64", "BOT@64", "LAB"]
SYM, ACT = "TOPK@64", "BOT@64"
TOL = 1e-9
MARGIN = 0.5
T_DF = {5: 2.570582, 8: 2.306004}
BOOT_SEED_2L = 20261127
BOOT_SEED_IMG = 20261024
SPLIT_SEED = 20261110


# ---------------------------------------------------------------- helpers
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def npz_get(path, key):
    with np.load(path, allow_pickle=False) as z:
        return z[key]


def read_json(path):
    with open(path) as fh:
        return json.load(fh)


def psnr_images(recon, ref):
    """Per image PSNR = 10 log10(1 / MSE); MSE over 3x32x32 values in [0, 1] (float64)."""
    assert recon.shape == ref.shape and recon.dtype == np.uint8 and ref.dtype == np.uint8
    n = ref.shape[0]
    out = np.empty(n, dtype=np.float64)
    step = 500
    for a in range(0, n, step):
        r = recon[a:a + step].astype(np.float64) / 255.0
        t = ref[a:a + step].astype(np.float64) / 255.0
        mse = ((r - t) ** 2).reshape(r.shape[0], -1).mean(axis=1)
        out[a:a + step] = 10.0 * np.log10(1.0 / mse)
    return out


def psnr_floor(floor_u8, ref):
    """PSNR of one uint8 floor image (3, 32, 32) against each image of ref."""
    rec = np.broadcast_to(floor_u8, ref.shape)
    return psnr_images(np.ascontiguousarray(rec), ref)


def two_level_bootstrap(mats, seed=BOOT_SEED_2L, chunks=40, per=250):
    """Part 6.3, plain loop. mats: name -> (K, n) array. One generator, shared draws."""
    shapes = {m.shape for m in mats.values()}
    assert len(shapes) == 1
    K, n = shapes.pop()
    rng = np.random.default_rng(seed)
    vals = {name: np.empty(chunks * per, dtype=np.float64) for name in mats}
    b = 0
    for _ in range(chunks):
        si = rng.integers(0, K, size=(per, K))
        ii = rng.integers(0, n, size=(per, n))
        for r in range(per):
            img = ii[r]
            for name, M in mats.items():
                total = 0.0
                for k in range(K):
                    total += M[si[r, k], img].mean()
                vals[name][b] = total / K
            b += 1
    return vals


def image_bootstrap(vecs, seed=BOOT_SEED_IMG, chunks=40, per=250, n=10000):
    """Sealed Part 7.1: default_rng(20261024).integers(0, 10000, size=(250, 10000)), 40 times."""
    rng = np.random.default_rng(seed)
    vals = {name: np.empty(chunks * per, dtype=np.float64) for name in vecs}
    b = 0
    for _ in range(chunks):
        ii = rng.integers(0, n, size=(per, n))
        for r in range(per):
            for name, v in vecs.items():
                vals[name][b] = v[ii[r]].mean()
            b += 1
    return vals


def ci95(vals):
    lo, hi = np.percentile(vals, [2.5, 97.5])  # numpy default: linear
    return [float(lo), float(hi)]


def reading_R(d, ci):
    lo, hi = ci
    if lo > 0 and d >= MARGIN:
        return "R1"
    if lo > 0 and d < MARGIN:
        return "R2"
    if lo <= 0 and hi >= 0:
        return "R3"
    if hi < 0:
        return "R4"
    raise RuntimeError("no reading")


R_TEXT = {"R1": "L+ replicates", "R2": "leak advantage, below margin",
          "R3": "no leak difference shown", "R4": "reversed"}


def seed_stats(seed_means):
    v = np.asarray(seed_means, dtype=np.float64)
    K = len(v)
    m = float(v.mean())
    sd = float(v.std(ddof=1))
    t = T_DF[K - 1]
    half = t * sd / math.sqrt(K)
    return sd, [m - half, m + half]


def sealed_rule(d_acc, acc_ci, acc_seeds, d_leak, leak_ci, leak_seeds):
    """Sealed Part 7.4 and 7.5."""
    a_plus = acc_ci[0] > -1.0
    a_minus = d_acc <= -1.0 and acc_ci[1] < 0 and all(x < 0 for x in acc_seeds)
    if a_plus and not a_minus:
        A = "A+"
    elif a_minus and not a_plus:
        A = "A-"
    else:
        A = "A?"
    hits = []
    if d_leak >= MARGIN and leak_ci[0] > 0 and all(x > 0 for x in leak_seeds):
        hits.append("L+")
    if leak_ci[0] > -MARGIN and leak_ci[1] < MARGIN:
        hits.append("L=")
    if d_leak <= -MARGIN and leak_ci[1] < 0 and all(x < 0 for x in leak_seeds):
        hits.append("L-")
    L = hits[0] if len(hits) == 1 else "L?"
    table = {("A+", "L+"): "S", ("A+", "L="): "E", ("A+", "L-"): "X", ("A+", "L?"): "U",
             ("A-", "L+"): "T", ("A-", "L="): "A", ("A-", "L-"): "A", ("A-", "L?"): "U"}
    case = table.get((A, L), "U")
    return A, L, case, hits


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    out = {"script": "audit/recheck.py", "numpy": np.__version__,
           "python": sys.version.split()[0], "tolerance": TOL}

    # 1. hashes ------------------------------------------------------------
    freeze = read_json(os.path.join(F, "FREEZE.json"))
    hashes = []
    for name, want in freeze["sha256"].items():
        got = sha256_file(os.path.join(F, name))
        hashes.append({"file": name, "want": want, "got": got, "ok": got == want})
    for name, want in freeze["data"]["npz_sha256"].items():
        got = sha256_file(os.path.join(DATA, name))
        hashes.append({"file": "SS/data/" + name, "want": want, "got": got, "ok": got == want})
    got = sha256_file(os.path.join(SS, "FREEZE.json"))
    hashes.append({"file": "SS/FREEZE.json (sealed_freeze_sha256)",
                   "want": freeze["sealed_freeze_sha256"], "got": got,
                   "ok": got == freeze["sealed_freeze_sha256"]})
    sealed_freeze = read_json(os.path.join(SS, "FREEZE.json"))
    hashes.append({"file": "npz record equal to sealed FREEZE.json data.npz_sha256",
                   "want": "equal", "got": "equal" if sealed_freeze["data"]["npz_sha256"]
                   == freeze["data"]["npz_sha256"] else "different",
                   "ok": sealed_freeze["data"]["npz_sha256"] == freeze["data"]["npz_sha256"]})
    # Part 2: byte copies (hash only, no read of the code)
    for name in ["sebits.py", "selection.json"]:
        a = sha256_file(os.path.join(F, name))
        b = sha256_file(os.path.join(SS, name))
        hashes.append({"file": name + " byte copy of SS/" + name, "want": b, "got": a, "ok": a == b})
    out["hashes"] = hashes

    # 2. data and per-image values ------------------------------------------
    x_test = npz_get(os.path.join(DATA, "cifar10_test_images.npz"), "x")
    y_test = npz_get(os.path.join(DATA, "cifar10_test_labels.npz"), "y").astype(np.int64)
    assert x_test.shape == (10000, 3, 32, 32) and x_test.dtype == np.uint8
    assert y_test.shape == (10000,)
    n = len(y_test)

    P = {"L": {a: {} for a in ARMS}, "S": {a: {} for a in ARMS}}
    C = {a: {} for a in ARMS}
    nbits = {a: {} for a in ARMS}
    best_epoch = {"L": {a: {} for a in ARMS}, "S": {a: {} for a in ARMS}}
    dec_dev_psnr = {}
    nonfinite_losses = []
    pred_info = {}
    for s in NEW:
        sd = os.path.join(RES, "s%d" % s)
        preds = {}
        for a in ARMS:
            mp = os.path.join(sd, "msgs", a + ".npz")
            preds[a] = npz_get(mp, "pred_test").astype(np.int64)
            C[a][s] = (preds[a] == y_test).astype(np.float64)
            nbits[a][s] = int(npz_get(mp, "nbits"))
            for size in ["L", "S"]:
                tag = "%s1_%s" % (size, a)
                P[size][a][s] = psnr_images(npz_get(os.path.join(sd, "dec", tag + ".npz"), "recon_test"), x_test)
                dj = read_json(os.path.join(sd, "dec", tag + ".json"))
                best_epoch[size][a][s] = int(dj["best_epoch"])
                dec_dev_psnr[(s, tag)] = dj.get("dev_psnr")
        # extra consistency checks of the predictions (informational)
        bot_logits = npz_get(os.path.join(sd, "msgs", "BOT@64.npz"), "logits_test")
        lab_bits = npz_get(os.path.join(sd, "msgs", "LAB.npz"), "bits_test")
        pred_info[str(s)] = {
            "BOT pred == argmax(logits_test) (first max)": bool(np.array_equal(preds["BOT@64"], np.argmax(bot_logits, axis=1))),
            "LAB pred == LAB message (bits >> 4)": bool(np.array_equal(preds["LAB"], (lab_bits[:, 0] >> 4).astype(np.int64))),
            "LAB pred == TOPK pred": bool(np.array_equal(preds["LAB"], preds["TOPK@64"])),
        }
        # finite losses in every decoder history (V5 part)
        for fn in sorted(os.listdir(os.path.join(sd, "dec"))):
            if fn.endswith(".json"):
                dj = read_json(os.path.join(sd, "dec", fn))
                for h in dj.get("hist", []):
                    for key in ("train_mse", "dev_mse"):
                        if not math.isfinite(float(h[key])):
                            nonfinite_losses.append([s, fn, h["epoch"], key])
    out["prediction_checks"] = pred_info

    # extra integrity checks (informational): duplicate reconstructions, D_dev PSNR record, arm bits
    recon_hash = {}
    devpsnr_maxdiff = 0.0
    for s in NEW:
        dd = os.path.join(RES, "s%d" % s, "dec")
        for fn in sorted(os.listdir(dd)):
            if fn.endswith(".npz"):
                with np.load(os.path.join(dd, fn), allow_pickle=False) as z:
                    recon_hash.setdefault(hashlib.sha256(z["recon_test"].tobytes()).hexdigest(), []).append("s%d/%s" % (s, fn))
                    pdev = z["psnr_dev"]
                dj = read_json(os.path.join(dd, fn[:-4] + ".json"))
                devpsnr_maxdiff = max(devpsnr_maxdiff, abs(float(pdev.mean()) - float(dj["dev_psnr"])))
    for s in SEALED:
        for a in [SYM, ACT]:
            p = os.path.join(SS, "results", "s%d" % s, "dec", "S1_%s.npz" % a)
            recon_hash.setdefault(hashlib.sha256(npz_get(p, "recon_test").tobytes()).hexdigest(), []).append("SS s%d/S1_%s" % (s, a))
    arm_bits = {}
    for s in NEW:
        for a, fn in [("TOPK@64", "TOPK@64.json"), ("BOT@64", "BOT_c16_s2_q1.json"), ("LAB", "LAB.json")]:
            arm_bits.setdefault(a, []).append(read_json(os.path.join(RES, "s%d" % s, "arms", fn))["bits"])
    out["integrity"] = {
        "duplicate_reconstructions": [v for v in recon_hash.values() if len(v) > 1],
        "distinct_reconstruction_arrays": len(recon_hash),
        "max_abs_diff_mean_psnr_dev_vs_json_dev_psnr": devpsnr_maxdiff,
        "arm_json_bits": arm_bits,
        "test_label_class_counts": np.bincount(y_test, minlength=10).tolist(),
    }

    # sealed reconstructions
    PS_sealed = {a: {} for a in [SYM, ACT]}
    for s in SEALED:
        for a in [SYM, ACT]:
            p = os.path.join(SS, "results", "s%d" % s, "dec", "S1_%s.npz" % a)
            PS_sealed[a][s] = psnr_images(npz_get(p, "recon_test"), x_test)
    PL_sealed1 = {}
    for a in [SYM, ACT]:
        p = os.path.join(SS, "results", "s20261111", "dec", "L1_%s.npz" % a)
        PL_sealed1[a] = psnr_images(npz_get(p, "recon_test"), x_test)

    for size in P:
        for a in ARMS:
            for s in NEW:
                assert np.all(np.isfinite(P[size][a][s])), (size, a, s)

    # 3. primary and 4. accuracy ---------------------------------------------
    D_L = np.stack([P["L"][ACT][s] - P["L"][SYM][s] for s in NEW])
    D_S = np.stack([P["S"][ACT][s] - P["S"][SYM][s] for s in NEW])
    D_A = np.stack([100.0 * (C[SYM][s] - C[ACT][s]) for s in NEW])
    boot = two_level_bootstrap({"L": D_L, "A": D_A, "S": D_S})

    def leak_row(D, vals):
        d = float(D.mean())
        seeds = [float(D[k].mean()) for k in range(D.shape[0])]
        ci = ci95(vals)
        sd, tci = seed_stats(seeds)
        r = reading_R(d, ci)
        return {"d_leak": d, "d_leak_ci": ci, "d_leak_seeds": seeds, "seed_sd": sd,
                "t_ci": tci, "reading": r, "ci_low_ge_margin": bool(ci[0] >= MARGIN)}

    primary = leak_row(D_L, boot["L"])
    primary.update({"reading_final": primary["reading"], "reading_text": R_TEXT[primary["reading"]],
                    "sym": SYM, "act": ACT, "decoder": "L",
                    "bits": {SYM: nbits[SYM][NEW[0]], ACT: nbits[ACT][NEW[0]]}})
    acc_ci = ci95(boot["A"])
    accuracy = {"d_acc": float(D_A.mean()), "d_acc_ci": acc_ci,
                "d_acc_seeds": [float(D_A[k].mean()) for k in range(len(NEW))],
                "outcome": "A+ replicates" if acc_ci[0] > -1.0 else "A+ does not replicate"}

    # 5. secondary ------------------------------------------------------------
    sec = {}
    sec["decoder_S_new_seeds"] = leak_row(D_S, boot["S"])

    abar = D_A.mean(axis=0)
    dbar_L = D_L.mean(axis=0)
    dbar_S = D_S.mean(axis=0)
    ib = image_bootstrap({"A": abar, "L": dbar_L, "S": dbar_S})
    for size, dbar, D in [("L", dbar_L, D_L), ("S", dbar_S, D_S)]:
        d_acc = float(abar.mean())
        a_ci = ci95(ib["A"])
        a_seeds = [float(D_A[k].mean()) for k in range(len(NEW))]
        d_leak = float(dbar.mean())
        l_ci = ci95(ib[size])
        l_seeds = [float(D[k].mean()) for k in range(len(NEW))]
        A, L, case, hits = sealed_rule(d_acc, a_ci, a_seeds, d_leak, l_ci, l_seeds)
        sec["sealed_rule_" + size] = {"d_acc": d_acc, "d_acc_ci": a_ci, "d_acc_seeds": a_seeds,
                                       "d_leak": d_leak, "d_leak_ci": l_ci, "d_leak_seeds": l_seeds,
                                       "A": A, "L": L, "case": case, "L_hits": hits}

    D9 = np.stack([PS_sealed[ACT][s] - PS_sealed[SYM][s] for s in SEALED] + [D_S[k] for k in range(len(NEW))])
    boot9 = two_level_bootstrap({"S9": D9})
    pooled = leak_row(D9, boot9["S9"])
    pooled["seeds"] = SEALED + NEW
    sec["pooled_S_sealed_and_new"] = pooled
    sec["sealed_seed1_L"] = float((PL_sealed1[ACT] - PL_sealed1[SYM]).mean())
    sec["decoder_gap_L_minus_S"] = {
        "d_leak_seeds": [primary["d_leak_seeds"][k] - sec["decoder_S_new_seeds"]["d_leak_seeds"][k]
                         for k in range(len(NEW))],
        "psnr": {a: [float(P["L"][a][s].mean() - P["S"][a][s].mean()) for s in NEW] for a in ARMS},
    }

    # floors (sealed Parts 2.2 and 5.4) ---------------------------------------
    train = np.load(os.path.join(DATA, "cifar10_train.npz"), allow_pickle=False)
    x_tr, y_tr = train["x"], train["y"].astype(np.int64)
    rng = np.random.default_rng(SPLIT_SEED)
    task_l, dev_l, aux_l = [], [], []
    for c in range(10):
        idx = np.flatnonzero(y_tr == c)
        perm = rng.permutation(idx)
        task_l.append(perm[:3500])
        dev_l.append(perm[3500:4000])
        aux_l.append(perm[4000:5000])
    task_i = np.sort(np.concatenate(task_l))
    dev_i = np.sort(np.concatenate(dev_l))
    aux_i = np.sort(np.concatenate(aux_l))
    assert len(task_i) == 35000 and len(dev_i) == 5000 and len(aux_i) == 10000
    x_aux, y_aux = x_tr[aux_i], y_tr[aux_i]
    x_dev = x_tr[dev_i]
    targets = np.concatenate([x_aux, x_aux[:, :, :, ::-1]])  # D_aux and its horizontal flips
    t_lab = np.concatenate([y_aux, y_aux])

    def mean_img_u8(imgs):
        m = imgs.astype(np.float64).reshape(imgs.shape[0], -1).sum(axis=0) / imgs.shape[0] / 255.0
        return np.round(255.0 * m).reshape(imgs.shape[1:]).clip(0, 255).astype(np.uint8)

    def mean_img_u8_halfup(imgs):
        s = imgs.astype(np.int64).reshape(imgs.shape[0], -1).sum(axis=0)
        q = (2 * s + imgs.shape[0]) // (2 * imgs.shape[0])  # exact round half up
        return q.reshape(imgs.shape[1:]).astype(np.uint8)

    floors = {}
    floor_variants = {}
    for label, fn in [("half_even_float64", mean_img_u8), ("half_up_exact", mean_img_u8_halfup)]:
        f0 = fn(targets)
        fy = np.stack([fn(targets[t_lab == c]) for c in range(10)])
        F0_test = float(psnr_floor(f0, x_test).mean())
        FY_test = float(psnr_images(np.ascontiguousarray(fy[y_test]), x_test).mean())
        F0_dev = float(psnr_floor(f0, x_dev).mean())
        floor_variants[label] = {"F0_test": F0_test, "FY_test": FY_test, "F0_dev": F0_dev}
        if label == "half_even_float64":
            floors = {"F0": F0_test, "FY": FY_test}
            f0_main = f0
    # V1 closed form on D_dev (extra): mean target image per LAB message label, applied to the D_dev LAB message
    closed_dev = {}
    for s in NEW:
        mp = os.path.join(RES, "s%d" % s, "msgs", "LAB.npz")
        lab_t = np.concatenate([npz_get(mp, "bits_aux")[:, 0] >> 4, npz_get(mp, "bits_auxF")[:, 0] >> 4]).astype(np.int64)
        lab_dev = (npz_get(mp, "bits_dev")[:, 0] >> 4).astype(np.int64)
        imgs = np.zeros((10, 3, 32, 32), dtype=np.uint8)
        present = []
        for c in range(10):
            sel = lab_t == c
            if sel.any():
                imgs[c] = mean_img_u8(targets[sel])
                present.append(c)
        assert set(np.unique(lab_dev)).issubset(present)
        closed_dev[s] = float(psnr_images(np.ascontiguousarray(imgs[lab_dev]), x_dev).mean())
    out["floors_detail"] = {"variants": floor_variants, "split_sizes": [len(task_i), len(dev_i), len(aux_i)],
                            "V1_closed_dev_recomputed": {str(s): closed_dev[s] for s in NEW}}

    # 6. validity ----------------------------------------------------------------
    planned_units = ["N", "head", "topk", "BOT_c16_s2_q1", "msgs",
                     "L1_TOPK@64", "L1_BOT@64", "L1_LAB", "S1_TOPK@64", "S1_BOT@64", "S1_LAB"]
    ctrl_units = ["L1_BOT@64_ctrl", "L1_TOPK@64_ctrl"]
    dec_units = [u for u in planned_units if u[:3] in ("L1_", "S1_")]
    last = {}
    all_records = []
    for s in NEW:
        with open(os.path.join(RES, "s%d" % s, "checks.jsonl")) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec["_seed"] = s
                all_records.append(rec)
                last[(s, rec["check"], rec.get("where"))] = rec
    ok_false_all = [r for r in all_records if r.get("ok") is not True]
    by_check = {}
    for (s, chk, where), rec in last.items():
        by_check.setdefault(chk, []).append(rec)

    vstate = {v: {"ok": True, "records": 0, "fail": []} for v in ["V1", "V2", "V3", "V4", "V5", "V6"]}
    for v in ["V1", "V2", "V3", "V4", "V6"]:
        vstate[v]["records"] = len(by_check.get(v, []))
        for rec in by_check.get(v, []):
            if rec.get("ok") is not True:
                vstate[v]["ok"] = False
                vstate[v]["fail"].append([rec["_seed"], rec.get("where")])
    v1_list, v2_list = [], []
    rederive = []
    n_dev_acc = []
    for s in NEW:
        units = planned_units + (ctrl_units if s == NEW[0] else [])
        # V1
        for where in ["L1_LAB", "S1_LAB"]:
            rec = last.get((s, "V1", where))
            if rec is None:
                vstate["V1"]["ok"] = False
                vstate["V1"]["fail"].append([s, where, "absent"])
                continue
            ok2 = rec["dev_psnr"] >= rec["closed"] - 0.2
            rederive.append({"seed": s, "check": "V1", "where": where, "record_ok": rec["ok"], "rederived_ok": ok2,
                             "dev_psnr_matches_dec_json": rec["dev_psnr"] == dec_dev_psnr[(s, where)],
                             "closed_recomputed": closed_dev[s],
                             "closed_diff": abs(closed_dev[s] - rec["closed"])})
            if not ok2:
                vstate["V1"]["ok"] = False
            v1_list.append({"seed": s, "where": where, "dev_psnr": rec["dev_psnr"], "closed": rec["closed"], "ok": rec["ok"]})
        # V2 (first new seed only)
        if s == NEW[0]:
            for where in ctrl_units:
                rec = last.get((s, "V2", where))
                if rec is None:
                    vstate["V2"]["ok"] = False
                    vstate["V2"]["fail"].append([s, where, "absent"])
                    continue
                ok2 = rec["dev_psnr"] <= rec["F0"] + 0.2
                rederive.append({"seed": s, "check": "V2", "where": where, "record_ok": rec["ok"], "rederived_ok": ok2,
                                 "F0_dev_recomputed": floor_variants["half_even_float64"]["F0_dev"],
                                 "F0_diff": abs(floor_variants["half_even_float64"]["F0_dev"] - rec["F0"])})
                if not ok2:
                    vstate["V2"]["ok"] = False
                v2_list.append({"seed": s, "where": where, "dev_psnr": rec["dev_psnr"], "F0": rec["F0"], "ok": rec["ok"]})
        # V3: start and end of each unit after N
        for u in units:
            if u == "N":
                continue
            for pos in ["start", "end"]:
                rec = last.get((s, "V3", "%s %s" % (u, pos)))
                if rec is None or rec.get("ok") is not True:
                    vstate["V3"]["ok"] = False
                    vstate["V3"]["fail"].append([s, "%s %s" % (u, pos), "absent" if rec is None else "not ok"])
        # V4
        for a in ARMS:
            rec = last.get((s, "V4", "msgs " + a))
            want_bits = {"TOPK@64": 58, "BOT@64": 64, "LAB": 4}[a]
            if rec is None or rec.get("ok") is not True or rec.get("bits") != want_bits or nbits[a][s] != want_bits:
                vstate["V4"]["ok"] = False
                vstate["V4"]["fail"].append([s, "msgs " + a])
        # V5
        missing = []
        for u in units:
            if not os.path.isfile(os.path.join(RES, "s%d" % s, "done", u + ".json")):
                missing.append("done/" + u)
        for u in dec_units + (ctrl_units if s == NEW[0] else []):
            if not os.path.isfile(os.path.join(RES, "s%d" % s, "dec", u + ".npz")):
                missing.append("dec/" + u + ".npz")
        for a in ARMS:
            if not os.path.isfile(os.path.join(RES, "s%d" % s, "msgs", a + ".npz")):
                missing.append("msgs/" + a + ".npz")
        nf = [x for x in nonfinite_losses if x[0] == s]
        vstate["V5"]["records"] += 1
        if missing or nf:
            vstate["V5"]["ok"] = False
            vstate["V5"]["fail"].append([s, missing, nf])
        # V6
        recs6 = [r for (ss, chk, w), r in last.items() if ss == s and chk == "V6"]
        nj = read_json(os.path.join(RES, "s%d" % s, "N.json"))
        n_dev_acc.append(nj["dev_acc"])
        if len(recs6) != 1 or recs6[0].get("ok") is not True or not recs6[0]["dev_acc"] >= 0.80 \
                or recs6[0]["dev_acc"] != nj["dev_acc"]:
            vstate["V6"]["ok"] = False
            vstate["V6"]["fail"].append([s, "V6"])
    valid = all(vstate[v]["ok"] for v in vstate)
    out["validity"] = {"vstate": vstate, "rederived": rederive,
                       "ok_false_records_all": len(ok_false_all),
                       "records_total": len(all_records), "records_last": len(last),
                       "records_all_by_check": {c: sum(1 for r in all_records if r["check"] == c)
                                                for c in sorted({r["check"] for r in all_records})},
                       "nonfinite_losses": nonfinite_losses}

    # descriptive -----------------------------------------------------------------
    desc = {"N_dev_acc": n_dev_acc}
    for a in ARMS:
        accs = [float(C[a][s].mean()) for s in NEW]
        desc[a] = {"acc_seeds": accs, "acc": float(np.mean([C[a][s] for s in NEW])),
                   "psnr_L_seeds": [float(P["L"][a][s].mean()) for s in NEW],
                   "psnr_L": float(np.mean([P["L"][a][s] for s in NEW])),
                   "psnr_S_seeds": [float(P["S"][a][s].mean()) for s in NEW],
                   "psnr_S": float(np.mean([P["S"][a][s] for s in NEW])),
                   "best_epoch_L": [best_epoch["L"][a][s] for s in NEW],
                   "best_epoch_S": [best_epoch["S"][a][s] for s in NEW]}
    desc["V1"] = v1_list
    desc["V2"] = v2_list

    mine = {"seeds": NEW, "floors": floors,
            "checks": {v: {"ok": vstate[v]["ok"], "records": vstate[v]["records"]} for v in vstate},
            "check_failures": {v: vstate[v]["fail"] for v in vstate},
            "failed_records_all": [{k: v for k, v in r.items() if k != "_seed"} for r in ok_false_all],
            "valid": valid, "primary": primary, "accuracy": accuracy, "secondary": sec,
            "descriptive": desc}
    out["recomputed"] = mine
    out["message_bits"] = {a: {str(s): nbits[a][s] for s in NEW} for a in ARMS}
    # implied t values of results.json (diagnostic only)

    # compare with results.json ------------------------------------------------------
    ref = read_json(os.path.join(RES, "results.json"))
    rows, skipped = [], []

    def compare(path, r, m):
        if isinstance(r, dict):
            if not isinstance(m, dict):
                rows.append({"path": path, "ref": r, "mine": m, "diff": None, "agree": False, "kind": "type"})
                return
            for k in r:
                if k in m:
                    compare(path + "." + k if path else k, r[k], m[k])
                else:
                    skipped.append(path + "." + k if path else k)
            for k in m:
                if k not in r:
                    rows.append({"path": path + "." + k, "ref": None, "mine": m[k], "diff": None,
                                 "agree": False, "kind": "absent in results.json"})
            return
        if isinstance(r, list):
            if not isinstance(m, list) or len(r) != len(m):
                rows.append({"path": path, "ref": r, "mine": m, "diff": None, "agree": False, "kind": "length"})
                return
            if len(r) == 0:
                rows.append({"path": path, "ref": [], "mine": [], "diff": None, "agree": True, "kind": "empty"})
                return
            for i, (ri, mi) in enumerate(zip(r, m)):
                compare("%s[%d]" % (path, i), ri, mi)
            return
        if isinstance(r, bool) or isinstance(m, bool) or isinstance(r, str) or r is None:
            rows.append({"path": path, "ref": r, "mine": m, "diff": None, "agree": r == m and type(r) == type(m),
                         "kind": "exact"})
            return
        if isinstance(r, int) and isinstance(m, (int, np.integer)):
            rows.append({"path": path, "ref": r, "mine": int(m), "diff": abs(r - int(m)), "agree": r == int(m),
                         "kind": "int"})
            return
        d = abs(float(r) - float(m))
        rows.append({"path": path, "ref": float(r), "mine": float(m), "diff": d, "agree": d <= TOL, "kind": "float"})

    compare("", ref, mine)
    floats = [r for r in rows if r["kind"] == "float"]
    worst = max(floats, key=lambda r: r["diff"])
    dis = [r for r in rows if not r["agree"]]

    # implied Student t of results.json, for the record
    implied_t = {}
    for name, node, K in [("primary", ref["primary"], 6), ("decoder_S_new_seeds", ref["secondary"]["decoder_S_new_seeds"], 6),
                          ("pooled_S_sealed_and_new", ref["secondary"]["pooled_S_sealed_and_new"], 9)]:
        se = node["seed_sd"] / math.sqrt(K)
        implied_t[name] = [(node["d_leak"] - node["t_ci"][0]) / se, (node["t_ci"][1] - node["d_leak"]) / se]

    # table.md: rebuild it from the recomputed values and compare the text --------------
    def f3(v):
        return "%.3f" % v

    def f2(v):
        return "%.2f" % v

    p, a_, sS, s9 = primary, accuracy, sec["decoder_S_new_seeds"], sec["pooled_S_sealed_and_new"]
    lines = [
        "# Results: replication of the leakage result (TOPK@64 against BOT@64)", "",
        "Seeds: %s. Valid: %s. Checks: %s." % (", ".join(str(s) for s in NEW), valid,
                                                 ", ".join("%s %s" % (v, "pass" if vstate[v]["ok"] else "FAIL") for v in vstate)),
        "",
        "| Row | Value | 95% CI (two-level) | Seed values | Reading |",
        "|---|---|---|---|---|",
        "| PRIMARY Δleak, Decoder-L (dB) | %s | %s to %s | %s | %s: %s |" % (
            f3(p["d_leak"]), f3(p["d_leak_ci"][0]), f3(p["d_leak_ci"][1]), ", ".join(f3(v) for v in p["d_leak_seeds"]),
            p["reading"], p["reading_text"]),
        "| Δacc (points; secondary) | %s | %s to %s | %s | %s |" % (
            f2(a_["d_acc"]), f2(a_["d_acc_ci"][0]), f2(a_["d_acc_ci"][1]), ", ".join(f2(v) for v in a_["d_acc_seeds"]),
            a_["outcome"]),
        "| Δleak, Decoder-S, new seeds (secondary) | %s | %s to %s | %s | %s (secondary) |" % (
            f3(sS["d_leak"]), f3(sS["d_leak_ci"][0]), f3(sS["d_leak_ci"][1]), ", ".join(f3(v) for v in sS["d_leak_seeds"]),
            sS["reading"]),
        "| Δleak, Decoder-S, 3 sealed + 6 new seeds (descriptive) | %s | %s to %s | %s | %s (descriptive) |" % (
            f3(s9["d_leak"]), f3(s9["d_leak_ci"][0]), f3(s9["d_leak_ci"][1]), ", ".join(f3(v) for v in s9["d_leak_seeds"]),
            s9["reading"]),
        "",
        "- Seed SD of the primary Δleak: %s dB. Student t 95%% interval over the seed means: %s to %s dB. "
        "CI lower bound at or above the margin: %s." % (f3(p["seed_sd"]), f3(p["t_ci"][0]), f3(p["t_ci"][1]),
                                                         p["ci_low_ge_margin"]),
        "- Sealed rule (image bootstrap only) on the new seeds: Decoder-L %s (CI %s to %s), Decoder-S %s (CI %s to %s)." % (
            sec["sealed_rule_L"]["L"], f3(sec["sealed_rule_L"]["d_leak_ci"][0]), f3(sec["sealed_rule_L"]["d_leak_ci"][1]),
            sec["sealed_rule_S"]["L"], f3(sec["sealed_rule_S"]["d_leak_ci"][0]), f3(sec["sealed_rule_S"]["d_leak_ci"][1])),
        "- Floors on test: F0 %s dB, FY %s dB." % (f3(floors["F0"]), f3(floors["FY"])),
        "- Sealed seed-1 Δleak with Decoder-L (reference): %s dB." % f3(sec["sealed_seed1_L"]),
        "",
        "| Arm | Test acc | PSNR Decoder-L (dB) | PSNR Decoder-S (dB) |",
        "|---|---|---|---|",
    ]
    for a in ARMS:
        lines.append("| %s | %.4f | %s | %s |" % (a, desc[a]["acc"], f3(desc[a]["psnr_L"]), f3(desc[a]["psnr_S"])))
    lines += ["", "N D_dev accuracy per seed: %s." % ", ".join("%.4f" % v for v in n_dev_acc)]
    expect = "\n".join(lines)
    with open(os.path.join(RES, "table.md")) as fh:
        actual = fh.read()
    act_lines = actual.rstrip("\n").split("\n")
    exp_lines = expect.split("\n")
    table_diff = [{"line": i + 1, "results_table": al, "recomputed": el}
                  for i, (al, el) in enumerate(zip(act_lines, exp_lines)) if al != el]
    if len(act_lines) != len(exp_lines):
        table_diff.append({"line": "count", "results_table": len(act_lines), "recomputed": len(exp_lines)})
    num_re = re.compile(r"-?\d+\.\d+")
    table_numbers = len(num_re.findall(actual))

    out["comparison"] = {
        "rows": len(rows), "agree": sum(1 for r in rows if r["agree"]), "disagree": len(dis),
        "float_rows": len(floats), "max_float_diff": worst["diff"], "max_float_diff_path": worst["path"],
        "disagreements": dis, "not_compared": skipped, "implied_t_in_results_json": implied_t,
    }
    out["table_md"] = {"lines": len(act_lines), "equal": not table_diff, "differences": table_diff,
                       "decimal_numbers_in_table": table_numbers}
    out["rows"] = rows
    out["seconds"] = time.time() - t0
    out["PASS"] = (not dis) and (not table_diff) and all(h["ok"] for h in hashes) and valid \
        and len(ok_false_all) == 0

    with open(os.path.join(HERE, "recheck.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    # grouped table for AUDIT.md: one row per scalar, one row per list -------------------------
    groups = {}
    order = []
    for r in rows:
        key = re.sub(r"\[\d+\]", "[]", r["path"]) if r["kind"] in ("float", "int") else r["path"]
        key = re.sub(r"\[\d+\]\.", "[].", key)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(r)

    def fmt(v):
        if isinstance(v, float):
            return "%.6g" % v if abs(v) < 1e-3 and v != 0 else ("%.6f" % v).rstrip("0").rstrip(".")
        return json.dumps(v)

    md = ["| Quantity | results.json value | Recomputed value | Largest difference | Agree |", "|---|---|---|---|---|"]
    for key in order:
        g = groups[key]
        diffs = [r["diff"] for r in g if r["diff"] is not None]
        md_diff = ("%.2e" % max(diffs)) if diffs and any(r["kind"] == "float" for r in g) else (
            "0" if diffs else "exact")
        agree = "yes" if all(r["agree"] for r in g) else "NO"
        if len(g) == 1:
            md.append("| `%s` | %s | %s | %s | %s |" % (key, fmt(g[0]["ref"]), fmt(g[0]["mine"]), md_diff, agree))
        else:
            md.append("| `%s` (%d values) | %s | %s | %s | %s |" % (
                key, len(g), ", ".join(fmt(r["ref"]) for r in g), ", ".join(fmt(r["mine"]) for r in g), md_diff, agree))
    with open(os.path.join(HERE, "table_rows.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")

    print(json.dumps({k: out["comparison"][k] for k in ["rows", "agree", "disagree", "float_rows",
                                                          "max_float_diff", "max_float_diff_path"]}, indent=1))
    print("hashes ok:", all(h["ok"] for h in hashes), "valid:", valid, "ok_false_all:", len(ok_false_all))
    print("table.md equal:", not table_diff, "groups:", len(order))
    print("not compared:", skipped)
    print("implied t:", implied_t)
    print("floor variants:", json.dumps(floor_variants))
    print("integrity:", json.dumps(out["integrity"]))
    print("PASS:", out["PASS"], "seconds: %.1f" % out["seconds"])
    if dis:
        for r in dis[:40]:
            print("DISAGREE", r["path"], r["ref"], r["mine"], r["diff"])
    if table_diff:
        for d in table_diff:
            print("TABLE", d)


if __name__ == "__main__":
    main()
