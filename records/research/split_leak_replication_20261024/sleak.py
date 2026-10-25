#!/usr/bin/env python3
"""Replication of the leakage result of split_equal_bits_20261023 with new seeds (CONFIRMATORY code).

The rule and every fixed choice are in PREREGISTRATION.md. FREEZE.json holds the sha256 of this file and of the
other frozen files. sebits.py is a byte copy of the sealed study's file; this driver imports it and calls its units
and metrics without a change. Run each job through run_sleak.sh. Run Python with -I -B.

  sleak.py test               unit tests of test_sleak.py (synthetic data; no CIFAR-10 file is read)
  sleak.py freeze [label]     check the copies, run the tests and write FREEZE.json (a label only for a re-freeze)
  sleak.py seed <run seed>    one seed job on MPS: N, TOPK head, formats and LAB, BOT@64, messages, decoders
  sleak.py analysis           the only job that reads the test labels (CPU)
"""
import argparse
import importlib.util
import json
import os
import sys
import time
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import sebits as S  # noqa: E402  (byte copy of the sealed file; FREEZE.json checks it)

SEALED = ROOT.parent / "split_equal_bits_20261023"
S.DATA_DIR = SEALED / "data"  # the CIFAR-10 npz files of the sealed study, read only
S.RES = ROOT / "results"

FROZEN = ("PREREGISTRATION.md", "sebits.py", "selection.json", "sleak.py", "test_sleak.py", "run_sleak.sh",
          "wait_and_run.sh", "test_run_sleak.zsh")
SEALED_SHA = {"sebits.py": "db0f25b468ee55046a4ab94b3398c875b5157409f6c2577fef0496045de08696",
              "selection.json": "b5a24e75a717249de91e0b5f82d1ef9f453e76070d151db8c524b789cd0ac907"}
NPZ = ("cifar10_train.npz", "cifar10_test_images.npz", "cifar10_test_labels.npz")

NEW_SEEDS = (20261121, 20261122, 20261123, 20261124, 20261125, 20261126)
SEALED_SEEDS = (20261111, 20261112, 20261113)
BOOT2_SEED = 20261127
N_BOOT2 = 10000
BOOT2_CHUNK = 250
MARGIN = 0.5  # dB
ACC_MARGIN = 1.0  # points
V1_TOL = 0.2  # dB, as sealed Part 7.3
T975 = {1: 12.706205, 2: 4.302653, 3: 3.182446, 4: 2.776445, 5: 2.570582, 6: 2.446912, 7: 2.364624, 8: 2.306004,
        9: 2.262157, 10: 2.228139}  # two-sided 95% Student t quantiles

MSG_NAMES = ["TOPK@64", "BOT@64", "LAB"]
DECS = [("TOPK@64", "L", 1, False), ("BOT@64", "L", 1, False), ("LAB", "L", 1, False),
        ("TOPK@64", "S", 1, False), ("BOT@64", "S", 1, False), ("LAB", "S", 1, False)]
CTRL_DECS = [("BOT@64", "L", 1, True), ("TOPK@64", "L", 1, True)]  # first new seed only (V2)

READING_TEXT = {"R1": "L+ replicates", "R2": "leak advantage, below margin", "R3": "no leak difference shown",
                "R4": "reversed: BOT leaks less", "void": "void: a validity check failed"}


# ------------------------------------------------------------------------------------------------ plan and freeze

def decs_for(seed):
    return DECS + (CTRL_DECS if seed == NEW_SEEDS[0] else [])


def bot_star(sel):
    return sel["BOT"][str(S.BSTAR)]


def planned_units(seed, sel):
    return ["N", "head", "topk", bot_star(sel), "msgs"] + [S.dec_tag(*d) for d in decs_for(seed)]


def check_freeze(files_used):
    fz = S.read_json(ROOT / "FREEZE.json")
    for f in FROZEN:
        if S.sha256_file(ROOT / f) != fz["sha256"][f]:
            raise S.CheckFail(f"freeze: {f} differs from FREEZE.json")
    for f in files_used:
        if S.sha256_file(S.DATA_DIR / f) != fz["data"]["npz_sha256"][f]:
            raise S.CheckFail(f"freeze: {S.DATA_DIR / f} differs from FREEZE.json")
    return fz


def run_tests():
    spec = importlib.util.spec_from_file_location("test_sleak", ROOT / "test_sleak.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["test_sleak"] = mod
    spec.loader.exec_module(mod)
    suite = unittest.defaultTestLoader.loadTestsFromModule(mod)
    return unittest.TextTestRunner(verbosity=1, stream=sys.stdout).run(suite)


def cmd_freeze(amend=None):
    old = None
    if (ROOT / "FREEZE.json").exists():
        if not amend:
            raise S.CheckFail("FREEZE.json exists; a re-freeze needs an amendment label (sleak.py freeze <label>)")
        old = S.read_json(ROOT / "FREEZE.json")
        if (ROOT / f"FREEZE_{amend}.json").exists():
            raise S.CheckFail(f"FREEZE_{amend}.json exists")
    for f, want in SEALED_SHA.items():
        if S.sha256_file(ROOT / f) != want or S.sha256_file(SEALED / f) != want:
            raise S.CheckFail(f"freeze: {f} is not the byte copy of the sealed file")
    data = S.read_json(SEALED / "FREEZE.json")["data"]
    for f in NPZ:
        if S.sha256_file(S.DATA_DIR / f) != data["npz_sha256"][f]:
            raise S.CheckFail(f"freeze: {S.DATA_DIR / f} differs from the sealed FREEZE.json")
    r = run_tests()
    if r.failures or r.errors:
        raise S.CheckFail("unit tests failed; no freeze")
    if old is not None:
        (ROOT / "FREEZE.json").replace(ROOT / f"FREEZE_{amend}.json")
    S.write_json(ROOT / "FREEZE.json", {
        "time": S.now(), "sha256": {f: S.sha256_file(ROOT / f) for f in FROZEN},
        "tests": {"run": r.testsRun, "failures": 0, "errors": 0},
        "data": data, "data_dir": str(S.DATA_DIR), "sealed_freeze_sha256": S.sha256_file(SEALED / "FREEZE.json"),
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__},
        "amends": f"FREEZE_{amend}.json" if old else None,
        "rule": "Every job checks these hashes and the npz hashes and stops on a difference."})
    S.log("FREEZE.json written")


# ------------------------------------------------------------------------------------------------ seed job

def v1_large(ctx):
    """V1 for Decoder-L: the closed form of sebits.unit_decoder (class means of the 20,000 decoder targets for each
    LAB message label), applied to the D_dev LAB messages, against the D_dev PSNR of L1_LAB."""
    spec = S.spec_from_state(torch.load(ctx.out / "arms" / "LAB.pt", weights_only=True, map_location="cpu")["spec"])
    z = np.load(ctx.out / "msgs" / "LAB.npz", allow_pickle=False)
    X = {s: S.dec_in_from_packed(spec, z[f"bits_{s}"]) for s in ("aux", "auxF", "dev")}
    aux = ctx.data.images("aux")
    t_u8 = torch.cat([aux, aux.flip(3)]).cpu().numpy()
    dev_np = ctx.data.images("dev").cpu().numpy()
    lab_tr = torch.cat([X["aux"], X["auxF"]]).argmax(1).numpy()
    lab_dev = X["dev"].argmax(1).numpy()
    means = np.stack([t_u8[lab_tr == c].astype(np.float64).mean(0) if (lab_tr == c).any()
                      else t_u8.astype(np.float64).mean(0) for c in range(10)])
    closed = float(S.psnr_u8(np.round(means).astype(np.uint8)[lab_dev], dev_np)[0].mean())
    got = S.read_json(ctx.out / "dec" / "L1_LAB.json")["dev_psnr"]
    ok = bool(got >= closed - V1_TOL)
    S.append_jsonl(ctx.out / "checks.jsonl", {"check": "V1", "where": "L1_LAB", "ok": ok, "dev_psnr": got,
                                              "closed": closed, "time": S.now()})
    S.log(f"V1 (Decoder-L): D_dev PSNR {got:.3f} dB, closed form {closed:.3f} dB, ok {ok}")


def seed_units(ctx):
    sel = ctx.sel
    bot = bot_star(sel)

    def n_unit():
        S.unit_N(ctx)
        S.check_v6(ctx)

    def lab_large(d):
        S.unit_decoder(ctx, *d)
        v1_large(ctx)

    units = [("N", n_unit), ("head", lambda: S.unit_head(ctx)),
             ("topk", lambda: S.unit_topk(ctx, fixed=sel["TOPK"])),
             (bot, lambda: S.unit_bot(ctx, bot, *S.parse_cfg(bot))),
             ("msgs", lambda: S.unit_msgs(ctx, MSG_NAMES))]
    for d in decs_for(ctx.seed):
        fn = (lambda d=d: lab_large(d)) if d == ("LAB", "L", 1, False) else (lambda d=d: S.unit_decoder(ctx, *d))
        units.append((S.dec_tag(*d), fn))
    return units


def run_seed(seed):
    if seed not in NEW_SEEDS:
        raise S.CheckFail(f"{seed} is not a run seed of PREREGISTRATION.md")
    check_freeze(NPZ[:2])
    data = S.load_data("confirm")
    ctx = S.Ctx("seed", seed, S.seed_dir(seed), data)
    if ctx.sel is None:
        raise S.CheckFail("selection.json does not exist")
    if (ctx.out / "N.json").exists():
        S.check_v6(ctx)
    units = seed_units(ctx)
    if [u for u, _ in units] != planned_units(seed, ctx.sel):
        raise S.CheckFail("the unit list differs from the plan")
    S.run_units(ctx, units, 0)


# ------------------------------------------------------------------------------------------------ statistics

def two_level(D, seed=None, draws=None):
    """Bootstrap over seeds, then images (PREREGISTRATION.md Part 6). D: [K seeds, n images]. For each draw: K seed
    indices with replacement, then n image indices with replacement; one image draw serves every drawn seed. The
    draw value is the mean of D over the drawn seeds and the drawn images. Chunks of 250 draws: first the seed
    indices of the chunk, size (250, K), then the image indices, size (250, n)."""
    D = np.asarray(D, dtype=np.float64)
    K, n = D.shape
    draws = N_BOOT2 if draws is None else draws
    rng = np.random.default_rng(BOOT2_SEED if seed is None else seed)
    out = []
    for c in range(0, draws, BOOT2_CHUNK):
        m = min(BOOT2_CHUNK, draws - c)
        si = rng.integers(0, K, size=(m, K))
        ii = rng.integers(0, n, size=(m, n))
        W = np.bincount((ii + n * np.arange(m)[:, None]).ravel(), minlength=m * n).reshape(m, n) / n
        Dw = D @ W.T  # [K, m]: the mean over the drawn images, for each seed
        out.append(Dw[si, np.arange(m)[:, None]].mean(1))
    return np.concatenate(out)


def t_interval(v):
    v = np.asarray(v, dtype=np.float64)
    k = len(v)
    h = T975[k - 1] * v.std(ddof=1) / np.sqrt(k)
    return float(v.mean() - h), float(v.mean() + h)


def reading(d, ci):
    """Primary rule (PREREGISTRATION.md Part 7)."""
    if ci[0] > 0 and d >= MARGIN:
        return "R1"
    if ci[0] > 0:
        return "R2"
    if ci[1] < 0:
        return "R4"
    return "R3"


def leak_row(D, seed=None):
    D = np.asarray(D, dtype=np.float64)
    m = two_level(D, seed)
    ci = S.ci95(m)
    d = float(D.mean())
    seeds = D.mean(1).tolist()
    return {"d_leak": d, "d_leak_ci": ci, "d_leak_seeds": seeds, "seed_sd": float(np.std(seeds, ddof=1)),
            "t_ci": t_interval(seeds), "reading": reading(d, ci), "ci_low_ge_margin": bool(ci[0] >= MARGIN)}


def acc_row(C_sym, C_act):
    A = 100.0 * (np.asarray(C_sym, np.float64) - np.asarray(C_act, np.float64))
    ci = S.ci95(two_level(A))
    return {"d_acc": float(A.mean()), "d_acc_ci": ci, "d_acc_seeds": A.mean(1).tolist(),
            "outcome": "A+ replicates" if ci[0] > -ACC_MARGIN else "A+ not replicated"}


# ------------------------------------------------------------------------------------------------ analysis

def collect_checks(seeds, sel):
    """V1 to V6 from the checks.jsonl of each seed, and V5 from the done markers and the decoder outputs. For each
    seed, check and place ("where"), only the last record counts: a rerun of a unit replaces the records of the
    attempt before it (PREREGISTRATION.md Part 9.3). Also returns every failed record, replaced or not."""
    recs = {k: [] for k in ("V1", "V2", "V3", "V4", "V5", "V6")}
    failed_all = []
    need = {"V1": [], "V2": [], "V4": [], "V6": []}
    for seed in seeds:
        out = S.seed_dir(seed)
        p = out / "checks.jsonl"
        last = {}
        for line in (p.read_text().splitlines() if p.exists() else []):
            r = dict(json.loads(line), seed=seed)
            last[(r["check"], r.get("where"))] = r
            if not r.get("ok", False):
                failed_all.append(r)
        for (k, _), r in last.items():
            recs[k].append(r)
        need["V1"] += [(seed, "S1_LAB"), (seed, "L1_LAB")]
        need["V2"] += [(seed, S.dec_tag(*d)) for d in decs_for(seed) if d[3]]
        need["V4"] += [(seed, f"msgs {a}") for a in MSG_NAMES]
        need["V6"].append((seed, None))
        miss = [u for u in planned_units(seed, sel) if not (out / "done" / f"{u}.json").exists()]
        miss += [S.dec_tag(*d) for d in decs_for(seed) if not (out / "dec" / f"{S.dec_tag(*d)}.npz").exists()]
        recs["V5"].append({"seed": seed, "ok": not miss, "missing": sorted(set(miss))})
    ok = {}
    for k, v in recs.items():
        present = all(any(r["seed"] == s and (w is None or r.get("where") == w) for r in v) for s, w in need.get(k, []))
        ok[k] = bool(v) and present and all(r.get("ok", False) for r in v)
    return recs, ok, failed_all


def psnr_of(path, test_u8):
    return S.psnr_u8(np.load(path, allow_pickle=False)["recon_test"], test_u8)[0]


def analysis():
    check_freeze(NPZ)
    t_start = time.time()
    data = S.load_data("analysis")
    sel = S.read_json(S.ROOT / "selection.json")
    test_u8 = data.images("test").cpu().numpy()
    y_test = data.labels("test")
    aux = data.images("aux").cpu().numpy()
    tr = np.concatenate([aux, aux[:, :, :, ::-1]]).astype(np.float64)
    y_aux2 = np.concatenate([data.labels("aux")] * 2)
    f0 = np.round(tr.mean(0)).astype(np.uint8)
    fy = np.stack([np.round(tr[y_aux2 == c].mean(0)).astype(np.uint8) for c in range(10)])
    del tr
    floors = {"F0": float(S.psnr_u8(np.broadcast_to(f0, test_u8.shape), test_u8)[0].mean()),
              "FY": float(S.psnr_u8(fy[y_test], test_u8)[0].mean())}
    seeds = NEW_SEEDS
    recs, ok, failed_all = collect_checks(seeds, sel)
    valid = all(ok.values())
    R = {"time": S.now(), "seeds": list(seeds), "floors": floors,
         "checks": {k: {"ok": ok[k], "records": len(recs[k])} for k in recs},
         "check_failures": {k: [r for r in v if not r.get("ok", False)] for k, v in recs.items()},
         "failed_records_all": failed_all, "valid": valid}
    S.RES.mkdir(exist_ok=True)
    if not ok["V5"]:
        R["primary"] = {"reading": "void", "reading_text": READING_TEXT["void"], "reason": "V5: a planned output is absent"}
        S.write_json(S.RES / "results.json", R)
        (S.RES / "table.md").write_text("# Results\n\nVoid: V5 failed (a planned output is absent).\n")
        S.log("analysis: void (V5 failed)")
        return R
    corr, ps = {}, {}
    for a in MSG_NAMES:
        corr[a] = np.stack([(np.load(S.seed_dir(s) / "msgs" / f"{a}.npz", allow_pickle=False)["pred_test"] == y_test)
                            .astype(np.float64) for s in seeds])
    for d in DECS:
        tag = S.dec_tag(*d)
        ps[tag] = np.stack([psnr_of(S.seed_dir(s) / "dec" / f"{tag}.npz", test_u8) for s in seeds])
    prim = leak_row(ps["L1_BOT@64"] - ps["L1_TOPK@64"])
    prim["reading_final"] = prim["reading"] if valid else "void"
    prim["reading_text"] = READING_TEXT[prim["reading_final"]]
    R["primary"] = dict(prim, sym="TOPK@64", act="BOT@64", decoder="L", bits={"TOPK@64": None, "BOT@64": None})
    for a in ("TOPK@64", "BOT@64"):
        R["primary"]["bits"][a] = int(np.load(S.seed_dir(seeds[0]) / "msgs" / f"{a}.npz", allow_pickle=False)["nbits"])
    R["accuracy"] = acc_row(corr["TOPK@64"], corr["BOT@64"])
    sec = {"decoder_S_new_seeds": leak_row(ps["S1_BOT@64"] - ps["S1_TOPK@64"])}
    sec["sealed_rule_L"] = S.compare(corr["TOPK@64"], corr["BOT@64"], ps["L1_TOPK@64"], ps["L1_BOT@64"])
    sec["sealed_rule_S"] = S.compare(corr["TOPK@64"], corr["BOT@64"], ps["S1_TOPK@64"], ps["S1_BOT@64"])
    sealed = {}
    for tag in ("S1_TOPK@64", "S1_BOT@64"):
        files = [SEALED / "results" / f"s{s}" / "dec" / f"{tag}.npz" for s in SEALED_SEEDS]
        if all(f.exists() for f in files):
            sealed[tag] = np.stack([psnr_of(f, test_u8) for f in files])
    if len(sealed) == 2:
        D9 = np.concatenate([sealed["S1_BOT@64"] - sealed["S1_TOPK@64"], ps["S1_BOT@64"] - ps["S1_TOPK@64"]])
        sec["pooled_S_sealed_and_new"] = dict(leak_row(D9), seeds=list(SEALED_SEEDS) + list(seeds))
    fl = [SEALED / "results" / f"s{SEALED_SEEDS[0]}" / "dec" / f"L1_{a}.npz" for a in ("BOT@64", "TOPK@64")]
    if all(f.exists() for f in fl):
        sec["sealed_seed1_L"] = float((psnr_of(fl[0], test_u8) - psnr_of(fl[1], test_u8)).mean())
    sec["decoder_gap_L_minus_S"] = {
        "d_leak_seeds": (np.asarray(prim["d_leak_seeds"]) - np.asarray(sec["decoder_S_new_seeds"]["d_leak_seeds"])).tolist(),
        "psnr": {a: (ps[f"L1_{a}"].mean(1) - ps[f"S1_{a}"].mean(1)).tolist() for a in MSG_NAMES}}
    R["secondary"] = sec
    desc = {"N_dev_acc": [S.read_json(S.seed_dir(s) / "N.json")["dev_acc"] for s in seeds]}
    for a in MSG_NAMES:
        desc[a] = {"acc_seeds": corr[a].mean(1).tolist(), "acc": float(corr[a].mean()),
                   "psnr_L_seeds": ps[f"L1_{a}"].mean(1).tolist(), "psnr_L": float(ps[f"L1_{a}"].mean()),
                   "psnr_S_seeds": ps[f"S1_{a}"].mean(1).tolist(), "psnr_S": float(ps[f"S1_{a}"].mean()),
                   "best_epoch_L": [S.read_json(S.seed_dir(s) / "dec" / f"L1_{a}.json")["best_epoch"] for s in seeds],
                   "best_epoch_S": [S.read_json(S.seed_dir(s) / "dec" / f"S1_{a}.json")["best_epoch"] for s in seeds]}
    desc["V1"] = [{k: r.get(k) for k in ("seed", "where", "dev_psnr", "closed", "ok")} for r in recs["V1"]]
    desc["V2"] = [{k: r.get(k) for k in ("seed", "where", "dev_psnr", "F0", "ok")} for r in recs["V2"]]
    desc["minutes"] = (time.time() - t_start) / 60.0
    R["descriptive"] = desc
    S.write_json(S.RES / "results.json", R)
    (S.RES / "table.md").write_text(table_md(R))
    S.log(f"analysis: primary {prim['reading_final']} ({prim['reading_text']}): d_leak {prim['d_leak']:+.3f} dB, "
          f"CI {prim['d_leak_ci'][0]:+.3f} to {prim['d_leak_ci'][1]:+.3f}; {desc['minutes']:.1f} min")
    return R


def table_md(R):
    f = S.fmt
    p, a, sec = R["primary"], R["accuracy"], R["secondary"]
    L = ["# Results: replication of the leakage result (TOPK@64 against BOT@64)", "",
         f"Seeds: {', '.join(str(s) for s in R['seeds'])}. Valid: {R['valid']}. "
         f"Checks: {', '.join(k + ' ' + ('pass' if v['ok'] else 'FAIL') for k, v in R['checks'].items())}.", "",
         "| Row | Value | 95% CI (two-level) | Seed values | Reading |", "|---|---|---|---|---|",
         f"| PRIMARY Δleak, Decoder-L (dB) | {f(p['d_leak'])} | {f(p['d_leak_ci'][0])} to {f(p['d_leak_ci'][1])} | "
         f"{', '.join(f(v) for v in p['d_leak_seeds'])} | {p['reading_final']}: {p['reading_text']} |",
         f"| Δacc (points; secondary) | {f(a['d_acc'], 2)} | {f(a['d_acc_ci'][0], 2)} to {f(a['d_acc_ci'][1], 2)} | "
         f"{', '.join(f(v, 2) for v in a['d_acc_seeds'])} | {a['outcome']} |"]
    s = sec["decoder_S_new_seeds"]
    L.append(f"| Δleak, Decoder-S, new seeds (secondary) | {f(s['d_leak'])} | {f(s['d_leak_ci'][0])} to "
             f"{f(s['d_leak_ci'][1])} | {', '.join(f(v) for v in s['d_leak_seeds'])} | {s['reading']} (secondary) |")
    if "pooled_S_sealed_and_new" in sec:
        s = sec["pooled_S_sealed_and_new"]
        L.append(f"| Δleak, Decoder-S, 3 sealed + {len(R['seeds'])} new seeds (descriptive) | {f(s['d_leak'])} | "
                 f"{f(s['d_leak_ci'][0])} to {f(s['d_leak_ci'][1])} | {', '.join(f(v) for v in s['d_leak_seeds'])} | "
                 f"{s['reading']} (descriptive) |")
    L += ["", f"- Seed SD of the primary Δleak: {f(p['seed_sd'])} dB. Student t 95% interval over the seed means: "
          f"{f(p['t_ci'][0])} to {f(p['t_ci'][1])} dB. CI lower bound at or above the margin: {p['ci_low_ge_margin']}.",
          f"- Sealed rule (image bootstrap only) on the new seeds: Decoder-L {sec['sealed_rule_L']['L']} "
          f"(CI {f(sec['sealed_rule_L']['d_leak_ci'][0])} to {f(sec['sealed_rule_L']['d_leak_ci'][1])}), "
          f"Decoder-S {sec['sealed_rule_S']['L']} "
          f"(CI {f(sec['sealed_rule_S']['d_leak_ci'][0])} to {f(sec['sealed_rule_S']['d_leak_ci'][1])}).",
          f"- Floors on test: F0 {f(R['floors']['F0'])} dB, FY {f(R['floors']['FY'])} dB."]
    if "sealed_seed1_L" in sec:
        L.append(f"- Sealed seed-1 Δleak with Decoder-L (reference): {f(sec['sealed_seed1_L'])} dB.")
    d = R["descriptive"]
    L += ["", "| Arm | Test acc | PSNR Decoder-L (dB) | PSNR Decoder-S (dB) |", "|---|---|---|---|"]
    for a in MSG_NAMES:
        L.append(f"| {a} | {f(d[a]['acc'], 4)} | {f(d[a]['psnr_L'])} | {f(d[a]['psnr_S'])} |")
    L += ["", f"N D_dev accuracy per seed: {', '.join(f(v, 4) for v in d['N_dev_acc'])}.", ""]
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------ main

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["test", "freeze", "seed", "analysis"])
    ap.add_argument("arg", nargs="?")
    a = ap.parse_args(argv)
    if a.cmd == "test":
        r = run_tests()
        return 0 if not (r.failures or r.errors) else 1
    name = f"seed_{a.arg}" if a.cmd == "seed" else a.cmd
    try:
        if a.cmd == "freeze":
            cmd_freeze(a.arg)
        elif a.cmd == "seed":
            if not torch.backends.mps.is_available():
                raise S.CheckFail("MPS is not available")
            S.DEV = torch.device("mps")
            S.log(f"job {name} start (device {S.DEV})")
            run_seed(int(a.arg))
            S.log(f"job {name} end")
        else:
            S.log("job analysis start (device cpu)")
            analysis()
            S.log("job analysis end")
        return 0
    except S.StopRule as e:
        S.log(f"STOP RULE: {e}")
        S.write_json(S.RES / f"STOP_{name}.json", {"time": S.now(), "stop_rule": str(e)})
        return 3
    except S.CheckFail as e:
        S.log(f"CHECK FAIL: {e}")
        S.write_json(S.RES / f"CHECKFAIL_{name}.json", {"time": S.now(), "check": str(e)})
        return 4


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    finally:  # run_sleak.sh reads this file: zsh loses the exit code of a job that was paused
        if os.environ.get("SLEAK_EXIT_FILE"):
            Path(os.environ["SLEAK_EXIT_FILE"]).write_text(f"{rc}\n")
    sys.exit(rc)
