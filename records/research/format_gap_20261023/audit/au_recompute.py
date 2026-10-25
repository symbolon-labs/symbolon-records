"""Independent audit (AU) of the format-gap confirmatory run.

This script recomputes the primary rows, the case and the "beats" rows
from confirm/fresh_probs.npz. It uses its own statistics code. It does
not import ugap.py or uconfirm.py. It loads no model and uses the CPU only.

Run (from the folder format_gap_20261023):
    ../../.venv/bin/python -I -B audit/au_recompute.py
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NPZ = os.path.join(ROOT, "confirm", "fresh_probs.npz")
SEL = os.path.join(ROOT, "dev", "dev_selection.json")
OUT = os.path.join(HERE, "au_recompute.json")

EPS = 1e-6
N_DRAWS = 10_000
CHUNK = 500
# The audit seed. It is different from the harness seed (20261023) on purpose.
AU_SEED = 991_024
CLOSE_MEAN = 0.005
CLOSE_UPPER = 0.010
BEATS_MARGIN = 0.02
EXPECTED_POOLS = {"V": 1991, "S0": 1838, "S1": 4000}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_arms(z):
    """Return y, pool and a dict arm -> (seeds, matrix of item NLL [S, n])."""
    y = z["y"].astype(np.float64)
    pool = z["pool"].astype(str)
    groups = {}
    for key in z.files:
        if not key.startswith("p_"):
            continue
        arm, seed = key[2:].rsplit("_seed", 1)
        groups.setdefault(arm, []).append(int(seed))
    arms = {}
    for arm, seeds in groups.items():
        seeds = sorted(seeds)
        rows = []
        for s in seeds:
            p = np.asarray(z[f"p_{arm}_seed{s}"], dtype=np.float64)
            if not np.all(np.isfinite(p)):
                raise SystemExit(f"non-finite probability in {arm} seed {s}")
            q = np.clip(p, EPS, 1.0 - EPS)
            rows.append(-(y * np.log(q) + (1.0 - y) * np.log1p(-q)))
        arms[arm] = (seeds, np.vstack(rows))
    return y, pool, arms


def item_nll(arms, arm):
    """NLL_i(arm): the mean over the seeds of the arm, for each item."""
    return arms[arm][1].mean(axis=0)


def counts_from_indices(idx, size):
    """Row-wise counts of the indices in idx [c, k] over range(size)."""
    c = idx.shape[0]
    flat = (idx + size * np.arange(c)[:, None]).ravel()
    return np.bincount(flat, minlength=c * size).reshape(c, size).astype(np.float64)


def two_level(arms, arm_names, rows, item_mask=None, seed=AU_SEED):
    """Two-level paired bootstrap (prereg 6.1), own implementation.

    Each draw resamples the items with replacement (the same items for
    every arm). For each arm, it also resamples the seeds of that arm with
    replacement, independently. The statistic of an arm in a draw is the
    mean over the drawn items of the mean over the drawn seeds.
    The weighted form below is equal to that index form: the mean over the
    drawn items of x_i is w @ x with w = item counts / n, and the mean over
    the drawn seeds is the seed counts / S.
    """
    rng = np.random.Generator(np.random.Philox(seed))
    mats = {}
    for a in arm_names:
        m = arms[a][1]
        mats[a] = m if item_mask is None else m[:, item_mask]
    n = next(iter(mats.values())).shape[1]
    stats = {a: np.empty(N_DRAWS) for a in arm_names}
    first_draws = []
    done = 0
    while done < N_DRAWS:
        c = min(CHUNK, N_DRAWS - done)
        item_idx = rng.integers(0, n, size=(c, n))
        w = counts_from_indices(item_idx, n) / n  # [c, n]
        seed_idx = {}
        for a in arm_names:
            s = mats[a].shape[0]
            seed_idx[a] = rng.integers(0, s, size=(c, s))
            sw = counts_from_indices(seed_idx[a], s) / s  # [c, S]
            per_seed_item_mean = w @ mats[a].T  # [c, S]
            stats[a][done:done + c] = np.sum(per_seed_item_mean * sw, axis=1)
        if done == 0:
            # Keep three draws for the slow index-form check.
            for b in range(3):
                first_draws.append((item_idx[b].copy(),
                                    {a: seed_idx[a][b].copy() for a in arm_names}))
        done += c
    # Slow check: the index form of the statistic for the first draws.
    max_diff = 0.0
    for b, (ii, sd) in enumerate(first_draws):
        for a in arm_names:
            slow = mats[a][sd[a]][:, ii].mean(axis=0).mean()
            max_diff = max(max_diff, abs(slow - stats[a][b]))
    out = {}
    for name, (x, c_arm) in rows.items():
        point = float(mats[x].mean(axis=0).mean() - mats[c_arm].mean(axis=0).mean())
        d = stats[x] - stats[c_arm]
        lo, hi = np.percentile(d, [2.5, 97.5])
        out[name] = {"x": x, "c": c_arm, "mean": point,
                     "ci_lo": float(lo), "ci_hi": float(hi)}
    return out, max_diff, n


def item_only_ci(d, seed):
    """Item-only bootstrap of the mean of d: 10,000 draws, percentile CI."""
    rng = np.random.Generator(np.random.Philox(seed))
    n = d.shape[0]
    means = np.empty(N_DRAWS)
    done = 0
    while done < N_DRAWS:
        c = min(CHUNK, N_DRAWS - done)
        idx = rng.integers(0, n, size=(c, n))
        means[done:done + c] = d[idx].mean(axis=1)
        done += c
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def beats_row(arms, worse, better, seed):
    """Confirmatory "beats" rule: arm `better` beats arm `worse`.

    d_i = NLL_i(worse) - NLL_i(better). The rule needs mean(d) >= 0.02,
    an item-only CI lower bound above 0, and each per-seed mean above 0
    (seed k of one arm with seed k of the other arm).
    """
    d = item_nll(arms, worse) - item_nll(arms, better)
    lo, hi = item_only_ci(d, seed)
    sw, mw = arms[worse]
    sb, mb = arms[better]
    k = min(len(sw), len(sb))
    per_seed = [float((mw[j] - mb[j]).mean()) for j in range(k)]
    mean = float(d.mean())
    passed = bool(mean >= BEATS_MARGIN and lo > 0 and all(v > 0 for v in per_seed))
    return {"better": better, "worse": worse, "mean": mean, "ci_lo": lo,
            "ci_hi": hi, "per_seed": per_seed,
            "seed_pairs": [[sb[j], sw[j]] for j in range(k)], "beats": passed}


def closes(r):
    return bool(r["mean"] <= CLOSE_MEAN and r["ci_hi"] < CLOSE_UPPER)


def remains(r):
    return bool((not closes(r)) and r["ci_lo"] > 0)


def gains(r):
    return bool(r["ci_lo"] > 0)


def decide_case(valid, lab):
    """Steps of prereg 10.2, in order."""
    if not valid:
        return "0"
    if lab["F3_closes_C"]:
        return "R"
    if lab["Fstar_closes_C"]:
        return "A1" if lab["Fstar_gains"] else "A2"
    if lab["Fstar_remains_C"] and lab["Fstar_gains"]:
        return "B"
    if lab["Fstar_remains_C"]:
        return "C"
    return "D"


def seed_spread(arms, names, rows, seeds):
    """Run the own bootstrap with more audit seeds (Monte Carlo spread)."""
    spread = {k: {"ci_lo": [], "ci_hi": []} for k in rows}
    for s in seeds:
        b, _, _ = two_level(arms, names, rows, seed=s)
        for k in rows:
            spread[k]["ci_lo"].append(b[k]["ci_lo"])
            spread[k]["ci_hi"].append(b[k]["ci_hi"])
    return {k: {"seeds": list(seeds),
                "ci_lo_min": min(v["ci_lo"]), "ci_lo_max": max(v["ci_lo"]),
                "ci_hi_min": min(v["ci_hi"]), "ci_hi_max": max(v["ci_hi"])}
            for k, v in spread.items()}


def spec_stream(arms, names, rows, seed=20261023, chunk=250):
    """Try the draw stream as G16 of the prereg describes it (text only).

    numpy default_rng(seed); chunks of 250 draws; in each chunk, first the
    item indices (250 x n), then the seed indices (250 x S) of each arm, in
    sorted order of the arm names. G16 does not name the integer call, so
    this uses Generator.integers. An exact match is extra evidence only.
    """
    rng = np.random.default_rng(seed)
    n = arms[names[0]][1].shape[1]
    stats = {a: np.empty(N_DRAWS) for a in names}
    done = 0
    while done < N_DRAWS:
        c = min(chunk, N_DRAWS - done)
        item_idx = rng.integers(0, n, size=(c, n))
        w = counts_from_indices(item_idx, n) / n
        for a in names:
            m = arms[a][1]
            s = m.shape[0]
            sw = counts_from_indices(rng.integers(0, s, size=(c, s)), s) / s
            stats[a][done:done + c] = np.sum((w @ m.T) * sw, axis=1)
        done += c
    out = {}
    for k, (x, c_arm) in rows.items():
        lo, hi = np.percentile(stats[x] - stats[c_arm], [2.5, 97.5])
        out[k] = [float(lo), float(hi)]
    return out


def v4_items(pool):
    """V4 checks that the fresh files allow (no sealed file is read)."""
    with open(os.path.join(ROOT, "fresh", "fresh_items.json")) as f:
        items = json.load(f)["items"]
    with open(os.path.join(ROOT, "fresh", "fresh_manifest.json")) as f:
        man = json.load(f)
    order_ok = [it["pool"] for it in items] == pool.tolist()
    norm = [" ".join(it["question"].lower().split()) for it in items]
    dup = len(norm) - len(set(norm))
    return {"n_items_file": len(items), "pool_order_equals_npz": bool(order_ok),
            "manifest_counts": man["counts"],
            "manifest_counts_equal_expected": man["counts"] == EXPECTED_POOLS,
            "duplicate_fresh_questions_simple_norm": int(dup),
            "spent_question_check": "not done by AU (needs the sealed items; "
                                    "taken from results.json V4)"}


def compare(own):
    """Compare the own values with confirm/results.json (read only here)."""
    with open(os.path.join(ROOT, "confirm", "results.json")) as f:
        res = json.load(f)
    rule = res["rule"]
    prim, sec = rule["primary"], rule["secondary"]
    harness_rows = {
        "gap_Fstar_LprobeR": prim["gap_star"],
        "gap_F3_LprobeR": prim["gap_ref"],
        "gain_Fstar": prim["gain_star"],
        "gap_Fstar_Lprobe": sec[f"gap({own['F_star']}, L-probe)"],
        "gap_F3_Lprobe": sec["gap(F3, L-probe)"],
        "gain_Fother": sec[f"gap(F3, {own['F_other']})"],
        "gap_Lprobe_LprobeR": sec["gap(L-probe, L-probe-R)"],
    }
    rows = {}
    for k, h in harness_rows.items():
        o = own["rows"][k]
        rows[k] = {"au_mean": o["mean"], "h_mean": h["mean"],
                   "d_mean": abs(o["mean"] - h["mean"]),
                   "au_ci": [o["ci_lo"], o["ci_hi"]], "h_ci": h["ci95"],
                   "d_ci": max(abs(o["ci_lo"] - h["ci95"][0]),
                               abs(o["ci_hi"] - h["ci95"][1]))}
    lab = own["labels"]
    labels = {
        "gap_star closes": (lab["Fstar_closes_C"], prim["gap_star"]["closes"]),
        "gap_star remains": (lab["Fstar_remains_C"], prim["gap_star"]["remains"]),
        "gap_ref closes": (lab["F3_closes_C"], prim["gap_ref"]["closes"]),
        "gap_ref remains": (lab["F3_remains_C"], prim["gap_ref"]["remains"]),
        "gain_star gains": (lab["Fstar_gains"], prim["gain_star"]["gains"]),
        "F* vs L-probe closes": (lab["Fstar_closes_Lprobe"],
                                 harness_rows["gap_Fstar_Lprobe"]["closes"]),
        "F* vs L-probe remains": (lab["Fstar_remains_Lprobe"],
                                  harness_rows["gap_Fstar_Lprobe"]["remains"]),
        "F3 vs L-probe closes": (lab["F3_closes_Lprobe"],
                                 harness_rows["gap_F3_Lprobe"]["closes"]),
        "F3 vs L-probe remains": (lab["F3_remains_Lprobe"],
                                  harness_rows["gap_F3_Lprobe"]["remains"]),
    }
    # Beats rows: the harness row (a, b) means "a beats b", with
    # mean_b_minus_a = mean of NLL_i(b) - NLL_i(a).
    hb = {(r["a"], r["b"]): r for r in sec["beats"]}
    fs = own["F_star"]
    pairs = {"F3_beats_L": ("F3", "L"), "Fstar_beats_L": (fs, "L"),
             "L_beats_F3": ("L", "F3"), "L_beats_Fstar": ("L", fs)}
    beats = {}
    for k, key in pairs.items():
        o, h = own["beats"][k], hb[key]
        beats[k] = {"au_mean": o["mean"], "h_mean": h["mean_b_minus_a"],
                    "d_mean": abs(o["mean"] - h["mean_b_minus_a"]),
                    "au_ci": [o["ci_lo"], o["ci_hi"]], "h_ci": h["ci95_item"],
                    "d_ci": max(abs(o["ci_lo"] - h["ci95_item"][0]),
                                abs(o["ci_hi"] - h["ci95_item"][1])),
                    "d_per_seed": max(abs(a - b) for a, b in
                                      zip(o["per_seed"], h["per_seed"])),
                    "au_beats": o["beats"], "h_beats": h["beats"]}
    means = {a: abs(own["mean_item_nll"][a] - v) for a, v in rule["means"].items()}
    validity = rule["validity"]
    valid = all(validity.values()) and own["v4_pool_counts_ok"]
    au_case = own["case_if_V1_to_V3_pass"] if valid else "0"
    share_d = abs(own["share_of_F3_gap_removed"] - prim["share_of_F3_gap_removed"])
    return {
        "harness_status": res["status"], "harness_boot_seed": rule["boot_seed"],
        "harness_validity": validity, "harness_case": rule["case"],
        "harness_case_name": rule["case_name"], "au_case": au_case,
        "harness_F_star": rule["F_star"], "harness_F_other": rule["F_other"],
        "harness_C_star": rule["C_star"],
        "harness_pool_sizes": res["sizes"]["per_pool"],
        "rows": rows, "labels": {k: {"au": a, "h": b, "equal": a == b}
                                 for k, (a, b) in labels.items()},
        "beats": beats, "max_d_arm_mean_nll": max(means.values()),
        "d_share": share_d,
        "max_d_point": max([r["d_mean"] for r in rows.values()]
                           + [r["d_mean"] for r in beats.values()]
                           + [max(means.values()), share_d]),
        "max_d_ci": max([r["d_ci"] for r in rows.values()]
                        + [r["d_ci"] for r in beats.values()]),
        "max_d_per_seed_beats": max(r["d_per_seed"] for r in beats.values()),
    }


def main():
    with open(SEL) as f:
        sel = json.load(f)
    f_star, f_other = sel["F_star"], sel["F_other"]
    z = np.load(NPZ, allow_pickle=False)
    y, pool, arms = load_arms(z)
    n = y.shape[0]

    # V4 (the part that the arrays can show): the pool counts.
    pool_counts = {p: int(np.sum(pool == p)) for p in sorted(set(pool.tolist()))}
    v4_counts_ok = pool_counts == EXPECTED_POOLS and n == sum(EXPECTED_POOLS.values())

    c_star = "L-probe-R"
    names = sorted({"F3", f_star, f_other, c_star, "L-probe", "L"})
    rows = {
        "gap_Fstar_LprobeR": (f_star, c_star),
        "gap_F3_LprobeR": ("F3", c_star),
        "gain_Fstar": ("F3", f_star),
        "gap_Fstar_Lprobe": (f_star, "L-probe"),
        "gap_F3_Lprobe": ("F3", "L-probe"),
        "gain_Fother": ("F3", f_other),
        "gap_Lprobe_LprobeR": ("L-probe", c_star),
    }
    boot, slow_diff, _ = two_level(arms, names, rows)

    # Point values from the plain definition, as a second path.
    direct = {k: float(np.mean(item_nll(arms, x) - item_nll(arms, c)))
              for k, (x, c) in rows.items()}
    point_path_diff = max(abs(direct[k] - boot[k]["mean"]) for k in rows)

    lab = {
        "F3_closes_C": closes(boot["gap_F3_LprobeR"]),
        "F3_remains_C": remains(boot["gap_F3_LprobeR"]),
        "Fstar_closes_C": closes(boot["gap_Fstar_LprobeR"]),
        "Fstar_remains_C": remains(boot["gap_Fstar_LprobeR"]),
        "Fstar_gains": gains(boot["gain_Fstar"]),
        "F3_closes_Lprobe": closes(boot["gap_F3_Lprobe"]),
        "F3_remains_Lprobe": remains(boot["gap_F3_Lprobe"]),
        "Fstar_closes_Lprobe": closes(boot["gap_Fstar_Lprobe"]),
        "Fstar_remains_Lprobe": remains(boot["gap_Fstar_Lprobe"]),
    }
    # The case if V1 to V3 pass (taken from results.json in the compare step).
    case_if_valid = decide_case(v4_counts_ok, lab)
    share = boot["gain_Fstar"]["mean"] / boot["gap_F3_LprobeR"]["mean"]

    beats = {
        "F3_beats_L": beats_row(arms, "L", "F3", AU_SEED + 1),
        "Fstar_beats_L": beats_row(arms, "L", f_star, AU_SEED + 2),
        "L_beats_F3": beats_row(arms, "F3", "L", AU_SEED + 3),
        "L_beats_Fstar": beats_row(arms, f_star, "L", AU_SEED + 4),
    }

    mean_nll = {a: float(item_nll(arms, a).mean()) for a in sorted(arms)}
    seeds = {a: arms[a][0] for a in sorted(arms)}

    out = {
        "status": "AUDIT (AU) recompute, own code, CPU only",
        "npz_sha256": sha256(NPZ),
        "script_sha256": sha256(os.path.abspath(__file__)),
        "F_star": f_star, "F_other": f_other, "C_star": c_star,
        "n_items": n, "pool_counts": pool_counts,
        "v4_pool_counts_ok": bool(v4_counts_ok),
        "seeds": seeds, "mean_item_nll": mean_nll,
        "bootstrap": {"draws": N_DRAWS, "generator": "numpy Philox",
                      "seed": AU_SEED, "percentiles": "numpy linear",
                      "slow_check_max_abs_diff": float(slow_diff),
                      "point_two_paths_max_abs_diff": float(point_path_diff)},
        "rows": boot, "labels": lab,
        "share_of_F3_gap_removed": share,
        "case_if_V1_to_V3_pass": case_if_valid,
        "beats": beats,
    }
    # The own numbers go to disk first. Only then does the script read the
    # harness output.
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)

    out["v4_items"] = v4_items(pool)
    out["seed_spread"] = seed_spread(arms, names, rows,
                                     [AU_SEED + 100 + j for j in range(20)])
    all_arms = sorted(arms)
    out["spec_stream_all_arms"] = spec_stream(arms, all_arms, rows)
    out["compare"] = compare(out)
    sc = out["compare"]
    h_ci = {k: sc["rows"][k]["h_ci"] for k in rows}
    out["spec_stream_max_d_vs_harness"] = max(
        max(abs(a - b) for a, b in zip(out["spec_stream_all_arms"][k], h_ci[k]))
        for k in rows)
    labels_equal = all(v["equal"] for v in sc["labels"].values())
    beats_equal = all(v["au_beats"] == v["h_beats"] for v in sc["beats"].values())
    v4_ok = (out["v4_pool_counts_ok"] and out["v4_items"]["pool_order_equals_npz"]
             and out["v4_items"]["manifest_counts_equal_expected"]
             and out["v4_items"]["duplicate_fresh_questions_simple_norm"] == 0)
    checks = {
        "point_values_agree_1e-9": sc["max_d_point"] <= 1e-9,
        "ci_bounds_agree_1e-3": sc["max_d_ci"] <= 1e-3,
        "labels_equal": labels_equal,
        "beats_equal": beats_equal,
        "case_equal": sc["au_case"] == sc["harness_case"],
        "F_star_F_other_C_star_equal": (sc["harness_F_star"], sc["harness_F_other"],
                                         sc["harness_C_star"]) == (f_star, f_other, c_star),
        "v4_own_checks": bool(v4_ok),
        "bootstrap_index_form_check": slow_diff < 1e-12,
    }
    checks = {k: bool(v) for k, v in checks.items()}
    out["checks"] = checks
    out["verdict"] = "audit passes" if all(checks.values()) else "audit fails"
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("checks", checks, "->", out["verdict"])
    print("max_d_point", sc["max_d_point"], "max_d_ci", sc["max_d_ci"],
          "spec_stream_max_d", out["spec_stream_max_d_vs_harness"])
    print(json.dumps({k: out[k] for k in ("pool_counts", "v4_pool_counts_ok",
                                           "case_if_V1_to_V3_pass",
                                           "share_of_F3_gap_removed")}, indent=1))
    for k, r in boot.items():
        print(f"{k:22s} mean {r['mean']:+.9f}  CI [{r['ci_lo']:+.6f}, {r['ci_hi']:+.6f}]")
    for k, r in beats.items():
        print(f"{k:16s} mean {r['mean']:+.6f}  CI [{r['ci_lo']:+.6f}, {r['ci_hi']:+.6f}]"
              f"  per-seed {[round(v, 6) for v in r['per_seed']]}  beats={r['beats']}")
    print("labels", lab)
    print("slow check", slow_diff, "two paths", point_path_diff)


if __name__ == "__main__":
    sys.exit(main())
