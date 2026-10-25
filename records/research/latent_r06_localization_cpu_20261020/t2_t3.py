"""EXPLORATORY. R.06 localization CPU tests T2 (encoder variants) and T3 (residual pull).

CPU only. Read-only inputs. Writes only into this folder. See PLAN.md.
Modes: selftest (synthetic data only), freeze, run.
"""
import argparse, hashlib, json, os, sys
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(2)

ROOT = Path(__file__).resolve().parent
R = ROOT.parent
INPUTS = {
    "ids": "latent_binding_alignment_source_20260911/inputs.json",
    "captures": "latent_binding_alignment_source_20260911/run/captures.pt",
    "features": "latent_binding_alignment_source_20260911/run/features.pt",
    "stages": "latent_binding_identity_localization_20260910/stages.pt",
    "align_candidate": "latent_binding_training_alignment_20260911/run/candidate.pt",
    "d0_candidate": "latent_binding_training_fold_recovery_20260913/run/fold0/step0/candidate.pt",
    "d0_coordinates": "latent_binding_training_fold_recovery_20260913/coordinates.pt",
    "folds": "latent_binding_training_fold_recovery_20260913/folds.json",
    "novel_fit": "latent_binding_constrained_code_extended_20260911/run/novel_fit.pt",
    "familiar_fit": "latent_binding_constrained_code_extended_20260911/run/familiar_fit.pt",
}
EPS = np.finfo(float).eps


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load(key):
    return torch.load(R / INPUTS[key], map_location="cpu", weights_only=True)


def target(ids):
    a = np.zeros((len(ids), 24))
    for i, name in enumerate(ids):
        a[i, ord(name[0]) - 65] = 1
        a[i, 16 + int(name[1])] = 1
    return a


def coef_of(X, Y):
    return np.linalg.pinv(Y, rcond=max(Y.shape) * EPS) @ X


def centered_norm(X):
    return np.linalg.norm(X - X.mean(0))


def rho_in(X, Y):
    return float(np.linalg.norm(Y @ coef_of(X, Y) - X) / centered_norm(X))


def rho_loo(X, Y):
    sq = 0.0
    for i in range(len(X)):
        m = np.arange(len(X)) != i
        sq += float(np.square(Y[i] @ coef_of(X[m], Y[m]) - X[i]).sum())
    return float(np.sqrt(sq) / centered_norm(X))


def rho_null(X, Y, n=1000, seed=20261020):
    rng = np.random.default_rng(seed)
    vals = [rho_in(X, Y[rng.permutation(len(Y))]) for _ in range(n)]
    return float(np.median(vals))


def novel_residual(X, Y, Z, zids):
    coef = coef_of(X, Y)
    rms = np.sqrt(np.square(X - X.mean(0)).sum() / len(X))
    return {z: float(np.linalg.norm(target([z])[0] @ coef - Z[k]) / rms) for k, z in enumerate(zids)}


def ridge(x, y, z):
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x - xm, y - ym
    lam = 0.001 * np.square(xc).sum() / len(x)
    w = xc.T @ np.linalg.solve(xc @ xc.T + lam * np.eye(len(x)), yc)
    return (z - xm) @ w + ym, float(lam)


def decode(scores):
    return [chr(65 + int(a)) + str(int(b)) for a, b in zip(scores[:, :16].argmax(1), scores[:, 16:].argmax(1))]


def probe(X, ids, Z, zids):
    Y = target(ids)
    s, lam = ridge(X, Y, Z)
    pred = decode(s)
    loo = []
    for i in range(len(X)):
        m = np.arange(len(X)) != i
        loo.append(decode(ridge(X[m], Y[m], X[i:i + 1])[0])[0])
    return {"lambda": lam,
            "novel_correct": sum(p == t for p, t in zip(pred, zids)),
            "novel_predictions": dict(zip(zids, pred)),
            "loo_correct": sum(p == t for p, t in zip(loo, ids)),
            "loo_errors": {t: p for p, t in zip(loo, ids) if p != t}}


def encode(h, q, K, V, alpha_fixed=None, scale_fixed=None):
    hl = F.layer_norm(h, (h.shape[-1],))
    k = F.normalize(hl @ K.T, dim=-1, eps=1e-8)
    qq = F.normalize(q, dim=-1, eps=1e-8)
    alpha = (2.0 * (k @ qq)).softmax(-1)
    if alpha_fixed is not None:
        alpha = alpha_fixed.expand_as(alpha)
    u = (alpha[..., None] * (hl @ V.T)).sum(1)
    s = u.square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
    if scale_fixed is not None:
        s = torch.full_like(s, float(scale_fixed))
    return u / s, alpha, s, hl


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def dedupe(stage, meta):
    out = {}
    for i, m in enumerate(meta):
        v = stage[i]
        if m["id"] in out:
            assert torch.equal(out[m["id"]], v), m["id"]
        else:
            out[m["id"]] = v.clone()
    ids = sorted(out)
    return ids, torch.stack([out[i] for i in ids])


def gate(results, name, ok, detail):
    results["gates"][name] = {"pass": bool(ok), **detail}
    if not ok:
        results["stopped_at"] = name
        raise StopIteration(name)


def compute():
    res = {"label": "EXPLORATORY", "gates": {}, "stopped_at": None}
    ids = json.loads((R / INPUTS["ids"]).read_text())["ids"]
    Y = target(ids)
    cap = load("captures")
    assert torch.equal(cap["first"], cap["repeat"])
    H = cap["first"].double()
    feat = load("features")["wire"].double()
    st = load("stages")
    banks = {}
    for b in ["novel", "familiar"]:
        meta = st[b]["meta"]
        sids, sf = dedupe(st[b]["stages"]["sender_field"], meta)
        _, w64 = dedupe(st[b]["stages"]["wire64"], meta)
        _, sel = dedupe(st[b]["stages"]["selected_3"], meta)
        banks[b] = {"ids": sids, "h": sf.double().reshape(-1, 2, 2048), "wire": w64.double(), "sel": sel.double(),
                    "wire_rows": st[b]["stages"]["wire64"].double(), "sel_rows": st[b]["stages"]["selected_3"].double()}
    align = load("align_candidate")
    d0 = load("d0_candidate")
    enc_equal = all(torch.equal(align[k], d0[k]) for k in ["resampler.queries", "resampler.key.weight", "resampler.value.weight"])
    q = d0["resampler.queries"].double()[0]
    K = d0["resampler.key.weight"].double()
    V = d0["resampler.value.weight"].double()

    try:
        # Gate 1: encoder parity.
        Ea_tr, al_tr, s_tr, hl_tr = encode(H, q, K, V)
        err_tr = float((Ea_tr - feat).abs().max())
        enc = {}
        errs = {"training_vs_features": err_tr}
        for b in banks:
            e, al, s, hl = encode(banks[b]["h"], q, K, V)
            enc[b] = (e, hl)
            errs[f"{b}_vs_wire64"] = float((e - banks[b]["wire"]).abs().max())
        gate(res, "gate1_encoder_parity", enc_equal and max(errs.values()) <= 1e-5,
             {"max_abs_error": errs, "align_equals_d0_encoder": enc_equal})

        # Variants.
        a_mean = al_tr.mean(0)
        s_mean = float(s_tr.mean())
        variants_tr = {"S_raw": H.reshape(56, -1), "S_LN": hl_tr.reshape(56, -1), "E_a": Ea_tr,
                       "E_b": encode(H, q, K, V, alpha_fixed=a_mean)[0],
                       "E_c": encode(H, q, K, V, scale_fixed=s_mean)[0],
                       "E_d": encode(H, q, K, V, alpha_fixed=a_mean, scale_fixed=s_mean)[0]}
        hN = banks["novel"]["h"]
        eN, hlN = enc["novel"]
        variants_nv = {"S_raw": hN.reshape(8, -1), "S_LN": hlN.reshape(8, -1), "E_a": eN,
                       "E_b": encode(hN, q, K, V, alpha_fixed=a_mean)[0],
                       "E_c": encode(hN, q, K, V, scale_fixed=s_mean)[0],
                       "E_d": encode(hN, q, K, V, alpha_fixed=a_mean, scale_fixed=s_mean)[0]}
        nids = banks["novel"]["ids"]
        X = {k: v.numpy() for k, v in variants_tr.items()}
        Z = {k: v.numpy() for k, v in variants_nv.items()}

        # Gate 2: record residuals.
        r_a, r_s = rho_in(X["E_a"], Y), rho_in(X["S_raw"], Y)
        gate(res, "gate2_record_residuals", abs(r_a - 0.292580) <= 1e-4 and abs(r_s - 0.113833) <= 1e-4,
             {"rho_in_E_a": r_a, "rho_in_S_raw": r_s, "record_E_a": 0.292580, "record_S_raw": 0.113833,
              "design_rank": int(np.linalg.matrix_rank(Y))})

        # Gate 3: record probe on E_a.
        pa = probe(X["E_a"], ids, Z["E_a"], nids)
        wrong = {k: v for k, v in pa["novel_predictions"].items() if k != v}
        gate(res, "gate3_record_probe", pa["novel_correct"] == 6 and wrong == {"A0": "H3", "A2": "M3"} and pa["loo_correct"] == 44,
             {"novel_correct": pa["novel_correct"], "novel_errors": wrong, "loo_correct": pa["loo_correct"], "lambda": pa["lambda"]})

        # T2 metrics.
        t2 = {"alpha_mean_training": a_mean.tolist(), "scale_mean_training": s_mean, "variants": {}}
        for k in ["S_raw", "S_LN", "E_a", "E_b", "E_c", "E_d"]:
            t2["variants"][k] = {"dim": int(X[k].shape[1]), "rho_in": rho_in(X[k], Y), "rho_loo": rho_loo(X[k], Y),
                                 "rho_null_median": rho_null(X[k], Y),
                                 "novel_residual": novel_residual(X[k], Y, Z[k], nids),
                                 "probe": probe(X[k], ids, Z[k], nids)}
        v = t2["variants"]
        d_nl = v["E_a"]["rho_in"] - v["E_d"]["rho_in"]
        d_lin = v["E_d"]["rho_in"] - v["S_LN"]["rho_in"]
        if d_nl >= 0.10 and d_nl >= 2 * d_lin:
            reading = "H3, nonlinear"
        elif d_lin >= 0.10 and d_lin >= 2 * d_nl:
            reading = "H3, linear"
        else:
            reading = "H3, mixed"
        pd = v["E_d"]["probe"]["novel_predictions"]
        secondary = pd["A0"] == "A0" and pd["A2"] == "A2" and all(pd[i] == i for i in nids if i not in ("A0", "A2"))
        t2.update({"d_nl": d_nl, "d_lin": d_lin, "reading": reading, "secondary_E_d_fixes_A0_A2": secondary})
        res["T2"] = t2

        # Gate 4: linear map from slot to selected value.
        Wv = d0["reader.value.weight"].double()
        P = d0["projector.weight"].double()
        base = load("d0_coordinates")[0]["base"]
        base_equal = torch.equal(d0["reader.value.weight"], base)
        A = Wv @ P
        rel = {}
        for b in banks:
            pred = banks[b]["wire_rows"] @ A.T
            ref = banks[b]["sel_rows"]
            rel[b] = float((pred - ref).abs().max() / ref.abs().max())
        gate(res, "gate4_slot_to_code_map", base_equal and max(rel.values()) <= 1e-5,
             {"relative_error": rel, "d0_value_equals_base": base_equal})

        # Gate 5: oracle join.
        fits = {"novel": load("novel_fit"), "familiar": load("familiar_fit")}
        join, ok5 = {}, True
        for b in banks:
            init = fits[b]["initial"].double()
            order = []
            for r in range(init.shape[0]):
                d = (banks[b]["sel"] - init[r]).abs().amax(-1)
                hits = [banks[b]["ids"][j] for j in range(len(d)) if float(d[j]) == 0.0]
                ok5 &= len(hits) == 1
                order.append(hits[0] if hits else None)
            join[b] = order
        gate(res, "gate5_oracle_join", ok5 and sorted(join["novel"]) == nids, {"oracle_row_ids": join})

        # T3.
        A_np = A.numpy()
        Xs = feat.numpy()
        t3 = {"rows": {}}
        for b in banks:
            for r, x_id in enumerate(join[b]):
                k = banks[b]["ids"].index(x_id)
                slot = banks[b]["wire"][k].numpy()
                if b == "familiar":
                    m = np.array([i != x_id for i in ids])
                    coef = coef_of(Xs[m], Y[m])
                    slot_rec = Xs[ids.index(x_id)]
                else:
                    coef = coef_of(Xs, Y)
                    slot_rec = None
                f = target([x_id])[0] @ coef
                rr = slot - f
                delta = (fits[b]["terminal"].double()[r] - fits[b]["initial"].double()[r]).numpy()
                t3["rows"][x_id] = {"bank": b, "f": f, "r": rr, "delta": delta,
                                    "slot_equals_training_feature_max_abs": None if slot_rec is None else float(np.abs(slot - slot_rec).max())}
        out = {}
        for x_id, row in t3["rows"].items():
            same = [y for y, o in t3["rows"].items() if o["bank"] == row["bank"] and y != x_id]
            k = cos(row["delta"], -(A_np @ row["r"]))
            kc = max(cos(row["delta"], -(A_np @ t3["rows"][y]["r"])) for y in same)
            out[x_id] = {"bank": row["bank"], "kappa": k, "kappa_cross": kc,
                         "norm_Af": float(np.linalg.norm(A_np @ row["f"])), "norm_Ar": float(np.linalg.norm(A_np @ row["r"])),
                         "norm_delta": float(np.linalg.norm(row["delta"])),
                         "slot_equals_training_feature_max_abs": row["slot_equals_training_feature_max_abs"]}
        nov = [out[i] for i in nids]
        fam = [o for o in out.values() if o["bank"] == "familiar"]
        med_n = float(np.median([o["kappa"] for o in nov]))
        med_f = float(np.median([o["kappa"] for o in fam]))
        n_support = sum(o["kappa"] >= 0.5 and o["kappa"] > o["kappa_cross"] for o in nov)
        n_low = sum(o["kappa"] <= 0.2 for o in nov)
        if n_support >= 6 and med_n - med_f >= 0.3:
            r3 = "Support for H4 (residual pull)"
        elif n_low >= 6:
            r3 = "No support (does not reject H4; T5 decides)"
        else:
            r3 = "Inconclusive"
        decodable = [i for i in nids if pa["novel_predictions"][i] == i]
        res["T3"] = {"per_id": out, "median_kappa_novel": med_n, "median_kappa_familiar": med_f,
                     "n_novel_support_condition": n_support, "n_novel_kappa_le_0p2": n_low, "reading": r3,
                     "slot_decodable_novel_ids": decodable,
                     "kappa_slot_decodable_novel": {i: out[i]["kappa"] for i in decodable}}
    except StopIteration:
        pass
    return res


def selftest():
    rng = np.random.default_rng(1)
    folds = json.loads((R / INPUTS["folds"]).read_text())
    ids = json.loads((R / INPUTS["ids"]).read_text())["ids"]
    Y = target(ids)
    a, b = rng.normal(size=(16, 32)), rng.normal(size=(8, 32))
    X = np.stack([a[ord(i[0]) - 65] + b[int(i[1])] for i in ids])
    assert rho_in(X, Y) < 1e-10 and rho_loo(X, Y) < 1e-10, "additive data must fit exactly"
    assert rho_null(X, Y, n=50) > 0.3, "permuted labels must not fit"
    assert np.linalg.matrix_rank(Y) == folds["full_additive_rank"] == 22
    Xn = X + 0.01 * rng.normal(size=X.shape)
    p = probe(Xn, ids, Xn[:3], ids[:3])
    assert p["loo_correct"] == 56 and p["novel_correct"] == 3, p
    h = torch.from_numpy(rng.normal(size=(5, 2, 16)))
    qv, Kv, Vv = torch.from_numpy(rng.normal(size=8)), torch.from_numpy(rng.normal(size=(8, 16))), torch.from_numpy(rng.normal(size=(4, 16)))
    e, al, s, _ = encode(h, qv, Kv, Vv)
    e2 = encode(h, qv, Kv, Vv, alpha_fixed=al)[0]
    assert torch.allclose(e, e2)
    lin = encode(h, qv, Kv, Vv, alpha_fixed=al.mean(0), scale_fixed=1.0)[0]
    hl = F.layer_norm(h, (16,))
    assert torch.allclose(lin, (al.mean(0)[None, :, None] * (hl @ Vv.T)).sum(1)), "E_d must be linear in LN states"
    v = rng.normal(size=6)
    assert abs(cos(v, v) - 1) < 1e-12 and abs(cos(v, -v) + 1) < 1e-12
    print("SELFTEST PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["selftest", "freeze", "run"])
    mode = ap.parse_args().mode
    if mode == "selftest":
        selftest()
    elif mode == "freeze":
        frz = {"label": "EXPLORATORY", "code": {n: sha(ROOT / n) for n in ["PLAN.md", "t2_t3.py"]},
               "inputs": {k: {"path": v, "sha256": sha(R / v)} for k, v in INPUTS.items()},
               "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
                           "threads": torch.get_num_threads(), "device": "cpu"}}
        with open(ROOT / "FREEZE.json", "x") as f:
            json.dump(frz, f, indent=1)
        print("FROZEN", sha(ROOT / "FREEZE.json"))
    else:
        frz = json.loads((ROOT / "FREEZE.json").read_text())
        assert frz["code"] == {n: sha(ROOT / n) for n in ["PLAN.md", "t2_t3.py"]}, "code changed after freeze"
        assert all(sha(R / v["path"]) == v["sha256"] for v in frz["inputs"].values()), "input changed after freeze"
        res = compute()
        res["freeze_sha256"] = sha(ROOT / "FREEZE.json")
        with open(ROOT / "RESULT.json", "x") as f:
            json.dump(res, f, indent=1)
        print(json.dumps({"gates": {k: v["pass"] for k, v in res["gates"].items()}, "stopped_at": res["stopped_at"],
                          "T2_reading": res.get("T2", {}).get("reading"), "T3_reading": res.get("T3", {}).get("reading")}, indent=1))


if __name__ == "__main__":
    main()
