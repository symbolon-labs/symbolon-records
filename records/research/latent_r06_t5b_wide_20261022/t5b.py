"""EXPLORATORY. R.06 test T5b (factored code at full boundary width) and control C1b.

CPU only. Reads sealed inputs and the T5 results; writes only into this folder. See PLAN.md.
Uses the byte copy of the frozen T5 script (t5.py) as a module.
Modes: smoke (gates, 3 steps, no code scored), freeze, run --arm t5b|c1b.
"""
import argparse, json, os, resource, sys, time
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import t5  # noqa: E402  (byte copy of the frozen T5 script)
import torch  # noqa: E402

R = ROOT.parent
BURNED = {"A0", "A2", "C4", "I2", "K0", "K3", "M0", "M3"}
EXTRA = {
    "t5_freeze": "latent_r06_t5_factored_oracle_20261020/FREEZE.json",
    "t5_result": "latent_r06_t5_factored_oracle_20261020/RESULT_t5.json",
    "c1_result": "latent_r06_t5_factored_oracle_20261020/RESULT_c1.json",
    "t5_fold0": "latent_r06_t5_factored_oracle_20261020/run/t5/fold0.json",
}
CODE = ["PLAN.md", "t5b.py", "t5.py", "seam.py", "complement.py", "test_t5b.py", "preflight.sh", "guard.sh", "launch.sh"]
LOCAL = ["complement.pt", "COMPLEMENT.json"]


def jl(rel):
    return json.loads((R / rel).read_text())


# ---------------------------------------------------------------- pure parts

class Wide(t5.Factored):
    """T5 factored value in the 64-d column space plus a factored offset in the 896-d complement."""

    def __init__(self, c0, a0, b0, s, vh, Q):
        super().__init__(c0, a0, b0, s, vh)
        self.register_buffer("Q", Q.double().clone())
        n = Q.shape[1]
        self.psi_c = torch.nn.Parameter(torch.zeros(1, n, dtype=torch.float64))
        self.psi_a = torch.nn.Parameter(torch.zeros(16, n, dtype=torch.float64))
        self.psi_b = torch.nn.Parameter(torch.zeros(8, n, dtype=torch.float64))

    def values(self, L, D):
        v = super().values(L, D)
        off = (self.psi_c + self.psi_a[L] + self.psi_b[D]) @ self.Q.T
        return v, off


def make_render_wide(state):
    base = t5.make_render(state)

    def render(pair):
        v, off = pair
        return base(v) + off.float()[:, None, :]
    return render


# ---------------------------------------------------------------- gates

def gate_branch():
    a, c = jl(EXTRA["t5_result"]), jl(EXTRA["c1_result"])
    F, H, N = a["totals"]["fit_joint"], a["totals"]["heldout_joint"], c["totals"]["heldout_joint"]
    ok = bool(a["completed"] and c["completed"] and ((F >= 605 and H <= 56) or F < 605))
    return {"F": F, "H": H, "N": N, "pass": ok}


def gate_copies():
    f = jl(EXTRA["t5_freeze"])["code"]
    g = {n: t5.sha(ROOT / n) == f[n] for n in ["t5.py", "seam.py"]}
    g["pass"] = all(g.values())
    return g


def gate_coordinates(state):
    co = t5.tload("coordinates")
    alpha = state["reader.gate"].double() * state["reader.norm_scale"].double()
    A = alpha * state["reader.output.weight"].double()
    rec = co["u"].double() @ torch.diag(co["s"].double()) @ co["vh"].double()
    err = float((rec - A).abs().max())
    return {"max_abs": err, "pass": err <= 1e-9}


def gate_complement(Q, expected_sha):
    u = t5.tload("coordinates")["u"].double()
    e1 = float((Q.T @ Q - torch.eye(Q.shape[1], dtype=torch.float64)).abs().max())
    e2 = float((u.T @ Q).abs().max())
    s = t5.sha(ROOT / "complement.pt")
    return {"qtq_max_abs": e1, "utq_max_abs": e2, "sha256": s, "pass": e1 <= 1e-12 and e2 <= 1e-12 and s == expected_sha}


def build(st, fit_rows, Q):
    names = [st.names[i] for i in fit_rows]
    uniq = sorted(set(names))
    codes = torch.stack([st.d0[st.rows[st.names.index(n)]["feature"]] for n in uniq])
    c0, a0, b0 = t5.additive_init(codes, uniq)
    return Wide(c0, a0, b0, st.s, st.vh, Q), t5.Factored(c0, a0, b0, st.s, st.vh)


def gate_start_identity(st, render_wide, Q):
    fold = st.folds[0]
    wide, fac = build(st, fold["fit_rows"], Q)
    L, D = st.LD(st.names)
    with torch.no_grad():
        same_render = bool(torch.equal(render_wide(wide.values(L, D)), st.render(fac.values(L, D))))
    names = [st.names[i] for i in fold["fit_rows"]]
    FL, FD = st.LD(names)
    hist = t5.fit(st.w.forward, render_wide, wide, st.q[fold["fit_rows"]], st.tokens_of(names), FL, FD, 1, 0.0, "gate6_lr0")
    ref = jl(EXTRA["t5_fold0"])["history"][0]
    same_hist = hist[0]["loss"] == ref["loss"] and hist[0]["teacher_forced_correct"] == ref["teacher_forced_correct"]
    unchanged = all(float(p.detach().abs().max()) == 0.0 for p in wide.parameters())
    return {"render_equal_224": same_render, "loss": hist[0]["loss"], "t5_loss": ref["loss"],
            "teacher_forced": hist[0]["teacher_forced_correct"], "t5_teacher_forced": ref["teacher_forced_correct"],
            "params_unchanged_lr0": unchanged, "pass": same_render and same_hist and unchanged}


def gate_scope(st):
    train = set(st.names)
    ok = len(train) == 56 and not (train & BURNED)
    return {"training_ids": len(train), "burned_overlap": sorted(train & BURNED), "pass": ok}


def all_gates(st, render_wide, Q, complement_sha):
    g = {"g1_branch": gate_branch(), "g2_code_copies": gate_copies()}
    for k in ["g1_branch", "g2_code_copies"]:
        assert g[k]["pass"], g
    g["g3_t5_mechanical"] = st.gates()  # asserts inside (T5 gates 1 to 4, amendments 1 to 3)
    g["g4_coordinates"] = gate_coordinates(st.state)
    assert g["g4_coordinates"]["pass"], g
    g["g5_complement"] = gate_complement(Q, complement_sha)
    assert g["g5_complement"]["pass"], g
    g["g6_start_identity"] = gate_start_identity(st, render_wide, Q)
    assert g["g6_start_identity"]["pass"], g
    g["g7_scope"] = gate_scope(st)
    assert g["g7_scope"]["pass"], g
    return g


# ---------------------------------------------------------------- modes

def load_Q():
    c = torch.load(ROOT / "complement.pt", map_location="cpu", weights_only=True)
    assert c["candidate_sha256"] == t5.D0_SHA
    return c["Q"].double()


def fit_wide(st, render_wide, fit_rows, arm, steps, tag, Q):
    model, _ = build(st, fit_rows, Q)
    names = [st.names[i] for i in fit_rows]
    train = set(st.names)
    targets = [t5.shift_map(sorted(train))[n] for n in names] if arm == "c1b" else names
    assert set(targets) <= train and not (set(targets) & BURNED)
    L, D = st.LD(names)
    hist = t5.fit(st.w.forward, render_wide, model, st.q[fit_rows], st.tokens_of(targets), L, D, steps, st.lr, tag)
    assert all(p.grad is None and not p.requires_grad for p in st.model.parameters())
    return model, hist


def smoke():
    out_path = ROOT / "SMOKE.json"
    assert not out_path.exists()
    (ROOT / "smoke.pid").write_text(str(os.getpid()))
    t0 = time.monotonic()
    st = t5.Setup()
    res = {"label": "EXPLORATORY", "mode": "smoke", "setup_seconds": time.monotonic() - t0}
    try:
        render_wide = make_render_wide(st.state)
        Q = load_Q()
        res["gates"] = all_gates(st, render_wide, Q, json.loads((ROOT / "COMPLEMENT.json").read_text())["sha256"])
        res["seconds_per_scored_row"] = res["gates"]["g3_t5_mechanical"]["parity_seconds_two_scorings"] / (2 * res["gates"]["g3_t5_mechanical"]["parity_rows"])
        _, hist = fit_wide(st, render_wide, st.folds[0]["fit_rows"], "t5b", 3, "smoke_fold0", Q)
        res["fit_steps"] = len(hist)
        res["fit_rows"] = len(st.folds[0]["fit_rows"])
        res["seconds_per_step"] = [h["seconds"] for h in hist]
        res["no_code_scored"] = True
    finally:
        res["end"] = st.close()
        res["elapsed_s"] = time.monotonic() - t0
        res["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        t5.write_new(out_path, res)
    print(json.dumps({k: res[k] for k in ["setup_seconds", "seconds_per_scored_row", "seconds_per_step", "elapsed_s"] if k in res}, indent=1))


def inputs_map():
    m = {k: v for k, v in t5.INPUTS.items()}
    m.update(EXTRA)
    return m


def freeze():
    assert (ROOT / "SMOKE.json").exists()
    frz = {"label": "EXPLORATORY", "code": {n: t5.sha(ROOT / n) for n in CODE},
           "local": {n: t5.sha(ROOT / n) for n in LOCAL},
           "inputs": {k: {"path": v, "sha256": t5.sha(R / v)} for k, v in inputs_map().items()},
           "smoke_sha256": t5.sha(ROOT / "SMOKE.json"), "steps": t5.STEPS, "seed": t5.SEED,
           "runtime": {"python": sys.version.split()[0], "torch": torch.__version__}}
    t5.write_new(ROOT / "FREEZE.json", frz)
    print("FROZEN", t5.sha(ROOT / "FREEZE.json"))


def guard_files():
    frz = json.loads((ROOT / "FREEZE.json").read_text())
    assert frz["code"] == {n: t5.sha(ROOT / n) for n in CODE}, "code changed after freeze"
    assert frz["local"] == {n: t5.sha(ROOT / n) for n in LOCAL}, "complement changed after freeze"
    assert all(t5.sha(R / v["path"]) == v["sha256"] for v in frz["inputs"].values()), "input changed after freeze"
    return t5.sha(ROOT / "FREEZE.json"), frz["local"]["complement.pt"]


def run(arm):
    freeze_sha, comp_sha = guard_files()
    out_dir = ROOT / "run" / arm
    out_dir.mkdir(parents=True)
    (out_dir / "pid").write_text(str(os.getpid()))
    t0 = time.monotonic()
    st = t5.Setup()
    res = {"label": "EXPLORATORY", "arm": arm, "freeze_sha256": freeze_sha, "completed": False, "folds": {}}
    try:
        render_wide = make_render_wide(st.state)
        Q = load_Q()
        res["gates"] = all_gates(st, render_wide, Q, comp_sha)
        t5.event("gates_pass", arm=arm)
        center = st.state["reader.query_center"]
        tok, ids = st.bank.tokens, st.b.IDS
        for k, fold in enumerate(st.folds):
            fr, hr = fold["fit_rows"], fold["validation_rows"]
            model, hist = fit_wide(st, render_wide, fr, arm, t5.STEPS, f"{arm}_fold{k}", Q)
            hn = [st.names[i] for i in hr]
            assert set(hn) <= set(st.names) and not (set(hn) & BURNED)
            HL, HD = st.LD(hn)
            fk = {"fit_rows": len(fr), "heldout_rows": len(hr), "history": hist}
            with torch.no_grad():
                hv = model.values(HL, HD)
                hp = render_wide(hv)
                fk["matched"] = t5.counts(st.score(st.q[hr], hp, "matched"), hn, tok, ids)
                if arm == "t5b":
                    fn = [st.names[i] for i in fr]
                    FL, FD = st.LD(fn)
                    fk["fit"] = t5.counts(st.score(st.q[fr], render_wide(model.values(FL, FD)), "fit"), fn, tok, ids)
                    hs = sorted(set(hn))
                    wmap = {n: hs[(i + 1) % len(hs)] for i, n in enumerate(hs)}
                    wn = [wmap[n] for n in hn]
                    WL, WD = st.LD(wn)
                    ow = st.score(st.q[hr], render_wide(model.values(WL, WD)), "wrong")
                    fk["wrong_intended"] = t5.counts(ow, hn, tok, ids)
                    fk["wrong_inserted"] = t5.counts(ow, wn, tok, ids)
                    delta = (hp[:, 0, :] - center[0]).double()
                    gen = torch.Generator().manual_seed(t5.SEED + k)
                    noise = torch.randn(delta.shape, generator=gen, dtype=torch.float64)
                    noise = noise / noise.norm(dim=-1, keepdim=True) * delta.norm(dim=-1, keepdim=True)
                    rp = center[None] + noise.float()[:, None, :]
                    fk["random_intended"] = t5.counts(st.score(st.q[hr], rp, "random"), hn, tok, ids)
                    zero = (torch.zeros_like(hv[0]), torch.zeros_like(hv[1]))
                    fk["zero_intended"] = t5.counts(st.score(st.q[hr], render_wide(zero), "zero"), hn, tok, ids)
                    fk["receiver_only_intended"] = t5.counts(st.score(st.q[hr], None, "receiver_only"), hn, tok, ids)
                    share = (hv[1].float().norm(dim=-1) / delta.float().norm(dim=-1)).tolist()
                    fk["complement_share_by_id"] = {n: s for n, s in zip(hn, share)}
            torch.save({k2: getattr(model, k2).detach() for k2 in ["theta_c", "theta_a", "theta_b", "psi_c", "psi_a", "psi_b", "c0", "a0", "b0"]},
                       out_dir / f"fold{k}_code.pt")
            t5.write_new(out_dir / f"fold{k}.json", fk)
            res["folds"][k] = {kk: v for kk, v in fk.items() if kk != "history"}
            t5.event("fold_done", arm=arm, fold=k)
        res["totals"] = {"heldout_joint": sum(f["matched"]["joint"] for f in res["folds"].values()),
                         "heldout_sequential": sum(f["matched"]["sequential"] for f in res["folds"].values())}
        if arm == "t5b":
            res["totals"]["fit_joint"] = sum(f["fit"]["joint"] for f in res["folds"].values())
            res["totals"]["fit_rows"] = sum(f["fit"]["rows"] for f in res["folds"].values())
        res["completed"] = True
    except BaseException as exc:
        res.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        res["end"] = st.close()
        res["elapsed_s"] = time.monotonic() - t0
        res["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        t5.write_new(ROOT / f"RESULT_{arm}.json", res)
        t5.event("job_end", arm=arm, completed=res["completed"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke", "freeze", "run"])
    ap.add_argument("--arm", choices=["t5b", "c1b"])
    a = ap.parse_args()
    if a.mode == "smoke":
        smoke()
    elif a.mode == "freeze":
        freeze()
    else:
        assert a.arm
        run(a.arm)
