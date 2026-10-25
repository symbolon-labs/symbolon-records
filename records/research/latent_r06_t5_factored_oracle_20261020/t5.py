"""EXPLORATORY. R.06 test T5 (factored oracle at the frozen interface) and control C1.

CPU only. Reads sealed inputs; writes only into this folder. See PLAN.md.
Modes: smoke (fold 0, gates, 3 steps, no code scored), freeze, run --arm t5|c1.
"""
import argparse, hashlib, importlib.util, json, os, resource, sys, time
from pathlib import Path
sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
R = ROOT.parent
sys.path.insert(0, str(ROOT))
from seam import Boundary  # noqa: E402

INPUTS = {
    "rows": "latent_binding_training_fold_recovery_20260913/inputs.json",
    "folds": "latent_binding_training_fold_recovery_20260913/folds.json",
    "features": "latent_binding_training_fold_recovery_20260913/features.pt",
    "candidate": "latent_binding_shared_id_query_mechanics_20260908/run/candidate.pt",
    "coordinates": "latent_binding_constrained_code_extended_20260911/coordinates.pt",
    "parity": "latent_binding_training_fold_recovery_20260913/run/fold0/step0/scores.pt",
    "memories": "latent_binding_training_fold_recovery_20260913/memories.pt",
    "familiar_prefill": "latent_binding_fresh_known_native_20260910/run/prefill_correct.pt",
    "fold_core": "latent_binding_shared_id_native_corrected_20260908/core.py",
    "chain": "latent_binding_literal_id_swap_20260910/runtime.py",
    "binding": "latent_binding_v1_20260906/binding.py",
    "channel": "latent_channel_v1_20260906/channel.py",
    "oracle_seam": "latent_binding_constrained_code_extended_20260911/seam.py",
}
CODE = ["PLAN.md", "t5.py", "seam.py", "test_t5.py", "guard.sh", "preflight.sh", "launch.sh"]
RECEIVER_HASH = "6a3e1782ebad58d1e9323d5eeaff191d112848b5f0058394a998d1ac54c492b3"
D0_SHA = "d0f16cb9705ba53daf6cc8f2bfad03a08efa96cbb4f77dd836c9533e948cba87"
STEPS = 200
SEED = 60113
BATCH = 8


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def jload(key):
    return json.loads((R / INPUTS[key]).read_text())


def tload(key):
    return torch.load(R / INPUTS[key], map_location="cpu", weights_only=True)


def event(name, **kw):
    print(json.dumps(dict(event=name, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kw)), flush=True)


def write_new(path, obj):
    with open(path, "x") as f:
        json.dump(obj, f, indent=1)


def parts(name):
    return ord(name[0]) - 65, int(name[1])


# ---------------------------------------------------------------- pure parts

def additive_init(codes, names):
    """Minimum-norm least squares of [1 | letter one-hot | digit one-hot] (25 columns)."""
    X = torch.zeros(len(names), 25, dtype=torch.float64)
    for i, n in enumerate(names):
        l, d = parts(n)
        X[i, 0] = 1
        X[i, 1 + l] = 1
        X[i, 17 + d] = 1
    coef = torch.linalg.pinv(X) @ codes.double()
    return coef[0], coef[1:17], coef[17:25]


class Factored(torch.nn.Module):
    def __init__(self, c0, a0, b0, s, vh):
        super().__init__()
        for k, v in dict(c0=c0, a0=a0, b0=b0, s=s, vh=vh).items():
            self.register_buffer(k, v.double().clone())
        self.theta_c = torch.nn.Parameter(torch.zeros(1, 64, dtype=torch.float64))
        self.theta_a = torch.nn.Parameter(torch.zeros(16, 64, dtype=torch.float64))
        self.theta_b = torch.nn.Parameter(torch.zeros(8, 64, dtype=torch.float64))

    def values(self, L, D):
        base = self.c0[None] + self.a0[L] + self.b0[D]
        off = ((self.theta_c + self.theta_a[L] + self.theta_b[D]) / self.s) @ self.vh
        return base + off


def make_render(state):
    W = state["reader.output.weight"].double()
    gate, scale, center = state["reader.gate"], state["reader.norm_scale"], state["reader.query_center"]

    def render(value):
        mapped = F.linear(value.double(), W).float()
        return center[None] + ((gate * mapped) * scale)[:, None, :]
    return render


def fit(forward, render, model, q, tgt, L, D, steps, lr, tag):
    """Full-pass Adam; batch loss = summed token CE / (2 x rows)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(q)
    history = []
    for step in range(steps):
        t0 = time.monotonic()
        opt.zero_grad(set_to_none=True)
        total, correct = 0.0, torch.zeros(2, dtype=torch.long)
        for start in range(0, n, BATCH):
            qq, tt = q[start:start + BATCH], tgt[start:start + BATCH]
            seq = torch.cat([qq, tt[:, :1]], 1)
            with torch.enable_grad():
                logits = forward(seq, render(model.values(L[start:start + BATCH], D[start:start + BATCH])), 2)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tt.reshape(-1), reduction="sum") / (2 * n)
                loss.backward()
            total += float(loss.detach())
            correct += (logits.detach().argmax(-1) == tt).sum(0)
        grads = [p.grad for p in model.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        gnorm = float(torch.sqrt(sum(g.square().sum() for g in grads)))
        assert gnorm > 0
        opt.step()
        history.append(dict(step_before_update=step, loss=total, gradient_norm=gnorm,
                            teacher_forced_correct=correct.tolist(), seconds=time.monotonic() - t0))
        if (step + 1) % 10 == 0 or step < 3:
            event("fit_step", tag=tag, step=step + 1, loss=total, seconds=history[-1]["seconds"])
    return history


def counts(out, targets, tokens, ids_list):
    """Joint and constrained-sequential counts (oracle rule)."""
    lp = out["token_lp"]
    groups = [torch.where(tokens[:, 0] == p)[0] for p in dict.fromkeys(tokens[:, 0].tolist())]
    reps = torch.tensor([int(g[0]) for g in groups])
    seq = []
    for i in range(len(lp)):
        g = groups[int(lp[i, reps, 0].argmax())]
        seq.append(int(g[lp[i, g, 1].argmax()]))
    target = torch.tensor([ids_list.index(t) for t in targets])
    return {"joint": int((out["prediction"] == target).sum()), "sequential": int((torch.tensor(seq) == target).sum()),
            "rows": len(targets)}


def shift_map(names):
    s = sorted(names)
    return {n: s[(i + 1) % len(s)] for i, n in enumerate(s)}


# ---------------------------------------------------------------- model parts

def chain():
    spec = importlib.util.spec_from_file_location("sealed_oracle_parent", R / INPUTS["chain"])
    r = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r)
    assert Path(r.b.__file__).resolve() == (R / INPUTS["binding"]).resolve()
    assert Path(r.b.c.__file__).resolve() == (R / INPUTS["channel"]).resolve()
    assert Path(r.c.p.__file__).resolve() == (R / INPUTS["fold_core"]).resolve()
    return r


class Setup:
    def __init__(self):
        assert sha(R / INPUTS["candidate"]) == D0_SHA
        assert open(ROOT / "seam.py", "rb").read() == open(R / INPUTS["oracle_seam"], "rb").read()
        self.r = chain()
        self.b = self.r.b
        self.b.c.seed_all(SEED)
        x = jload("rows")
        self.rows, self.q = x["rows"], torch.tensor(x["query_ids"])
        self.names = [r["id"] for r in self.rows]
        self.folds = jload("folds")["folds"]
        self.state = tload("candidate")
        feat = tload("features")["projected"].double()
        self.d0 = F.linear(feat, self.state["reader.value.weight"].double())
        self.d0_rows = self.d0[[r["feature"] for r in self.rows]]
        co = tload("coordinates")
        self.s, self.vh, self.lr = co["s"].double(), co["vh"].double(), float(co["lr"])
        self.render = make_render(self.state)
        self.model, tok = self.b.c.load("receiver")
        self.bank = self.b.AnswerBank(tok)
        assert self.bank.tokens.tolist() == x["candidate_tokens"]
        assert all(self.b.IDS[r["label"]] == r["id"] for r in self.rows)
        self.before = self.b.c.parameter_hash(self.model)
        assert self.before == RECEIVER_HASH
        assert all(not p.requires_grad and p.grad is None for p in self.model.parameters())
        assert all(not m._forward_hooks for m in self.model.modules())
        self.six = torch.cat([self.q[:8], self.bank.tokens[:8, :1]], 1)
        with torch.no_grad():
            self.stock = self.model(input_ids=self.six, use_cache=False, logits_to_keep=1).logits.clone()
        self.w = None

    def tokens_of(self, names):
        return self.bank.tokens[[self.b.IDS.index(n) for n in names]]

    def LD(self, names):
        p = [parts(n) for n in names]
        return torch.tensor([a for a, _ in p]), torch.tensor([d for _, d in p])

    def score(self, q, payload, tag):
        a = self.b.score(self.w.forward, q, self.bank, payload)
        z = self.b.score(self.w.forward, q, self.bank, payload)
        for k in ["token_lp", "lp", "prediction"]:
            assert torch.equal(a[k], z[k]), (tag, k)
        return a

    def native_parity(self, idx):
        """Amended gate 3: native d0 bridge path against the oracle seam, same machine."""
        mem, prior = tload("memories"), tload("familiar_prefill")["first"]["capture"]
        feat = tload("features")["projected"]
        m = feat[[self.rows[i]["feature"] for i in idx]]
        labels = [self.rows[i]["label"] for i in idx]
        seq = torch.cat([self.q[idx], self.bank.tokens[labels][:, :1]], 1)
        wires = mem["wire"][idx]
        bridge = self.r.c.bridge()
        bridge.requires_grad_(False)
        w = self.b.Receiver(self.model, bridge)
        chk = dict(template_hard_scores_exact_descriptive=True, routes_exact=True, memory_slot_equals_feature=True,
                   outside_sites_exact=True, payload_exact=True)
        hard_equal, score_diff = True, 0.0
        native = []
        try:
            with torch.no_grad():
                for start in range(0, len(idx), BATCH):
                    ix = idx[start:start + BATCH]
                    z = w.forward(seq[start:start + BATCH], wires[start:start + BATCH], 2).clone()
                    cap = {k: v.clone() for k, v in bridge.reader.capture.items()}
                    template = [mem["template_indices"][i] for i in ix]
                    chk["template_hard_scores_exact_descriptive"] &= bool(torch.equal(cap["hard"][:, :5], prior["hard"][template]) and torch.equal(cap["scores"][:, :5], prior["scores"][template]))
                    hard_equal &= bool(torch.equal(cap["hard"][:, :5], prior["hard"][template]))
                    score_diff = max(score_diff, float((cap["scores"][:, :5].double() - prior["scores"][template].double()).abs().max()))
                    er = torch.tensor([mem["routes"][i] for i in ix])
                    chk["routes_exact"] &= bool(torch.equal(cap["hard"][:, 3:5].argmax(-1), er[:, None].expand(-1, 2)))
                    chk["memory_slot_equals_feature"] &= bool(torch.equal(cap["memory"][torch.arange(len(ix)), 2 * er], m[start:start + BATCH]))
                    chk["outside_sites_exact"] &= bool(torch.equal(cap["output"][:, :3], cap["hidden"][:, :3]) and torch.equal(cap["output"][:, 5:], cap["hidden"][:, 5:]))
                    chk["payload_exact"] &= bool(torch.equal(self.render(self.d0_rows[ix]), cap["output"][:, 3:5]))
                    native.append(z)
        finally:
            w.hook.remove()
        seam = Boundary(self.model)
        rendered = []
        try:
            with torch.no_grad():
                for start in range(0, len(idx), BATCH):
                    rendered.append(seam.forward(seq[start:start + BATCH], self.render(self.d0_rows[idx[start:start + BATCH]]), 2).clone())
        finally:
            seam.close()
        native, rendered = torch.cat(native), torch.cat(rendered)
        chk["max_abs_logit"] = float((native - rendered).abs().max())
        chk["same_argmax"] = bool(torch.equal(native.argmax(-1), rendered.argmax(-1)))
        chk["rows"] = len(idx)
        chk["template_hard_equal_descriptive"] = hard_equal
        chk["template_scores_max_abs_descriptive"] = score_diff
        chk["pass"] = all(v for k, v in chk.items() if isinstance(v, bool) and not k.endswith("_descriptive")) and chk["max_abs_logit"] <= 1e-4
        return chk

    def gates(self):
        g = {}
        ref = tload("parity")
        idx = [int(i) for i in ref["indices"]]
        assert idx == self.folds[0]["validation_rows"]
        g["gate3_native_parity_same_machine"] = self.native_parity(idx)
        assert g["gate3_native_parity_same_machine"]["pass"], g
        assert all(not m._forward_hooks for m in self.model.modules())
        self.w = Boundary(self.model)
        with torch.no_grad():
            self.w.observe = True
            disabled = self.w.forward(self.six)
            own = self.w.capture["after"][:, 3:5].clone()
            selfout = self.w.forward(self.six, own)
            self.w.observe = False
        g["stock_disabled_self_exact"] = bool(torch.equal(self.stock, disabled) and torch.equal(self.stock, selfout))
        assert g["stock_disabled_self_exact"], g
        names = [self.names[i] for i in idx]
        payload = self.render(self.d0_rows[idx])
        t0 = time.monotonic()
        out = self.score(self.q[idx], payload, "d0_cross_platform")
        g["parity_seconds_two_scorings"] = time.monotonic() - t0
        g["parity_rows"] = len(idx)
        # Descriptive only (amendment of 2026-10-20): the reference comes from Linux x86_64.
        g["cross_platform_max_abs_lp_descriptive"] = float((out["lp"] - ref["first"]["lp"]).abs().max())
        g["cross_platform_same_predictions_descriptive"] = bool(torch.equal(out["prediction"], ref["first"]["prediction"]))
        one = self.b.score(self.w.forward, self.q[idx[:1]], self.bank, payload[:1])
        direct = self.b.direct_score(self.w.forward, self.q[idx[:1]], self.bank, payload[:1])
        g["direct_max_abs_descriptive"] = float((one["lp"] - direct).abs().max())
        g["direct_same_argmax"] = bool(torch.equal(one["prediction"], direct.argmax(-1)))
        g["cross_batch_max_abs_descriptive"] = float((out["lp"][:1] - direct).abs().max())
        assert g["direct_same_argmax"], g  # amendment 3: the difference is descriptive
        g["d0_parity_names_checked"] = sorted(set(names))
        return g

    def close(self):
        if self.w is not None:
            self.w.close()
        assert all(not m._forward_hooks for m in self.model.modules())
        with torch.no_grad():
            restored = self.model(input_ids=self.six, use_cache=False, logits_to_keep=1).logits.clone()
        after = self.b.c.parameter_hash(self.model)
        ok = bool(torch.equal(self.stock, restored)) and after == self.before
        ok = ok and all(not p.requires_grad and p.grad is None for p in self.model.parameters())
        return {"restored_exact": ok, "hash_after": after}

    def fold_fit(self, fit_rows, arm, steps, tag):
        names = [self.names[i] for i in fit_rows]
        uniq = sorted(set(names))
        c0, a0, b0 = additive_init(torch.stack([self.d0[self.rows[self.names.index(n)]["feature"]] for n in uniq]), uniq)
        model = Factored(c0, a0, b0, self.s, self.vh)
        targets = [shift_map(sorted(set(self.names)))[n] for n in names] if arm == "c1" else names
        L, D = self.LD(names)
        hist = fit(self.w.forward, self.render, model, self.q[fit_rows], self.tokens_of(targets), L, D, steps, self.lr, tag)
        assert all(p.grad is None and not p.requires_grad for p in self.model.parameters())
        return model, hist


# ---------------------------------------------------------------- modes

def guard_files():
    frz = json.loads((ROOT / "FREEZE.json").read_text())
    assert frz["code"] == {n: sha(ROOT / n) for n in CODE}, "code changed after freeze"
    assert all(sha(R / v["path"]) == v["sha256"] for v in frz["inputs"].values()), "input changed after freeze"
    return sha(ROOT / "FREEZE.json")


def smoke():
    out_path = ROOT / "SMOKE.json"
    assert not out_path.exists()
    (ROOT / "smoke.pid").write_text(str(os.getpid()))
    t0 = time.monotonic()
    st = Setup()
    res = {"label": "EXPLORATORY", "mode": "smoke", "setup_seconds": time.monotonic() - t0}
    try:
        res["gates"] = st.gates()
        res["seconds_per_scored_row"] = res["gates"]["parity_seconds_two_scorings"] / (2 * res["gates"]["parity_rows"])
        fold = st.folds[0]
        _, hist = st.fold_fit(fold["fit_rows"], "t5", 3, "smoke_fold0")
        res["fit_steps"] = len(hist)
        res["fit_rows"] = len(fold["fit_rows"])
        res["seconds_per_step"] = [h["seconds"] for h in hist]
        res["no_code_scored"] = True
    finally:
        res["end"] = st.close()
        res["elapsed_s"] = time.monotonic() - t0
        res["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write_new(out_path, res)
    print(json.dumps({k: res[k] for k in ["setup_seconds", "seconds_per_scored_row", "seconds_per_step", "elapsed_s"] if k in res}, indent=1))


def freeze():
    assert (ROOT / "SMOKE.json").exists()
    frz = {"label": "EXPLORATORY", "code": {n: sha(ROOT / n) for n in CODE},
           "inputs": {k: {"path": v, "sha256": sha(R / v)} for k, v in INPUTS.items()},
           "smoke_sha256": sha(ROOT / "SMOKE.json"), "steps": STEPS, "seed": SEED,
           "runtime": {"python": sys.version.split()[0], "torch": torch.__version__}}
    write_new(ROOT / "FREEZE.json", frz)
    print("FROZEN", sha(ROOT / "FREEZE.json"))


def run(arm):
    freeze_sha = guard_files()
    out_dir = ROOT / "run" / arm
    out_dir.mkdir(parents=True)
    (out_dir / "pid").write_text(str(os.getpid()))
    t0 = time.monotonic()
    st = Setup()
    res = {"label": "EXPLORATORY", "arm": arm, "freeze_sha256": freeze_sha, "completed": False, "folds": {}}
    try:
        res["gates"] = st.gates()
        event("gates_pass", arm=arm)
        for k, fold in enumerate(st.folds):
            fr, hr = fold["fit_rows"], fold["validation_rows"]
            model, hist = st.fold_fit(fr, arm, STEPS, f"{arm}_fold{k}")
            hn = [st.names[i] for i in hr]
            HL, HD = st.LD(hn)
            fk = {"fit_rows": len(fr), "heldout_rows": len(hr), "history": hist}
            with torch.no_grad():
                hv = model.values(HL, HD)
                fk["matched"] = counts(st.score(st.q[hr], st.render(hv), "matched"), hn, st.bank.tokens, st.b.IDS)
                if arm == "t5":
                    fn = [st.names[i] for i in fr]
                    FL, FD = st.LD(fn)
                    fk["fit"] = counts(st.score(st.q[fr], st.render(model.values(FL, FD)), "fit"), fn, st.bank.tokens, st.b.IDS)
                    hs = sorted(set(hn))
                    wmap = {n: hs[(i + 1) % len(hs)] for i, n in enumerate(hs)}
                    wn = [wmap[n] for n in hn]
                    WL, WD = st.LD(wn)
                    ow = st.score(st.q[hr], st.render(model.values(WL, WD)), "wrong")
                    fk["wrong_intended"] = counts(ow, hn, st.bank.tokens, st.b.IDS)
                    fk["wrong_inserted"] = counts(ow, wn, st.bank.tokens, st.b.IDS)
                    gen = torch.Generator().manual_seed(SEED + k)
                    noise = torch.randn(hv.shape, generator=gen, dtype=torch.float64)
                    noise = noise / noise.norm(dim=-1, keepdim=True) * hv.norm(dim=-1, keepdim=True)
                    fk["random_intended"] = counts(st.score(st.q[hr], st.render(noise), "random"), hn, st.bank.tokens, st.b.IDS)
                    fk["zero_intended"] = counts(st.score(st.q[hr], st.render(torch.zeros_like(hv)), "zero"), hn, st.bank.tokens, st.b.IDS)
                    fk["receiver_only_intended"] = counts(st.score(st.q[hr], None, "receiver_only"), hn, st.bank.tokens, st.b.IDS)
            torch.save({"theta_c": model.theta_c.detach(), "theta_a": model.theta_a.detach(), "theta_b": model.theta_b.detach(),
                        "c0": model.c0, "a0": model.a0, "b0": model.b0}, out_dir / f"fold{k}_code.pt")
            write_new(out_dir / f"fold{k}.json", fk)
            res["folds"][k] = {kk: v for kk, v in fk.items() if kk != "history"}
            event("fold_done", arm=arm, fold=k)
        # The descriptive novel-bank fit is not run (PLAN.md section 5, amendment of 2026-10-20).
        key = "matched"
        res["totals"] = {"heldout_joint": sum(f[key]["joint"] for f in res["folds"].values()),
                         "heldout_sequential": sum(f[key]["sequential"] for f in res["folds"].values())}
        if arm == "t5":
            res["totals"]["fit_joint"] = sum(f["fit"]["joint"] for f in res["folds"].values())
            res["totals"]["fit_rows"] = sum(f["fit"]["rows"] for f in res["folds"].values())
        res["completed"] = True
    except BaseException as exc:
        res.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        res["end"] = st.close()
        res["elapsed_s"] = time.monotonic() - t0
        res["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write_new(ROOT / f"RESULT_{arm}.json", res)
        event("job_end", arm=arm, completed=res["completed"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke", "freeze", "run"])
    ap.add_argument("--arm", choices=["t5", "c1"])
    a = ap.parse_args()
    if a.mode == "smoke":
        smoke()
    elif a.mode == "freeze":
        freeze()
    else:
        assert a.arm
        run(a.arm)
