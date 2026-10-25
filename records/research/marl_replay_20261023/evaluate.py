"""Task rollouts and monitor rollouts for one team (design sections 4, 6 and 7).

Outputs in the team folder:
- outcomes.npz: per-episode success, length, return and first catch for each task arm
  on the evaluation set E (2,000 episodes), and the MAC accept masks.
- eval.json: the seeds, the checks V2 and V3 (and V4 for team 1 of W1), the MAC
  metrics, the field table S4, the tag timing and the summary rates.
- W1 only: monitor.npz (fit set and test arms, 1,000 episodes x 30 steps, raw
  delivered messages, positions and views), silent.npz (per-step rows of the task
  rollouts that occurred, for the silent-failure table) and influence.npz.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

import beacon as B
import channel as CH
from team import Team

ARMS = ["C", "RT", "RE", "RN", "Z", "MAC-C", "MAC-RT", "MAC-RE", "RT5", "RE5"]
SILENT_ARMS = ["C", "RT", "RE", "RN", "Z", "RT5", "RE5"]
MON_ARMS = ["C", "RT", "RE", "RN", "RT5", "RE5"]
INFLUENCE_ARMS = ["C", "RT", "RE"]
N_E, N_GATE, N_DONOR, N_MON, N_CAL = 2000, 2000, 2000, 1000, 1000
T_MON = 30
N_FIT = 5000
N_REF = 5
FIELD_EPISODES = 200
AUX = 1_000_000  # offset for the streams that the design table does not name


def seeds(s):
    """Every stream of team s (design section 7, plus the fixed extra streams)."""
    s = int(s)
    return dict(weights=s, train_env=100000 + s,
                gate=400000 + s, gate_actions=400000 + s + AUX,
                E=500000 + s, E_actions=510000 + s, E_random=600000 + s, E_donor_perm=700000 + s,
                fit=800000 + s, fit_actions=800000 + s + AUX, fit_subsample=810000 + s,
                test=820000 + s, test_actions=820000 + s + AUX,
                test_random=600000 + s + AUX, test_donor_perm=700000 + s + AUX,
                cal=830000 + s, cal_actions=830000 + s + AUX,
                influence_refs=840000 + s, donor_pool=900000 + s)


def gen(seed):
    return torch.Generator().manual_seed(int(seed))


def uniforms(seed, n, T):
    return torch.rand(n, T, generator=gen(seed))


@torch.no_grad()
def messages(team, data, T):
    """The speaker's messages for steps 0..T-1 of a set: (n, T, 64) float32."""
    n = data["n"]
    return team.speaker(data["bstate"][:, :T].reshape(n * T, 4)).reshape(n, T, B.MSG_DIM)


def gate_eval(team, s):
    """Gate set (2,000 task episodes), arms C and Z."""
    sd = seeds(s)
    data = B.make_set(sd["gate"], N_GATE, B.T_MAX)
    u = uniforms(sd["gate_actions"], N_GATE, B.T_MAX)
    m = messages(team, data, B.T_MAX)
    c = B.rollout(team.listener, data, m, u)
    z = B.rollout(team.listener, data, torch.zeros_like(m), u)
    nc, nz = int(c["success"].sum()), int(z["success"].sum())
    return dict(n=N_GATE, n_C=nc, n_Z=nz, success_C=nc / N_GATE, success_Z=nz / N_GATE,
                set_seed=sd["gate"], action_seed=sd["gate_actions"])


def gates_pass(g, cond):
    """G-a: success(C) >= 0.80; G-b: success(C) - success(Z) >= 0.20. Integer counts, no rounding."""
    n, nc, nz = g["n"], g["n_C"], g["n_Z"]
    ga = 100 * nc >= 80 * n
    gb = 100 * (nc - nz) >= 20 * n
    return dict(G_a=bool(ga), G_b=bool(gb), passed=bool(ga and (gb or cond == "W0")))


def load_team(path):
    ck = torch.load(path, weights_only=True)
    team = Team(ck["seed"])
    team.load_state_dict(ck["state"])
    team.eval()
    return team, ck


def ragged(rec, delivered, length):
    """Rows of the steps that occurred, episode-major: (sum(length), 69) float32."""
    T = delivered.shape[1]
    mask = torch.arange(T).unsqueeze(0) < length.unsqueeze(1)
    rows = torch.cat([delivered, rec["pos"], rec["view"]], -1)[mask]
    return rows.numpy().astype(np.float32)


def task_streams(team, s):
    """Evaluation set E, donor pool and the delivered streams of all ten arms."""
    sd = seeds(s)
    T = B.T_MAX
    E = B.make_set(sd["E"], N_E, T)
    D = B.make_set(sd["donor_pool"], N_DONOR, T)
    mE, mD = messages(team, E, T), messages(team, D, T)
    perm = torch.randperm(N_DONOR, generator=gen(sd["E_donor_perm"]))[:N_E]
    key = CH.team_key(s)
    tagsE = CH.sign_stream(key, E["seed"], mE)
    tagsD = CH.sign_stream(key, D["seed"], mD)
    RT, RE = CH.replay_t(mE), CH.replay_e(mD, perm, T)
    st = {"C": mE, "RT": RT, "RE": RE, "RN": CH.random_sphere(gen(sd["E_random"]), (N_E, T)),
          "Z": torch.zeros_like(mE), "RT5": CH.delay(mE), "RE5": CH.splice(mE, RE)}
    acc = {}
    st["MAC-C"], acc["MAC-C"] = CH.through_mac(key, E["seed"], mE, tagsE)
    st["MAC-RT"], acc["MAC-RT"] = CH.through_mac(key, E["seed"], RT, CH.wire_tags_rt(tagsE))
    st["MAC-RE"], acc["MAC-RE"] = CH.through_mac(key, E["seed"], RE, CH.wire_tags_re(tagsD, perm, T))
    return dict(E=E, D=D, mE=mE, mD=mD, perm=perm, key=key, tagsE=tagsE, tagsD=tagsD, streams=st, acc=acc)


def check_v3(ts):
    """V3: every RT and RE message is a byte copy of a sent message; norm 8 within 1e-4."""
    st, mE, mD, perm = ts["streams"], ts["mE"], ts["mD"], ts["perm"]
    rt_copy = bool(torch.equal(st["RT"], mE[:, :1].expand_as(mE)))
    re_copy = bool(torch.equal(st["RE"], mD[perm]))
    rt5 = bool(torch.equal(st["RT5"][:, 5:], mE[:, :-5]) and torch.equal(st["RT5"][:, :5], mE[:, :5]))
    re5 = bool(torch.equal(st["RE5"][:, 5:], mD[perm][:, 5:]) and torch.equal(st["RE5"][:, :5], mE[:, :5]))
    # byte level: compare the 256-byte strings of a sample of messages
    a, b, d = st["RT"].numpy(), mE.numpy(), mD.numpy()
    byte_ok = all(CH.msg_bytes(a[e, t]) == CH.msg_bytes(b[e, 0]) for e in range(0, N_E, 97) for t in range(0, B.T_MAX, 7))
    r = st["RE"].numpy()
    byte_ok &= all(CH.msg_bytes(r[e, t]) == CH.msg_bytes(d[int(perm[e]), t]) for e in range(0, N_E, 97) for t in range(0, B.T_MAX, 7))
    dev = {k: float((st[k].norm(dim=-1) - B.MSG_NORM).abs().max()) for k in ["RT", "RE", "RN", "RT5", "RE5", "C"]}
    norm_ok = all(v <= 1e-4 for v in dev.values())
    return dict(rt_copy=rt_copy, re_copy=re_copy, rt5_copy=rt5, re5_copy=re5, byte_sample_ok=bool(byte_ok),
                max_norm_dev=dev, passed=bool(rt_copy and re_copy and rt5 and re5 and byte_ok and norm_ok))


def field_table(ts):
    """S4: accept fractions of C, RT, RT5, RE and RE5 traffic for the four tag field sets
    (first 200 episodes of E, all 50 steps). Holds by construction; measured here."""
    n, T = FIELD_EPISODES, B.T_MAX
    key, E, D, perm = ts["key"], ts["E"], ts["D"], ts["perm"][:n]
    mE, mD = ts["mE"][:n], ts["mD"]
    out = {}
    for fs in CH.FIELD_SETS:
        tE = CH.sign_stream(key, E["seed"], mE, fs)
        tD = CH.sign_stream(key, D["seed"], mD[perm], fs, episodes=perm)  # donor nonces of d(e)
        RE = mD[perm]
        traffic = {"C": (mE, tE), "RT": (CH.replay_t(mE), CH.wire_tags_rt(tE)),
                   "RT5": (CH.delay(mE), CH.wire_tags_delay(tE)),
                   "RE": (RE, tD), "RE5": (CH.splice(mE, RE), CH.wire_tags_splice(tE, tD))}
        out[fs] = {}
        for name, (msgs, tags) in traffic.items():
            _, acc = CH.through_mac(key, E["seed"], msgs, tags, fs)
            out[fs][name] = dict(accept_all_steps=float(acc.float().mean()),
                                 accept_steps_ge1=float(acc[:, 1:].float().mean()),
                                 accept_steps_ge5=float(acc[:, 5:].float().mean()))
    return out


def tag_timing(ts, n=100_000):
    """Mean time to compute one tag and to check one tag, over 100,000 tags."""
    a = ts["mE"].numpy()
    key, seed = ts["key"], ts["E"]["seed"]
    items = [(CH.session_nonce(seed, i % N_E), i % B.T_MAX, CH.msg_bytes(a[i % N_E, i % B.T_MAX])) for i in range(n)]
    t0 = time.perf_counter()
    tags = [CH.tag(key, nc, t, mb) for nc, t, mb in items]
    t1 = time.perf_counter()
    ok = sum(CH.Receiver(key).accept(nc, t, mb, tg) for (nc, t, mb), tg in zip(items, tags))
    t2 = time.perf_counter()
    return dict(n=n, sign_us=(t1 - t0) / n * 1e6, verify_us=(t2 - t1) / n * 1e6, verify_ok=int(ok),
                tag_bytes=32, message_bytes=256)


def summary(o):
    succ = o["success"].double()
    caught = o["success"]
    return dict(success=float(succ.mean()), ret=float(o["ret"].mean()),
                steps_to_catch=float(o["length"][caught].float().mean()) if caught.any() else None)


def run_team(team_dir, s, cond, timing_only=False):
    """All evaluation for one trained team. timing_only: run everything, write nothing
    except the wall times (J0 timing on an untrained network)."""
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    team_dir = Path(team_dir) if team_dir is not None else None
    tm = {}
    t0 = time.perf_counter()
    if timing_only:
        team = Team(s); team.eval()
    else:
        team, _ = load_team(team_dir / "team.pt")
    sd = seeds(s)
    ts = task_streams(team, s)
    tm["streams_and_mac"] = time.perf_counter() - t0
    uE = uniforms(sd["E_actions"], N_E, B.T_MAX)
    want_rec = cond == "W1"
    outs = {}
    t1 = time.perf_counter()
    for arm in ARMS:
        outs[arm] = B.rollout(team.listener, ts["E"], ts["streams"][arm], uE, record=want_rec and arm in SILENT_ARMS)
    tm["task_rollouts"] = time.perf_counter() - t1
    acc = ts["acc"]
    # V2: MAC-C has 0 rejections and equals C; MAC-RE equals Z.
    def same(a, b):
        return bool(torch.equal(outs[a]["success"], outs[b]["success"]) and torch.equal(outs[a]["length"], outs[b]["length"]))
    v2 = dict(mac_c_rejections=int((~acc["MAC-C"]).sum()), mac_c_equals_c=same("MAC-C", "C"),
              mac_re_equals_z=same("MAC-RE", "Z"),
              mac_rt_accepts_only_step0=bool(acc["MAC-RT"][:, 0].all() and not acc["MAC-RT"][:, 1:].any()),
              mac_re_rejects_all=bool(not acc["MAC-RE"].any()))
    v2["passed"] = bool(v2["mac_c_rejections"] == 0 and v2["mac_c_equals_c"] and v2["mac_re_equals_z"])
    v3 = check_v3(ts)
    mac = {}
    for arm in ["MAC-C", "MAC-RT", "MAC-RE"]:
        occ = torch.arange(B.T_MAX).unsqueeze(0) < outs[arm]["length"].unsqueeze(1)
        rej = ~acc[arm] & occ
        mac[arm] = dict(frac_messages_rejected=float(rej.sum() / occ.sum()),
                        frac_episodes_with_rejection=float(rej.any(1).float().mean()))
    t2 = time.perf_counter()
    fields = field_table(ts)
    tm["field_table"] = time.perf_counter() - t2
    res = dict(seed=int(s), condition=cond, seeds=sd, arms={a: summary(outs[a]) for a in ARMS},
               V2=v2, V3=v3, mac=mac, field_table_S4=fields)
    if cond == "W1" and int(s) == 1 and not timing_only:
        res["tag_timing"] = tag_timing(ts)
        team2, _ = load_team(team_dir / "team.pt")
        again = B.rollout(team2.listener, ts["E"], messages(team2, ts["E"], B.T_MAX), uE)
        res["V4"] = dict(passed=bool(torch.equal(again["success"], outs["C"]["success"])
                                     and torch.equal(again["length"], outs["C"]["length"])
                                     and torch.equal(again["ret"], outs["C"]["ret"])))
    npz = {f"{k}__{a}": outs[a][k].numpy() for a in ARMS for k in ["success", "length", "ret", "first_catch"]}
    npz.update({f"accept__{a}": acc[a].numpy() for a in acc})
    npz["E_donor_perm"] = ts["perm"].numpy()
    if want_rec:
        t3 = time.perf_counter()
        mon, silent, infl = monitor_sets(team, s, ts)
        silent.update({f"E__{a}__rows": ragged(outs[a], ts["streams"][a], outs[a]["length"]) for a in SILENT_ARMS})
        silent.update({f"E__{a}__length": outs[a]["length"].numpy() for a in SILENT_ARMS})
        silent.update({f"E__{a}__success": outs[a]["success"].numpy() for a in SILENT_ARMS})
        silent.update({f"E__{a}__rejected_any": (~acc[a] & (torch.arange(B.T_MAX).unsqueeze(0) < outs[a]["length"].unsqueeze(1))).any(1).numpy()
                       for a in ["MAC-C", "MAC-RT", "MAC-RE"]})
        silent.update({f"E__{a}__success": outs[a]["success"].numpy() for a in ["MAC-C", "MAC-RT", "MAC-RE"]})
        tm["monitor_sets"] = time.perf_counter() - t3
    tm["total"] = time.perf_counter() - t0
    res["seconds"] = tm
    if timing_only:
        return res
    np.savez(team_dir / "outcomes.npz", **npz)
    if want_rec:
        np.savez(team_dir / "monitor.npz", **mon)
        np.savez(team_dir / "silent.npz", **silent)
        np.savez(team_dir / "influence.npz", **infl)
    (team_dir / "eval.json").write_text(json.dumps(res, indent=2))
    return res


def monitor_sets(team, s, ts):
    """Monitor fit set (arm C) and test arms: termination off, exactly 30 steps. Calibration
    set for the silent-failure table (1,000 task episodes, arm C)."""
    sd = seeds(s)
    T = T_MON
    mon, silent, infl = {}, {}, {}
    F = B.make_set(sd["fit"], N_MON, T)
    mF = messages(team, F, T)
    rf = B.rollout(team.listener, F, mF, uniforms(sd["fit_actions"], N_MON, T), terminate=False, record=True)
    mon.update(fit__delivered=mF.numpy(), fit__pos=rf["pos"].numpy(), fit__view=rf["view"].numpy())
    mon["fit_subsample"] = torch.randperm(N_MON * T, generator=gen(sd["fit_subsample"]))[:N_FIT].numpy()
    X = B.make_set(sd["test"], N_MON, T)
    mX = messages(team, X, T)
    permT = torch.randperm(N_DONOR, generator=gen(sd["test_donor_perm"]))[:N_MON]
    REx = CH.replay_e(ts["mD"], permT, T)
    st = {"C": mX, "RT": CH.replay_t(mX), "RE": REx, "RN": CH.random_sphere(gen(sd["test_random"]), (N_MON, T)),
          "RT5": CH.delay(mX), "RE5": CH.splice(mX, REx)}
    uX = uniforms(sd["test_actions"], N_MON, T)
    refs = CH.random_sphere(gen(sd["influence_refs"]), (N_MON, T, N_REF))
    for arm in MON_ARMS:
        r = B.rollout(team.listener, X, st[arm], uX, terminate=False, record=True,
                      influence_refs=refs if arm in INFLUENCE_ARMS else None)
        mon.update({f"test_{arm}__delivered": st[arm].numpy(), f"test_{arm}__pos": r["pos"].numpy(),
                    f"test_{arm}__view": r["view"].numpy(), f"test_{arm}__caught_any": r["success"].numpy()})
        if arm in INFLUENCE_ARMS:
            infl[f"{arm}__step_kl"] = r["influence"].numpy()
            infl[f"{arm}__episode"] = r["influence"].mean(1).numpy()
    mon["test_donor_perm"] = permT.numpy()
    Cal = B.make_set(sd["cal"], N_CAL, B.T_MAX)
    mC = messages(team, Cal, B.T_MAX)
    rc = B.rollout(team.listener, Cal, mC, uniforms(sd["cal_actions"], N_CAL, B.T_MAX), record=True)
    silent.update(cal__rows=ragged(rc, mC, rc["length"]), cal__length=rc["length"].numpy(),
                  cal__success=rc["success"].numpy())
    return mon, silent, infl
