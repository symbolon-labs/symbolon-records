"""Synthetic unit tests (design section 15.3, tests 1 to 7, and extra checks 8 to 20).

They use untrained networks and random arrays only. No test runs an optimizer step
on a team, and no test reads a result file.

Run from ember/:
    .venv/bin/python -B research/marl_replay_20261023/test_marl.py
"""
import importlib.util
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import beacon as B  # noqa: E402
import channel as CH  # noqa: E402
import evaluate as EV  # noqa: E402
import monitors as MON  # noqa: E402
import team as TM  # noqa: E402

LAB = ROOT.parent / "latent_sabotage_harness_20261011" / "detector_suite.py"


def lab():
    spec = importlib.util.spec_from_file_location("lab_ds_test", LAB)
    mod = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(mod)
    return mod


def small_streams(team, s, n=40, T=12):
    """A small version of evaluate.task_streams (same functions, fewer episodes)."""
    sd = EV.seeds(s)
    E = B.make_set(sd["E"], n, T)
    D = B.make_set(sd["donor_pool"], n, T)
    mE, mD = EV.messages(team, E, T), EV.messages(team, D, T)
    perm = torch.randperm(n, generator=EV.gen(sd["E_donor_perm"]))
    key = CH.team_key(s)
    tE, tD = CH.sign_stream(key, E["seed"], mE), CH.sign_stream(key, D["seed"], mD)
    return E, D, mE, mD, perm, key, tE, tD


class T01_MAC(unittest.TestCase):
    def test_accept_and_four_rejections(self):
        key = CH.team_key(1)
        m = CH.random_sphere(torch.Generator().manual_seed(0), (1, 3))[0].numpy()
        n1, n2 = CH.session_nonce(500001, 0), CH.session_nonce(500001, 1)
        tg = CH.tag(key, n1, 2, CH.msg_bytes(m[2]))
        self.assertTrue(CH.Receiver(key).accept(n1, 2, CH.msg_bytes(m[2]), tg))
        self.assertFalse(CH.Receiver(key).accept(n1, 3, CH.msg_bytes(m[2]), tg))  # another step
        self.assertFalse(CH.Receiver(key).accept(n2, 2, CH.msg_bytes(m[2]), tg))  # another session
        b = bytearray(CH.msg_bytes(m[2])); b[17] ^= 0x01  # one changed bit
        self.assertFalse(CH.Receiver(key).accept(n1, 2, bytes(b), tg))
        forged = CH.tag(CH.team_key(999), n1, 2, CH.msg_bytes(m[2]))  # wrong key
        self.assertFalse(CH.Receiver(key).accept(n1, 2, CH.msg_bytes(m[2]), forged))

    def test_one_message_per_nonce_and_step(self):
        key = CH.team_key(1)
        n1 = CH.session_nonce(5, 0)
        mb = CH.msg_bytes(np.ones(64, dtype=np.float32))
        tg = CH.tag(key, n1, 0, mb)
        rx = CH.Receiver(key)
        self.assertTrue(rx.accept(n1, 0, mb, tg))
        self.assertFalse(rx.accept(n1, 0, mb, tg))

    def test_tag_covers_message_bytes(self):
        key = CH.team_key(3)
        nc = CH.session_nonce(1, 1)
        a = CH.msg_bytes(np.zeros(64, dtype=np.float32))
        b = CH.msg_bytes(np.full(64, 1e-30, dtype=np.float32))
        self.assertNotEqual(CH.tag(key, nc, 0, a), CH.tag(key, nc, 0, b))
        self.assertEqual(len(a), 256)
        self.assertEqual(len(CH.tag(key, nc, 0, a)), 32)


class T02_FieldTable(unittest.TestCase):
    def test_pattern(self):
        team = TM.Team(7)
        E, D, mE, mD, perm, key, _, _ = small_streams(team, 7)
        expect = {"bytes": dict(C=1, RT=1, RT5=1, RE=1, RE5=1),
                  "bytes+nonce": dict(C=1, RT=1, RT5=1, RE=0, RE5=0),
                  "bytes+counter": dict(C=1, RT=0, RT5=0, RE=1, RE5=1),
                  "full": dict(C=1, RT=0, RT5=0, RE=0, RE5=0)}
        for fs, exp in expect.items():
            tE = CH.sign_stream(key, E["seed"], mE, fs)
            tD = CH.sign_stream(key, D["seed"], mD[perm], fs, episodes=perm)
            RE = mD[perm]
            traffic = {"C": (mE, tE), "RT": (CH.replay_t(mE), CH.wire_tags_rt(tE)),
                       "RT5": (CH.delay(mE), CH.wire_tags_delay(tE)), "RE": (RE, tD),
                       "RE5": (CH.splice(mE, RE), CH.wire_tags_splice(tE, tD))}
            for name, (msgs, tags) in traffic.items():
                _, acc = CH.through_mac(key, E["seed"], msgs, tags, fs)
                start = 1 if name == "RT" else (5 if name in ("RT5", "RE5") else 0)
                self.assertEqual(float(acc[:, start:].float().mean()), exp[name], f"{fs} {name}")
                if name in ("RT5", "RE5"):
                    self.assertTrue(bool(acc[:, :5].all()))
            if fs == "full":
                _, acc = CH.through_mac(key, E["seed"], CH.replay_t(mE), CH.wire_tags_rt(tE), fs)
                self.assertTrue(bool(acc[:, 0].all()))


class T03_Arms(unittest.TestCase):
    def test_byte_copies_and_norms(self):
        team = TM.Team(3)
        E, D, mE, mD, perm, key, _, _ = small_streams(team, 3)
        RT, RE = CH.replay_t(mE), CH.replay_e(mD, perm, mE.shape[1])
        a, b, r, d = RT.numpy(), mE.numpy(), RE.numpy(), mD.numpy()
        for e in range(a.shape[0]):
            for t in range(a.shape[1]):
                self.assertEqual(CH.msg_bytes(a[e, t]), CH.msg_bytes(b[e, 0]))
                self.assertEqual(CH.msg_bytes(r[e, t]), CH.msg_bytes(d[int(perm[e]), t]))
        RN = CH.random_sphere(torch.Generator().manual_seed(1), (50, 7))
        self.assertLess(float((RN.norm(dim=-1) - 8).abs().max()), 1e-4)
        self.assertLess(float((mE.norm(dim=-1) - 8).abs().max()), 1e-4)
        d5 = CH.delay(mE)
        self.assertTrue(torch.equal(d5[:, :5], mE[:, :5]) and torch.equal(d5[:, 5:], mE[:, :-5]))
        s5 = CH.splice(mE, RE)
        self.assertTrue(torch.equal(s5[:, :5], mE[:, :5]) and torch.equal(s5[:, 5:], RE[:, 5:]))


class T04_MonitorsEqualLab(unittest.TestCase):
    def test_knn_pca_equal_fit_score(self):
        fit_score = lab().fit_score
        rng = np.random.default_rng(4)
        fit = rng.standard_normal((800, 64)) * 2
        test = rng.standard_normal((64, 64)) * 2
        atk = rng.standard_normal((64, 64)) * 2 + 0.3
        for det in ["knn", "pca"]:
            st, sa = fit_score(det, fit[:, None, :], test[:, None, :], atk[:, None, :])
            if det == "knn":
                mt, ma = MON.knn_scores(fit, test, chunk=17), MON.knn_scores(fit, atk, chunk=64)
            else:
                p = MON.PCA(fit)
                mt, ma = p.scores(test, chunk=10), p.scores(atk)
            self.assertLess(float(np.max(np.abs(st - mt))), 1e-9, det)
            self.assertLess(float(np.max(np.abs(sa - ma))), 1e-9, det)

    def test_knn_on_sphere_messages(self):
        fit_score = lab().fit_score
        g = torch.Generator().manual_seed(5)
        fit = CH.random_sphere(g, (500,)).double().numpy()
        x = CH.random_sphere(g, (64,)).double().numpy()
        _, sa = fit_score("knn", fit[:, None, :], x[:1, None, :], x[:, None, :])
        self.assertLess(float(np.max(np.abs(sa - MON.knn_scores(fit, x)))), 1e-9)


class T05_CommonRandomNumbers(unittest.TestCase):
    def test_same_beacon_path_in_all_arms(self):
        team = TM.Team(2)
        E, D, mE, mD, perm, key, tE, tD = small_streams(team, 2)
        T = mE.shape[1]
        u = EV.uniforms(EV.seeds(2)["E_actions"], E["n"], T)
        E2 = B.make_set(EV.seeds(2)["E"], E["n"], T)
        self.assertTrue(torch.equal(E["beacon_xy"], E2["beacon_xy"]) and torch.equal(E["pos0"], E2["pos0"]))
        seen = []
        orig = B.local_view

        def spy(bxy, pos):
            seen[-1].append(bxy.clone())
            return orig(bxy, pos)
        B.local_view = spy
        try:
            for st in [mE, CH.replay_t(mE), torch.zeros_like(mE)]:
                seen.append([])
                B.rollout(team.listener, E, st, u)
        finally:
            B.local_view = orig
        for other in seen[1:]:
            for a, b in zip(seen[0], other):
                self.assertTrue(torch.equal(a, b))


class T06_Environment(unittest.TestCase):
    def test_catch_distance(self):
        data = dict(n=2, T=1, pos0=torch.tensor([[0.5, 0.5], [0.5, 0.5]]),
                    beacon_xy=torch.tensor([[[0.5 + 0.0999, 0.5]] * 2, [[0.5 + 0.1001, 0.5]] * 2]))
        lis = lambda obs: torch.tensor([[50.0] + [0.0] * 8]).expand(obs.shape[0], 9)  # always action 0 (+x)
        out = B.rollout(lis, data, torch.zeros(2, 1, 64), torch.full((2, 1), 0.5))
        self.assertEqual(out["success"].tolist(), [True, False])  # 0.0599 <= 0.06 < 0.0601
        self.assertEqual(out["length"].tolist(), [1, 1])

    def test_clip_to_arena(self):
        p = B.listener_move(torch.tensor([[0.99, 0.01], [0.0, 1.0]]), torch.tensor([1, 3]))
        self.assertTrue(bool((p >= 0).all() and (p <= 1).all()))
        self.assertAlmostEqual(float(p[0, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(p[1, 0]), 0.0, places=6)

    def test_moves_length(self):
        L = B.MOVES.norm(dim=-1)
        self.assertTrue(torch.allclose(L[:8], torch.full((8,), 0.04)))
        self.assertEqual(float(L[8]), 0.0)

    def test_reflection(self):
        xy = torch.tensor([[0.005, 0.5], [0.995, 0.5], [0.5, 0.005], [0.5, 0.995]])
        th = torch.tensor([math.pi, 0.0, -math.pi / 2, math.pi / 2])
        nxy, nth = B.beacon_move(xy, th, torch.zeros(4))
        self.assertTrue(torch.allclose(nxy, torch.tensor([[0.01, 0.5], [0.99, 0.5], [0.5, 0.01], [0.5, 0.99]]), atol=1e-6))
        self.assertTrue(torch.allclose(torch.cos(nth), torch.tensor([1.0, -1.0, 0.0, 0.0]), atol=1e-6))
        self.assertTrue(torch.allclose(torch.sin(nth), torch.tensor([0.0, 0.0, 1.0, -1.0]), atol=1e-6))

    def test_beacon_speed_and_inside(self):
        d = B.make_set(11, 300, 50)
        step = (d["beacon_xy"][:, 1:] - d["beacon_xy"][:, :-1]).norm(dim=-1)
        self.assertLessEqual(float(step.max()), 0.015 + 1e-6)
        self.assertTrue(bool((d["beacon_xy"] >= 0).all() and (d["beacon_xy"] <= 1).all()))
        self.assertTrue(bool(((d["pos0"] - d["beacon_xy"][:, 0]).norm(dim=-1) >= 0.3).all()))
        self.assertTrue(bool((d["beacon_xy"][:, 0] >= 0.1).all() and (d["beacon_xy"][:, 0] <= 0.9).all()))

    def test_local_view(self):
        pos = torch.tensor([[0.5, 0.5]] * 3)
        b = torch.tensor([[0.6, 0.5], [0.5, 0.65], [0.5, 0.6501]])
        v = B.local_view(b, pos)
        self.assertTrue(torch.allclose(v[0], torch.tensor([0.1 / 0.15, 0.0, 1.0]), atol=1e-6))
        self.assertEqual(float(v[1, 2]), 1.0)
        self.assertTrue(torch.equal(v[2], torch.zeros(3)))

    def test_return_is_catch_plus_distance_gain(self):
        """The shaped return telescopes: 1{catch} + d_start - d_end, so it lies in [-1.5, 2.5]."""
        data = B.make_set(3, 50, 50)
        team = TM.Team(1)
        m = EV.messages(team, data, 50)
        out = B.rollout(team.listener, data, m, EV.uniforms(9, 50, 50))
        d0 = (data["pos0"] - data["beacon_xy"][:, 0]).norm(dim=-1).double()
        d_end = out["success"].double() + d0 - out["ret"]
        self.assertTrue(bool((d_end >= -1e-5).all() and (d_end <= 1.4143).all()))
        self.assertTrue(bool((d_end[out["success"]] <= 0.06 + 1e-5).all()))


class T07_Seeds(unittest.TestCase):
    def test_teams_share_nothing(self):
        a, b = EV.seeds(1), EV.seeds(2)
        self.assertFalse(set(a.values()) & set(b.values()))
        all_seeds = [v for s in list(range(1, 21)) + [9001, 9002] for v in EV.seeds(s).values()]
        self.assertEqual(len(all_seeds), len(set(all_seeds)))
        Ea, Eb = B.make_set(a["E"], 2000, 2), B.make_set(b["E"], 2000, 2)
        pa = set(map(tuple, Ea["pos0"].tolist())); pb = set(map(tuple, Eb["pos0"].tolist()))
        self.assertFalse(pa & pb)
        ra = CH.random_sphere(EV.gen(a["E_random"]), (10, 5)); rb = CH.random_sphere(EV.gen(b["E_random"]), (10, 5))
        self.assertFalse(torch.equal(ra, rb))
        da = torch.randperm(2000, generator=EV.gen(a["E_donor_perm"]))
        db = torch.randperm(2000, generator=EV.gen(b["E_donor_perm"]))
        self.assertFalse(torch.equal(da, db))

    def test_design_seed_values(self):
        s = EV.seeds(5)
        for k, v in dict(weights=5, train_env=100005, gate=400005, E=500005, E_actions=510005, E_random=600005,
                         E_donor_perm=700005, fit=800005, fit_subsample=810005, test=820005, cal=830005,
                         influence_refs=840005, donor_pool=900005).items():
            self.assertEqual(s[k], v, k)


class T08_Networks(unittest.TestCase):
    def test_param_counts(self):
        t = TM.Team(1)
        self.assertEqual(TM.n_params(t.speaker), 8640)
        self.assertEqual(TM.n_params(t.listener), 9225)
        self.assertEqual(TM.n_params(t.critic), 4865)

    def test_message_norm(self):
        t = TM.Team(1)
        with torch.no_grad():
            m = t.speaker(torch.rand(100, 4))
        self.assertLess(float((m.norm(dim=-1) - 8).abs().max()), 1e-4)

    def test_init_is_seeded(self):
        a, b, c = TM.Team(4), TM.Team(4), TM.Team(5)
        for (k, x), (_, y), (_, z) in zip(a.state_dict().items(), b.state_dict().items(), c.state_dict().items()):
            self.assertTrue(torch.equal(x, y), k)
        self.assertFalse(all(torch.equal(x, z) for x, z in zip(a.state_dict().values(), c.state_dict().values())))


class T09_GradientThroughMessage(unittest.TestCase):
    def test_speaker_gets_gradient_except_blackout(self):
        torch.manual_seed(0)
        team = TM.Team(1)
        n = 64
        batch = dict(bstate=torch.rand(n, 4), pos=torch.rand(n, 2), view=torch.zeros(n, 3),
                     blackout=torch.zeros(n, dtype=torch.bool), action=torch.randint(0, 9, (n,)),
                     reward=torch.randn(4, 16), done=torch.zeros(4, 16, dtype=torch.bool), v_last=torch.zeros(16))
        loss, _, _ = TM.loss_terms(team, batch)
        loss.backward()
        g = sum(float(p.grad.abs().sum()) for p in team.speaker.parameters())
        self.assertGreater(g, 0.0)
        team.zero_grad()
        batch["blackout"] = torch.ones(n, dtype=torch.bool)
        loss, _, _ = TM.loss_terms(team, batch)
        loss.backward()
        g = sum(float(p.grad.abs().sum()) if p.grad is not None else 0.0 for p in team.speaker.parameters())
        self.assertEqual(g, 0.0)


class T10_GAE(unittest.TestCase):
    def test_hand_computed(self):
        r = torch.tensor([[1.0], [0.0], [2.0]])
        v = torch.tensor([[0.5], [0.2], [0.1]])
        done = torch.tensor([[False], [True], [False]])
        vl = torch.tensor([0.3])
        adv = TM.gae(r, v, vl, done)
        g, l = TM.GAMMA, TM.LAMBDA
        d2 = 2.0 + g * 0.3 - 0.1
        d1 = 0.0 - 0.2
        d0 = 1.0 + g * 0.2 - 0.5
        self.assertAlmostEqual(float(adv[2]), d2, places=6)
        self.assertAlmostEqual(float(adv[1]), d1, places=6)
        self.assertAlmostEqual(float(adv[0]), d0 + g * l * d1, places=6)


class T11_TrainEnv(unittest.TestCase):
    def test_truncation_and_reset(self):
        g = torch.Generator().manual_seed(0)
        env = B.TrainEnv(64, g, 0.15)
        stay = torch.full((64,), 8)
        ever = torch.zeros(64, dtype=torch.bool)
        for k in range(B.T_MAX - 1):
            _, done, caught, trunc, _ = env.step(stay)
            ever |= caught
            self.assertFalse(bool(trunc.any()))
        _, done, caught, trunc, final_in = env.step(stay)
        self.assertTrue(torch.equal(trunc, ~(ever | caught)))
        self.assertGreater(int(trunc.sum()), 0)
        self.assertEqual(final_in.shape, (64, 9))
        self.assertTrue(bool((env.t[trunc] == 0).all()))

    def test_blackout_stream_same_in_W1_and_W0(self):
        g1, g0 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
        e1, e0 = B.TrainEnv(16, g1, 0.15), B.TrainEnv(16, g0, 0.0)
        self.assertTrue(torch.equal(e1.pos, e0.pos))
        self.assertFalse(bool(e0.blackout.any()))
        self.assertTrue(torch.equal(torch.rand(4, generator=g1), torch.rand(4, generator=g0)))


class T12_InverseCDF(unittest.TestCase):
    def test_frequencies(self):
        p = torch.tensor([[0.1, 0.0, 0.2, 0.0, 0.0, 0.3, 0.0, 0.0, 0.4]]).expand(200000, 9)
        u = torch.rand(200000, generator=torch.Generator().manual_seed(0))
        a = B.inverse_cdf(p, u)
        f = torch.bincount(a, minlength=9).float() / 200000
        self.assertTrue(torch.allclose(f, p[0], atol=0.005))
        self.assertEqual(int(B.inverse_cdf(torch.full((1, 9), 1 / 9), torch.tensor([0.9999999]))), 8)


class T13_MacArmsEqualControls(unittest.TestCase):
    def test_mac_c_equals_c_and_mac_re_equals_z(self):
        team = TM.Team(6)
        E, D, mE, mD, perm, key, tE, tD = small_streams(team, 6)
        T = mE.shape[1]
        u = EV.uniforms(1, E["n"], T)
        dc, ac = CH.through_mac(key, E["seed"], mE, tE)
        RE = CH.replay_e(mD, perm, T)
        dr, ar = CH.through_mac(key, E["seed"], RE, CH.wire_tags_re(tD, perm, T))
        self.assertTrue(bool(ac.all()) and not bool(ar.any()))
        oc, om = B.rollout(team.listener, E, mE, u), B.rollout(team.listener, E, dc, u)
        oz, oe = B.rollout(team.listener, E, torch.zeros_like(mE), u), B.rollout(team.listener, E, dr, u)
        for k in ["success", "length", "ret"]:
            self.assertTrue(torch.equal(oc[k], om[k]))
            self.assertTrue(torch.equal(oz[k], oe[k]))


class T14_Context(unittest.TestCase):
    def test_previous_message_and_ragged(self):
        rng = np.random.default_rng(0)
        d = rng.standard_normal((3, 5, 64)); p = rng.random((3, 5, 2)); v = rng.random((3, 5, 3))
        X = MON.context_padded(d, p, v).reshape(3, 5, 133)
        self.assertTrue(np.array_equal(X[:, 0, 64:128], d[:, 0]))
        self.assertTrue(np.array_equal(X[:, 3, 64:128], d[:, 2]))
        length = np.array([2, 5, 1])
        rows = np.concatenate([np.concatenate([d[e, :L], p[e, :L], v[e, :L]], -1) for e, L in enumerate(length)])
        R = MON.context_ragged(rows, length)
        ref = np.concatenate([X[e, :L] for e, L in enumerate(length)])
        self.assertTrue(np.array_equal(R, ref))
        self.assertTrue(np.array_equal(MON.episode_max_ragged(np.arange(8.0), length), np.array([1.0, 6.0, 7.0])))

    def test_block_scaler(self):
        rng = np.random.default_rng(1)
        X = np.concatenate([rng.standard_normal((500, 64)) * 3, rng.standard_normal((500, 64)), rng.random((500, 2)),
                            rng.random((500, 3)) * 0.1], 1)
        sc = MON.BlockScaler(X, MON.block_sizes())
        Z = sc(X)
        for a, w in [(0, 64), (64, 64), (128, 2), (130, 3)]:
            self.assertAlmostEqual(float(Z[:, a:a + w].var(0).sum()), 1.0, places=9)
        self.assertLess(float(np.abs(Z.mean(0)).max()), 1e-12)

    def test_participation_ratio(self):
        rng = np.random.default_rng(2)
        self.assertAlmostEqual(MON.participation_ratio(rng.standard_normal((100000, 4))), 4.0, delta=0.05)
        x = np.zeros((100, 3)); x[:, 0] = rng.standard_normal(100)
        self.assertAlmostEqual(MON.participation_ratio(x), 1.0, places=9)


class T15_Influence(unittest.TestCase):
    def test_kl_nonnegative_and_zero_for_constant_listener(self):
        team = TM.Team(8)
        E = B.make_set(1, 20, 6)
        m = EV.messages(team, E, 6)
        refs = CH.random_sphere(torch.Generator().manual_seed(2), (20, 6, 5))
        out = B.rollout(team.listener, E, m, torch.rand(20, 6), terminate=False, record=True, influence_refs=refs)
        self.assertTrue(bool((out["influence"] >= -1e-6).all()))
        const = lambda obs: torch.zeros(*obs.shape[:-1], 9)
        out = B.rollout(const, E, m, torch.rand(20, 6), terminate=False, influence_refs=refs)
        self.assertLess(float(out["influence"].abs().max()), 1e-6)


class T16_Stats(unittest.TestCase):
    def test_t_values_both_scripts(self):
        import analysis as AN
        import stats_check as SC
        for p, df, ref in [(0.975, 19, 2.093), (1 - 0.0125 / 2, 19, 2.759), (1 - 0.0125 / 2, 17, 2.793)]:
            self.assertAlmostEqual(AN.t_quantile(p, df), ref, delta=5e-4)
            self.assertAlmostEqual(SC.t_q(p, df), ref, delta=5e-4)
        self.assertAlmostEqual(AN.t_quantile(0.975, 1000), 1.962, delta=1e-3)

    def test_auroc_agree(self):
        import analysis as AN
        import stats_check as SC
        rng = np.random.default_rng(3)
        neg, pos = rng.standard_normal(1000), rng.standard_normal(1000) + 0.4
        a1 = AN.auroc(neg, pos)
        a2, ties = SC.pair_auroc(neg, pos)
        self.assertEqual(ties, 0)
        self.assertLess(abs(a1 - a2), 1e-12)
        self.assertLess(abs(a1 - AN.lab_auroc(neg, pos)), 1e-12)
        negt = np.round(neg, 1); post = np.concatenate([np.round(pos[:500], 1), negt[:500]])
        a3, ties = SC.pair_auroc(negt, post)
        self.assertGreater(ties, 0)
        self.assertEqual(ties, AN.cross_ties(negt, post))
        self.assertLess(abs(AN.auroc(negt, post) - a3), 1e-12)
        self.assertLess(abs(AN.tpr_at_fpr(neg, pos) - SC.tpr99(neg, pos)), 1e-12)

    def test_verdicts(self):
        import analysis as AN
        mk = lambda m, lo, hi: dict(mean=m, lo=lo, hi=hi)
        self.assertEqual(AN.verdict_harm(mk(0.12, 0.06, 0.18)), "HARM")
        self.assertEqual(AN.verdict_harm(mk(0.12, 0.04, 0.2)), "UNCLEAR")
        self.assertEqual(AN.verdict_harm(mk(0.01, -0.02, 0.049)), "NO HARM")
        self.assertEqual(AN.verdict_mac(mk(0.06, 0.01, 0.1)), "MAC HELPS")
        self.assertEqual(AN.verdict_mac(mk(-0.06, -0.1, -0.01)), "MAC HURTS")
        self.assertEqual(AN.verdict_mac(mk(0.04, 0.01, 0.07)), "NO CLEAR EFFECT")
        self.assertEqual(AN.verdict_monitor(mk(0.95, 0.85, 0.99), 0.6), "FLAGS")
        self.assertEqual(AN.verdict_monitor(mk(0.95, 0.85, 0.99), 0.4), "RANK ONLY")
        self.assertEqual(AN.verdict_monitor(mk(0.5, 0.45, 0.69), 0.01), "CANNOT FLAG")
        self.assertEqual(AN.verdict_monitor(mk(0.5, 0.45, 0.71), 0.01), "RANK ONLY")
        self.assertEqual(AN.verdict_fallback(mk(0.06, 0.01, 0.1)), "FALLBACK TRAINING HELPS")


class T17_JobPlan(unittest.TestCase):
    def test_plan(self):
        import jobs as J
        self.assertEqual(J.plan_jobs(1.0)["teams_per_job"], 20)
        self.assertEqual(J.plan_jobs(5.0)["teams_per_job"], 10)
        self.assertEqual(J.plan_jobs(15.0)["teams_per_job"], 5)
        self.assertEqual(J.parse_seeds("1-3,7"), [1, 2, 3, 7])


class T18_GateRule(unittest.TestCase):
    def test_gates(self):
        g = lambda c, z: dict(n=2000, n_C=c, n_Z=z)
        self.assertTrue(EV.gates_pass(g(1600, 1200), "W1")["passed"])
        self.assertFalse(EV.gates_pass(g(1600, 1201), "W1")["passed"])
        self.assertTrue(EV.gates_pass(g(1600, 1590), "W0")["passed"])
        self.assertFalse(EV.gates_pass(g(1599, 0), "W1")["passed"])
        import stats_check as SC
        bad = 0
        for c in range(1600, 2001):
            for cond in ("W1", "W0"):
                ga, p = SC.gate_pass(g(c, c - 400), cond)
                bad += int(not p or p != EV.gates_pass(g(c, c - 400), cond)["passed"])
                ga, p = SC.gate_pass(g(c, c - 399), cond)
                bad += int(p != EV.gates_pass(g(c, c - 399), cond)["passed"])
        self.assertEqual(bad, 0)


class T19_RolloutModes(unittest.TestCase):
    def test_monitor_rollout_runs_all_steps(self):
        team = TM.Team(2)
        E = B.make_set(4, 30, 30)
        m = EV.messages(team, E, 30)
        out = B.rollout(team.listener, E, m, torch.rand(30, 30), terminate=False, record=True)
        self.assertTrue(bool((out["length"] == 30).all()))
        self.assertEqual(out["pos"].shape, (30, 30, 2))
        self.assertEqual(out["view"].shape, (30, 30, 3))


class T20_Determinism(unittest.TestCase):
    def test_rollout_repeats(self):
        torch.use_deterministic_algorithms(True)
        team = TM.Team(1)
        E = B.make_set(500001, 100, 50)
        m = EV.messages(team, E, 50)
        u = EV.uniforms(510001, 100, 50)
        a, b = B.rollout(team.listener, E, m, u), B.rollout(team.listener, E, EV.messages(team, E, 50), u)
        for k in ["success", "length", "ret"]:
            self.assertTrue(torch.equal(a[k], b[k]))


class T21_TrainerCollect(unittest.TestCase):
    """The rollout and the loss pass of the trainer, without an optimizer step."""

    def test_collect_and_loss(self):
        saved = TM.N_ENVS
        TM.N_ENVS = 32
        try:
            tr = TM.Trainer(77777, "W1")
            before = {k: v.clone() for k, v in tr.team.state_dict().items()}
            batch = tr.collect()
            self.assertEqual(batch["bstate"].shape, (TM.ROLLOUT * 32, 4))
            self.assertEqual(batch["reward"].shape, (TM.ROLLOUT, 32))
            loss, parts, m = TM.loss_terms(tr.team, batch)
            self.assertTrue(bool(torch.isfinite(loss)))
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(tr.team.parameters(), TM.CLIP)
            self.assertTrue(bool(torch.isfinite(gn)))
            for k, v in tr.team.state_dict().items():
                self.assertTrue(torch.equal(v, before[k]), k)
            self.assertEqual(tr.updates, 0)
        finally:
            TM.N_ENVS = saved


class T22_PipelineSmoke(unittest.TestCase):
    """End to end on small sizes with untrained teams: evaluation, monitor scores,
    analysis.py and stats_check.py. No optimizer step. The gate files are synthetic."""

    def test_pipeline(self):
        import json
        import tempfile
        import analysis as AN
        import stats_check as SC
        names = ["N_E", "N_GATE", "N_DONOR", "N_MON", "N_CAL", "N_FIT", "FIELD_EPISODES"]
        saved = {k: getattr(EV, k) for k in names}
        paths = (AN.ROOT, AN.RUNS, SC.ROOT, SC.RUNS)
        try:
            EV.N_E = EV.N_GATE = EV.N_DONOR = 60
            EV.N_MON = EV.N_CAL = 40
            EV.N_FIT, EV.FIELD_EPISODES = 300, 6
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                runs = root / "runs"
                for cond in ["W1", "W0"]:
                    for s in [1, 2, 3]:
                        d = runs / f"{cond}_s{s:02d}"
                        d.mkdir(parents=True)
                        team = TM.Team(s)
                        torch.save(dict(seed=s, condition=cond, budget="D1", updates=0, state=team.state_dict()), d / "team.pt")
                        (d / "train.json").write_text(json.dumps(dict(status="ok")))
                        g = dict(n=2000, n_C=1800, n_Z=200 + 20 * s, success_C=0.9, success_Z=(200 + 20 * s) / 2000)
                        g.update(EV.gates_pass(g, cond))
                        (d / "gate.json").write_text(json.dumps(g))
                        res = EV.run_team(d, s, cond)
                        self.assertTrue(res["V2"]["passed"])
                        self.assertTrue(res["V3"]["passed"])
                        if cond == "W1":
                            MON.score_team(d)
                            # make the RN AUROC differ between teams (the interval level of P on RN matters)
                            sc = dict(np.load(d / "scores.npz"))
                            for mname in MON.MONITORS:
                                k = f"test__{mname}__RN__episode"
                                sc[k] = sc[f"test__{mname}__C__episode"] + (s - 2) * 0.3 * np.std(sc[k]) + np.linspace(-1, 1, len(sc[k])) * np.std(sc[k])
                            np.savez(d / "scores.npz", **sc)
                AN.ROOT, AN.RUNS, SC.ROOT, SC.RUNS = root, runs, root, runs
                out = AN.main()
                self.assertTrue(out["validity"]["V4"]["passed"])
                self.assertEqual(out["validity"]["V1"]["n_pass"], 3)
                self.assertGreater(out["monitors"]["P-PCA|RN"]["sd"], 0.0)
                ft = out["field_table_S4"]
                self.assertEqual(ft["full"]["RE"]["accept_steps_ge5_max"], 0.0)
                self.assertEqual(ft["bytes"]["RE"]["accept_steps_ge5_min"], 1.0)
                self.assertEqual(SC.main(), 0)
            r = MON.synthetic_timing(n=40, T=30, n_e=60, t_max=50, n_cal=40, n_fit=300)
            self.assertIn("seconds_test", r)
        finally:
            for k, v in saved.items():
                setattr(EV, k, v)
            AN.ROOT, AN.RUNS, SC.ROOT, SC.RUNS = paths


if __name__ == "__main__":
    torch.set_num_threads(1)
    res = unittest.main(exit=False, verbosity=1).result
    sys.exit(0 if res.wasSuccessful() else 1)
