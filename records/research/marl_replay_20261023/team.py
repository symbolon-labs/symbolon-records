"""Speaker, listener and critic, and the A2C update with gradients through the message
(design section 3).

The speaker reads only the beacon state, so its message at a step is a function of
that state. Thus the rollout runs without gradients, and one batched loss pass over
the 8,192 rows recomputes the messages with gradients (DIAL-style: the listener's
policy loss flows back through the message into the speaker).
"""
import math
import time

import torch
import torch.nn as nn

import beacon as B

HIDDEN = 64
N_ENVS = 512
ROLLOUT = 16
GAMMA, LAMBDA = 0.99, 0.95
ENT_COEF, VF_COEF = 0.01, 0.5
LR, ADAM_EPS = 3e-4, 1e-5
CLIP = 0.5
BLACKOUT_P = {"W1": 0.15, "W0": 0.0}
BUDGETS = {"D1": 489, "D2": 977, "D3": 1954}  # updates; 8,192 environment steps each


def mlp(n_in, n_out, g, head_gain):
    """Two hidden layers of 64 with tanh. Orthogonal init: gain sqrt(2) for the hidden
    layers, head_gain for the output layer, all biases 0."""
    layers = [nn.Linear(n_in, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
              nn.Linear(HIDDEN, n_out)]
    for lin, gain in zip([layers[0], layers[2], layers[4]], [math.sqrt(2), math.sqrt(2), head_gain]):
        nn.init.orthogonal_(lin.weight, gain=gain, generator=g)
        nn.init.zeros_(lin.bias)
    return nn.Sequential(*layers)


class Speaker(nn.Module):
    def __init__(self, g):
        super().__init__()
        self.net = mlp(4, B.MSG_DIM, g, 1.0)

    def forward(self, bstate):
        u = self.net(bstate)
        return B.MSG_NORM * u / u.norm(dim=-1, keepdim=True)


class Team(nn.Module):
    """The weights stream (seed s) initializes speaker, listener and critic in this order."""

    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(int(seed))
        self.speaker = Speaker(g)
        self.listener = mlp(B.LISTENER_IN, B.N_ACTIONS, g, 0.01)
        self.critic = mlp(B.CRITIC_IN, 1, g, 1.0)


def n_params(module):
    return sum(p.numel() for p in module.parameters())


class NaNError(RuntimeError):
    pass


def gae(rewards, values, v_last, done):
    """Generalized advantage estimates. rewards, values, done: (T, n); v_last: (n,).
    done marks the end of an episode at that step (catch or truncation). The caller
    adds gamma * V(final state) to the reward of a truncated step before this call."""
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(v_last)
    for t in reversed(range(T)):
        nextv = v_last if t == T - 1 else values[t + 1]
        nonterm = 1.0 - done[t].float()
        delta = rewards[t] + GAMMA * nextv * nonterm - values[t]
        last = delta + GAMMA * LAMBDA * nonterm * last
        adv[t] = last
    return adv


def loss_terms(team, batch):
    """One batched pass over the rollout rows. Returns (loss, parts, messages)."""
    bs, pos, view = batch["bstate"], batch["pos"], batch["view"]
    m = team.speaker(bs)
    delivered = m * (~batch["blackout"]).float().unsqueeze(-1)
    logits = team.listener(torch.cat([pos, view, delivered], -1))
    logp_all = torch.log_softmax(logits, -1)
    logp = logp_all.gather(-1, batch["action"].unsqueeze(-1)).squeeze(-1)
    entropy = -(logp_all.exp() * logp_all).sum(-1).mean()
    values = team.critic(torch.cat([bs, pos, view], -1)).squeeze(-1)
    T, n = batch["reward"].shape
    adv = gae(batch["reward"], values.detach().view(T, n), batch["v_last"], batch["done"]).flatten()
    returns = adv + values.detach()
    adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
    pg = -(logp * adv_n).mean()
    vloss = ((values - returns) ** 2).mean()
    loss = pg + VF_COEF * vloss - ENT_COEF * entropy
    return loss, dict(pg=pg.item(), vloss=vloss.item(), entropy=entropy.item()), m


class Trainer:
    """A2C for one team. train_until(k) runs updates until k updates are done; calls in
    sequence continue the same run (constant learning rate, so a run to D2 passes
    through the exact D1 state)."""

    def __init__(self, seed, condition):
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        self.seed, self.condition = int(seed), condition
        self.team = Team(seed)
        self.g = torch.Generator().manual_seed(100000 + int(seed))
        self.env = B.TrainEnv(N_ENVS, self.g, BLACKOUT_P[condition])
        self.opt = torch.optim.Adam(self.team.parameters(), lr=LR, eps=ADAM_EPS)
        self.updates = 0
        self.log = []
        self.ep_success, self.ep_len = [], []
        self.seconds = 0.0

    @torch.no_grad()
    def collect(self):
        rows = {k: [] for k in ["bstate", "pos", "view", "blackout", "action", "reward", "done"]}
        team, env = self.team, self.env
        for _ in range(ROLLOUT):
            bs, pos, view = env.obs()
            m = team.speaker(bs)
            if not torch.isfinite(m).all():
                raise NaNError(f"message NaN at update {self.updates}")
            delivered = m * (~env.blackout).float().unsqueeze(-1)
            probs = torch.softmax(team.listener(torch.cat([pos, view, delivered], -1)), -1)
            u = torch.rand(env.n, generator=self.g)
            a = B.inverse_cdf(probs, u)
            for k, v in zip(["bstate", "pos", "view", "blackout", "action"], [bs, pos, view, env.blackout, a]):
                rows[k].append(v)
            ep_t = env.t + 1
            reward, done, caught, truncated, final_in = env.step(a)
            if truncated.any():
                reward = reward.clone()
                reward[truncated] += GAMMA * team.critic(final_in[truncated]).squeeze(-1)
            rows["reward"].append(reward); rows["done"].append(done)
            self.ep_success.extend(caught[done].tolist())
            self.ep_len.extend(ep_t[done].tolist())
        batch = {k: torch.stack(v) for k, v in rows.items()}
        bs, pos, view = env.obs()
        batch["v_last"] = team.critic(torch.cat([bs, pos, view], -1)).squeeze(-1)
        for k in ["bstate", "pos", "view", "blackout", "action"]:
            batch[k] = batch[k].flatten(0, 1)
        return batch

    def update(self):
        batch = self.collect()
        loss, parts, m = loss_terms(self.team, batch)
        if not torch.isfinite(loss) or not torch.isfinite(m).all():
            raise NaNError(f"loss or message NaN at update {self.updates}")
        self.opt.zero_grad()
        loss.backward()
        gn = nn.utils.clip_grad_norm_(self.team.parameters(), CLIP)
        if not torch.isfinite(gn):
            raise NaNError(f"gradient NaN at update {self.updates}")
        self.opt.step()
        self.updates += 1
        return parts, float(gn)

    def train_until(self, n_updates, log_every=50):
        t0 = time.perf_counter()
        while self.updates < n_updates:
            parts, gn = self.update()
            if self.updates % log_every == 0 or self.updates == n_updates:
                k = min(len(self.ep_success), 2000)
                self.log.append(dict(update=self.updates, steps=self.updates * N_ENVS * ROLLOUT,
                                     train_success_last2000=(sum(self.ep_success[-k:]) / k) if k else None,
                                     train_len_last2000=(sum(self.ep_len[-k:]) / k) if k else None,
                                     grad_norm=gn, **parts))
        self.seconds += time.perf_counter() - t0
        return self.log[-1] if self.log else None
