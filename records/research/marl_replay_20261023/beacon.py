"""Beacon: a vectorized environment for one speaker and one listener (design section 2).

The beacon motion does not depend on the agents. Thus an evaluation set is a
fixed table of starts and beacon paths, and all arms of one team read the same
table (common random numbers, design section 7). Training uses `TrainEnv`,
which draws new episodes from one generator and resets each episode at its end.

All values are float32. All random draws use an explicit torch.Generator.
"""
import math

import torch

BEACON_LO, BEACON_HI = 0.1, 0.9
BEACON_SPEED = 0.015
HEADING_SD = 0.3
MIN_START_DIST = 0.3
STEP_LEN = 0.04
CATCH_DIST = 0.06
VIEW_RADIUS = 0.15
T_MAX = 50
N_ACTIONS = 9
MSG_DIM = 64
MSG_NORM = 8.0
LISTENER_IN = 2 + 3 + MSG_DIM  # own position, local view, message
CRITIC_IN = 4 + 2 + 3  # beacon state, own position, local view

# Actions 0..7: compass directions at k * 45 degrees (0 = +x, 2 = +y). Action 8: stay.
_ang = torch.arange(8, dtype=torch.float64) * (math.pi / 4)
MOVES = torch.cat([torch.stack([torch.cos(_ang), torch.sin(_ang)], 1) * STEP_LEN,
                   torch.zeros(1, 2, dtype=torch.float64)]).float()


def local_view(beacon_xy, pos):
    """((beacon - pos) / 0.15, 1) if the distance is at most 0.15, else (0, 0, 0)."""
    diff = beacon_xy - pos
    inside = (diff.norm(dim=-1) <= VIEW_RADIUS).unsqueeze(-1).float()
    return torch.cat([diff / VIEW_RADIUS, torch.ones_like(diff[..., :1])], -1) * inside


def beacon_state(xy, theta):
    """Speaker observation (x, y, cos theta, sin theta)."""
    return torch.cat([xy, torch.cos(theta).unsqueeze(-1), torch.sin(theta).unsqueeze(-1)], -1)


def beacon_move(xy, theta, eps):
    """One beacon step: heading += 0.3 * eps, move 0.015, then reflect at the walls."""
    th = theta + HEADING_SD * eps
    c, s = torch.cos(th), torch.sin(th)
    x = xy[:, 0] + BEACON_SPEED * c
    y = xy[:, 1] + BEACON_SPEED * s
    lo, hi = x < 0, x > 1
    x = torch.where(lo, -x, torch.where(hi, 2 - x, x))
    c = torch.where(lo | hi, -c, c)
    lo, hi = y < 0, y > 1
    y = torch.where(lo, -y, torch.where(hi, 2 - y, y))
    s = torch.where(lo | hi, -s, s)
    return torch.stack([x, y], 1), torch.atan2(s, c)


def listener_move(pos, action):
    return (pos + MOVES[action]).clamp(0.0, 1.0)


def sample_starts(g, n):
    """Beacon uniform in [0.1, 0.9]^2, heading uniform in [0, 2 pi), listener uniform
    in the arena at a distance of at least 0.3 from the beacon (rejection sample:
    each round draws new candidates for the rows that are still open, in row order)."""
    xy = BEACON_LO + (BEACON_HI - BEACON_LO) * torch.rand(n, 2, generator=g)
    theta = 2 * math.pi * torch.rand(n, generator=g)
    pos = torch.empty(n, 2)
    todo = torch.arange(n)
    while len(todo):
        cand = torch.rand(len(todo), 2, generator=g)
        ok = (cand - xy[todo]).norm(dim=-1) >= MIN_START_DIST
        pos[todo[ok]] = cand[ok]
        todo = todo[~ok]
    return xy, theta, pos


def make_set(seed, n, T):
    """A fixed episode set: starts and beacon paths for T steps (states 0..T).

    Draw order from one generator: starts (sample_starts), then heading noise (n, T).
    """
    g = torch.Generator().manual_seed(int(seed))
    xy, theta, pos0 = sample_starts(g, n)
    eps = torch.randn(n, T, generator=g)
    xys, ths = [xy], [theta]
    for t in range(T):
        xy, theta = beacon_move(xy, theta, eps[:, t])
        xys.append(xy); ths.append(theta)
    bxy = torch.stack(xys, 1)
    bth = torch.stack(ths, 1)
    return dict(seed=int(seed), n=n, T=T, pos0=pos0, beacon_xy=bxy, beacon_theta=bth,
                bstate=beacon_state(bxy, bth))


def inverse_cdf(probs, u):
    """Action index from probabilities (n, 9) and uniforms (n,) by the inverse CDF."""
    cdf = probs.cumsum(-1)
    return (cdf < u.unsqueeze(-1)).sum(-1).clamp(max=N_ACTIONS - 1)


@torch.no_grad()
def rollout(listener, data, delivered, u, terminate=True, record=False, influence_refs=None):
    """Run the listener on a fixed set with a fixed delivered-message stream.

    delivered: (n, T, 64), u: (n, T) action uniforms. With terminate=True an episode
    ends at the first catch (task rollout). With terminate=False every episode runs
    exactly T steps (monitor rollout); the catch then only counts.
    record=True returns the per-step position (before the move) and local view.
    influence_refs (n, T, R, 64): random reference messages for the influence score.
    """
    n, T = delivered.shape[:2]
    bxy = data["beacon_xy"]
    pos = data["pos0"].clone()
    alive = torch.ones(n, dtype=torch.bool)
    success = torch.zeros(n, dtype=torch.bool)
    length = torch.full((n,), T, dtype=torch.long)
    ret = torch.zeros(n, dtype=torch.float64)
    first_catch = torch.full((n,), -1, dtype=torch.long)
    rec_pos, rec_view, kl = [], [], []
    for t in range(T):
        view = local_view(bxy[:, t], pos)
        obs = torch.cat([pos, view, delivered[:, t]], -1)
        logits = listener(obs)
        probs = torch.softmax(logits, -1)
        if record:
            rec_pos.append(pos.clone()); rec_view.append(view)
        if influence_refs is not None:
            R = influence_refs.shape[2]
            obs_r = torch.cat([pos.unsqueeze(1).expand(n, R, 2), view.unsqueeze(1).expand(n, R, 3),
                               influence_refs[:, t]], -1)
            q = torch.softmax(listener(obs_r), -1).mean(1)
            logp = torch.log_softmax(logits, -1)
            kl.append((probs * (logp - torch.log(q))).sum(-1))
        a = inverse_cdf(probs, u[:, t])
        new = listener_move(pos, a)
        d_prev = (pos - bxy[:, t]).norm(dim=-1)
        d_catch = (new - bxy[:, t]).norm(dim=-1)
        caught = d_catch <= CATCH_DIST
        d_next = torch.where(caught, d_catch, (new - bxy[:, t + 1]).norm(dim=-1))
        r = caught.double() + (d_prev - d_next).double()
        ret = ret + torch.where(alive, r, torch.zeros_like(r))
        newly = caught & alive & (first_catch < 0)
        first_catch = torch.where(newly, torch.full_like(first_catch, t), first_catch)
        if terminate:
            end = caught & alive
            success = success | end
            length = torch.where(end, torch.full_like(length, t + 1), length)
            alive = alive & ~caught
            pos = torch.where(alive.unsqueeze(-1), new, pos)
        else:
            success = success | caught
            pos = new
    out = dict(success=success, length=length, ret=ret, first_catch=first_catch)
    if record:
        out["pos"] = torch.stack(rec_pos, 1)
        out["view"] = torch.stack(rec_view, 1)
    if influence_refs is not None:
        out["influence"] = torch.stack(kl, 1)
    return out


class TrainEnv:
    """512 parallel training episodes. Each episode resets at a catch or after T_MAX steps.

    Draw order for each step (from the one training generator): the caller draws the
    action uniforms, then `step` draws the heading noise for all rows, then new
    starts for the rows that ended (row order), then one blackout uniform per new
    episode. The blackout uniform is drawn in both conditions, so W1 and W0 read the
    same stream positions; it applies only if blackout_p > 0.
    """

    def __init__(self, n, g, blackout_p):
        self.n, self.g, self.blackout_p = n, g, blackout_p
        self.xy, self.theta, self.pos = sample_starts(g, n)
        self.t = torch.zeros(n, dtype=torch.long)
        self.blackout = torch.rand(n, generator=g) < blackout_p

    def obs(self):
        return beacon_state(self.xy, self.theta), self.pos, local_view(self.xy, self.pos)

    def step(self, action):
        """Return reward, done, caught, truncated and the critic input of the state after
        the step (before any reset; used to bootstrap truncated rows)."""
        eps = torch.randn(self.n, generator=self.g)
        new = listener_move(self.pos, action)
        d_prev = (self.pos - self.xy).norm(dim=-1)
        d_catch = (new - self.xy).norm(dim=-1)
        caught = d_catch <= CATCH_DIST
        nxy, nth = beacon_move(self.xy, self.theta, eps)
        d_next = torch.where(caught, d_catch, (new - nxy).norm(dim=-1))
        reward = caught.float() + (d_prev - d_next)
        self.t = self.t + 1
        truncated = ~caught & (self.t >= T_MAX)
        done = caught | truncated
        final_critic_in = torch.cat([beacon_state(nxy, nth), new, local_view(nxy, new)], -1)
        self.xy, self.theta, self.pos = nxy, nth, new
        idx = torch.nonzero(done).flatten()
        if len(idx):
            xy, th, pos = sample_starts(self.g, len(idx))
            self.xy = self.xy.clone(); self.theta = self.theta.clone(); self.pos = self.pos.clone()
            self.xy[idx], self.theta[idx], self.pos[idx] = xy, th, pos
            self.t[idx] = 0
            self.blackout = self.blackout.clone()
            self.blackout[idx] = torch.rand(len(idx), generator=self.g) < self.blackout_p
        return reward, done, caught, truncated, final_critic_in
