"""Environments and behaviour-policy families used in the experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import core

HERE = Path(__file__).resolve().parent


@dataclass
class Env:
    name: str
    P: np.ndarray
    Rsas: np.ndarray
    gamma: float
    s0: int
    horizon: int
    policies: dict = field(default_factory=dict)  # name -> (S, A) policy

    @property
    def S(self):
        return self.P.shape[0]

    @property
    def A(self):
        return self.P.shape[1]

    @property
    def R(self):
        return core.sa_reward(self.P, self.Rsas)

    @property
    def d0(self):
        return np.eye(self.S)[self.s0]

    def rho(self, pi):
        return core.rho(self.P, self.R, pi, self.gamma, self.d0)


def perturb(pi_star, alpha):
    """Upstream 'perturb_policy': optimal action keeps 1-alpha, rest spread uniformly."""
    A = pi_star.shape[1]
    return (1 - alpha) * pi_star + alpha * (1 - pi_star) / (A - 1)


def gridworld():
    G = np.load(HERE / "gridworld.npz")
    env = Env("gridworld", G["P"], G["Rsas"], float(G["gamma"]), int(G["s0"]), horizon=50)
    pi_star, _ = core.optimal_policy(env.P, env.R, env.gamma)
    env.policies = {
        "poor": perturb(pi_star, 0.7),
        "medium": perturb(pi_star, 0.5),
        "good": perturb(pi_star, 0.1),
    }
    env.pi_star = pi_star
    return env


def random_mdp(rng, S=50, A=4, connectivity=4, gamma=0.95, horizon=50, K=3, tau=0.02):
    """Random MDP in the style of Laroche et al. (2019), with K diverse behaviour policies.

    Each (s, a) leads to `connectivity` random successors with random probabilities. The goal is the
    hardest-to-reach state from s0 = 0 (lowest optimal value); entering it yields reward 1, after which
    the agent stays there with zero reward.
    Behaviour policy i mixes a softmax over Q* with its own random deterministic policy:
        pi_i = lam_i * softmax(Q*/tau) + (1 - lam_i) * sigma_i,
    so the policies differ both in quality and in which parts of the state space they cover.
    """
    P = np.zeros((S, A, S))
    for s in range(S):
        for a in range(A):
            nxt = rng.choice(S, size=connectivity, replace=False)
            P[s, a, nxt] = rng.random(connectivity)
            P[s, a] /= P[s, a].sum()
    best = None
    for goal in range(1, S):
        Pg = P.copy(); Pg[goal] = 0; Pg[goal, :, goal] = 1
        Rg = np.zeros((S, A, S)); Rg[:, :, goal] = 1; Rg[goal] = 0
        _, Q = core.optimal_policy(Pg, core.sa_reward(Pg, Rg), gamma)
        v = Q[0].max()
        if v > 1e-3 and (best is None or v < best[0]):
            best = (v, goal, Pg, Rg)
    _, goal, P, Rsas = best
    env = Env("random_mdp", P, Rsas, gamma, 0, horizon)
    pi_star, Q = core.optimal_policy(P, env.R, gamma)
    z = (Q - Q.max(1, keepdims=True)) / tau
    soft = np.exp(z); soft /= soft.sum(1, keepdims=True)
    lams = np.linspace(0.3, 0.9, K)
    for i, lam in enumerate(lams):
        sigma = np.zeros((S, A)); sigma[np.arange(S), rng.integers(A, size=S)] = 1
        env.policies[f"pi{i + 1}"] = lam * soft + (1 - lam) * sigma
    env.pi_star = pi_star
    env.goal = goal
    return env


def detour(p_goal=0.3, r_safe=0.5, r_side=0.4, gamma=0.95, eps_other=0.05):
    """Five-state counterexample in which naive pooling is unsafe.

    s0 --a0--> sA --any--> end, reward r_safe          (the reference's route)
    s0 --a1--> sB --a0--> goal w.p. p_goal (reward 1), else end   (the reference's action in sB)
                  --a1--> end, reward r_side          (the other policy's action in sB)
    The reference always plays a0. The other policy plays a1 in s0 and mostly a1 in sB, so
    (s0, a1) is well covered by foreign data while (sB, a0) is rarely seen and therefore bootstrapped.
    """
    S, A = 5, 2
    s0, sA, sB, goal, end = range(5)
    P = np.zeros((S, A, S)); R = np.zeros((S, A, S))
    P[s0, 0, sA] = 1; P[s0, 1, sB] = 1
    P[sA, :, end] = 1; R[sA, :, end] = r_safe
    P[sB, 0, goal] = p_goal; P[sB, 0, end] = 1 - p_goal; R[sB, 0, goal] = 1
    P[sB, 1, end] = 1; R[sB, 1, end] = r_side
    P[goal, :, goal] = 1; P[end, :, end] = 1
    env = Env("detour", P, R, gamma, s0, horizon=3)
    ref = np.zeros((S, A)); ref[:, 0] = 1
    other = np.zeros((S, A)); other[:, 1] = 1; other[sB] = [eps_other, 1 - eps_other]
    env.policies = {"reference": ref, "other": other}
    env.pi_star, _ = core.optimal_policy(P, env.R, gamma)
    return env
