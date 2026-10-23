"""Core tabular routines for SPIBB with multiple behaviour policies.

Everything is plain numpy so that thousands of runs are cheap. Conventions:
  P     : (S, A, S) transition tensor
  Rsas  : (S, A, S) reward r(s, a, s')   (assumed known, as in Simao & Spaan 2019)
  pi    : (S, A) stochastic policy
  data  : int array (n, 4) with columns  (s, a, s', policy_index)
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- evaluation
def sa_reward(P, Rsas):
    return np.einsum("ijk,ijk->ij", P, Rsas)


def policy_value(P, R, pi, gamma):
    """State values V^pi for (S,A,S) transitions P and (S,A) rewards R."""
    S = P.shape[0]
    P_pi = np.einsum("sa,sat->st", pi, P)
    r_pi = np.einsum("sa,sa->s", pi, R)
    return np.linalg.solve(np.eye(S) - gamma * P_pi, r_pi)


def q_values(P, R, pi, gamma):
    V = policy_value(P, R, pi, gamma)
    return R + gamma * np.einsum("sat,t->sa", P, V)


def rho(P, R, pi, gamma, d0):
    """Expected discounted return under initial distribution d0 (vector of length S)."""
    return float(d0 @ policy_value(P, R, pi, gamma))


def optimal_policy(P, R, gamma, iters=10_000, tol=1e-10):
    S, A, _ = P.shape
    V = np.zeros(S)
    for _ in range(iters):
        Q = R + gamma * np.einsum("sat,t->sa", P, V)
        V_new = Q.max(1)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    Q = R + gamma * np.einsum("sat,t->sa", P, V)
    pi = np.zeros((S, A))
    pi[np.arange(S), Q.argmax(1)] = 1.0
    return pi, Q


# --------------------------------------------------------------------------- data
def sample_dataset(P, pi, s0, n_traj, horizon, rng, label=0):
    """Trajectories of fixed length starting in s0 (int) or drawn from d0 (array)."""
    S, A, _ = P.shape
    cum_pi = np.cumsum(pi, 1)
    cum_P = np.cumsum(P, 2)
    out = np.empty((n_traj * horizon, 4), dtype=np.int32)
    if np.ndim(s0) == 0:
        s = np.full(n_traj, int(s0))
    else:
        s = rng.choice(S, size=n_traj, p=s0)
    for t in range(horizon):
        a = (rng.random(n_traj)[:, None] > cum_pi[s]).sum(1)
        a = np.minimum(a, A - 1)
        s2 = (rng.random(n_traj)[:, None] > cum_P[s, a]).sum(1)
        s2 = np.minimum(s2, S - 1)
        idx = np.arange(n_traj) * horizon + t
        out[idx, 0], out[idx, 1], out[idx, 2], out[idx, 3] = s, a, s2, label
        s = s2
    return out


def counts(data, S, A):
    N = np.zeros((S, A, S))
    np.add.at(N, (data[:, 0], data[:, 1], data[:, 2]), 1)
    return N


def mle_model(Nsas):
    """MLE transitions; never-observed pairs get an all-zero row (as in the upstream code)."""
    Nsa = Nsas.sum(2, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        P_hat = np.where(Nsa > 0, Nsas / np.maximum(Nsa, 1), 0.0)
    return P_hat, Nsa[..., 0]


def behaviour_cloning(data, S, A):
    """Count-based BC estimate of the (state-wise) mixture policy; uniform where unseen."""
    Nsa = np.zeros((S, A))
    np.add.at(Nsa, (data[:, 0], data[:, 1]), 1)
    Ns = Nsa.sum(1, keepdims=True)
    return np.where(Ns > 0, Nsa / np.maximum(Ns, 1), 1.0 / A)


# --------------------------------------------------------------------------- algorithms
def pi_b_spibb(P_hat, R_hat, pi_ref, bootstrap, gamma, max_iter=1000, eps=1e-9, keep_ref_on_ties=True):
    """Pi_b-SPIBB (Laroche et al. 2019, Alg. 1) by constrained policy iteration in the MLE MDP.

    bootstrap : boolean (S, A) mask of the bootstrapped set B.
    On B the new policy copies pi_ref; the remaining mass of pi_ref goes to the best non-bootstrapped
    action under the current Q.
    keep_ref_on_ties: if the best non-bootstrapped action is not strictly better (by eps) than the reference's
    own non-bootstrapped mixture, keep pi_ref in that state. This only changes the choice among Q-ties, so the
    output is still Pi_b-optimal, but it avoids arbitrary deviations in states the model knows nothing about
    (e.g. all-zero values of a pessimistic model). keep_ref_on_ties=False reproduces Laroche et al./upstream.
    """
    S, A = pi_ref.shape
    free_mass = np.where(bootstrap, 0.0, pi_ref).sum(1)  # mass that may be re-allocated
    has_free = (~bootstrap).any(1)
    pi = pi_ref.copy()
    cur_best = np.full(S, -1)
    for _ in range(max_iter):
        Q = q_values(P_hat, R_hat, pi, gamma)
        Qm = np.where(bootstrap, -np.inf, Q)
        best = Qm.argmax(1)
        # keep the current argmax unless strictly better (avoids cycling on ties)
        keep = (cur_best >= 0) & (Qm[np.arange(S), best] <= Qm[np.arange(S), np.maximum(cur_best, 0)] + eps)
        best = np.where(keep, cur_best, best)
        move = has_free.copy()
        if keep_ref_on_ties:
            ref_val = np.where(bootstrap, 0.0, pi_ref * Q).sum(1) / np.maximum(free_mass, 1e-300)
            move &= Q[np.arange(S), best] > ref_val + eps
            best = np.where(move, best, -1)
        new = np.where(bootstrap, pi_ref, 0.0)
        rows = np.where(move)[0]
        new[rows, best[rows]] += free_mass[rows]
        new[~move] = pi_ref[~move]
        if np.allclose(new, pi, atol=1e-12):
            break
        pi, cur_best = new, np.where(move, best, -1)
    return pi


def pessimistic_model(P_hat, bootstrap):
    """Zero out the rows of bootstrapped pairs: taking (s, a) in B ends the episode with reward 0.

    With rewards in [0, R_max] this makes the model pessimistic exactly on B, which removes the
    beta_new term of the pooled-data bound (Theorem 2 / Corollary in the paper)."""
    Pp = P_hat.copy()
    Pp[bootstrap] = 0.0
    return Pp


def bootstrapped_mass(P, pi, pi_ref, bootstrap, gamma, d0):
    """beta_pi(B) = (1-gamma) * sum_s d^pi(s) * sum_{a:(s,a) in B} pi_ref(a|s)  (normalised occupancy, true P)."""
    S = P.shape[0]
    P_pi = np.einsum("sa,sat->st", pi, P)
    occ = np.linalg.solve((np.eye(S) - gamma * P_pi).T, d0)  # unnormalised discounted occupancy
    m_b = np.where(bootstrap, pi_ref, 0.0).sum(1)
    return float((1 - gamma) * occ @ m_b)


def beta_hat_mc(data, pi_ref, bootstrap, gamma, horizon):
    """Monte-Carlo estimate of beta_B(pi_ref) from the reference's own trajectories (truncated at horizon)."""
    if len(data) == 0:
        return np.nan
    m_b = np.where(bootstrap, pi_ref, 0.0).sum(1)
    x = m_b[data[:, 0]].reshape(-1, horizon)
    return float((1 - gamma) * (x @ (gamma ** np.arange(horizon))).mean())


def beta_hat_model(P_hat, pi_ref, bootstrap, gamma, d0):
    """Model-based estimate of beta_B(pi_ref) in the (pooled) MLE model; used only when D_ref is empty."""
    return bootstrapped_mass(P_hat, pi_ref, pi_ref, bootstrap, gamma, d0)


def basic_rl(P_hat, R_hat, gamma):
    return optimal_policy(P_hat, R_hat, gamma)[0]


# --------------------------------------------------------------------------- reference selection
def mc_returns(data, Rsas, gamma, horizon):
    """Discounted (truncated) return of every trajectory in a single-policy dataset."""
    r = Rsas[data[:, 0], data[:, 1], data[:, 2]].reshape(-1, horizon)
    return r @ (gamma ** np.arange(horizon))


def select_reference(datasets, Rsas, gamma, horizon, delta, g_max, rule="lcb"):
    """Pick the behaviour policy with the best (lower-confidence) Monte-Carlo return on its own data.

    Hoeffding with a union bound over the K policies: half-width g_max*sqrt(log(2K/delta)/(2 n_i)).
    rule = 'mean' picks the empirical argmax, 'lcb' the lower-confidence-bound argmax.
    """
    K = len(datasets)
    scores, means = [], []
    for D in datasets:
        if len(D) == 0:
            scores.append(-np.inf); means.append(np.nan); continue
        G = mc_returns(D, Rsas, gamma, horizon)
        m = G.mean()
        half = g_max * np.sqrt(np.log(2 * K / delta) / (2 * len(G)))
        means.append(m)
        scores.append(m - half if rule == "lcb" else m)
    return int(np.argmax(scores)), np.array(means)


# --------------------------------------------------------------------------- theory helpers
def n_wedge_theory(zeta, delta, gamma, v_max, S, A):
    """Count threshold of Laroche et al. (2019), Eq. (6)/(Thm 2) form."""
    return 32 * v_max ** 2 / (zeta ** 2 * (1 - gamma) ** 2) * np.log(2 * S * A * 2 ** S / delta)
