"""One experimental run: sample per-policy datasets, run every method, return true performances."""
from __future__ import annotations

import numpy as np

from . import core

METHODS = [
    "spibb_own",        # standard Pi_b-SPIBB with pi_j on D_j only (= augmented MDP with per-slice counts)
    "spibb_own_full",   # standard Pi_b-SPIBB with pi_j on n trajectories all collected by pi_j (same budget)
    "mb_spibb",         # Pi_b-SPIBB with pi_j on the pooled dataset D (MLE model)
    "mb_spibb_p",       # ablation: pooled counts, pessimistic on the bootstrapped set, no guard
    "mb_spibb_pg",      # ours (MB-SPIBB): pessimistic; if estimated beta_ref > tau fall back to SPIBB on D_ref
    "basic_rl",         # greedy policy of the pooled MLE MDP (no safety mechanism)
    "spibb_bc",         # Pi_b-SPIBB on D with the behaviour-cloned (state-wise mixture) policy as reference
    "mb_spibb_pg_select",  # ours: MB-SPIBB w.r.t. the reference selected by MC returns (LCB)
]


def allocate(n_total, shares):
    """Split n_total trajectories according to shares (largest remainder)."""
    raw = np.asarray(shares, float) * n_total
    n = np.floor(raw).astype(int)
    rest = n_total - n.sum()
    order = np.argsort(-(raw - n))
    n[order[:rest]] += 1
    return n


def run_once(env, names, n_per_policy, ref, nwedges, rng, delta_select=0.1, methods=METHODS,
             own_full_n=None, diagnostics=False, tau=0.05):
    """Return (rows, extra): rows has one dict per method x n_wedge with the true performance of the output."""
    S, A, H, g = env.S, env.A, env.horizon, env.gamma
    pis = [env.policies[k] for k in names]
    j = names.index(ref)
    datasets = [core.sample_dataset(env.P, pi, env.s0, int(n), H, rng, label=i) if n > 0
                else np.zeros((0, 4), np.int32) for i, (pi, n) in enumerate(zip(pis, n_per_policy))]
    pooled = np.concatenate(datasets)
    P_pool, N_pool = core.mle_model(core.counts(pooled, S, A))
    R_pool = core.sa_reward(P_pool, env.Rsas)
    P_own, N_own = core.mle_model(core.counts(datasets[j], S, A))
    R_own = core.sa_reward(P_own, env.Rsas)

    out = {}   # (method, nwedge) -> (policy, reference policy, bootstrap mask)
    if "basic_rl" in methods:
        out[("basic_rl", None)] = (core.basic_rl(P_pool, R_pool, g), pis[j], None)
    g_max = env.Rsas.max() / (1 - g)
    sel = None
    if any(m.endswith("select") for m in methods):
        sel, _ = core.select_reference(datasets, env.Rsas, g, H, delta_select, g_max, rule="lcb")
    pi_bc = core.behaviour_cloning(pooled, S, A) if "spibb_bc" in methods else None
    if "spibb_own_full" in methods:
        nf = int(own_full_n if own_full_n is not None else sum(n_per_policy))
        D_full = core.sample_dataset(env.P, pis[j], env.s0, nf, H, rng, label=j)
        P_full, N_full = core.mle_model(core.counts(D_full, S, A))
        R_full = core.sa_reward(P_full, env.Rsas)

    def pess(B, ref_pi):
        Pp = core.pessimistic_model(P_pool, B)
        return core.pi_b_spibb(Pp, core.sa_reward(Pp, env.Rsas), ref_pi, B, g)

    own_models = {}

    def own_spibb(k, nw):
        """Fallback: standard SPIBB on the reference's own data (certified by the original theorem)."""
        if k not in own_models:
            Pk, Nk = core.mle_model(core.counts(datasets[k], S, A))
            own_models[k] = (Pk, core.sa_reward(Pk, env.Rsas), Nk)
        Pk, Rk, Nk = own_models[k]
        return core.pi_b_spibb(Pk, Rk, pis[k], Nk < nw, g)

    def guarded(B, k, nw):
        b = core.beta_hat_mc(datasets[k], pis[k], B, g, H)
        if np.isnan(b):
            b = core.beta_hat_model(P_pool, pis[k], B, g, env.d0)
        return (pess(B, pis[k]) if b <= tau else own_spibb(k, nw)), b

    for nw in nwedges:
        B_pool, B_own = N_pool < nw, N_own < nw
        if "spibb_own" in methods:
            out[("spibb_own", nw)] = (core.pi_b_spibb(P_own, R_own, pis[j], B_own, g), pis[j], B_own)
        if "spibb_own_full" in methods:
            B_full = N_full < nw
            out[("spibb_own_full", nw)] = (core.pi_b_spibb(P_full, R_full, pis[j], B_full, g), pis[j], B_full)
        if "mb_spibb" in methods:
            out[("mb_spibb", nw)] = (core.pi_b_spibb(P_pool, R_pool, pis[j], B_pool, g), pis[j], B_pool)
        if "mb_spibb_p" in methods:
            out[("mb_spibb_p", nw)] = (pess(B_pool, pis[j]), pis[j], B_pool)
        if "mb_spibb_pg" in methods:
            pi_g, b_hat = guarded(B_pool, j, nw)
            out[("mb_spibb_pg", nw)] = (pi_g, pis[j], B_pool)
        if "mb_spibb_pg_select" in methods:
            out[("mb_spibb_pg_select", nw)] = (guarded(B_pool, sel, nw)[0], pis[sel], B_pool)
        if "spibb_bc" in methods:
            out[("spibb_bc", nw)] = (core.pi_b_spibb(P_pool, R_pool, pi_bc, B_pool, g), pi_bc, B_pool)
        if "mb_spibb_select" in methods:
            out[("mb_spibb_select", nw)] = (core.pi_b_spibb(P_pool, R_pool, pis[sel], B_pool, g), pis[sel], B_pool)
        if "mb_spibb_p_select" in methods:
            out[("mb_spibb_p_select", nw)] = (pess(B_pool, pis[sel]), pis[sel], B_pool)

    rows = []
    for (m, nw), (pi, ref_pi, B) in out.items():
        row = dict(method=m, nwedge=nw, value=env.rho(pi))
        if diagnostics and B is not None:
            row["beta_new"] = core.bootstrapped_mass(env.P, pi, ref_pi, B, g, env.d0)
            row["beta_ref"] = core.bootstrapped_mass(env.P, ref_pi, ref_pi, B, g, env.d0)
            row["frac_B"] = float(B.mean())
        rows.append(row)
    extra = dict(bc_value=env.rho(pi_bc) if pi_bc is not None else np.nan,
                 selected=names[sel] if sel is not None else "")
    return rows, extra
