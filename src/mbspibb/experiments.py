"""Experiments for 'Safe Policy Improvement with Multiple Behavior Policies'.

  python -m mbspibb.experiments e0 --runs 2000     # counterexample: naive pooling can be unsafe
  python -m mbspibb.experiments e1 --runs 1000     # dataset size, every reference, gridworld
  python -m mbspibb.experiments e2 --runs 500      # share of the reference's own data, gridworld
  python -m mbspibb.experiments e3 --runs 300      # random MDPs (new MDP every run)
  python -m mbspibb.experiments e4 --runs 300      # guard level tau, random MDPs

Results are written to results/<exp>.csv (one row per run x reference x method x N_wedge).
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import envs
from .runner import METHODS, allocate, run_once

ROOT = Path(__file__).resolve().parents[2]
NWEDGES = [5, 10, 20, 50, 100]
N_TOTAL = [9, 30, 60, 150, 300, 600, 1500, 3000]


def _rows(env, names, n_alloc, ref, nwedges, rng, meta, methods=METHODS, diagnostics=True):
    rows, extra = run_once(env, names, n_alloc, ref, nwedges, rng, methods=methods, diagnostics=diagnostics)
    vals = {k: env.rho(p) for k, p in env.policies.items()}
    for r in rows:
        r.update(meta)
        r.update(ref=ref, ref_value=vals[ref], best_value=max(vals.values()), opt_value=env.rho(env.pi_star),
                 bc_value=extra["bc_value"], selected=extra["selected"],
                 selected_value=vals.get(extra["selected"], np.nan))
    return rows


def job_e1(args):
    run, seed = args
    env = envs.gridworld()
    names = list(env.policies)
    rows = []
    rng = np.random.default_rng(seed)
    for n in N_TOTAL:
        alloc = allocate(n, [1 / 3] * 3)
        for ref in names:
            rows += _rows(env, names, alloc, ref, NWEDGES, rng, dict(run=run, n_total=n, share=1 / 3))
    return rows


def job_e2(args):
    run, seed = args
    env = envs.gridworld()
    names = list(env.policies)
    rng = np.random.default_rng(seed)
    rows = []
    ref = "medium"
    for n in [90, 300]:
        for partner in ["good", "poor", "both"]:
            for p in [0.0, 0.1, 0.2, 1 / 3, 0.5, 0.75, 1.0]:
                rest = 1 - p
                shares = {"medium": p, "good": 0.0, "poor": 0.0}
                if partner == "both":
                    shares["good"] = shares["poor"] = rest / 2
                else:
                    shares[partner] = rest
                alloc = allocate(n, [shares[k] for k in names])
                rows += _rows(env, names, alloc, ref, [10, 20, 50], rng,
                              dict(run=run, n_total=n, share=p, partner=partner),
                              methods=["spibb_own", "mb_spibb", "mb_spibb_p", "mb_spibb_pg", "basic_rl"])
    return rows


def job_e3(args):
    run, seed = args
    rng = np.random.default_rng(seed)
    env = envs.random_mdp(rng)
    names = list(env.policies)
    rows = []
    for n in N_TOTAL:
        alloc = allocate(n, [1 / 3] * 3)
        for ref in names:
            rows += _rows(env, names, alloc, ref, NWEDGES, rng, dict(run=run, n_total=n, share=1 / 3))
    return rows


def job_e0(args):
    """Counterexample (detour MDP): failure probability of naive pooling vs. foreign-data volume."""
    run, seed = args
    env = envs.detour()
    names = list(env.policies)
    rng = np.random.default_rng(seed)
    rows = []
    for n_other in [10, 20, 40, 80, 160, 320, 640]:
        rows += _rows(env, names, [20, n_other], "reference", [3, 5, 10, 20, 50], rng,
                      dict(run=run, n_total=20 + n_other, n_other=n_other),
                      methods=["spibb_own", "mb_spibb", "mb_spibb_p", "mb_spibb_pg", "basic_rl"])
    return rows


def job_e4(args):
    """Sensitivity of MB-SPIBB to the guard level tau (random MDPs, every reference, N_wedge = 20)."""
    run, seed = args
    rng = np.random.default_rng(seed)
    env = envs.random_mdp(rng)
    names = list(env.policies)
    rows = []
    for n in [60, 150, 300, 600, 1500]:
        alloc = allocate(n, [1 / 3] * 3)
        for ref in names:
            # same data for every tau: re-seed the data stream per (n, ref)
            sub = int(rng.integers(2**31))
            for tau in [0.01, 0.02, 0.05, 0.1, 0.2, 1.0]:
                r, _ = run_once(env, names, alloc, ref, [20], np.random.default_rng(sub),
                                methods=["mb_spibb_pg"], tau=tau)
                vals = {k: env.rho(p) for k, p in env.policies.items()}
                for x in r:
                    x.update(run=run, n_total=n, ref=ref, tau=tau, ref_value=vals[ref],
                             opt_value=env.rho(env.pi_star))
                rows += r
    return rows


JOBS = {"e4": job_e4, "e0": job_e0, "e1": job_e1, "e2": job_e2, "e3": job_e3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=list(JOBS))
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    seeds = np.random.SeedSequence(args.seed).generate_state(args.runs)
    t = time.time()
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(JOBS[args.exp], list(enumerate(seeds)), chunksize=2)):
            rows += r
            if (i + 1) % max(1, args.runs // 10) == 0:
                print(f"{args.exp}: {i + 1}/{args.runs} runs  ({time.time() - t:.0f}s)", flush=True)
    out = ROOT / "results" / f"{args.exp}.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out, len(rows), "rows")


if __name__ == "__main__":
    main()
