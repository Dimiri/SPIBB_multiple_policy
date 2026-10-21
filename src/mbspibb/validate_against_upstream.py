"""Check that core.pi_b_spibb reproduces the upstream SPIBB values (results/e1_reproduction.csv) on identical data."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external" / "parametric_spibb"))
import reproduce_spibb as up
from spibb_with_smt.src.pmdp.Policy import policy_to_numpy
from spibb_with_smt.src.spibb.trajectory_gen import generate_dataset
from mbspibb import core

pmdp, valuation = up.make_gridworld()
G = np.load(ROOT / "src/mbspibb/gridworld.npz")
P, Rsas, gamma, s0 = G["P"], G["Rsas"], float(G["gamma"]), int(G["s0"])
S, A, _ = P.shape
R = core.sa_reward(P, Rsas)
d0 = np.eye(S)[s0]
pi_star, _ = core.optimal_policy(P, R, gamma)
pi_b = 0.5 * pi_star + (1 - pi_star) * 0.5 / (A - 1)
print("baseline", core.rho(P, R, pi_b, gamma, d0), "optimal", core.rho(P, R, pi_star, gamma, d0))
ref = {}
for line in (ROOT / "results/e1_reproduction.csv").read_text().splitlines()[1:]:
    s, n, v = line.split(","); ref[(int(s), int(n))] = float(v)
maxdiff = 0
for seed in [1, 2]:
    data = generate_dataset(pmdp=pmdp, actual_valuation=valuation, policy=up.perturb_policy(pmdp=pmdp, optimal_policy=up.policy_iter(transitions=P, sa_rewards=R, discount=gamma).policy, alpha=up.RationalValue.from_frac(50, 100)),
                            trajectory_length=50, num_trajectories=100000, generator=np.random.default_rng(seed=seed))
    for n in up.TRAJ_COUNTS:
        Nsas = core.counts(np.c_[data[: n * 50], np.zeros(n * 50, int)], S, A)
        P_hat, Nsa = core.mle_model(Nsas)
        pi = core.pi_b_spibb(P_hat, core.sa_reward(P_hat, Rsas), pi_b, Nsa < 200, gamma, keep_ref_on_ties=False)
        v = core.rho(P, R, pi, gamma, d0)
        maxdiff = max(maxdiff, abs(v - ref[(seed, n)]))
        print(seed, n, round(v, 4), round(ref[(seed, n)], 4))
print("max abs diff vs upstream:", maxdiff)
