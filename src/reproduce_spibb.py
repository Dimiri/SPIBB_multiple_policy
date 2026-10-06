"""Reproduce standard SPIBB on Gridworld (E0/E1) by calling the upstream code directly.

Usage:  python src/reproduce_spibb.py --seeds 1 2 3
Upstream code is expected at external/parametric_spibb (see README).
"""
import argparse
import sys
from math import ceil, floor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "parametric_spibb"))

from spibb_with_smt.src.ExactPolicyIteration import policy_iter
from spibb_with_smt.src.benchmark_code.benchmark_gridworld import GridworldParamMDPFactory
from spibb_with_smt.src.pmdp.PolicyEval import get_state_values_stoch_policy
from spibb_with_smt.src.pmdp.pmdp_float_instantiation import (
    instantiate_transitions_using_params, sas_to_sa_reward_valuation, sas_to_sa_reward_transition)
from spibb_with_smt.src.smt_utils.expression.PolynomialExpr import RationalValue, Variable
from spibb_with_smt.src.smt_utils.expression.Valuation import Valuation
from spibb_with_smt.src.spibb.ConstrainedPolicyIteration import constrained_policy_iter, EnumPolicyUpdateMethod
from spibb_with_smt.src.spibb.behavioural_policy_creation import perturb_policy
from spibb_with_smt.src.spibb.trajectory_gen import generate_dataset
from spibb_with_smt.src.spibb.transition_counter import ParamFreeTransitionCounts
from spibb_with_smt.src.pmdp.pmdp_float_instantiation import EnumUnknownDistributionValue

TRAJ_COUNTS = [1, 2, 5, 8, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]


def make_gridworld():
    """Same 5x5 gridworld as upstream step_0_prepare_benchmarks.py."""
    x_max = y_max = 5
    walls = [[[x_max / 2., 0], [x_max / 2., ceil(y_max / 2.)]],
             [[x_max / 2. + 1., floor(y_max / 2.)], [x_max / 2. + 1., ceil(y_max / 2.) + 1.]]]
    pmdp = GridworldParamMDPFactory(x_max=x_max, y_max=y_max, walls=walls, x_end=x_max - 1,
                                    y_end=y_max - 1, env_type=0).get_pmdp(discount=RationalValue.from_frac(95, 100))
    valuation = Valuation(values={Variable("p_success"): RationalValue.from_frac(75, 100)})
    return pmdp, valuation


def value(pmdp, transitions, sa_rewards, policy):
    v = get_state_values_stoch_policy(sa_rewards=sa_rewards, discount=pmdp.get_discount().as_float(),
                                      transitions=transitions, policy=policy)
    return float(v[pmdp.get_initial_state()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--n_wedge", type=int, default=200)
    ap.add_argument("--traj_len", type=int, default=50)
    ap.add_argument("--out", default=str(ROOT / "results" / "e1_reproduction.csv"))
    args = ap.parse_args()

    pmdp, valuation = make_gridworld()
    P = instantiate_transitions_using_params(pmdp=pmdp, valuation=valuation, unknown_sa_pair_fix=None, det_counts=None)
    R = sas_to_sa_reward_valuation(pmdp=pmdp, valuation=valuation, det_counts=None)
    optimal = policy_iter(transitions=P, sa_rewards=R, discount=pmdp.get_discount().as_float()).policy
    baseline = perturb_policy(pmdp=pmdp, optimal_policy=optimal, alpha=RationalValue.from_frac(50, 100))
    print(f"baseline {value(pmdp, P, R, baseline):.2f}   optimal {value(pmdp, P, R, optimal):.2f}")

    rows = ["seed,trajCount,improvedValue"]
    for seed in args.seeds:
        data = generate_dataset(pmdp=pmdp, actual_valuation=valuation, policy=baseline, trajectory_length=args.traj_len,
                                num_trajectories=max(TRAJ_COUNTS), generator=np.random.default_rng(seed=seed))
        counts, used = ParamFreeTransitionCounts(pmdp=pmdp), 0
        for n in TRAJ_COUNTS:
            counts.update(data=data[used:n * args.traj_len]); used = n * args.traj_len
            P_hat = counts.get_probs(unknown_sa_pair_fix=EnumUnknownDistributionValue.ZERO)
            R_hat = sas_to_sa_reward_transition(pmdp=pmdp, transitions=P_hat)
            new = constrained_policy_iter(transitions=P_hat, sa_rewards=R_hat, discount=pmdp.get_discount().as_float(),
                                          cons_policy=baseline,  # <- the baseline SPIBB copies on uncertain pairs
                                          uncertainty_set=counts.get_uncertainty(min_count=args.n_wedge),
                                          update_method=EnumPolicyUpdateMethod.LAROCHE).policy
            v = value(pmdp, P, R, new)
            rows.append(f"{seed},{n},{v}")
            print(f"seed {seed}  trajectories {n:>6}  value {v:.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(rows) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
