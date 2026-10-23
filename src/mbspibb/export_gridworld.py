"""Export the upstream 5x5 Gridworld (same as reproduce_spibb.py) to plain numpy arrays."""
import sys
from math import ceil, floor
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "external" / "parametric_spibb"))
from spibb_with_smt.src.benchmark_code.benchmark_gridworld import GridworldParamMDPFactory
from spibb_with_smt.src.pmdp.pmdp_float_instantiation import instantiate_transitions_using_params, pmdp_sas_reward_to_numpy
from spibb_with_smt.src.smt_utils.expression.PolynomialExpr import RationalValue, Variable
from spibb_with_smt.src.smt_utils.expression.Valuation import Valuation


def main(out):
    x_max = y_max = 5
    walls = [[[x_max / 2., 0], [x_max / 2., ceil(y_max / 2.)]],
             [[x_max / 2. + 1., floor(y_max / 2.)], [x_max / 2. + 1., ceil(y_max / 2.) + 1.]]]
    pmdp = GridworldParamMDPFactory(x_max=x_max, y_max=y_max, walls=walls, x_end=x_max - 1, y_end=y_max - 1,
                                    env_type=0).get_pmdp(discount=RationalValue.from_frac(95, 100))
    val = Valuation(values={Variable("p_success"): RationalValue.from_frac(75, 100)})
    P = instantiate_transitions_using_params(pmdp=pmdp, valuation=val, unknown_sa_pair_fix=None, det_counts=None)
    Rsas = pmdp_sas_reward_to_numpy(pmdp=pmdp)
    np.savez(out, P=np.asarray(P, float), Rsas=np.asarray(Rsas, float), s0=pmdp.get_initial_state(),
             gamma=pmdp.get_discount().as_float())
    print(P.shape, Rsas.shape, pmdp.get_initial_state())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "src" / "mbspibb" / "gridworld.npz"))
