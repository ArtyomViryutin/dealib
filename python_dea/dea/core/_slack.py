__all__ = ["slack"]

import numpy as np
from numpy.typing import ArrayLike, NDArray

from python_dea.dea._options import RTS, Orientation
from python_dea.dea._utils import post_process_data, pre_process_data
from python_dea.dea._wrappers import Efficiency
from python_dea.linprog import LPP, simplex


def construct_lpp(x: NDArray[float], y: NDArray[float], rts: RTS) -> LPP:
    m, k = x.shape
    n = y.shape[0]

    lpp = LPP()

    lpp.c = np.zeros(k)
    lpp.b_ub = np.zeros(m + n)
    lpp.A_ub = np.vstack((x, -y))

    if rts != RTS.crs:
        rts_constraint = np.ones(k)
        if rts == RTS.vrs:
            lpp.A_eq = np.array([rts_constraint])
            lpp.b_eq = np.ones(1)
        elif rts == RTS.drs:
            lpp.A_ub = np.vstack((lpp.A_ub, rts_constraint))
            lpp.b_ub = np.append(lpp.b_ub, [1])
        elif rts == RTS.irs:
            lpp.A_ub = np.vstack((lpp.A_ub, -rts_constraint))
            lpp.b_ub = np.append(lpp.b_ub, [-1])
    return lpp


def solve_slack(
    x: NDArray[float], y: NDArray[float], eff: Efficiency
) -> Efficiency:
    m, k = x.shape
    n = y.shape[0]

    lpp = construct_lpp(x, y, eff.rts)

    for i in range(k):
        lpp.b_ub[:m] = x[:, i]
        lpp.b_ub[m : m + n] = -y[:, i]
        if eff.orientation == Orientation.input:
            lpp.b_ub[:m] *= eff.eff[i]
        else:
            lpp.b_ub[m : m + n] *= eff.eff[i]

        lpp_result = simplex(lpp, opt_f=False, opt_slacks=True)

        eff.objval[i] = lpp_result.f
        eff.lambdas[i] = lpp_result.x
        eff.slack[i] = lpp_result.slack[: m + n]

    return eff


def slack(
    inputs: ArrayLike,
    outputs: ArrayLike,
    eff: Efficiency,
    transpose: bool = False,
):
    x, y, x_std, y_std = pre_process_data(inputs, outputs, transpose)

    eff = solve_slack(x, y, eff)

    post_process_data(eff, x_std, y_std, model=eff.model)

    return eff
