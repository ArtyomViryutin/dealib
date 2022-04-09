__all__ = ["add"]

import numpy as np
from numpy.typing import ArrayLike, NDArray
from tqdm import trange

from python_dea.dea._utils import post_process_data, pre_process_data
from python_dea.linprog import LPP, simplex

from .._options import RTS, Model, Orientation
from .._wrappers import Efficiency


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


def solve_add(
    x: NDArray[float], y: NDArray[float], rts: RTS, tol: float
) -> Efficiency:
    m, k = x.shape
    n = y.shape[0]

    lpp = construct_lpp(x, y, rts)

    eff = Efficiency(Model.envelopment, Orientation.input, rts, k, m, n)

    eff.eff = np.ones(k)

    for i in trange(k, desc=f"Computing {rts} additive model"):
        lpp.b_ub[:m] = x[:, i]
        lpp.b_ub[m : m + n] = -y[:, i]

        lpp_result = simplex(lpp, opt_f=False, opt_slacks=True, tol=tol)

        eff.objval[i] = lpp_result.f
        eff.lambdas[i] = lpp_result.x
        eff.slack[i] = lpp_result.slack[: m + n]

    return eff


def add(
    inputs: ArrayLike,
    outputs: ArrayLike,
    rts: RTS,
    tol: float = 1e-9,
) -> Efficiency:
    rts = RTS.get(rts)

    x, y, x_std, y_std = pre_process_data(inputs, outputs, tol)

    eff = solve_add(x, y, rts, tol)

    post_process_data(eff, x_std, y_std, model=Model.envelopment)

    return eff
