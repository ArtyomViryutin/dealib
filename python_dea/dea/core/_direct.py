__all__ = ["direct"]

import numpy as np
from numpy.typing import NDArray

from python_dea.linprog import LPP, simplex

from .._options import RTS, Model, Orientation
from .._utils import post_process_data, pre_process_data
from .._wrappers import Efficiency
from ._slack import slack


def construct_lpp(
    x: NDArray[float],
    y: NDArray[float],
    orientation: Orientation,
    rts: RTS,
) -> LPP:
    lpp = LPP()
    m, k = x.shape
    n = y.shape[0]

    lpp.c = np.zeros(k + 1)
    lpp.c[-1] = 1
    lpp.b_ub = np.zeros(m + n)
    lpp.A_ub = np.hstack((np.vstack((x, -y)), np.zeros(m + n)[:, np.newaxis]))

    if rts != RTS.crs:
        rts_constraint = np.ones(k + 1)
        rts_constraint[-1] = 0
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


def direct(
    inputs: NDArray[float],
    outputs: NDArray[float],
    direct_: NDArray[float],
    orientation: Orientation = Orientation.input,
    rts: RTS = RTS.vrs,
    two_phase: bool = False,
    transpose: bool = False,
    eps: float = 1e-6,
    tol: float = 1e-9,
) -> Efficiency:
    orientation = Orientation.get(orientation)
    rts = RTS.get(rts)

    x, y, x_std, y_std = pre_process_data(inputs, outputs, transpose, tol)

    if orientation == Orientation.input:
        direct_ = np.divide(direct_, x_std)
    else:
        direct_ = np.divide(direct_, y_std)

    m, k = x.shape
    n = y.shape[0]

    lpp = construct_lpp(x, y, orientation, rts)

    eff = Efficiency(Model.envelopment, orientation, rts, k, m, n)

    eff.direct = np.asarray(direct_)
    for i in range(k):
        lpp.b_ub[:m] = x[:, i]
        lpp.b_ub[m : m + n] = -y[:, i]
        if orientation == Orientation.input:
            lpp.A_ub[:m, -1] = eff.direct
            lpp.A_ub[m : m + n, -1] = 0
        else:
            lpp.A_ub[:m, -1] = 0
            lpp.A_ub[m : m + n, -1] = eff.direct

        if two_phase:
            lpp_result = simplex(lpp, opt_f=True, opt_slacks=False, tol=tol)
        else:
            lpp_result = simplex(
                lpp, opt_f=True, opt_slacks=True, eps=eps, tol=tol
            )
        eff.objval[i] = lpp_result.x[-1]
        eff.lambdas[i, :] = lpp_result.x[:-1]
        eff.slack[i] = lpp_result.slack[: m + n]

    if two_phase:
        eff = slack(x, y, eff, transpose=True, tol=tol)

    m = np.outer(eff.objval, eff.direct)
    if eff.orientation == Orientation.input:
        x_t = x.transpose()
        not_nulls = x_t != 0
        m[not_nulls] /= x_t[not_nulls]
        m[np.logical_not(not_nulls)] = np.inf
        eff.eff = 1 - m
    else:
        y_t = y.transpose()
        not_nulls = y_t != 0
        m[not_nulls] /= y_t[not_nulls]
        m[np.logical_not(not_nulls)] = np.inf
        eff.eff = 1 + m

    if eff.eff.shape[1] == 1:
        eff.eff = eff.eff.flatten()

    post_process_data(eff, x_std, y_std, eff.model)
    return eff
