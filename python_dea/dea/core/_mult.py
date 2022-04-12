__all__ = ["mult"]

from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from python_dea.dea._options import RTS, Orientation
from python_dea.dea._wrappers import Efficiency
from python_dea.linprog import simplex
from python_dea.linprog.wrappers import LPP


def construct_lpp(
    xref: NDArray[float],
    yref: NDArray[float],
    rts: RTS,
) -> LPP:
    lpp = LPP()
    k, m = xref.shape
    n = yref.shape[1]
    lpp.c = np.zeros(m + n)
    lpp.A_ub = np.vstack(
        (
            np.hstack((-xref, yref)),
            -np.eye(m + n),
        )
    )
    lpp.b_ub = np.hstack((np.zeros(k), -np.full(m + n, 1e-6)))
    lpp.A_eq = np.array([np.zeros(m + n)])
    lpp.b_eq = np.ones(1)
    if rts == RTS.irs:
        lpp.c = np.append(lpp.c, 1)
        lpp.A_ub = np.hstack(
            (lpp.A_ub, np.hstack((np.ones(k), np.zeros(m + n)))[:, np.newaxis])
        )
        lpp.A_eq = np.array([np.append(lpp.A_eq[0], 0)])
    elif rts == RTS.drs:
        lpp.c = np.append(lpp.c, -1)
        lpp.A_ub = np.hstack(
            (
                lpp.A_ub,
                np.hstack((-np.ones(k), np.zeros(m + n)))[:, np.newaxis],
            )
        )
        lpp.A_eq = np.array([np.append(lpp.A_eq[0], 0)])
    elif rts == RTS.vrs:
        lpp.c = np.append(lpp.c, [1, -1])
        lpp.A_ub = np.hstack(
            (
                lpp.A_ub,
                np.hstack((np.ones(k), np.zeros(m + n)))[:, np.newaxis],
                np.hstack((-np.ones(k), np.zeros(m + n)))[:, np.newaxis],
            )
        )
        lpp.A_eq = np.array([np.append(lpp.A_eq[0], np.zeros(2))])
    return lpp


# def find_inefficient(x: NDArray[float], y: NDArray[float]) -> NDArray[float]:
#     k = x.shape[1]
#     inefficient_dmu = []
#     for i in range(k):
#         for j in range(k):
#             if np.all(x[:, i] > x[:, j]) and np.all(y[:, i] < y[:, j]):
#                 inefficient_dmu.append(i)
#                 break
#     return np.asarray(inefficient_dmu, dtype=int)


def solve_mult(
    x: NDArray[float],
    y: NDArray[float],
    orientation: Orientation,
    rts: RTS,
) -> Efficiency:
    k, m = x.shape
    n = y.shape[1]

    lpp = construct_lpp(x, y, rts)
    # inefficient_dmu = find_inefficient(x, y)
    #
    # efficient_dmu = np.ones(k, dtype=bool)
    # efficient_dmu[inefficient_dmu] = False
    # A_ub = np.delete(A_ub, inefficient_dmu, axis=0)
    # b_ub = np.delete(b_ub, inefficient_dmu)
    # current_efficient_count = 0
    # current_inefficient_count = inefficient_dmu.size

    eff = Efficiency(rts, orientation, k, m, n, dual=True)

    for i in range(k):
        if orientation == Orientation.input:
            lpp.c[m : m + n] = y[i, :]
            lpp.A_eq[0][:m] = x[i, :]
        else:
            lpp.c[:m] = -x[i, :]
            lpp.A_eq[0][m : m + n] = y[i, :]
        lpp_result = simplex(
            lpp,
            opt_f=True,
            opt_slacks=False,
        )
        eff.eff[i] = abs(lpp_result.f)
        eff.lambdas[i] = lpp_result.x[: m + n]
        eff.slack[i] = lpp_result.slack[:k]

        # slack[i, efficient_dmu] = lpp_result.slack[:k - current_inefficient_count]
        #
        # if efficient_dmu[i]:
        #     if np.abs(1 - efficiency[i]) > 1e-9:
        #         efficient_dmu[i] = False
        #         A_ub = np.delete(A_ub, current_efficient_count, axis=0)
        #         b_ub = np.delete(b_ub, current_efficient_count)
        #         current_inefficient_count += 1
        #     else:
        #         current_efficient_count += 1
    eff.objval = eff.eff
    return eff


def mult(
    x: Union[ArrayLike, NDArray[float]],
    y: Union[ArrayLike, NDArray[float]],
    rts: RTS = RTS.vrs,
    orientation: Orientation = Orientation.input,
    transpose: bool = False,
) -> Efficiency:
    rts = RTS.get(rts)
    orientation = Orientation.get(orientation)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if transpose:
        x = x.transpose()
        y = y.transpose()

    if x.shape[0] != y.shape[0]:
        raise ValueError("Number of units must be the same in 'x' and 'y'")

    xm, ym = x.mean(axis=0), y.mean(axis=0)
    if min(xm) < 1e-4 or max(xm) > 10000 or min(ym) < 1e-4 or max(ym) > 10000:
        scaling = True
        xx, yy = x.std(axis=0), y.std(axis=0)
        xx[xx < 1e-9] = 1
        yy[yy < 1e-9] = 1
        x = np.divide(x, xx)
        y = np.divide(y, yy)
    else:
        scaling = False
        xx = yy = None

    eff = solve_mult(x, y, orientation, rts)

    if scaling:
        eff.lambdas = np.divide(eff.lambdas, np.hstack((xx, yy)))

    eps = 1e-6
    eff.lambdas[np.abs(eff.lambdas) < eps] = 0
    eff.lambdas[np.abs(eff.lambdas - 1) < eps] = 1

    eff.slack[np.abs(eff.slack) < eps] = 0

    eff.eff = eff.objval.copy()
    eff.eff[np.abs(eff.eff) < eps] = 0
    eff.eff[np.abs(eff.eff - 1) < eps] = 1

    return eff
