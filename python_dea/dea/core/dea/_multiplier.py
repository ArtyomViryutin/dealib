__all__ = ["solve_multiplier"]

import numpy as np
from numpy.typing import NDArray
from tqdm import trange

from python_dea.dea._options import RTS, Model, Orientation
from python_dea.dea._wrappers import Efficiency
from python_dea.linprog import simplex
from python_dea.linprog.wrappers import LPP


def construct_lpp(
    x: NDArray[float],
    y: NDArray[float],
    rts: RTS,
) -> LPP:
    lpp = LPP()
    m, k = x.shape
    n = y.shape[0]
    lpp.c = np.zeros(m + n)
    lpp.A_ub = np.vstack(
        (
            np.hstack((-x.transpose(), y.transpose())),
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


def solve_multiplier(
    x: NDArray[float],
    y: NDArray[float],
    orientation: Orientation,
    rts: RTS,
) -> Efficiency:
    m, k = x.shape
    n = y.shape[0]

    lpp = construct_lpp(x, y, rts)
    # inefficient_dmu = find_inefficient(x, y)
    #
    # efficient_dmu = np.ones(k, dtype=bool)
    # efficient_dmu[inefficient_dmu] = False
    # A_ub = np.delete(A_ub, inefficient_dmu, axis=0)
    # b_ub = np.delete(b_ub, inefficient_dmu)
    # current_efficient_count = 0
    # current_inefficient_count = inefficient_dmu.size

    eff = Efficiency(Model.multiplier, orientation, rts, k, m, n)

    for i in trange(k, desc=f"Computing {orientation}-{rts} multiplier model"):
        if orientation == Orientation.input:
            lpp.c[m : m + n] = y[:, i]
            lpp.A_eq[0][:m] = x[:, i]
        else:
            lpp.c[:m] = -x[:, i]
            lpp.A_eq[0][m : m + n] = y[:, i]
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
