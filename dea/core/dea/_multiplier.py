__all__ = ["solve_multiplier"]

import numpy as np
from tqdm import trange

from dea.linprog import simplex

from ._options import RTS, Orientation
from ._wrappers import DEAResult


def construct_lpp(
    x: np.ndarray,
    y: np.ndarray,
    orientation: Orientation,  # noqa
    rts: RTS,
    eps: float,
):
    m, k = x.shape
    n = y.shape[0]
    c = np.zeros(m + n)
    A_ub = np.vstack(
        (
            np.hstack((-x.transpose(), y.transpose())),
            -np.eye(m + n),
        )
    )
    b_ub = np.hstack((np.zeros(k), -np.full(m + n, eps)))
    A_eq = np.array([np.zeros(m + n)])
    b_eq = np.ones(1)
    if rts == RTS.irs:
        c = np.append(c, 1)
        A_ub = np.hstack(
            (A_ub, np.hstack((np.ones(k), np.zeros(m + n)))[:, np.newaxis])
        )
        A_eq = np.array([np.append(A_eq[0], 0)])
    elif rts == RTS.drs:
        c = np.append(c, -1)
        A_ub = np.hstack(
            (A_ub, np.hstack((-np.ones(k), np.zeros(m + n)))[:, np.newaxis])
        )
        A_eq = np.array([np.append(A_eq[0], 0)])
    elif rts == RTS.vrs:
        c = np.append(c, [1, -1])
        A_ub = np.hstack(
            (
                A_ub,
                np.hstack((np.ones(k), np.zeros(m + n)))[:, np.newaxis],
                np.hstack((-np.ones(k), np.zeros(m + n)))[:, np.newaxis],
            )
        )
        A_eq = np.array([np.append(A_eq[0], np.zeros(2))])
    return c, A_ub, b_ub, A_eq, b_eq


# def find_inefficient(x: np.ndarray, y: np.ndarray) -> np.ndarray:
#     k = x.shape[1]
#     inefficient_dmu = []
#     for i in range(k):
#         for j in range(k):
#             if np.all(x[:, i] > x[:, j]) and np.all(y[:, i] < y[:, j]):
#                 inefficient_dmu.append(i)
#                 break
#     return np.asarray(inefficient_dmu, dtype=int)


def solve_multiplier(
    x: np.ndarray,
    y: np.ndarray,
    orientation: Orientation,
    rts: RTS,
    eps: float,
    tol: float,
):
    m, k = x.shape
    n = y.shape[0]

    c, A_ub, b_ub, A_eq, b_eq = construct_lpp(x, y, orientation, rts, eps=eps)
    # inefficient_dmu = find_inefficient(x, y)
    #
    # efficient_dmu = np.ones(k, dtype=bool)
    # efficient_dmu[inefficient_dmu] = False
    # A_ub = np.delete(A_ub, inefficient_dmu, axis=0)
    # b_ub = np.delete(b_ub, inefficient_dmu)
    # current_efficient_count = 0
    # current_inefficient_count = inefficient_dmu.size

    efficiency = np.zeros(k)
    lambdas = np.zeros((k, m + n))
    slack = np.zeros((k, k))

    for i in trange(k, desc=f"Computing {orientation}-{rts} multiplier model"):
        if orientation == Orientation.input:
            c[m : m + n] = y[:, i]
            A_eq[0][:m] = x[:, i]
        else:
            c[:m] = -x[:, i]
            A_eq[0][m : m + n] = y[:, i]
        lpp_result = simplex(
            c,
            A_ub,
            b_ub,
            A_eq,
            b_eq,
            opt_f=True,
            opt_slacks=False,
            eps=eps,
            tol=tol,
        )
        efficiency[i] = abs(lpp_result.f)
        lambdas[i] = lpp_result.x[: m + n]
        slack[i] = lpp_result.slack[:k]
        # slack[i, efficient_dmu] = lpp_result.slack[:k - current_inefficient_count]
        #
        # if efficient_dmu[i]:
        #     if np.abs(1 - efficiency[i]) > tol:
        #         efficient_dmu[i] = False
        #         A_ub = np.delete(A_ub, current_efficient_count, axis=0)
        #         b_ub = np.delete(b_ub, current_efficient_count)
        #         current_inefficient_count += 1
        #     else:
        #         current_efficient_count += 1

    return DEAResult(efficiency, lambdas, slack)
