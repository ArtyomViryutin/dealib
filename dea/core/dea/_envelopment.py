__all__ = ["solve_envelopment"]

from typing import List, Optional

import numpy as np
from numpy.typing import NDArray
from tqdm import trange

from dea.linprog import simplex

from ._options import RTS, Orientation
from ._wrappers import DEAResult


def construct_lpp(
    x: NDArray[float],
    y: NDArray[float],
    orientation: Orientation,
    rts: RTS,
):
    m, k = x.shape
    n = y.shape[0]

    c = np.zeros(k + 1)
    c[-1] = 1
    b_ub = np.zeros(m + n)
    A_ub = np.hstack((np.vstack((x, y)), np.zeros(m + n)[:, np.newaxis]))

    A_eq = None
    b_eq = None
    if orientation == Orientation.input:
        c[-1] = -1
        A_ub[m:] *= -1
    else:
        A_ub[m:, :-1] *= -1
    if rts != RTS.crs:
        rts_constraint = np.ones(k + 1)
        rts_constraint[-1] = 0
        if rts == RTS.vrs:
            A_eq = np.array([rts_constraint])
            b_eq = np.ones(1)
        elif rts == RTS.drs:
            A_ub = np.vstack((A_ub, rts_constraint))
            b_ub = np.append(b_ub, [1])
        elif rts == rts.irs:
            A_ub = np.vstack((A_ub, -rts_constraint))
            b_ub = np.append(b_ub, [-1])
    return c, A_ub, b_ub, A_eq, b_eq


def find_inefficient_dmu(x: NDArray[float], y: NDArray[float]) -> List[int]:
    k = x.shape[1]
    inefficient_dmu = []
    for i in range(k):
        for j in range(k):
            if np.logical_and(x[:, i] > x[:, j], y[:, i] < y[:, j]).all():
                inefficient_dmu.append(i)
                break
    return inefficient_dmu


def solve_one_phase(
    c: NDArray[float],
    A_ub: NDArray[float],
    b_ub: NDArray[float],
    A_eq: Optional[NDArray[float]],
    b_eq: Optional[NDArray[float]],
    eps: float,
    tol: float,
):
    return simplex(
        c, A_ub, b_ub, A_eq, b_eq, opt_f=True, opt_slacks=True, eps=eps, tol=tol
    )


def solve_two_phase(
    c: NDArray[float],
    A_ub: NDArray[float],
    b_ub: NDArray[float],
    A_eq: Optional[NDArray[float]],
    b_eq: Optional[NDArray[float]],
    orientation: Orientation,
    m: int,
    n: int,
    eps: float,
    tol: float,
):
    lpp_result = simplex(
        c, A_ub, b_ub, A_eq, b_eq, opt_f=True, opt_slacks=False, eps=eps, tol=tol
    )
    e = lpp_result.x[-1]
    if orientation == Orientation.input:
        b_ub[:m] = -A_ub[:m, -1] * e
        A_ub[:m, -1] = 0
    else:
        b_ub[m : m + n] = A_ub[m : m + n, -1] * e
        A_ub[m : m + n, -1] = 0
    lpp_result = simplex(
        c, A_ub, b_ub, A_eq, b_eq, opt_f=False, opt_slacks=True, eps=eps, tol=tol
    )
    lpp_result.x[-1] = e
    return lpp_result


def solve_envelopment(
    x: NDArray[float],
    y: NDArray[float],
    orientation: Orientation,
    rts: RTS,
    two_phase: bool,
    eps: float,
    tol: float,
) -> DEAResult:
    m, k = x.shape
    n = y.shape[0]

    c, A_ub, b_ub, A_eq, b_eq = construct_lpp(x, y, orientation, rts)
    inefficient_dmu = find_inefficient_dmu(x, y)

    eff_dmu = np.ones(k, dtype=bool)
    eff_dmu[inefficient_dmu] = False
    c = np.delete(c, inefficient_dmu)
    A_ub = np.delete(A_ub, inefficient_dmu, axis=1)
    if A_eq is not None:
        A_eq = np.delete(A_eq, inefficient_dmu, axis=1)

    eff_dmu_count = 0
    efficiency = np.zeros(k)
    lambdas = np.zeros((k, k))
    slack = np.zeros((k, m + n))
    for i in trange(k, desc=f"Computing {orientation}-{rts} envelopment model"):
        if orientation == Orientation.input:
            A_ub[:m, -1] = -x[:, i]
            A_ub[m : m + n, -1] = 0
            b_ub[:m] = 0
            b_ub[m : m + n] = -y[:, i]
        else:
            A_ub[:m, -1] = 0
            A_ub[m : m + n, -1] = y[:, i]
            b_ub[:m] = x[:, i]
            b_ub[m : m + n] = 0
        if two_phase:
            lpp_result = solve_two_phase(
                c, A_ub, b_ub, A_eq, b_eq, orientation, m, n, eps=eps, tol=tol
            )
        else:
            lpp_result = solve_one_phase(c, A_ub, b_ub, A_eq, b_eq, eps=eps, tol=tol)
        efficiency[i] = lpp_result.x[-1]
        lambdas[i, eff_dmu] = lpp_result.x[:-1]
        slack[i] = lpp_result.slack[: m + n]
        slack[i] = lpp_result.slack[: m + n]

        if eff_dmu[i]:
            if np.abs(1 - efficiency[i]) > 1e-9:
                eff_dmu[i] = False
                c = np.delete(c, eff_dmu_count)
                A_ub = np.delete(A_ub, eff_dmu_count, axis=1)
                if A_eq is not None:
                    A_eq = np.delete(A_eq, eff_dmu_count, axis=1)
            else:
                eff_dmu_count += 1
    return DEAResult(efficiency, lambdas, slack)
